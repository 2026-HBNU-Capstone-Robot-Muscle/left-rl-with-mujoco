"""Offline only: all hardware access replaced by memory or mocks."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from primitive_worker import Worker, Rejected
from primitive_hardware import Hardware, clear_stopped_watchdogs


class Fake:
    def __init__(self):
        self.grip=True; self.open=True; self.at=False; self.turns=[]; self.calls=[]
        self.stopped=False
    def start(self,check): check()
    def tick(self,release,dt):
        self.calls.append((threading.get_ident(),release))
        return self.grip,self.open,self.at
    def rotate(self,n): self.turns.append(n)
    def finish_release(self, *, check):
        check()
        self.close()
    def close(self): self.stopped=True


class Tests(unittest.TestCase):
    def setUp(self):
        self.f=Fake()
        self.w=Worker(self.f,period=.005,settle=.015,report=lambda _:None)
        self.addCleanup(self.w.close)
    def send(self,n,t=0,timeout=1):
        return self.w.submit(n+'_upper_layer',t,'session',timeout)
    def end(self,j):
        self.assertTrue(j.done.wait(2))
        return j.result
    def grasp(self): self.assertTrue(self.end(self.send('grasp'))[0])
    def test_hold_and_rotate_then_open(self):
        self.grasp()
        ticks=len(self.f.calls); time.sleep(.04)
        self.assertGreater(len(self.f.calls),ticks)
        j=self.send('rotate',-1);time.sleep(.04)
        self.assertFalse(j.done.is_set())
        self.assertGreater(len(self.f.calls),ticks+2)
        self.assertFalse(any(release for _,release in self.f.calls))
        self.f.at=True;self.assertTrue(self.end(j)[0])
        self.assertEqual(self.f.turns,[-1])
        self.assertTrue(self.end(self.send('release'))[0])
        self.assertTrue(self.f.stopped)
        self.assertEqual(self.w.state,'OPEN')
        self.assertEqual(len(set(t for t,_ in self.f.calls)),1)
    def test_busy_wrong_order_and_cancel(self):
        with self.assertRaises(Rejected): self.send('rotate',1)
        self.f.grip=False;j=self.send('grasp')
        with self.assertRaises(Rejected): self.send('grasp')
        j.cancel.set();self.assertEqual(self.end(j)[1],'CANCELLED')
        self.assertTrue(self.f.stopped)
    def test_missing_confirmation_times_out(self):
        self.f.grip=False
        self.assertEqual(self.end(self.send('grasp',timeout=.04))[1],'TIMEOUT')
    def test_grip_loss_aborts_rotation(self):
        self.grasp();j=self.send('rotate',2);self.f.grip=False
        self.assertEqual(self.end(j)[1],'HARDWARE_FAULT')
        self.assertTrue(self.f.stopped)
    def test_previous_stop_failure_not_erased_by_later_wrist_check(self):
        self.f.grip=False
        calls=[]
        def close_once_fails():
            calls.append(1)
            if len(calls)==1: raise RuntimeError('finger torque OFF unconfirmed')
        self.f.close=close_once_fails
        job=self.send('grasp',timeout=.04)
        self.assertEqual(self.end(job)[1],'STOP_UNCONFIRMED')
        self.w.thread.join()
        self.assertFalse(self.w.close())
        self.assertEqual(len(calls),2)

    def test_original_failure_remains_visible_when_stop_unconfirmed(self):
        self.f.grip=False
        self.f.close=Mock(side_effect=RuntimeError('wrist confirmation failed'))
        job=self.send('grasp',timeout=.04)
        result=self.end(job)
        self.assertEqual(result[1],'STOP_UNCONFIRMED')
        self.assertIn('original TIMEOUT',result[2])
        self.assertIn('wrist confirmation failed',result[2])

    def test_foreign_execution_denied(self):
        self.grasp()
        with self.assertRaises(Rejected): self.w.submit('release_upper_layer',0,'other',1)


    def test_release_waits_for_adapters_full_finish(self):
        self.grasp()
        entered=threading.Event(); finish=threading.Event()
        def home(*,check):
            self.assertFalse(self.f.stopped)
            entered.set()
            finish.wait(1)
            check()
            self.f.close()
        self.f.finish_release=home
        job=self.send('release')
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(job.done.is_set())
        finally:
            finish.set()
        self.assertTrue(self.end(job)[0])
        self.assertEqual(self.w.state,'OPEN')

    def test_fault_does_not_attempt_home(self):
        self.f.finish_release=Mock()
        self.f.tick=Mock(side_effect=RuntimeError('communication lost'))
        self.assertEqual(self.end(self.send('grasp'))[1],'HARDWARE_FAULT')
        self.f.finish_release.assert_not_called()
        self.assertTrue(self.f.stopped)

    def test_home_failure_does_not_report_success(self):
        self.grasp()
        self.f.finish_release=Mock(side_effect=RuntimeError('home failed'))
        result=self.end(self.send('release'))
        self.assertFalse(result[0])
        self.assertIn('home failed',result[2])
        self.assertEqual(self.w.state,'FAULT')


    def test_recovery_latches_fault_before_return_and_always_stops(self):
        self.grasp()
        trigger=threading.Event(); entered=threading.Event(); finish=threading.Event()
        old_tick=self.f.tick
        def tick(*args):
            if trigger.is_set(): raise RuntimeError('IO deadline exceeded: test')
            return old_tick(*args)
        def recovery(*,check):
            entered.set(); finish.wait(1); check()
        self.f.tick=tick;self.f.recover_fingers_after_fault=recovery
        trigger.set()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(self.w.state,'FAULT')
            with self.assertRaises(Rejected): self.send('rotate',1)
        finally:
            finish.set()
        self.w.thread.join(2)
        self.assertFalse(self.w.thread.is_alive())
        self.assertTrue(self.f.stopped)

    def test_only_empty_header_or_target_range_can_try_recovery(self):
        from bus import BusError
        from primitive_hardware import WristTargetOutOfRange

        cases = (
            (BusError('XL330 ID 4: 응답 헤더를 찾지 못했습니다; received bytes=<none>'), True),
            (WristTargetOutOfRange('target=295'), True),
            (BusError('XL330 ID 4: 응답 헤더를 찾지 못했습니다; received bytes=ff'), False),
            (BusError('XL330 CRC mismatch; received bytes=ff ff'), False),
        )
        for fault, expected in cases:
            with self.subTest(fault=fault):
                hardware = Fake()
                hardware.can_recover_after_fault = Hardware.can_recover_after_fault
                hardware.recover_fingers_after_fault = Mock()
                errors = []
                worker = Worker(hardware, period=.005, settle=.015, report=errors.append)
                try:
                    job = worker.submit('grasp_upper_layer', 0, 'session', 1)
                    self.assertTrue(job.done.wait(2) and job.result[0])
                    hardware.tick = Mock(side_effect=fault)
                    worker.thread.join(2)
                    self.assertFalse(worker.thread.is_alive())
                    self.assertEqual(hardware.recover_fingers_after_fault.call_count, int(expected))
                    self.assertTrue(hardware.stopped)
                    self.assertTrue(errors[0].startswith('HARDWARE_FAULT'))
                finally:
                    worker.close()


class AdapterTests(unittest.TestCase):
    def test_real_model_and_feedback_without_port(self):
        root=Path(__file__).resolve().parent.parent
        cfg=json.loads((root/'config.provisional.json').read_text())
        cal=dict(verified=True,min_tick=50,max_tick=4000,cw_sign=-1,quarter_ticks=1024,tolerance=5,
                 grasp_ticks={str(m['id']):[m['provisional_close_tick'],m['initial_position_tick']-20]
                              for m in cfg['motors']})
        with tempfile.TemporaryDirectory() as d, patch('primitive_hardware.Bus',side_effect=AssertionError):
            p=Path(d)/'mock.json';p.write_text(json.dumps(cal))
            h=Hardware(p)
            from apply_policy import Guard,PositionCompensation
            rows=[dict(id=m['id'],position_tick=m['initial_position_tick'],velocity_tick_s=0,
                       current_ma=0,temperature_c=25,goal_position_tick=m['initial_position_tick'],
                       sample_start_s=time.monotonic()) for m in cfg['motors']]
            h.bus=Mock()
            h.bus.read_diagnostic.return_value=rows
            h.guard=Guard(cfg,rows,[dict(id=m['id'],temperature_limit_c=70,current_limit_ma=700)
                                     for m in cfg['motors']])
            h.comp=PositionCompensation(cfg,100)
            for r in rows:r['sample_start_s']=time.monotonic()
            grip,_,_=h.tick(False,.03)
            self.assertFalse(grip)
            self.assertEqual(set(h.bus.command_sync.call_args.args[0]),{1,2,3,4})
            for r in rows:r['sample_start_s']=time.monotonic()
            h.tick(True,.03)
            expected={m['id']:m['initial_position_tick'] for m in cfg['motors']}
            self.assertEqual(h.bus.command_sync.call_args.args[0],expected)
    def test_provisional_calibration_requires_explicit_opt_in(self):
        root=Path(__file__).resolve().parent.parent
        cfg=json.loads((root/'config.provisional.json').read_text(encoding='utf-8'))
        cal=dict(verified=False,provisional_confirmed_for_test=True,min_tick=352,max_tick=2400,
                 cw_sign=-1,quarter_ticks=1024,tolerance=10,reference_tick=2400,
                 grasp_ticks={str(m['id']):[m['provisional_close_tick'],m['initial_position_tick']-20]
                              for m in cfg['motors']})
        with tempfile.TemporaryDirectory() as d, patch('primitive_hardware.Policy'), patch('primitive_hardware.Bus',side_effect=AssertionError('port must not open')):
            path=Path(d)/'cal.json';path.write_text(json.dumps(cal))
            with self.assertRaisesRegex(ValueError,'explicit provisional'):
                Hardware(path)
            h=Hardware(path,allow_provisional=True)
            self.assertIsNone(h.bus)

    def test_provisional_wrist_bounds_follow_planned_targets_and_limit_motion(self):
        root = Path(__file__).resolve().parent.parent
        cal = json.loads((root/'primitive_calibration.local.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory, patch('primitive_hardware.Policy'), patch('primitive_hardware.Bus') as bus:
            path = Path(directory)/'cal.json'
            path.write_text(json.dumps(cal), encoding='utf-8')
            hardware = Hardware(path, allow_provisional=True)
            self.assertEqual(hardware.wrist_goal_range, (295, 3538))
            self.assertEqual((hardware.c['min_tick'], hardware.c['max_tick']), (265, 3568))
            hardware._read = Mock(return_value=2400)
            hardware._goal = Mock()
            hardware.wrist_armed = True
            hardware.rotate(2)
            self.assertEqual([hardware.wtarget]+hardware.rotation_remaining, [1860, 1319, 807, 295])

            # A 20-tick shift puts the same plan inside the feedback margin,
            # but outside its command range.
            hardware._goal.reset_mock()
            hardware._read.return_value = 2380
            with self.assertRaisesRegex(RuntimeError, 'target=275, allowed=295'):
                hardware.rotate(2)
            hardware._goal.assert_not_called()
            hardware._read.return_value = 265
            self.assertEqual(hardware._pos(), 265)
            hardware._read.return_value = 264
            with self.assertRaisesRegex(RuntimeError, 'outside calibrated range'):
                hardware._pos()

            cal['quarter_ticks'] = 1000
            path.write_text(json.dumps(cal), encoding='utf-8')
            recalculated = Hardware(path, allow_provisional=True)
            self.assertEqual(recalculated.wrist_goal_range, (344, 3511))
            self.assertEqual((recalculated.c['min_tick'], recalculated.c['max_tick']), (314, 3541))

            cal['quarter_ticks'] = 1200
            path.write_text(json.dumps(cal), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'single-turn encoder range'):
                Hardware(path, allow_provisional=True)
            bus.assert_not_called()

    def test_watchdog_reset_requires_off_and_rechecks(self):
        inventory=[dict(id=i,torque=0,watchdog=255,hardware_error=0) for i in range(1,5)]
        cleared=[dict(m,watchdog=0) for m in inventory]
        bus=Mock();bus.inventory.return_value=cleared
        with patch('builtins.print'):
            self.assertEqual(clear_stopped_watchdogs(bus,inventory),cleared)
        self.assertEqual([c.args for c in bus.write_checked.call_args_list],
                         [(i,98,0,1) for i in range(1,5)])
        bus.write_checked.reset_mock()
        inventory[0]['torque']=1
        with self.assertRaisesRegex(Exception,'cannot clear watchdog'):
            clear_stopped_watchdogs(bus,inventory)
        bus.write_checked.assert_not_called()

    def test_startup_rejects_enabled_finger_but_cleanup_still_sends_off(self):
        h=Hardware.__new__(Hardware);h.cfg={'port':'COM8','baud':1000000}
        h.wrist_armed=False;h._read=lambda address,size:0
        bus=Mock();bus.armed_ids=[];bus.verify_torque_off.return_value=[]
        bus.inventory.return_value=[dict(id=i,model_number=1200,mode=5,drive_mode=0,
            torque=int(i==2),watchdog=0,hardware_error=0,secondary_id=255,
            status_return_level=2,temperature_limit_c=70,shutdown_mask=0x24,
            current_limit_ma=1000,goal_current_raw=1000) for i in range(1,5)]
        with patch('primitive_hardware.Bus',return_value=bus), patch('primitive_hardware.sts_write'):
            with self.assertRaisesRegex(Exception,'ID2: startup requires'):
                h.start(lambda:None)
            self.assertEqual(bus.armed_ids,[1,2,3,4])
            h.close()
        bus.send_torque_off.assert_called_once()
        bus.verify_torque_off.assert_called_once()
        self.assertFalse(any(call.args[1]==116 for call in bus.write_checked.call_args_list))

    def test_wrist_outside_range_prevents_arm(self):
        h=Hardware.__new__(Hardware)
        h.cfg={'port':'COM8','baud':1000000}
        h.c={'min_tick':352,'max_tick':2400,'reference_tick':2400,'tolerance':10}
        h.wrist_armed=False
        h._read=lambda addr,n:0
        h._pos=Mock(side_effect=RuntimeError('wrist outside calibrated range'))
        bus=Mock();bus.inventory.return_value=[]
        with patch('primitive_hardware.Bus',return_value=bus), patch('primitive_hardware.preflight'):
            with self.assertRaisesRegex(Exception,'outside calibrated range'):
                h.start(lambda:None)
        bus.read_diagnostic.assert_not_called()
        bus.write_checked.assert_not_called()

    def test_initial_wrist_moves_to_reference_and_monitors_fingers(self):
        h=Hardware.__new__(Hardware);h.c={'min_tick':352,'max_tick':2400,'reference_tick':2400,'tolerance':10}
        h.bus=Mock();h.guard=Mock();h.wrist_armed=False
        positions=iter([1800,2200,2395]+[2400]*20)
        h._pos=lambda:next(positions)
        h._read=lambda addr,n:1
        h._goal=Mock()
        now=[0.]
        def sleep(seconds): now[0]+=seconds
        with patch('primitive_hardware.sts_write') as write:
            h._move_wrist_reference(lambda:None,clock=lambda:now[0],sleep=sleep)
        self.assertEqual(h._goal.call_args_list[0].args,(1800,))
        self.assertEqual(h._goal.call_args_list[1].args,(2400,))
        write.assert_called_once_with(h.bus.port,10,40,b'\x01',expect_reply=True)
        self.assertTrue(h.wrist_armed)
        self.assertEqual(h.wtarget,2400)
        self.assertGreater(h.bus.read_diagnostic.call_count,1)
        self.assertGreaterEqual(now[0],.3)

    def test_initial_wrist_timeout_stops_before_policy(self):
        h=Hardware.__new__(Hardware);h.c={'min_tick':352,'max_tick':2400,'reference_tick':2400,'tolerance':10}
        h.bus=Mock();h.guard=Mock();h.wrist_armed=False;h._pos=lambda:1800;h._read=lambda a,n:1;h._goal=Mock()
        now=[0.]
        def sleep(seconds): now[0]+=seconds
        with patch('primitive_hardware.sts_write'):
            with self.assertRaisesRegex(Exception,'not reached'):
                h._move_wrist_reference(lambda:None,timeout=.1,clock=lambda:now[0],sleep=sleep)
        self.assertTrue(h.wrist_armed)  # caller can turn torque off during fault cleanup

    def test_wrist_position_missing_header_retries_read_once(self):
        from bus import BusError
        h=Hardware.__new__(Hardware);h.bus=Mock();h.bus.port.timeout=.05
        h.c={'min_tick':352,'max_tick':3424}
        h._read=Mock(side_effect=[BusError('응답 헤더를 찾지 못했습니다'),2400])
        with patch('builtins.print'):
            self.assertEqual(h._pos(),2400)
        self.assertEqual(h._read.call_count,2)
        self.assertEqual(h.bus.port.timeout,.05)

    def test_wrist_position_retry_failure_keeps_fault_and_timeout(self):
        from bus import BusError
        h=Hardware.__new__(Hardware);h.bus=Mock();h.bus.port.timeout=.05
        h.c={'min_tick':352,'max_tick':3424}
        h._read=Mock(side_effect=BusError('응답 헤더를 찾지 못했습니다'))
        with self.assertRaisesRegex(BusError,'retry failed'):
            h._pos()
        self.assertEqual(h._read.call_count,2)
        self.assertEqual(h.bus.port.timeout,.05)

    def test_other_wrist_read_error_is_not_retried(self):
        from bus import BusError
        h=Hardware.__new__(Hardware);h.bus=Mock();h.bus.port.timeout=.05
        h.c={'min_tick':352,'max_tick':3424}
        h._read=Mock(side_effect=BusError('STS checksum error'))
        with self.assertRaisesRegex(BusError,'checksum error'):
            h._pos()
        h._read.assert_called_once_with(56,2)

    def test_wrist_range_failure_includes_actual_reading(self):
        h=Hardware.__new__(Hardware);h.c={'min_tick':352,'max_tick':2400}
        h._read=lambda addr,size:2403
        with self.assertRaisesRegex(Exception,r'actual=2403 tick, allowed=352\.\.2400'):
            h._pos()

    def test_user_reference_allows_both_quarters_but_blocks_extended_half(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.wrist_armed=True
        h.c=dict(min_tick=352,max_tick=3538,cw_sign=-1,quarter_ticks=1024,rotation_regrip_max_steps=3)
        h._pos=lambda:2400
        with patch.object(h,'_goal') as write:
            h.rotate(1);write.assert_called_with(1831, speed=550)
            h.rotate(-1);write.assert_called_with(2969, speed=550)
            write.reset_mock()
            with self.assertRaisesRegex(RuntimeError,r'target=295, allowed=352\.\.3538'):
                h.rotate(2)
            write.assert_not_called()

    def test_1124_turn_with_existing_limits(self):
        calibration=dict(min_tick=352,max_tick=3424,cw_sign=-1,quarter_ticks=1124,reference_tick=2400,tolerance=10,rotation_regrip_max_steps=3)
        h=Hardware.__new__(Hardware);h.bus=Mock();h.wrist_armed=True;h.c=calibration
        h._pos=lambda:2400
        with patch.object(h,'_goal') as goal:
            h.rotate(1)
            goal.assert_called_once_with(1776, speed=550)
            goal.reset_mock()
            with self.assertRaisesRegex(Exception,'target=3649'):
                h.rotate(-1)
            with self.assertRaisesRegex(Exception,'target=90'):
                h.rotate(2)
            goal.assert_not_called()

    def test_1124_quarter_targets_in_four_steps_if_range_allows(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.wrist_armed=True
        h.c=dict(min_tick=0,max_tick=4095,cw_sign=-1,quarter_ticks=1124,reference_tick=2400,tolerance=10,rotation_regrip_max_steps=3)
        h._pos=lambda:2400
        with patch.object(h,'_goal') as goal:
            h.rotate(2)
            goal.assert_called_once_with(1807, speed=550)
            self.assertEqual(h.rotation_remaining[-1],90)

    def test_close_turns_off_wrist_even_when_not_armed_by_program(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.bus.armed_ids=[];h.bus.verify_torque_off.return_value=[]
        h.wrist_armed=False
        h._read=lambda address,size:0
        port=h.bus.port;bus=h.bus
        with patch('primitive_hardware.sts_write') as write:
            h.close()
        write.assert_called_once_with(port,10,40,b'\x00')
        bus.close.assert_called_once()
        self.assertIsNone(h.bus)

    def test_close_retries_each_finger_if_broadcast_left_torque_on(self):
        h=Hardware.__new__(Hardware)
        bus=Mock();bus.armed_ids=[1,2,3,4]
        bus.verify_torque_off.side_effect=[['torque remains enabled'],[]]
        bus.read_int.return_value=1
        h.bus=bus;h.wrist_armed=False
        h.current_restore={};h.profiles={};h.gains={};h.igains={}
        h._read=lambda address,size:0
        with patch('primitive_hardware.sts_write'):
            h.close()
        self.assertEqual([c.args for c in bus.write_checked.call_args_list if c.args[1]==64],
                         [(i,64,0,1) for i in range(1,5)])
        self.assertEqual(bus.verify_torque_off.call_count,2)
        bus.close.assert_called_once()

    def test_close_reopens_port_and_verifies_even_without_grasp(self):
        h=Hardware.__new__(Hardware);h.bus=None;h.cfg={'port':'COM8','baud':1000000};h.wrist_armed=False
        bus=Mock();bus.armed_ids=[];bus.verify_torque_off.return_value=[]
        h._read=lambda address,size:0
        with patch('primitive_hardware.Bus',return_value=bus) as factory, patch('primitive_hardware.sts_write') as write:
            h.close()
        factory.assert_called_once_with('COM8',1000000)
        write.assert_called_once_with(bus.port,10,40,b'\x00')
        bus.close.assert_called_once()

    def test_close_reports_unconfirmed_wrist(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.bus.armed_ids=[];h.bus.verify_torque_off.return_value=[];h.wrist_armed=False
        h._read=lambda address,size:1
        with patch('primitive_hardware.sts_write'):
            with self.assertRaisesRegex(Exception,'wrist torque OFF unconfirmed'):h.close()
        self.assertIsNone(h.bus)

    def test_stop_commands_both_types_before_confirmation(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.bus.armed_ids=[]
        h.bus.verify_torque_off.return_value=[];h.wrist_armed=False
        order=[]
        h.bus.send_torque_off.side_effect=lambda:order.append('finger_send')
        h.bus.wait_for_quiet_input.side_effect=lambda:order.append('quiet')
        h._read=lambda address,size:order.append('wrist_read') or 0
        h.bus.verify_torque_off.side_effect=lambda:order.append('finger_verify') or []
        with patch('primitive_hardware.sts_write',side_effect=lambda *a:order.append('wrist_send')):
            h.close()
        self.assertEqual(order,['finger_send','wrist_send','quiet','wrist_read','finger_verify'])

    def test_noisy_line_never_reports_stop_confirmed(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.bus.armed_ids=[]
        h.bus.wait_for_quiet_input.side_effect=RuntimeError('noise')
        h.bus.verify_torque_off.return_value=[];h.wrist_armed=False;h._read=lambda a,n:0
        with patch('primitive_hardware.sts_write'):
            with self.assertRaisesRegex(Exception,'input not quiet'):
                h.close()

    def test_wrist_read_skips_one_empty_write_ack(self):
        from wrist_protocol import sts_read
        port=Mock()
        with patch('wrist_protocol.sts_request',return_value=b'') as req, patch('wrist_protocol.sts_status',return_value=b'\x00') as status:
            self.assertEqual(sts_read(port,10,40,1),b'\x00')
        req.assert_called_once()
        status.assert_called_once_with(port,10)

    def test_wrist_read_rejects_bad_length(self):
        from wrist_protocol import sts_read
        with patch('wrist_protocol.sts_request',return_value=b'\x01\x02'):
            with self.assertRaisesRegex(Exception,'expected 1 bytes, received 2'):
                sts_read(Mock(),10,40,1)

    def test_180_from_near_reference_uses_reference(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.wrist_armed=True
        h.c=dict(min_tick=295,max_tick=3424,cw_sign=-1,quarter_ticks=1024,reference_tick=2400,tolerance=10,rotation_regrip_max_steps=3)
        for actual in (2390,2399,2400,2409,2410):
            h._pos=lambda:actual
            with patch.object(h,'_goal') as goal:
                h.rotate(2)
                goal.assert_called_once_with(1860, speed=550)
                self.assertEqual(h.rotation_remaining[-1],295)

    def test_180_after_quarter_turn_still_checks_actual_range(self):
        h=Hardware.__new__(Hardware);h.bus=Mock();h.wrist_armed=True
        h.c=dict(min_tick=352,max_tick=3424,cw_sign=-1,quarter_ticks=1024,reference_tick=2400,tolerance=10,rotation_regrip_max_steps=3)
        h._pos=lambda:1376
        with patch.object(h,'_goal') as goal:
            with self.assertRaisesRegex(Exception,'actual=1376, base=1376, target=-729'):
                h.rotate(2)
            goal.assert_not_called()

    def test_checksum_error_reports_raw_packet_and_register(self):
        from wrist_protocol import sts_read, sts_packet
        class Port:
            timeout=.05
            def __init__(self):
                packet=bytearray(sts_packet(10,0,b'\x01'));packet[-1]^=1
                self.data=bytes(packet)
            def reset_input_buffer(self): pass
            def write(self,packet): return len(packet)
            def flush(self): pass
            def read(self,n): data,self.data=self.data[:n],self.data[n:];return data
        with self.assertRaisesRegex(Exception,'READ addr=40:.*체크섬 오류.*packet=ff ff 0a'):
            sts_read(Port(),10,40,1)

    def test_incomplete_wrist_response_shows_raw_data(self):
        from wrist_protocol import sts_read
        class Port:
            timeout=.05
            def __init__(self):self.data=b'\xff\xff'
            def reset_input_buffer(self):pass
            def write(self,data):return len(data)
            def flush(self):pass
            def read(self,n):data,self.data=self.data[:n],self.data[n:];return data
        with self.assertRaisesRegex(Exception,'READ addr=42:.*received bytes=ff ff'):
            sts_read(Port(),10,42,2)

    def test_wrist_mapping_limits_and_readback(self):
        h=Hardware.__new__(Hardware)
        h.bus=Mock();h.wrist_armed=True
        h.c=dict(min_tick=0,max_tick=4095,cw_sign=-1,quarter_ticks=1024,rotation_regrip_max_steps=3)
        h._pos=lambda:2400
        with patch('primitive_hardware.sts_write_position') as write:
            for turns in [1,-1,2]:
                first=round(1024*(47.5 if turns==2 else 50)/90)
                goal=2400-(1 if turns>0 else -1)*first;h._read=lambda a,n:goal
                h.rotate(turns)
                write.assert_called_with(h.bus.port,10,goal,550,10)
                travel=round(1024*{1:95,-1:100,2:185}[turns]/90)
                self.assertEqual(h.rotation_remaining[-1],2400-(1 if turns>0 else -1)*travel)
            h._pos=lambda:100
            with self.assertRaisesRegex(Exception,'outside range'): h.rotate(1)
            h._pos=lambda:2400;h._read=lambda a,n:0
            with self.assertRaisesRegex(Exception,'readback'): h.rotate(1)

if __name__=='__main__': unittest.main()
