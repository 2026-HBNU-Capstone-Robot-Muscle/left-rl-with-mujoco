"""Motor-free checks for the explicit, unloaded wrist return."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from core import Fault
from primitive_hardware import Hardware


class WristHomeTests(unittest.TestCase):
    def hardware(self):
        h=Hardware.__new__(Hardware)
        h.bus=None
        h.cfg={'port':'unused','baud':1000000}
        h.c={'reference_tick':2400,'tolerance':10}
        return h

    def test_return_uses_current_hold_then_reference_and_turns_off(self):
        h=self.hardware()
        bus=Mock()
        bus.read_register_sync.return_value={1:0,2:0,3:0,4:0}
        h._pos=Mock(side_effect=[3000,2400,2400,2400,2400])
        h._goal=Mock()
        torque=[0]
        h._read=Mock(side_effect=lambda addr,size:torque[0])
        def write(port,motor,addr,data,**kwargs):
            if addr==40:torque[0]=data[0]
        clock=[0.]
        def monotonic():
            clock[0]+=.1
            return clock[0]
        with patch('primitive_hardware.Bus',return_value=bus), \
             patch('primitive_hardware.sts_write',side_effect=write) as writes, \
             patch('primitive_hardware.time.monotonic',side_effect=monotonic), \
             patch('primitive_hardware.time.sleep'), \
             contextlib.redirect_stdout(io.StringIO()):
            h.return_wrist_to_reference()
        self.assertEqual([call.args[0] for call in h._goal.call_args_list],[3000,2400])
        self.assertEqual([call.args[3] for call in writes.call_args_list],[b'\x01',b'\x00'])
        self.assertIsNone(h.bus)
        bus.close.assert_called_once()

    def test_finger_torque_on_blocks_wrist_motion(self):
        h=self.hardware()
        bus=Mock()
        bus.read_register_sync.return_value={1:0,2:1,3:0,4:0}
        h._read=Mock(return_value=0)
        h._goal=Mock()
        with patch('primitive_hardware.Bus',return_value=bus), \
             patch('primitive_hardware.sts_write') as write, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Fault,'finger torque must be OFF'):
                h.return_wrist_to_reference()
        h._goal.assert_not_called()
        self.assertEqual([call.args[3] for call in write.call_args_list],[b'\x00'])
        self.assertIsNone(h.bus)
        bus.close.assert_called_once()


    def test_release_returns_on_live_connection_before_wrist_off(self):
        h = self.hardware()
        bus = h.bus = Mock()
        bus.armed_ids = [1, 2, 3, 4]
        bus.read_register_sync.return_value = dict.fromkeys(bus.armed_ids, 0)
        bus.verify_torque_off.return_value = []
        h.current_restore = {}
        h.profiles = h.gains = h.igains = {}
        h._pos = Mock(side_effect=[3538, 2800, 2400, 2400, 2400])
        operations = Mock()
        operations.attach_mock(bus.send_torque_off, 'finger_stop')
        operations.attach_mock(bus.read_register_sync, 'finger_read')
        h._goal = operations.wrist_goal
        clock = iter(range(100))
        torque = 1

        def write(port, motor, address, data):
            nonlocal torque
            operations.wrist_write(address, data)
            torque = data[0]

        h._read = lambda address, size: torque
        with patch('primitive_hardware.Bus') as new_bus, \
             patch('primitive_hardware.sts_write', side_effect=write), \
             patch('primitive_hardware.time.monotonic', side_effect=lambda: next(clock) * .2), \
             patch('primitive_hardware.time.sleep'):
            h.finish_release(check=lambda: None)

        new_bus.assert_not_called()
        from unittest.mock import call
        self.assertEqual(operations.mock_calls, [
            call.finger_stop(), call.finger_read(64, 1), call.wrist_goal(2400),
            call.finger_read(64, 1), call.finger_stop(), call.wrist_write(40, b'\x00'),
        ])
        self.assertIsNone(h.bus)
        bus.close.assert_called_once()

    def test_release_with_unconfirmed_finger_stop_does_not_move_wrist(self):
        h = self.hardware()
        h.bus = Mock()
        h.bus.armed_ids = [1, 2, 3, 4]
        h.bus.read_register_sync.return_value = {1: 0, 2: 1, 3: 0, 4: 0}
        h._goal = Mock()
        with self.assertRaisesRegex(Fault, 'finger torque must be OFF'):
            h.finish_release(check=lambda: None)
        h._goal.assert_not_called()
        h.bus.write_checked.assert_called_once_with(2, 64, 0, 1)

    def test_release_retries_only_enabled_finger_before_wrist_goal(self):
        from unittest.mock import call

        h = self.hardware()
        h.bus = Mock()
        h.bus.armed_ids = [1, 2, 3, 4]
        h.bus.read_register_sync.side_effect = [{1: 0, 2: 1, 3: 0, 4: 0},
                                                {1: 0, 2: 0, 3: 0, 4: 0}]
        h._read = Mock(return_value=1)
        h._pos = Mock(return_value=238)
        operations = Mock()
        operations.attach_mock(h.bus.write_checked, 'finger_write')
        operations.attach_mock(h.bus.read_register_sync, 'finger_read')
        h._move_released_wrist = operations.move_wrist
        h.close = operations.close
        check = Mock()

        h.finish_release(check=check)

        self.assertEqual(operations.mock_calls, [
            call.finger_read(64, 1), call.finger_write(2, 64, 0, 1),
            call.finger_read(64, 1), call.move_wrist(2400, timeout=15., check=check),
            call.close(retry_wrist_off=True),
        ])

    def test_release_stop_retries_wrist_once_and_requires_zero_readback(self):
        for final_torque in (0, 1):
            with self.subTest(final_torque=final_torque):
                h = self.hardware()
                bus = h.bus = Mock()
                bus.armed_ids = []
                bus.verify_torque_off.return_value = []
                h._read = Mock(side_effect=[1, final_torque])
                with patch('primitive_hardware.sts_write') as write:
                    if final_torque:
                        with self.assertRaisesRegex(Fault, 'wrist torque remains enabled'):
                            h.close(retry_wrist_off=True)
                    else:
                        h.close(retry_wrist_off=True)
                self.assertEqual([call.args[3] for call in write.call_args_list], [b'\x00', b'\x00'])
                self.assertEqual(bus.wait_for_quiet_input.call_count, 2)


if __name__=='__main__':unittest.main()
