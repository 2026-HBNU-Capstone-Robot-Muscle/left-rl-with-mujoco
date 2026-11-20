"""Offline segmented rotation and regrip state-machine tests."""
import unittest
from unittest.mock import Mock, patch
from core import Fault
from primitive_hardware import Hardware


class SegmentedRegripTests(unittest.TestCase):
    def setUp(self):
        self.now=0.;self.wrist=2400
        self.positions={i:2000 for i in range(1,5)}
        self.h=Hardware.__new__(Hardware)
        h=self.h
        h.c={'min_tick':352,'max_tick':3538,'cw_sign':-1,'quarter_ticks':1024,
             'reference_tick':2400,'tolerance':10,'rotation_regrip_max_steps':3,
             'grasp_ticks':{str(i):[1950,2050] for i in range(1,5)}}
        h.cfg={'motors':[{'id':i,'initial_position_tick':3000,'provisional_close_tick':2000}
                         for i in range(1,5)],'policy':{'action_smoothing':1.}}
        h.bus=Mock()
        h.bus.read_diagnostic.side_effect=self.rows
        h.bus.confirm_pending.return_value=1.
        h.guard=Mock();h.guard.ends={i:(2000,3000) for i in range(1,5)};h.guard.targets.side_effect=lambda goals:goals
        h.comp=Mock();h.comp.command.side_effect=lambda goals,rows,now,dt:dict(goals)
        h.policy=Mock();h.policy.predict.return_value=[0.,0.,0.,0.]
        h._pos=lambda:self.wrist
        h._goal=Mock()
        h.wrist_armed=True;h.wtarget=2400;h.smooth=None;h.last_goal=None
        h.rotation_phase=None;h.rotation_remaining=[]
        h.regrip_targets=None;h.regrip_lower=None;h.regrip_deadline=None
        self.clock=patch('primitive_hardware.time.monotonic',side_effect=lambda:self.now)
        self.obs=patch('primitive_hardware.observation',return_value=[0.]*8)
        self.action=patch('primitive_hardware.action_targets',return_value={i:2000 for i in range(1,5)})
        self.clock.start();self.obs.start();self.action.start()
        self.addCleanup(self.clock.stop);self.addCleanup(self.obs.stop);self.addCleanup(self.action.stop)

    def rows(self,**kwargs):
        return [dict(id=i,position_tick=self.positions[i],velocity_tick_s=0,current_ma=0,
                     temperature_c=25,sample_start_s=self.now,goal_position_tick=2000)
                for i in range(1,5)]

    def tick(self):return self.h.tick(False,.03)

    def test_90_degree_midpoint_then_final_regrip(self):
        h=self.h;h.rotate(1)
        self.assertEqual([h.wtarget]+h.rotation_remaining,[1831,1319])
        for position, expected in ((2400, 2000), (2258, 1975), (2115, 1950),
                                   (2115, 1950), (2258, 1950), (1973, 1925)):
            self.wrist = position
            self.assertFalse(self.tick()[2])
            self.assertEqual(h.bus.command_sync.call_args.args[0], dict.fromkeys(range(1, 5), expected))
        self.wrist=1831
        self.assertFalse(self.tick()[2])
        self.assertEqual(h.regrip_targets,{i:1900 for i in range(1,5)})
        self.assertEqual(h.bus.command_sync.call_args.args[0],{i:1900 for i in range(1,5)})
        self.assertEqual({i:h.guard.ends[i][0] for i in range(1,5)},
                         {i:1900 for i in range(1,5)})
        self.positions=dict.fromkeys(range(1,5),1900)
        self.assertEqual(h._goal.call_count,1)
        self.now=.03;self.assertFalse(self.tick()[2])
        self.assertEqual([c.args[0] for c in h._goal.call_args_list],[1831,1319])
        self.wrist=1575;self.now=2.5
        self.assertFalse(self.tick()[2])
        self.assertEqual(h.bus.command_sync.call_args.args[0], dict.fromkeys(range(1, 5), 1875))
        self.wrist=1319;self.now=2.97
        self.assertFalse(self.tick()[2])
        self.assertEqual(h.bus.command_sync.call_args.args[0], dict.fromkeys(range(1, 5), 1850))
        self.positions=dict.fromkeys(range(1, 5), 1850)
        self.now=3.
        self.assertFalse(self.tick()[2])
        final_targets=dict.fromkeys(range(1,5),1800)
        self.assertEqual(h.bus.command_sync.call_args.args[0],final_targets)
        self.assertEqual(h.rotation_phase,'final_regrip')
        self.positions=final_targets.copy()
        self.wrist=1280;self.now=3.03
        self.assertFalse(self.tick()[2])
        self.wrist=1319;self.now=3.06
        self.assertTrue(self.tick()[2])
        self.assertEqual(h.regrip_targets,final_targets)
        self.assertIsNone(h.rotation_phase)
        self.now=3.09
        self.assertTrue(self.tick()[2])
        self.assertEqual(h.bus.command_sync.call_args.args[0],final_targets)

    def test_180_degree_ramps_first_three_segments_then_final_regrip(self):
        h=self.h
        h.c['min_tick']=295  # Test fixture permits the full 185-degree travel.
        h.rotate(2)
        targets=[1860,1319,807,295]
        for n, (start, target) in enumerate(zip([2400]+targets, targets)):
            base_goal = 2000 - 75*n
            # Check that each segment has an independent base and never
            # applies another pull while the measured wrist stops or retreats.
            midway = start + (target-start)//2
            self.now = n*3 - .5
            self.wrist = midway
            self.assertFalse(self.tick()[2])
            expected_midway = base_goal - 38 if n < 3 else base_goal
            self.assertEqual(h.bus.command_sync.call_args.args[0],
                             dict.fromkeys(range(1, 5), expected_midway))
            self.wrist = (start + midway)//2
            self.now += .1
            self.assertFalse(self.tick()[2])
            self.assertEqual(h.bus.command_sync.call_args.args[0],
                             dict.fromkeys(range(1, 5), expected_midway))
            self.wrist=target;self.now=n*3
            self.assertFalse(self.tick()[2])
            if n<3:
                expected=2000-75*(n+1)
                self.assertEqual(h.regrip_targets,{i:expected for i in range(1,5)})
                self.assertEqual(h.bus.command_sync.call_args.args[0],{i:expected for i in range(1,5)})
                self.assertEqual({i:h.guard.ends[i][0] for i in range(1,5)},
                                 {i:expected for i in range(1,5)})
                self.positions=dict.fromkeys(range(1,5),expected)
                self.assertEqual(h._goal.call_count, n+1)
                if n == 1:
                    self.now = n*3 + .99
                    self.assertFalse(self.tick()[2])
                    self.assertEqual(h._goal.call_count, n+1)
                    self.now = n*3 + 1
                else:
                    self.now = n*3 + .03
                self.assertFalse(self.tick()[2])
                self.assertEqual(h._goal.call_count, n+2)
        self.assertEqual([c.args[0] for c in h._goal.call_args_list],targets)
        self.assertEqual([c.kwargs for c in h._goal.call_args_list],[dict(speed=550)]*4)
        final_targets=dict.fromkeys(range(1,5),1725)
        self.assertEqual(h.regrip_targets,final_targets)
        self.assertEqual(h.bus.command_sync.call_args.args[0],final_targets)
        self.positions=final_targets.copy()
        self.now+=.03
        self.assertTrue(self.tick()[2])

    def test_180_rejected_if_three_extra_pulls_not_configured(self):
        self.h.c['rotation_regrip_max_steps']=1
        with self.assertRaisesRegex(Fault,'3 regrips required'):
            self.h.rotate(2)
        self.h._goal.assert_not_called()

    def test_counterclockwise_targets_and_bad_range_prevent_any_command(self):
        h=self.h;h.rotate(-1)
        self.assertEqual([h.wtarget]+h.rotation_remaining,[2969,3538])
        self.wrist = 2685
        self.assertFalse(self.tick()[2])
        self.assertEqual(h.bus.command_sync.call_args.args[0], dict.fromkeys(range(1, 5), 1950))
        self.wrist = 2969
        self.tick()
        self.now = .03
        self.tick()
        self.wrist = 3254
        self.tick()
        self.assertEqual(h.bus.command_sync.call_args.args[0], dict.fromkeys(range(1, 5), 1875))
        h._goal.reset_mock();self.wrist=3538
        with self.assertRaisesRegex(Fault,'outside range'):h.rotate(-1)
        h._goal.assert_not_called()

    def test_position_limit_clips_regrip_target(self):
        self.h.cfg['motors'][3]['provisional_close_tick']=2050
        self.h.c['rotation_regrip_max_steps']=1
        self.h.rotate(1);self.wrist=1831;self.tick()
        self.assertEqual(self.h.regrip_targets[4],1950)
        self.assertEqual(self.h.regrip_targets[1],1900)

    def test_grip_loss_aborts_before_next_segment(self):
        self.h.rotate(1);self.wrist=1831;self.tick()
        self.positions[1]=2100;self.now=2
        with self.assertRaisesRegex(Fault,'grasp telemetry'):
            self.tick()
        self.assertEqual(self.h._goal.call_count,1)

    def test_drift_restarts_dwell_only_at_half_turn_halfway_point(self):
        h = self.h
        h.c['min_tick'] = 295
        h.rotate(2)
        self.wrist = 1860
        self.tick()
        self.now = .03
        self.tick()
        self.assertEqual(h.wtarget, 1319)
        self.wrist = 1319
        self.now = 1.
        self.tick()
        self.wrist = 1349
        self.now = 1.7
        self.tick()
        self.wrist = 1319
        self.now = 1.8
        self.tick()
        self.assertEqual(h.regrip_targets, dict.fromkeys(range(1, 5), 1850))
        self.now = 2.79
        self.tick()
        self.assertEqual(h._goal.call_count, 2)
        self.now = 2.8
        self.tick()
        self.assertEqual(h.wtarget, 807)

    def test_release_clears_extra_hold(self):
        self.h.rotate(1);self.wrist=1831;self.tick()
        self.h.tick(True,.03)
        self.assertIsNone(self.h.regrip_targets)
        self.assertIsNone(self.h.regrip_lower)
        self.assertEqual(self.h.bus.command_sync.call_args.args[0],{i:3000 for i in range(1,5)})


if __name__=='__main__':unittest.main()
