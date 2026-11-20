"""Offline initialization tests: memory registers only; never opens a port."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import apply_policy as policy
from bus import Bus
from core import Fault


class MemoryBus:
    def __init__(self, target=2000):
        self.targets={i:target for i in range(1,5)}
        self.positions={i:200 for i in self.targets}
        self.reg={(i,a):v for i in self.targets for a,v in ((64,0),(102,1000),(116,3500),(98,0))}
        self.armed_ids=[]
        self.events=[]
        self.bad_goal=None
        self.enable_changes_current=None
        self.before_sync=None
        self.reaches_goal=True
        self.prepare_hold=Mock(side_effect=AssertionError('must not prepare a present-position hold'))

    def read_int(self,i,addr,size):
        return self.reg[(i,addr)]

    def write_checked(self,i,addr,value,size):
        # Applying drive current is the point at which a stale goal could move
        # the hand. Every motor must already have its verified open goal then.
        if addr==102 and value:
            assert all(self.reg[j,116]==self.targets[j] for j in self.targets)
            assert all(self.reg[j,64]==1 for j in self.targets)
        self.events.append(('write',i,addr,value))
        self.reg[i,addr]=value
        if addr==64 and value==1 and i==self.enable_changes_current:
            self.reg[i,102]=1000

    def read_diagnostic(self):
        return [dict(id=i,position_tick=self.positions[i],velocity_tick_s=0,current_ma=0,
                     temperature_c=25,sample_start_s=time.monotonic(),
                     goal_position_tick=self.reg[i,116]) for i in self.targets]

    def read_register_sync(self,addr,size):
        if addr==70:return dict.fromkeys(self.targets,0)
        return {i:self.reg[i,addr] for i in self.targets}

    def command_sync(self,targets,rows,max_age):
        if self.before_sync:self.before_sync()
        self.events.append(('sync',dict(targets),{i:self.reg[i,102] for i in self.targets}))
        if len([e for e in self.events if e[0]=='sync'])==1:
            assert all(self.reg[i,102]==0 for i in self.targets)
        assert all(self.reg[i,64]==1 for i in self.targets)
        self.reg.update({(i,116):v for i,v in targets.items()})
        if self.bad_goal is not None:
            self.reg[1,116]+=self.bad_goal
            raise Fault('Sync goal readback mismatch')
        for i,v in targets.items():
            if self.reg[i,102] and self.reaches_goal:self.positions[i]=v


class InitialSetupTests(unittest.TestCase):
    def setUp(self):
        self.bus=MemoryBus()
        self.cfg={'motors':[dict(id=i,initial_position_tick=2000,provisional_close_tick=100)
                            for i in range(1,5)]}
        self.inv=[dict(id=i,temperature_limit_c=70,current_limit_ma=1000,
                       position_p_gain=850,position_i_gain=0,
                       profile_velocity_raw=80,profile_acceleration_raw=40)
                  for i in range(1,5)]
        self.args=SimpleNamespace(position_p=1000,profile_velocity=80,profile_acceleration=40)
        self.events=[]
        self.now=0.

    def run_setup(self,check=lambda:None,emit=None):
        rows=self.bus.read_diagnostic()
        guard=policy.Guard(self.cfg,rows,self.inv,initializing=True)
        real_move=policy.move_to_initial
        def sleep(seconds):self.now+=seconds
        def move(bus,cfg,guard,emit,check):
            return real_move(bus,cfg,guard,emit,clock=lambda:self.now,sleep=sleep,check=check)
        with patch('apply_policy.move_to_initial',side_effect=move):
            return policy.initial_setup(self.bus,self.cfg,self.inv,rows,guard,self.args,
                                        0,{},{},{},emit or self.events.append,check)

    def test_large_start_and_old_goal_differences_move_to_open(self):
        # Start is 1800 ticks from open and old goal is 3300 from present.
        guard=self.run_setup()
        self.assertFalse(guard.initializing)
        self.assertEqual(self.bus.positions,self.bus.targets)
        self.bus.prepare_hold.assert_not_called()
        self.assertTrue(any(e['type']=='initial_pose_reached' for e in self.events))

    def test_start_below_close_and_above_open_can_initialize(self):
        self.bus.positions[4]=50  # Below the provisional close tick of 100.
        self.bus.positions[3]=3000  # Above the provisional open tick of 2000.
        guard=self.run_setup()
        self.assertEqual(self.bus.positions,self.bus.targets)
        self.assertFalse(guard.initializing)
        self.assertTrue(any(e['type']=='initial_pose_reached' for e in self.events))

    def test_after_initialization_position_limit_still_applies(self):
        guard=self.run_setup()
        self.bus.positions[4]=95  # More than three ticks below close.
        with self.assertRaisesRegex(Fault,r'ID4: position limit: actual=95, allowed=97\.\.2033'):
            guard.check(self.bus.read_diagnostic())

    def test_initialization_keeps_current_and_temperature_checks(self):
        rows=self.bus.read_diagnostic()
        guard=policy.Guard(self.cfg,rows,self.inv,initializing=True)
        rows[3]['current_ma']=1001
        with self.assertRaisesRegex(Fault,'initialization current'):
            guard.check(rows)
        rows=self.bus.read_diagnostic()
        guard=policy.Guard(self.cfg,rows,self.inv,initializing=True)
        rows[3]['temperature_c']=69
        with self.assertRaisesRegex(Fault,'temperature'):
            guard.check(rows)

    def test_open_goals_confirmed_at_zero_current_before_any_drive_current(self):
        self.run_setup()
        sync=next(e for e in self.bus.events if e[0]=='sync')
        self.assertEqual(sync[1],self.bus.targets)
        self.assertEqual(sync[2],dict.fromkeys(range(1,5),0))
        self.assertFalse(any(e[0]=='write' and e[2]==116 for e in self.bus.events))
        self.assertTrue(all(self.bus.reg[i,102]==1000 for i in range(1,5)))

    def test_goal_write_failure_keeps_all_current_limits_zero(self):
        for delta in (1,1000):
            with self.subTest(delta=delta):
                self.bus=MemoryBus()
                self.bus.bad_goal=delta
                with self.assertRaisesRegex(Fault,'Sync goal readback mismatch'):
                    self.run_setup()
                self.assertTrue(all(self.bus.reg[i,102]==0 for i in range(1,5)))

    def test_current_limit_change_on_enable_stops_before_goal_or_current_release(self):
        self.bus.enable_changes_current=1
        with self.assertRaisesRegex(Fault,'current limit changed during enable'):
            self.run_setup()
        self.assertFalse(any(e[0]=='sync' for e in self.bus.events))
        self.assertFalse(any(e[0]=='write' and e[2]==102 and e[3] for e in self.bus.events))

    def test_cancel_after_goal_confirmation_does_not_release_current(self):
        cancel=[False]
        def emit(event):
            if event['type']=='initial_open_prepared':cancel[0]=True
        def check():
            if cancel[0]:raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):self.run_setup(check=check,emit=emit)
        self.assertTrue(all(self.bus.reg[i,102]==0 for i in range(1,5)))

    def test_cancel_during_initial_move_propagates(self):
        cancel=[False]
        def emit(event):
            if event['type']=='initial_move_state':cancel[0]=True
        def check():
            if cancel[0]:raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):self.run_setup(check=check,emit=emit)
        self.assertFalse(any(e.get('type')=='initial_pose_reached' for e in self.events))

    def test_out_of_range_initial_target_never_enables(self):
        self.cfg['motors'][0]['initial_position_tick']=4096
        with self.assertRaisesRegex(Fault,'Invalid initial open targets'):
            policy.initial_setup(self.bus,self.cfg,self.inv,self.bus.read_diagnostic(),
                                 Mock(),self.args,0,{},{},{},self.events.append)
        self.assertEqual(self.bus.events,[])

    def test_initial_pose_timeout_is_still_a_failure(self):
        self.bus.reaches_goal=False
        with self.assertRaisesRegex(Fault,'Initial position not reached'):
            self.run_setup()
        self.assertFalse(any(e.get('type')=='initial_pose_reached' for e in self.events))

    def test_normal_write_readback_still_requires_exact_value(self):
        bus=Bus.__new__(Bus)
        bus.port=Mock()
        bus.read_int=Mock(return_value=555)
        with patch('bus.dxl_write'):
            with self.assertRaisesRegex(Fault,'expected=556, actual=555'):
                bus.write_checked(1,116,556,4)


if __name__=='__main__':unittest.main()
