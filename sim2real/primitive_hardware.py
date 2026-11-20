"""Hardware adapter. All methods are invoked on the same primitive worker thread."""
import json
import time
from pathlib import Path

from apply_policy import (Bus, Guard, AGE, Policy, PositionCompensation, action_targets,
                          initial_pose_ready, observation, validate, preflight,
                          finite, move_to_initial, initial_setup, set_trial_current, restore_trial_current)
from bus import Fault, BusError
from wrist_protocol import sts_read, sts_write, sts_write_position

WRIST_SPEED = 600
WRIST_ROTATION_SPEED = 550
REGRIP_TICKS = 100
HALF_TURN_REGRIP_TICKS = 75
SECOND_QUARTER_REGRIP_TICKS = 50
FINAL_REGRIP_TICKS = 50
REGRIP_SECONDS = 1.0
WRIST_CUMULATIVE_ANGLES = {-1: (50, 100), 1: (50, 95), 2: (47.5, 95, 140, 185)}


def wrist_segment_targets(calibration, base, turns):
    direction = calibration['cw_sign'] * (1 if turns > 0 else -1)
    quarter_ticks = calibration['quarter_ticks']
    return [
        base + direction * round(quarter_ticks * angle / 90)
        for angle in WRIST_CUMULATIVE_ANGLES[turns]
    ]


class WristTargetOutOfRange(Fault):
    """A proposed wrist target was rejected before sending a motion command."""


def clear_stopped_watchdogs(bus, inventory):
    """Clear a latched watchdog only with torque OFF and no hardware error."""
    stale=[m for m in inventory if m['watchdog']]
    for m in stale:
        if m['torque']!=0 or m['hardware_error']!=0:
            raise Fault(f"ID{m['id']}: cannot clear watchdog with torque={m['torque']}, hardware_error=0x{m['hardware_error']:02X}")
    for m in stale:
        bus.write_checked(m['id'],98,0,1)
    if stale:
        inventory=bus.inventory()
        if any(m['watchdog']!=0 for m in inventory):
            raise Fault('Watchdog reset not confirmed for all fingers')
        print('Finger watchdog reset confirmed: '+', '.join(f"ID{m['id']}" for m in stale),flush=True)
    return inventory


class Hardware:
    def __init__(self, calibration, model=None, *, allow_provisional=False):
        root=Path(__file__).resolve().parent
        self.cfg=json.loads((root/'config.provisional.json').read_text(encoding='utf-8-sig'))
        validate(self.cfg,policy=True,allow_provisional=True)
        self.c=json.loads(Path(calibration).read_text(encoding='utf-8-sig'))
        c=self.c
        if c.get('verified') is not True and not (allow_provisional and c.get('provisional_confirmed_for_test') is True):
            raise ValueError('Verified calibration required; explicit provisional test opt-in is required for provisional data')
        stale=c.get('grasp_windows_need_measurement', [])
        if not isinstance(stale,list) or any(type(i) is not int or i not in (1,2,3,4) for i in stale):
            raise ValueError('Invalid grasp_windows_need_measurement')
        if stale:
            ids=', '.join(str(i) for i in stale)
            raise ValueError(f'파지 위치 미측정: ID {ids}. 풀리/목표 변경 후 실제 파지 위치를 측정해 grasp_ticks를 갱신하세요.')
        if type(c['cw_sign'])!=int or c['cw_sign'] not in (-1,1): raise ValueError('cw_sign')
        if not 0<=c['min_tick']<c['max_tick']<=4095 or not 0<c['quarter_ticks']<=2048:
            raise ValueError('wrist range')
        if not all(type(c[k])==int for k in ('min_tick','max_tick','quarter_ticks','tolerance')):
            raise ValueError('integer wrist calibration required')
        steps=c.get('rotation_regrip_max_steps',1)
        if type(steps) is not int or not 1<=steps<=3:
            raise ValueError('rotation_regrip_max_steps must be 1..3')
        max_regrip = steps * REGRIP_TICKS + SECOND_QUARTER_REGRIP_TICKS + FINAL_REGRIP_TICKS
        if any(m['provisional_close_tick'] < max_regrip for m in self.cfg['motors']):
            raise ValueError('rotation regrip would leave single-turn encoder range')
        if not c['quarter_ticks']<2048 or not 0<c['tolerance']<c['quarter_ticks']/4: raise ValueError('wrist tolerance')
        for m in self.cfg['motors']:
            lo,hi=c['grasp_ticks'][str(m['id'])]
            if not m['provisional_close_tick']-3<=lo<=hi<m['initial_position_tick']-10:
                raise ValueError('grasp window must exclude open pose')
        ref=c.get('reference_tick')
        if ref is not None and (type(ref) is not int or not c['min_tick']<=ref<=c['max_tick']):
            raise ValueError('Invalid wrist reference_tick')
        self.wrist_goal_range = (c['min_tick'], c['max_tick'])
        if 'provisional_rotation_margin_tick' in c:
            margin = c['provisional_rotation_margin_tick']
            if c.get('verified') is True or not allow_provisional:
                raise ValueError('Automatic wrist range requires provisional test calibration')
            if ref is None or type(margin) is not int or not 0 <= margin <= 100:
                raise ValueError('Automatic wrist range requires a reference and a margin of 0..100 ticks')
            planned = [ref]
            for turns, angles in WRIST_CUMULATIVE_ANGLES.items():
                if len(angles) - 1 <= steps:
                    planned.extend(wrist_segment_targets(c, ref, turns))
            low, high = min(planned), max(planned)
            if not 0 <= low - margin < high + margin <= 4095:
                raise ValueError('Wrist targets and margin must fit within the single-turn encoder range')
            self.wrist_goal_range = (low, high)
            c['min_tick'], c['max_tick'] = low - margin, high + margin
            print(f"Provisional wrist range: goals={low}..{high}, "
                  f"feedback={c['min_tick']}..{c['max_tick']} (margin={margin} ticks)", flush=True)
        self.policy=Policy(Path(model) if model else root.parent/'model'/self.cfg['policy']['filename'],self.cfg)
        self.io_phase='constructed'; self.last_rows=None
        self.bus=None; self.wrist_armed=False; self.wtarget=None; self.next_wtarget=None
        self.smooth=None; self.last_goal=None; self.last_feedback=None; self.last_confirmed=False
        self.rotation_phase=None; self.rotation_remaining=[]
        self.regrip_targets=None; self.regrip_lower=None; self.regrip_deadline=None; self.regrip_ramp_start=None

    def start(self,check):
        check()
        cfg=self.cfg
        self.last_rows=None; self.tick_timings={}
        self.io_phase='open_port'
        self.bus=bus=Bus(cfg['port'],cfg['baud'])
        self.gains={}; self.igains={}; self.profiles={}; self.current_restore={}
        # The previous process may have failed to release torque. Own all four
        # IDs before inventory so any startup failure still sends and verifies OFF.
        bus.armed_ids=[1,2,3,4]
        self.io_phase='finger_inventory'
        inv=bus.inventory()
        inv=clear_stopped_watchdogs(bus,inv)
        preflight(inv); check()
        self.io_phase='wrist_preflight'
        if self._read(40,1)!=0: raise Fault('wrist must start torque OFF')
        if self._read(33,1)!=0: raise Fault('STS must be in position mode')
        if self.c.get('reference_tick') is None:
            raise Fault('Wrist reference tick is not configured')
        self._pos()  # Refuse motion if the current position is outside the checked range.
        self.io_phase='finger_preflight'
        rows=bus.read_diagnostic()
        guard=Guard(cfg,rows,inv,initializing=True); guard.check(rows)
        self.policy.predict(observation(cfg,rows,time.monotonic(),allow_provisional=True))
        rows=bus.read_diagnostic(); guard.check(rows); check()
        bus.armed_ids=[1,2,3,4]
        inv=set_trial_current(bus,inv,self.current_restore,lambda record:check())
        rows=bus.read_diagnostic()
        guard=Guard(cfg,rows,inv,initializing=True); guard.check(rows)
        from types import SimpleNamespace
        args=SimpleNamespace(position_p=1000,profile_velocity=80,profile_acceleration=40)
        self.io_phase='finger_initial_setup'
        self.guard=initial_setup(bus,cfg,inv,rows,guard,args,0,self.gains,self.igains,self.profiles,
                                 lambda record:check(),check)
        # Keep the fingers open and monitored while the wrist moves to its reference.
        self.io_phase='wrist_reference'
        self._move_wrist_reference(check)
        self.comp=PositionCompensation(cfg,100)
        self.smooth=None; self.last_goal=None; self.next_wtarget=None
        self.rotation_phase=None; self.rotation_remaining=[]
        self.regrip_targets=None; self.regrip_lower=None; self.regrip_deadline=None; self.regrip_ramp_start=None

    def _move_wrist_reference(self,check,timeout=15.,clock=time.monotonic,sleep=time.sleep):
        reference=self.c.get('reference_tick')
        if type(reference) is not int or not self.c['min_tick']<=reference<=self.c['max_tick']:
            raise Fault('Wrist reference outside calibrated range')
        check()
        position=self._pos()
        # Hold the present position before enabling torque to avoid a jump.
        self.wrist_armed=True  # A goal write itself may enable torque.
        self._goal(position)
        sts_write(self.bus.port,10,40,b'\x01',expect_reply=True)
        if self._read(40,1)!=1: raise Fault('wrist enable unconfirmed')
        check()
        print(f'Moving wrist to reference {reference} tick...',flush=True)
        self._goal(reference)
        self.wtarget=reference
        began=clock(); settled=None
        while clock()-began<timeout:
            check()
            # Finger bus traffic maintains the watchdog and checks errors/current.
            try:
                finger_rows=self.bus.read_diagnostic(combined=True)
            except Exception as exc:
                raise Fault(f'wrist reference: finger sync read failed: {exc}') from exc
            self.guard.check(finger_rows)
            if self._read(40,1)!=1: raise Fault('wrist torque lost during initial move')
            position=self._pos()
            now=clock()
            reached=abs(position-reference)<=self.c['tolerance']
            settled=(now if settled is None else settled) if reached else None
            if settled is not None and now-settled>=.3:
                print(f'Wrist reference reached: {position} tick',flush=True)
                return
            sleep(.03)
        raise Fault('Wrist initial position not reached within 15s; policy was not started')

    def _read(self,addr,n):
        return int.from_bytes(sts_read(self.bus.port,10,addr,n),'little')

    def _pos(self):
        try:
            p=self._read(56,2)
        except BusError as first:
            # One extra READ is allowed for a missing STS status header. The
            # shorter retry keeps the entire control cycle below its deadline.
            if '응답 헤더를 찾지 못했습니다' not in str(first): raise
            port=self.bus.port; original_timeout=port.timeout
            try:
                port.timeout=min(original_timeout,.02)
                p=self._read(56,2)
            except BusError as second:
                raise BusError(f'wrist position read retry failed: {second}') from second
            finally:
                port.timeout=original_timeout
            print('Wrist position read recovered on retry.',flush=True)
        if not self.c['min_tick']<=p<=self.c['max_tick']:
            raise Fault(f"wrist outside calibrated range: actual={p} tick, allowed={self.c['min_tick']}..{self.c['max_tick']}")
        return p

    def _goal(self, tick, *, speed=WRIST_SPEED):
        sts_write_position(self.bus.port, 10, tick, speed, 10)
        if self._read(42,2)!=tick: raise Fault('wrist goal readback mismatch')

    def rotate(self,turns):
        if turns not in (-1,1,2): raise Fault('Invalid wrist turn')
        p=self._pos()
        reference=self.c.get('reference_tick')
        # Use the known reference when the measured position is within tolerance.
        base=(reference if reference is not None and abs(p-reference)<=self.c['tolerance'] else p)
        targets = wrist_segment_targets(self.c, base, turns)
        count = len(targets)
        if count-1>self.c.get('rotation_regrip_max_steps',1):
            raise Fault(f'{count-1} regrips required; configured maximum is '
                        f"{self.c.get('rotation_regrip_max_steps',1)}")
        low, high = getattr(self, 'wrist_goal_range', (self.c['min_tick'], self.c['max_tick']))
        if any(not low <= target <= high for target in targets):
            raise WristTargetOutOfRange(
                f'wrist target outside range: actual={p}, base={base}, '
                f'target={targets[-1]}, allowed={low}..{high}'
            )
        if not self.wrist_armed:
            self.wrist_armed=True
            self._goal(p)
            sts_write(self.bus.port,10,40,b'\x01',expect_reply=True)
            if self._read(40,1)!=1: raise Fault('wrist enable unconfirmed')
        self.rotation_quarter_turn = turns in (-1, 1)
        self.rotation_regrip_ticks = REGRIP_TICKS if self.rotation_quarter_turn else HALF_TURN_REGRIP_TICKS
        # Segment two is the logical 90-degree halfway point of a half turn.
        self.rotation_pause_target = targets[1] if turns == 2 else None
        self.rotation_remaining = targets[1:]
        self._start_wrist_segment(targets[0], p)
        print(f'Wrist segment 1/{count}: target {self.wtarget}', flush=True)

    def _start_wrist_segment(self, target, position):
        self.wtarget = target
        self.rotation_phase = 'turning'
        self.regrip_deadline = None
        self.regrip_ramp_limit = self.rotation_regrip_ticks if self.rotation_remaining else 0
        if self.rotation_quarter_turn and not self.rotation_remaining:
            self.regrip_ramp_limit = SECOND_QUARTER_REGRIP_TICKS
        self.regrip_ramp_start = position if self.regrip_ramp_limit else None
        self.regrip_ramp_base = None
        self.regrip_ramp_ticks = 0
        self._goal(target, speed=WRIST_ROTATION_SPEED)

    def _advance_regrip(self, rows, wrist_position, *, wrist_at):
        start = self.regrip_ramp_start
        travel = self.wtarget - start
        progress = 1.0 if wrist_at else (wrist_position - start) / travel
        pull = round(self.regrip_ramp_limit * max(0.0, min(1.0, progress)))
        if self.regrip_ramp_base is None:
            # Reuse one base for the segment so the pull cannot accumulate each tick.
            self.regrip_ramp_base = self.regrip_targets or self.last_goal
            if self.regrip_ramp_base is None:
                self.regrip_ramp_base = {row['id']: row['position_tick'] for row in rows}
        if pull > self.regrip_ramp_ticks:
            self._set_regrip_targets(self.regrip_ramp_base, pull, final=False)
            self.regrip_ramp_ticks = pull

        if wrist_at:
            self.regrip_ramp_start = None

    def _apply_final_regrip(self, rows):
        previous = self.last_goal
        if previous is None:
            previous = {row['id']: row['position_tick'] for row in rows}
        self._set_regrip_targets(previous, FINAL_REGRIP_TICKS, final=True)
        self.rotation_phase = 'final_regrip'
        print(f'Wrist final regrip: finger targets '
              f'{dict(sorted(self.regrip_targets.items()))}', flush=True)

    def _set_regrip_targets(self, previous, pull, *, final):
        budget = self.c.get('rotation_regrip_max_steps', 1) * self.rotation_regrip_ticks
        if self.rotation_quarter_turn and not self.rotation_remaining:
            budget += SECOND_QUARTER_REGRIP_TICKS
        if final:
            budget += FINAL_REGRIP_TICKS
        targets={}
        for motor in self.cfg['motors']:
            motor_id=motor['id']
            previous_goal=round(previous[motor_id])
            floor=min(motor['provisional_close_tick']-budget,previous_goal)
            if floor<0:
                raise Fault('Invalid rotation regrip floor')
            targets[motor_id]=max(floor,previous_goal-pull)

        self.regrip_targets=targets
        for motor_id,target in targets.items():
            low,high=self.guard.ends[motor_id]
            self.guard.ends[motor_id]=(min(low,target),high)
        self.regrip_lower={
            motor_id:max(self.guard.ends[motor_id][0]-3,
                         min(self.c['grasp_ticks'][str(motor_id)][0],target-3))
            for motor_id,target in targets.items()
        }

    def grasp_diagnostics(self):
        if self.last_feedback is None: return 'finger feedback unavailable'
        return f'goal_confirmed={self.last_confirmed}; finger target/actual/window: '+', '.join(
            f'ID{i}={self.last_goal.get(i) if self.last_goal else None}/'
            f'{self.last_feedback.get(i)}/{self.c["grasp_ticks"][str(i)]}'
            for i in (1,2,3,4))

    def _tick_stage(self, name):
        now=time.perf_counter()
        if self._active_tick_stage is not None:
            self.tick_timings[self._active_tick_stage]=self.tick_timings.get(self._active_tick_stage,0.)+now-self._tick_stage_start
        self._active_tick_stage=name
        if name is not None: self.io_phase=name
        self._tick_stage_start=now

    def fault_context(self):
        """Format only cached feedback; never query motors on the fault path."""
        rows=getattr(self,'last_rows',None)
        now=time.monotonic()
        fields=('position_tick','goal_position_tick','current_ma','input_voltage_v','hardware_error')
        return 'context='+json.dumps({
            'phase':getattr(self,'io_phase','unknown'),
            'rotation_phase':getattr(self,'rotation_phase',None),
            'wrist_target':getattr(self,'wtarget',None),
            'last_sent_goal':getattr(self,'last_goal',None),
            'cached_fingers':None if rows is None else [{
                'id':r['id'], 'sample_age_ms':round((now-r['sample_start_s'])*1000,2),
                **{key:r.get(key) for key in fields}} for r in rows],
        },ensure_ascii=False,sort_keys=True)

    def timing_diagnostics(self):
        timings=getattr(self,'tick_timings',{})
        retry=getattr(getattr(self,'bus',None),'last_diagnostic_retry',False)
        return f'finger_read_retry={retry}; tick_ms: '+', '.join(f'{name}={seconds*1000:.2f}'
                                      for name,seconds in timings.items())

    def tick(self,releasing,dt):
        # Measure wall time without logging or extra bus traffic in the loop.
        began=time.perf_counter()
        self.tick_timings={}
        self._active_tick_stage='setup'; self._tick_stage_start=began
        try:
            return self._tick(releasing,dt)
        finally:
            self._tick_stage(None)
            self.tick_timings['total']=time.perf_counter()-began

    def _tick(self,releasing,dt):
        b=self.bus; cfg=self.cfg
        self._tick_stage('finger_read')
        rows=b.read_diagnostic(combined=True)
        self.last_rows=rows
        self._tick_stage('guard_confirm')
        self.guard.check(rows)
        self.last_feedback={r['id']:r['position_tick'] for r in rows}
        confirmed=b.confirm_pending(rows,AGE) is not None
        self.last_confirmed=confirmed
        now=time.monotonic()
        if releasing:
            self.rotation_phase=None; self.rotation_remaining=[]
            self.regrip_targets=None; self.regrip_lower=None; self.regrip_deadline=None; self.regrip_ramp_start=None
        lower=self.regrip_lower or {int(i):limits[0] for i,limits in self.c['grasp_ticks'].items()}
        grip=confirmed and all(lower[r['id']]<=r['position_tick']<=
                               self.c['grasp_ticks'][str(r['id'])][1] for r in rows)
        self._tick_stage('wrist_read')
        wrist_position = self._pos() if self.wtarget is not None else None
        wrist_at = wrist_position is None or abs(wrist_position - self.wtarget) <= self.c['tolerance']
        self._tick_stage('rotation')
        if not releasing and self.rotation_phase is not None:
            if not grip: raise Fault('grasp telemetry left permitted window during rotation')
            ramp_active = self.rotation_phase == 'turning' and self.regrip_ramp_start is not None
            if ramp_active:
                self._advance_regrip(rows, wrist_position, wrist_at=wrist_at)
            if self.rotation_phase == 'turning' and wrist_at:
                if self.rotation_remaining or ramp_active:
                    # Confirm the segment's last finger command on the next tick.
                    # Only the logical halfway point adds a timed dwell.
                    pause = REGRIP_SECONDS if self.wtarget == self.rotation_pause_target else 0.
                    self.rotation_phase = 'regripping'
                    self.regrip_deadline = time.monotonic() + pause
                    action = f'holding for {pause:g}s' if pause else 'no timed pause'
                    print(f'Wrist segment reached: {action}; '
                          f'finger targets {dict(sorted(self.regrip_targets.items()))}', flush=True)
                else:
                    self._apply_final_regrip(rows)
            elif self.rotation_phase=='final_regrip' and wrist_at:
                # grip includes readback confirmation of the previous goal.
                self.rotation_phase=None
                print('Wrist rotation and regrip complete',flush=True)
            elif self.rotation_phase=='regripping' and not wrist_at:
                # Recheck arrival before advancing. At the timed midpoint
                # this also restarts the dwell without pulling again.
                self.rotation_phase='turning'; self.regrip_deadline=None
            elif self.rotation_phase=='regripping' and now>=self.regrip_deadline:
                if self.rotation_remaining:
                    self._start_wrist_segment(self.rotation_remaining.pop(0), wrist_position)
                    wrist_at=False
                    print(f'Wrist next segment: target {self.wtarget}',flush=True)
                else:
                    self._apply_final_regrip(rows)
        self._tick_stage('target_compute')
        if releasing:
            commands={m['id']:m['initial_position_tick'] for m in cfg['motors']}
        else:
            self._tick_stage('policy_inference')
            action=self.policy.predict(observation(cfg,rows,now,allow_provisional=True))
            self._tick_stage('target_compute')
            if len(action)!=4: raise Fault('incomplete policy output')
            a=[max(-1.,min(1.,finite(v))) for v in action]
            alpha=cfg['policy']['action_smoothing']
            self.smooth=a if self.smooth is None else [alpha*x+(1-alpha)*y for x,y in zip(a,self.smooth)]
            goals=self.guard.targets(action_targets(cfg,self.smooth))
            commands=self.comp.command(goals,rows,now,dt)
            if self.regrip_targets is not None:
                # Preserve an existing tighter PPO command; never add another regrip
                # to already compensated commands on every control cycle.
                commands={i:min(command,self.regrip_targets[i]) for i,command in commands.items()}
        opened=releasing and self.last_goal==commands and all(r['goal_position_tick']==commands[r['id']] for r in rows) and initial_pose_ready(cfg,rows)
        wrist_ok=wrist_at and self.rotation_phase is None
        self._tick_stage('finger_write')
        if not (releasing and opened):
            b.command_sync(commands,rows,AGE,deferred=True)
        self.last_goal=commands
        return grip,opened,wrist_ok

    @staticmethod
    def can_recover_after_fault(error):
        if isinstance(error, WristTargetOutOfRange):
            return True
        if isinstance(error, BusError):
            message = str(error)
            return ('응답 헤더를 찾지 못했습니다' in message
                    and message.endswith('received bytes=<none>'))
        return False

    def recover_fingers_after_fault(self, *, check, timeout=3.):
        """Open fingers after fresh checks on the existing connection."""
        bus = self.bus
        if bus is None or set(bus.armed_ids) != {1, 2, 3, 4}:
            raise Fault('recovery requires an existing armed connection')
        if self.guard.failed:
            raise Fault('recovery blocked by latched guard fault')

        deadline = time.monotonic() + timeout

        def bounded():
            check()
            if time.monotonic() >= deadline:
                raise TimeoutError('finger recovery deadline exceeded')

        def fresh():
            bounded()
            rows = bus.read_diagnostic(combined=True)
            self.guard.check(rows)
            if any(abs(row['current_ma']) > self.guard.current_stop[row['id']] for row in rows):
                raise Fault('recovery current above limit')
            bounded()
            return rows

        bounded()
        sts_write(bus.port, 10, 40, b'\x00', expect_reply=True)
        bounded()
        if self._read(40, 1) != 0:
            raise Fault('recovery wrist stop unconfirmed')
        self.wrist_armed = False
        bounded()
        if bus.read_register_sync(64, 1) != {1: 1, 2: 1, 3: 1, 4: 1}:
            raise Fault('recovery requires existing finger torque ON')
        bounded()
        watchdogs = bus.read_register_sync(98, 1)
        if set(watchdogs) != {1, 2, 3, 4} or any(value not in (0, 25) for value in watchdogs.values()):
            raise Fault('recovery watchdog state invalid')
        rows = fresh()
        pending = getattr(bus, '_pending_goal', None)
        if pending is not None:
            if {row['id']: row['goal_position_tick'] for row in rows} != pending:
                raise Fault('recovery previous goal mismatch')
            # The fresh read authorizes superseding the goal, while the original
            # primitive still reports failure.
            bus._pending_goal = None

        targets = {motor['id']: motor['initial_position_tick'] for motor in self.cfg['motors']}
        bounded()
        bus.command_sync(targets, rows, AGE)
        settled = None
        while True:
            rows = fresh()
            now = time.monotonic()
            if any(row['goal_position_tick'] != targets[row['id']] for row in rows):
                raise Fault('recovery goal changed')
            reached = initial_pose_ready(self.cfg, rows)
            settled = (now if settled is None else settled) if reached else None
            if settled is not None and now - settled >= .3:
                return
            time.sleep(.02)

    def finish_release(self, *, check):
        """Return the wrist on the existing connection, then stop and close."""
        check()
        bus = self.bus
        if bus is None or set(bus.armed_ids) != {1, 2, 3, 4}:
            raise Fault('Release requires the active four-finger connection')

        bus.send_torque_off()
        bus.wait_for_quiet_input()
        self._check_fingers_off(retry=check)
        check()
        if self._read(40, 1) != 1:
            raise Fault('Wrist torque lost before release return')

        reference = self.c['reference_tick']
        current = self._pos()
        if abs(current - reference) > self.c['tolerance']:
            self._move_released_wrist(reference, timeout=15., check=check)
        else:
            print(f'Wrist already at reference: {current} tick', flush=True)

        check()
        self.close(retry_wrist_off=True)

    def _check_fingers_off(self, *, retry=None):
        bus = self.bus
        off = {1: 0, 2: 0, 3: 0, 4: 0}
        states = bus.read_register_sync(64, 1)
        if retry is not None and states != off and states.keys() == off.keys():
            enabled = [motor_id for motor_id, torque in states.items() if torque == 1]
            if enabled:
                print(f'Finger torque OFF retry before wrist home: IDs {enabled}', flush=True)
                for motor_id in enabled:
                    retry()
                    bus.write_checked(motor_id, 64, 0, 1)
                retry()
                states = bus.read_register_sync(64, 1)
        if states != off:
            raise Fault(f'finger torque must be OFF before wrist home: actual={states}')

    def _move_released_wrist(self, reference, *, timeout, check):
        check()
        self._goal(reference)
        print(f'Moving released wrist to reference {reference} tick...', flush=True)
        deadline = time.monotonic() + timeout
        settled = None
        while time.monotonic() < deadline:
            check()
            if self._read(40, 1) != 1:
                raise Fault('wrist home torque lost')
            position = self._pos()
            now = time.monotonic()
            if abs(position - reference) > self.c['tolerance']:
                settled = None
            elif settled is None:
                settled = now
            elif now - settled >= .3:
                check()
                self._check_fingers_off()
                print(f'Wrist reference reached: {position} tick', flush=True)
                return
            time.sleep(.03)
        raise Fault(f'Wrist home did not reach reference within {timeout:g}s')

    def return_wrist_to_reference(self, timeout=15., *, check=lambda: None):
        """After the fingers are released and OFF, move the unloaded wrist to 2400."""
        check()
        if self.bus is not None: raise Fault('wrist home requires the worker to release COM first')
        reference=self.c['reference_tick']
        bus=Bus(self.cfg['port'],self.cfg['baud'])
        self.bus=bus
        stop_errors=[]
        try:
            self._check_fingers_off()
            if self._read(40,1)!=0:
                raise Fault('wrist must start torque OFF for home')
            current=self._pos()
            if abs(current-reference)<=self.c['tolerance']:
                print(f'Wrist already at reference: {current} tick',flush=True)
                return
            check()
            # Hold the present encoder position before enabling wrist torque.
            self._goal(current)
            sts_write(bus.port,10,40,b'\x01',expect_reply=True)
            if self._read(40,1)!=1: raise Fault('wrist home torque ON unconfirmed')
            self._move_released_wrist(reference, timeout=timeout, check=check)
        finally:
            try:sts_write(bus.port,10,40,b'\x00')
            except Exception as exc:stop_errors.append(f'wrist home torque OFF write: {exc}')
            try:
                if self._read(40,1)!=0: raise Fault('wrist home torque remains ON')
            except Exception as exc:stop_errors.append(f'wrist home torque OFF unconfirmed: {exc}')
            self.bus=None;bus.close()
            if stop_errors:raise Fault('; '.join(stop_errors))

    def close(self, *, retry_wrist_off=False):
        # On quit after a fault or before grasp, reopen only to check and stop ID10.
        if self.bus is None:
            self.bus=Bus(self.cfg['port'],self.cfg['baud'])
        b=self.bus; errors=[]
        try:
            # Send both protocols' OFF commands before waiting for any reply.
            # A Sync Write is unacknowledged: confirmation is always separate.
            try:b.send_torque_off()
            except Exception as exc:errors.append(f'finger torque OFF write unconfirmed: {exc}')
            try:sts_write(b.port,10,40,b'\x00')
            except Exception as exc:errors.append(f'wrist torque OFF write unconfirmed: {exc}')
            quiet=False
            try:
                b.wait_for_quiet_input()
                quiet=True
            except Exception as exc:errors.append(f'stop input not quiet: {exc}')
            try:
                wrist_torque = self._read(40, 1)
                if quiet and wrist_torque == 1 and retry_wrist_off:
                    print('Retrying wrist ID10 torque OFF after release.', flush=True)
                    sts_write(b.port, 10, 40, b'\x00')
                    b.wait_for_quiet_input()
                    wrist_torque = self._read(40, 1)
                if wrist_torque != 0: raise Fault('wrist torque remains enabled')
                if not quiet: raise Fault('input not quiet; response may be stale')
                print('Wrist ID10 torque OFF confirmed.',flush=True)
            except Exception as exc:errors.append(f'wrist torque OFF unconfirmed: {exc}')
            try:finger_errors=b.verify_torque_off()
            except Exception as exc:finger_errors=[f'finger torque OFF verification failed: {exc}']
            if b.armed_ids and quiet and finger_errors:
                # A broadcast Sync Write has no acknowledgement. If it did not
                # switch torque off, retry each motor with an acknowledged write.
                print('Retrying finger torque OFF by motor ID...',flush=True)
                for i in b.armed_ids:
                    try:
                        if b.read_int(i,64,1)!=0:
                            b.write_checked(i,64,0,1)
                    except Exception as exc:
                        print(f'ID{i} torque OFF retry failed: {exc}',flush=True)
                try:finger_errors=b.verify_torque_off()
                except Exception as exc:finger_errors=[f'finger torque OFF recheck failed: {exc}']
            if b.armed_ids and not quiet:
                finger_errors.append('finger torque OFF unconfirmed: input not quiet')
            errors.extend(finger_errors)
            if b.armed_ids and not finger_errors:
                errors.extend(restore_trial_current(b,self.current_restore,lambda record:None))
                for i in b.armed_ids:
                    settings=[]
                    if i in self.profiles:settings.extend(zip((112,108),self.profiles[i],(4,4)))
                    if i in self.gains:settings.append((84,self.gains[i],2))
                    if i in self.igains:settings.append((82,self.igains[i],2))
                    for addr,v,n in settings+[(98,0,1)]:
                        try:b.write_checked(i,addr,v,n)
                        except Exception as exc:errors.append(str(exc))
        finally:
            self.bus=None; self.wrist_armed=False
            b.close()
        if errors:raise Fault('; '.join(errors))
