"""Hardware adapter. All methods are invoked on the same primitive worker thread."""
import json
import time
from pathlib import Path

from apply_policy import (Bus, Guard, AGE, Policy, PositionCompensation, action_targets,
                          initial_pose_ready, observation, validate, preflight,
                          finite, move_to_initial, initial_setup, set_trial_current, restore_trial_current)
from bus import Fault, BusError
from wrist_protocol import sts_read, sts_write, sts_write_position

WRIST_SPEED = 600  # STS3215 raw speed setting; modestly below 650.
REGRIP_TICKS = 100
REGRIP_SECONDS = 2.0


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
        if any(m['provisional_close_tick']-steps*REGRIP_TICKS<0 for m in self.cfg['motors']):
            raise ValueError('rotation regrip would leave single-turn encoder range')
        if not c['quarter_ticks']<2048 or not 0<c['tolerance']<c['quarter_ticks']/4: raise ValueError('wrist tolerance')
        for m in self.cfg['motors']:
            lo,hi=c['grasp_ticks'][str(m['id'])]
            if not m['provisional_close_tick']-3<=lo<=hi<m['initial_position_tick']-10:
                raise ValueError('grasp window must exclude open pose')
        ref=c.get('reference_tick')
        if ref is not None and (type(ref) is not int or not c['min_tick']<=ref<=c['max_tick']):
            raise ValueError('Invalid wrist reference_tick')
        self.policy=Policy(Path(model) if model else root.parent/'model'/self.cfg['policy']['filename'],self.cfg)
        self.bus=None; self.wrist_armed=False; self.wtarget=None; self.next_wtarget=None
        self.smooth=None; self.last_goal=None; self.last_feedback=None; self.last_confirmed=False
        self.rotation_phase=None; self.rotation_remaining=[]
        self.regrip_targets=None; self.regrip_lower=None; self.regrip_deadline=None; self.regrip_segment_target=None

    def start(self,check):
        check()
        cfg=self.cfg
        self.bus=bus=Bus(cfg['port'],cfg['baud'])
        self.gains={}; self.igains={}; self.profiles={}; self.current_restore={}
        # The previous process may have failed to release torque. Own all four
        # IDs before inventory so any startup failure still sends and verifies OFF.
        bus.armed_ids=[1,2,3,4]
        inv=bus.inventory()
        inv=clear_stopped_watchdogs(bus,inv)
        preflight(inv); check()
        if self._read(40,1)!=0: raise Fault('wrist must start torque OFF')
        if self._read(33,1)!=0: raise Fault('STS must be in position mode')
        if self.c.get('reference_tick') is None:
            raise Fault('Wrist reference tick is not configured')
        self._pos()  # Refuse motion if the current position is outside the checked range.
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
        self.guard=initial_setup(bus,cfg,inv,rows,guard,args,0,self.gains,self.igains,self.profiles,
                                 lambda record:check(),check)
        # Keep the fingers open and monitored while the wrist moves to its reference.
        self._move_wrist_reference(check)
        self.comp=PositionCompensation(cfg,100)
        self.smooth=None; self.last_goal=None; self.next_wtarget=None
        self.rotation_phase=None; self.rotation_remaining=[]
        self.regrip_targets=None; self.regrip_lower=None; self.regrip_deadline=None; self.regrip_segment_target=None

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

    def _goal(self,tick):
        sts_write_position(self.bus.port,10,tick,WRIST_SPEED,10)
        if self._read(42,2)!=tick: raise Fault('wrist goal readback mismatch')

    def rotate(self,turns):
        if turns not in (-1,1,2): raise Fault('Invalid wrist turn')
        p=self._pos()
        reference=self.c.get('reference_tick')
        # Use the known reference when the measured position is within tolerance.
        base=(reference if reference is not None and abs(p-reference)<=self.c['tolerance'] else p)
        quarter=self.c['quarter_ticks']
        direction=self.c['cw_sign']*(1 if turns>0 else -1)
        count=2*abs(turns)
        if count-1>self.c.get('rotation_regrip_max_steps',1):
            raise Fault(f'{count-1} regrips required; configured maximum is '
                        f"{self.c.get('rotation_regrip_max_steps',1)}")
        if abs(turns)==1:
            # One logical cube quarter-turn uses 50+50 degrees of wrist travel
            # to compensate for the user-observed short physical cube turn.
            targets=[base+direction*round(quarter*5*k/9) for k in (1,2)]
        else:
            # Preserve the existing 180-degree command as four 45-degree steps.
            targets=[base+direction*(quarter*k//2) for k in range(1,5)]
        target=targets[-1]
        if any(not self.c['min_tick']<=t<=self.c['max_tick'] for t in targets):
            raise Fault(f"wrist target outside range: actual={p}, base={base}, target={target}, allowed={self.c['min_tick']}..{self.c['max_tick']}")
        if not self.wrist_armed:
            self.wrist_armed=True
            self._goal(p)
            sts_write(self.bus.port,10,40,b'\x01',expect_reply=True)
            if self._read(40,1)!=1: raise Fault('wrist enable unconfirmed')
        self.rotation_remaining=targets[1:]
        self.rotation_phase='turning'; self.regrip_deadline=None; self.regrip_segment_target=None
        self.wtarget=targets[0]
        self._goal(self.wtarget)
        print(f'Wrist segment 1/{count}: target {self.wtarget}',flush=True)

    def grasp_diagnostics(self):
        if self.last_feedback is None: return 'finger feedback unavailable'
        return f'goal_confirmed={self.last_confirmed}; finger target/actual/window: '+', '.join(
            f'ID{i}={self.last_goal.get(i) if self.last_goal else None}/'
            f'{self.last_feedback.get(i)}/{self.c["grasp_ticks"][str(i)]}'
            for i in (1,2,3,4))

    def tick(self,releasing,dt):
        b=self.bus; cfg=self.cfg
        rows=b.read_diagnostic(combined=True); self.guard.check(rows)
        self.last_feedback={r['id']:r['position_tick'] for r in rows}
        confirmed=b.confirm_pending(rows,AGE) is not None
        self.last_confirmed=confirmed
        now=time.monotonic()
        if releasing:
            self.rotation_phase=None; self.rotation_remaining=[]
            self.regrip_targets=None; self.regrip_lower=None; self.regrip_deadline=None; self.regrip_segment_target=None
        lower=self.regrip_lower or {int(i):limits[0] for i,limits in self.c['grasp_ticks'].items()}
        grip=confirmed and all(lower[r['id']]<=r['position_tick']<=
                               self.c['grasp_ticks'][str(r['id'])][1] for r in rows)
        wrist_at=self.wtarget is None or abs(self._pos()-self.wtarget)<=self.c['tolerance']
        if not releasing and self.rotation_phase is not None:
            if not grip: raise Fault('grasp telemetry left permitted window during rotation')
            if self.rotation_phase=='turning' and wrist_at:
                if self.rotation_remaining:
                    # A brief wrist drift restarts the 2-second hold but must
                    # not add another 100 ticks at the same 45-degree stop.
                    if self.regrip_segment_target!=self.wtarget:
                        motors={m['id']:m for m in cfg['motors']}
                        previous=self.regrip_targets or self.last_goal or {r['id']:r['position_tick'] for r in rows}
                        allowed=self.c.get('rotation_regrip_max_steps',1)
                        floors={i:min(motors[i]['provisional_close_tick']-allowed*REGRIP_TICKS,
                                      round(previous[i])) for i in motors}
                        if any(v<0 for v in floors.values()):raise Fault('Invalid rotation regrip floor')
                        self.regrip_targets={i:max(floors[i],round(previous[i])-REGRIP_TICKS) for i in motors}
                        self.guard.ends={i:(min(self.guard.ends[i][0],self.regrip_targets[i]),
                                            self.guard.ends[i][1]) for i in motors}
                        self.regrip_lower={i:max(self.guard.ends[i][0]-3,
                                                 min(self.c['grasp_ticks'][str(i)][0],self.regrip_targets[i]-3))
                                           for i in motors}
                        self.regrip_segment_target=self.wtarget
                    self.rotation_phase='regripping'
                    self.regrip_deadline=now+REGRIP_SECONDS
                    print(f'Wrist stopped: regripping for {REGRIP_SECONDS:g} seconds; '
                          f'finger targets {dict(sorted(self.regrip_targets.items()))}',flush=True)
                else:
                    self.rotation_phase=None
                    print('Wrist rotation and regrip complete',flush=True)
            elif self.rotation_phase=='regripping' and not wrist_at:
                # Require the wrist to return to the segment target before
                # starting a fresh full regrip interval.
                self.rotation_phase='turning'; self.regrip_deadline=None
            elif self.rotation_phase=='regripping' and now>=self.regrip_deadline:
                if self.rotation_remaining:
                    self.wtarget=self.rotation_remaining.pop(0)
                    self._goal(self.wtarget)
                    self.rotation_phase='turning'; self.regrip_deadline=None
                    wrist_at=False
                    print(f'Wrist next segment: target {self.wtarget}',flush=True)
        if releasing:
            commands={m['id']:m['initial_position_tick'] for m in cfg['motors']}
        else:
            action=self.policy.predict(observation(cfg,rows,now,allow_provisional=True))
            if len(action)!=4: raise Fault('incomplete policy output')
            a=[max(-1.,min(1.,finite(v))) for v in action]
            alpha=cfg['policy']['action_smoothing']
            self.smooth=a if self.smooth is None else [alpha*x+(1-alpha)*y for x,y in zip(a,self.smooth)]
            goals=self.guard.targets(action_targets(cfg,self.smooth))
            commands=self.comp.command(goals,rows,now,dt)
            if self.regrip_targets is not None:
                # Preserve an existing tighter PPO command; never add 100 ticks
                # to already compensated commands on every control cycle.
                commands={i:min(command,self.regrip_targets[i]) for i,command in commands.items()}
        opened=releasing and self.last_goal==commands and all(r['goal_position_tick']==commands[r['id']] for r in rows) and initial_pose_ready(cfg,rows)
        wrist_ok=wrist_at and self.rotation_phase is None
        if not (releasing and opened):
            b.command_sync(commands,rows,AGE,deferred=True)
        self.last_goal=commands
        return grip,opened,wrist_ok

    def return_wrist_to_reference(self, timeout=15.):
        """After the fingers are released and OFF, move the unloaded wrist to 2400."""
        if self.bus is not None: raise Fault('wrist home requires the worker to release COM first')
        reference=self.c['reference_tick']
        bus=Bus(self.cfg['port'],self.cfg['baud'])
        self.bus=bus
        stop_errors=[]
        try:
            off={1:0,2:0,3:0,4:0}
            if bus.read_register_sync(64,1)!=off:
                raise Fault('finger torque must be OFF before wrist home')
            if self._read(40,1)!=0:
                raise Fault('wrist must start torque OFF for home')
            current=self._pos()
            if abs(current-reference)<=self.c['tolerance']:
                print(f'Wrist already at reference: {current} tick',flush=True)
                return
            # Hold the present encoder position before enabling wrist torque.
            self._goal(current)
            sts_write(bus.port,10,40,b'\x01',expect_reply=True)
            if self._read(40,1)!=1: raise Fault('wrist home torque ON unconfirmed')
            self._goal(reference)
            print(f'Moving released wrist to reference {reference} tick...',flush=True)
            began=time.monotonic();settled=None
            while time.monotonic()-began<timeout:
                if self._read(40,1)!=1: raise Fault('wrist home torque lost')
                position=self._pos();now=time.monotonic()
                reached=abs(position-reference)<=self.c['tolerance']
                settled=(now if settled is None else settled) if reached else None
                if settled is not None and now-settled>=.3:
                    if bus.read_register_sync(64,1)!=off:
                        raise Fault('finger torque changed during wrist home')
                    print(f'Wrist reference reached: {position} tick',flush=True)
                    return
                time.sleep(.03)
            raise Fault('Wrist home did not reach reference within 15s')
        finally:
            try:sts_write(bus.port,10,40,b'\x00')
            except Exception as exc:stop_errors.append(f'wrist home torque OFF write: {exc}')
            try:
                if self._read(40,1)!=0: raise Fault('wrist home torque remains ON')
            except Exception as exc:stop_errors.append(f'wrist home torque OFF unconfirmed: {exc}')
            self.bus=None;bus.close()
            if stop_errors:raise Fault('; '.join(stop_errors))

    def close(self):
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
                if self._read(40,1)!=0: raise Fault('wrist torque remains enabled')
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
