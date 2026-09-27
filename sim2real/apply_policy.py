"""Bounded four-motor policy execution using user-approved provisional endpoints.
Default is shadow. --execute explicitly enables motors; calibration is NOT certified.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
from bus import Bus
from position_compensation import PositionCompensation
from core import Fault, observation, action_targets, states_by_id, finite, validate
from run import Policy, inspect_policy

ROOT=Path(__file__).resolve().parent
PERIOD=.1
SENSOR_PERIOD=.02
AGE=.1
INITIAL_TOLERANCE_TICK=10


class Guard:
    def __init__(self,cfg,rows,inventory,initializing=False):
        self.temperature_stop={m["id"]:m["temperature_limit_c"]-1 for m in inventory}
        self.current_stop={m["id"]:min(700,m["current_limit_ma"]) for m in inventory}
        self.ends={m['id']:(m['provisional_close_tick'],m['initial_position_tick']) for m in cfg['motors']}
        self.previous={r['id']:r['position_tick'] for r in rows}
        self.upper={i:max(self.previous[i],hi) for i,(lo,hi) in self.ends.items()}
        self.failed=False
        for i,(lo,hi) in self.ends.items():
            if not 0<=lo<hi<=4095 or not lo<=self.previous[i]<=(4095 if initializing else hi+64):
                raise Fault(f'ID{i}: outside provisional starting envelope')

    def check(self,rows):
        try:
            if self.failed: raise Fault('Latched fault')
            by=states_by_id(rows,time.monotonic(),AGE)
            now=time.monotonic()
            for i,r in by.items():
                lo,_=self.ends[i]
                if not lo-3<=r['position_tick']<=self.upper[i]+3: raise Fault(f'ID{i}: position limit')
                if abs(r['current_ma'])>self.current_stop[i]: raise Fault(f'ID{i}: configured current limit')
                if r['temperature_c']>=self.temperature_stop[i]:
                    raise Fault(f'ID{i}: temperature approaching configured device limit')
                # No arbitrary measured-speed or timed 600mA stall cutoff.
                # Device overload detection is checked by command_sync before goals.
            return by
        except BaseException:
            self.failed=True
            raise

    def targets(self,raw):
        if self.failed: raise Fault('Latched fault')
        if set(raw)!={1,2,3,4}: raise Fault('Incomplete action targets')
        out={}
        for i,(lo,hi) in self.ends.items():
            desired=max(lo,min(hi,finite(raw[i])))
            # Preserve the policy's absolute goal; only endpoint clipping and
            # encoder quantization are applied. Motor-internal profile remains.
            out[i]=max(lo,min(hi,round(desired)))
        self.previous=out
        return out


def initial_pose_ready(cfg,rows):
    by=states_by_id(rows,time.monotonic(),AGE)
    return all(abs(by[m['id']]['position_tick']-m['initial_position_tick'])<=INITIAL_TOLERANCE_TICK
               and abs(by[m['id']]['velocity_tick_s'])<=32 for m in cfg['motors'])


def move_to_initial(bus,cfg,guard,emit,timeout=15.,clock=time.monotonic,sleep=time.sleep):
    targets={m['id']:m['initial_position_tick'] for m in cfg['motors']}
    rows=bus.read_diagnostic(); guard.check(rows)
    emit({'type':'initial_move_attempt','targets':targets,'rows':rows})
    bus.command_sync(targets,rows,AGE)
    began=clock(); settled=None
    while clock()-began<timeout:
        rows=bus.read_diagnostic()
        emit({'type':'initial_move_state','rows':rows})
        guard.check(rows)
        errors=bus.read_register_sync(70,1)
        if any(errors.values()): raise Fault(f'Hardware error during initial move: {errors}')
        now=clock()
        reached=initial_pose_ready(cfg,rows)
        settled=(now if settled is None else settled) if reached else None
        if settled is not None and now-settled>=.3:
            emit({'type':'initial_pose_reached','targets':targets,'rows':rows})
            return rows
        sleep(.02)
    raise Fault('Initial position not reached within 15s; policy was not started')


def preflight(inv):
    if len(inv)!=4 or {m['id'] for m in inv}!={1,2,3,4}: raise Fault('Incomplete inventory')
    for m in inv:
        if m['model_number']!=1200 or m['mode']!=5 or m['drive_mode']!=0:
            raise Fault('Requires XL330-M288, mode5, drive0')
        if m['torque'] or m['watchdog'] or m['hardware_error']:
            raise Fault('All torque OFF, watchdog0 and no hardware error required')
        if m['secondary_id']!=255 or m['status_return_level']!=2:
            raise Fault('Secondary ID must be disabled; status return2 required')
        if not 1<m['temperature_limit_c']<=100 or m['shutdown_mask'] & 0x24 != 0x24:
            raise Fault('Device overheat and overload shutdown must be enabled; invalid temperature setting')
        if not 0<m['current_limit_ma']<=700: raise Fault('EEPROM current limit exceeds 700mA or zero')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--position-p',type=int,choices=range(0,16384),metavar='0..16383',default=None,help='Temporary position P gain; restored after confirmed torque off')
    p.add_argument('--position-compensation',type=int,choices=range(201),metavar='0..200',default=200,help='Maximum extra winding correction; requires motor I=0')
    p.add_argument('--position-i',type=int,choices=range(0,16384),metavar='0..16383',default=None,help='Temporary position I gain; restored after torque off')
    p.add_argument('--profile-velocity',type=int,choices=range(0,32768),metavar='0..32767',default=80,help='Motor profile velocity; 0 disables shaping')
    p.add_argument('--profile-acceleration',type=int,choices=range(0,32768),metavar='0..32767',default=30,help='Motor profile acceleration; 0 disables shaping')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--policy-period-ms',type=int,choices=[30,100],default=30,help='Policy execution period; current model trained at 30ms')
    args=p.parse_args()
    if not math.isfinite(args.seconds) or not 0<args.seconds<=30:
        p.error('seconds must be >0 and <=30')
    if args.position_compensation and args.position_i not in (None,0):
        p.error('Position compensation requires --position-i 0; disable compensation to test motor I')
    effective_i=0 if args.position_compensation else args.position_i
    cfg=json.loads((ROOT/'config.v5.json').read_text(encoding='utf-8'))
    validate(cfg,policy=True,allow_provisional=True)
    period=args.policy_period_ms/1000.
    sensor_period=period if args.policy_period_ms==30 else SENSOR_PERIOD
    cfg['period_s']=period
    path=ROOT.parents[1]/'ppo_model'/cfg['policy']['filename']
    policy=Policy(path,cfg)
    logpath=ROOT/'logs'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_apply_policy.jsonl')
    logpath.parent.mkdir(exist_ok=True)
    bus=None; code=0; rows=None; gain_restore={}; i_restore={}; profile_restore={}; last_commands=None
    with logpath.open('x',encoding='utf-8') as log:
        def emit(x):
            log.write(json.dumps(x,allow_nan=False)+'\n')
            if x.get('type') in ('fault','stop','metadata'): log.flush()
        emit({'type':'metadata','execute':args.execute,'config':cfg,'policy':inspect_policy(path),
              'period_s':period,'seconds':args.seconds,'provisional':True,'requested_position_p':args.position_p,'requested_position_i':args.position_i,'effective_position_i':effective_i,
              'position_compensation_max_tick':args.position_compensation,'compensation_rate_cap_tick_s':40,
              'initial_move':{'enabled':args.execute,'p_gain':500,'profile_velocity':20,'profile_acceleration':10,'tolerance_tick':INITIAL_TOLERANCE_TICK,'settle_s':.3,'timeout_s':15},
              'timing_note':f'{args.policy_period_ms}ms execution; selected 30ms model',
              'current_cap_ma':700,'software_speed_stop':None,'software_timed_stall_stop':None,
              'profile_velocity_raw':args.profile_velocity,'profile_acceleration_raw':args.profile_acceleration,
              'command_mode':'absolute-policy-position','max_step_tick':None,
              'smoothing_tau_s':0,'instant_tracking_error_stop':False,'temperature_stop_source':'device Temperature Limit(31) minus 1C',
              'fidelity_note':'Provisional position mapping; user-adjustable hardware profile, provisional tuning; execution timing and hardware servo response require verification.'})
        try:
            bus=Bus('COM8',1000000)
            inv=bus.inventory(); emit({'type':'inventory','motors':inv}); preflight(inv)
            emit({'type':'effective_protection','temperature_stop_c':{m['id']:m['temperature_limit_c']-1 for m in inv},
                  'current_cap_ma':{m['id']:min(700,m['current_limit_ma']) for m in inv},
                  'overload_stop':'device error, no prediction of hardware trip time'})
            rows=bus.read_diagnostic(); guard=Guard(cfg,rows,inv,initializing=args.execute); guard.check(rows)
            # Warm up with real telemetry before any motor writes.
            policy.predict(observation(cfg,rows,time.monotonic(),allow_provisional=True))
            rows=bus.read_diagnostic(); guard=Guard(cfg,rows,inv,initializing=args.execute); guard.check(rows)
            baseline={r['id']:r['position_tick'] for r in rows}
            if args.execute:
                bus.armed_ids=[1,2,3,4]
                for m in inv:
                    i=m['id']
                    # Use identical initialization dynamics for both gain trials.
                    gain_restore[i]=m['position_p_gain']
                    bus.write_checked(i,84,500,2)
                    bus.write_checked(i,102,min(700,m['current_limit_ma']),2)
                    profile_restore[i]=(m['profile_velocity_raw'],m['profile_acceleration_raw'])
                    bus.write_checked(i,112,20,4)
                    bus.write_checked(i,108,10,4)
                    hold=bus.prepare_hold(i,baseline[i])
                    emit({'type':'hold_prepared','id':i,'position_tick':hold})
                fresh=bus.read_diagnostic(); guard.check(fresh)
                if any(abs(r['position_tick']-baseline[r['id']])>2 for r in fresh):
                    raise Fault('Hand moved during preparation')
                for i in range(1,5): bus.write_checked(i,98,25,1)
                for i in range(1,5):
                    fresh=bus.read_diagnostic(); guard.check(fresh)
                    if any(abs(r['position_tick']-baseline[r['id']])>2 for r in fresh):
                        raise Fault('Hand moved during torque enable')
                    emit({'type':'enable_attempt','id':i})
                    bus.write_checked(i,64,1,1)
                print('Moving to saved initial pose...')
                rows=move_to_initial(bus,cfg,guard,emit)
                for m in inv:
                    i=m['id']
                    bus.write_checked(i,84,args.position_p if args.position_p is not None else m['position_p_gain'],2)
                    if effective_i is not None:
                        i_restore[i]=m['position_i_gain']
                        bus.write_checked(i,82,effective_i,2)
                    bus.write_checked(i,112,args.profile_velocity,4)
                    bus.write_checked(i,108,args.profile_acceleration,4)
                rows=bus.read_diagnostic()
                if not initial_pose_ready(cfg,rows):
                    raise Fault('Initial pose changed during gain setup; policy was not started')
                guard=Guard(cfg,rows,inv); guard.check(rows)
                emit({'type':'policy_start','rows':rows})
                print('Initial pose confirmed. Starting policy.')
            smoothing=float(cfg['policy'].get('action_smoothing',1.))
            if not math.isfinite(smoothing) or not 0<smoothing<=1: raise Fault('Invalid action smoothing')
            smooth_action=None
            compensation=PositionCompensation(cfg,args.position_compensation)
            began=time.perf_counter(); next_policy=began; cycles=0; last_update=None
            previous_cycle_start=None; next_sensor=began; timing_rows=[]; skipped_policy_slots=0
            while time.perf_counter()-began<args.seconds or getattr(bus,'_pending_goal',None) is not None:
                remaining=next_sensor-time.perf_counter()
                if remaining>0: time.sleep(remaining)
                cycle_started=time.perf_counter()
                lateness=max(0.,cycle_started-next_sensor)
                if lateness>AGE: raise Fault('Sensor scheduling deadline exceeded')
                interval=None if previous_cycle_start is None else cycle_started-previous_cycle_start
                previous_cycle_start=cycle_started
                rows=bus.read_diagnostic(combined=True)
                read_finished=time.perf_counter(); guard.check(rows)
                confirmation=bus.confirm_pending(rows,AGE)
                if last_commands is not None:
                    emit({'type':'tracking','error_tick':{r['id']:last_commands[r['id']]-r['position_tick'] for r in rows},'rows':rows})
                now=time.perf_counter()
                policy_due=now>=next_policy and now-began<args.seconds
                inference_ms=0.; write_ms=0.; policy_interval_ms=None
                if policy_due:
                    if now-next_policy>AGE: raise Fault('Policy scheduling deadline exceeded')
                    started=time.perf_counter()
                    obs=observation(cfg,rows,time.monotonic(),allow_provisional=True)
                    action=policy.predict(obs)
                    # Match the training environment: clip, then EMA, then scale.
                    if len(action)!=4: raise Fault('Expected four actions')
                    clipped=[max(-1.,min(1.,finite(v))) for v in action]
                    smooth_action=(clipped.copy() if smooth_action is None else
                                   [(1-smoothing)*previous+smoothing*current
                                    for previous,current in zip(smooth_action,clipped)])
                    raw=action_targets(cfg,smooth_action)
                    guard.check(rows)
                    policy_goals=guard.targets(raw)
                    policy_interval_ms=None if last_update is None else (now-last_update)*1000
                    dt=period if last_update is None else now-last_update
                    commands=compensation.command(policy_goals,rows,time.monotonic(),dt)
                    last_update=now
                    emit({'type':'policy','cycle':cycles,'rows':rows,'obs':obs,'action':action,
                          'smoothed_action':list(smooth_action),'action_smoothing':smoothing,
                          'raw_targets_tick':raw,'policy_goal_tick':policy_goals,'limited_targets_tick':commands,
                          'compensation_tick':dict(compensation.offset),
                          'goal_adjustment_tick':{i:commands[i]-raw[i] for i in commands},
                          'outside_initial_reference':[m['id'] for m in cfg['motors'] if next(r['position_tick'] for r in rows if r['id']==m['id'])>m['initial_position_tick']],
                          'inference_ms':(time.perf_counter()-started)*1000})
                    write_started=time.perf_counter()
                    if args.execute:
                        emit({'type':'command_attempt','targets':commands})
                        bus.command_sync(commands,rows,AGE,deferred=True)
                        last_commands=dict(commands)
                    write_ms=(time.perf_counter()-write_started)*1000
                    inference_ms=(write_started-started)*1000
                    cycles+=1; next_policy+=period
                    if next_policy<time.perf_counter():
                        skipped=math.floor((time.perf_counter()-next_policy)/period)+1
                        skipped_policy_slots+=skipped
                        next_policy+=skipped*period
                total=(time.perf_counter()-cycle_started)*1000
                metric={'type':'io_timing','policy_cycle':policy_due,'cycle_start_s':cycle_started,
                        'policy_interval_ms':policy_interval_ms,
                        'read_ms':(read_finished-cycle_started)*1000,'inference_transform_ms':inference_ms,
                        'write_ms':write_ms,'total_ms':total,'over_30ms':total>30,
                        'confirmation_ms':confirmation,
                        'sensor_interval_ms':None if interval is None else interval*1000,
                        'schedule_lateness_ms':lateness*1000}
                metric['console_ms']=None
                if policy_due:
                    console_started=time.perf_counter()
                    actual=dict(sorted((r['id'],r['position_tick']) for r in rows))
                    display_targets=dict(sorted(commands.items()))
                    print(f"cycle {cycles-1} | target {display_targets} | actual {actual} | "
                          f"read-to-send {total:.2f}ms | sent {args.execute}",flush=True)
                    metric['console_ms']=(time.perf_counter()-console_started)*1000
                metric['total_with_console_ms']=(time.perf_counter()-cycle_started)*1000
                emit(metric); timing_rows.append(metric)
                if total>AGE*1000: raise Fault('IO deadline exceeded (100ms)')
                next_sensor+=sensor_period
                if next_sensor<time.perf_counter():
                    next_sensor+=(math.floor((time.perf_counter()-next_sensor)/sensor_period)+1)*sensor_period
            def stats(key):
                values=[r[key] for r in timing_rows if r[key] is not None]
                return {'count':len(values),'mean_ms':sum(values)/len(values),'max_ms':max(values),
                        'over_30ms':sum(v>30 for v in values)} if values else None
            summary={'type':'timing_summary','sensor_target_ms':sensor_period*1000,'policy_target_ms':args.policy_period_ms,
                     'policy_updates':cycles,'skipped_policy_slots':skipped_policy_slots,
                     **{key:stats(key) for key in ['read_ms','total_ms','console_ms','total_with_console_ms','confirmation_ms','sensor_interval_ms','policy_interval_ms']}}
            emit(summary); print('TIMING:',json.dumps(summary)); log.flush()
        except BaseException as exc:
            code=1; print('STOP:',type(exc).__name__,str(exc))
            try: emit({'type':'fault','error':str(exc),'last_acquired_rows':rows})
            except Exception: pass
        finally:
            if bus:
                try:
                    errors=bus.stop()
                    for error in errors: print(error)
                    if errors: code=1
                    if bus.armed_ids and not errors:
                        for i,(velocity,acceleration) in profile_restore.items():
                            try:
                                bus.write_checked(i,112,velocity,4)
                                bus.write_checked(i,108,acceleration,4)
                                emit({'type':'profile_restored','id':i,'velocity':velocity,'acceleration':acceleration})
                            except Exception as exc:
                                code=1
                                emit({'type':'profile_restore_failed','id':i,'error':str(exc)})
                                print('Profile restoration failed:',i,exc)
                        for i,original_i in i_restore.items():
                            try:
                                bus.write_checked(i,82,original_i,2)
                                emit({'type':'i_gain_restored','id':i,'position_i':original_i})
                            except Exception as exc:
                                code=1
                                emit({'type':'i_gain_restore_failed','id':i,'error':str(exc)})
                                print('I gain restoration failed:',i,exc)
                        for i,original_gain in gain_restore.items():
                            try:
                                bus.write_checked(i,84,original_gain,2)
                                emit({'type':'gain_restored','id':i,'position_p':original_gain})
                            except Exception as exc:
                                code=1
                                emit({'type':'gain_restore_failed','id':i,'error':str(exc)})
                                print('Gain restoration failed:',i,exc)
                        for i in bus.armed_ids:
                            try: bus.write_checked(i,98,0,1)
                            except Exception as exc:
                                code=1; print('Torque OFF, watchdog cleanup failed:',i,exc)
                        print('All four motors: torque OFF confirmed.')
                    try: emit({'type':'stop','torque_off_confirmed':bool(bus.armed_ids) and not errors,'errors':errors})
                    except Exception: code=1
                finally: bus.close()
    print('LOG:',logpath)
    return code

if __name__=='__main__': raise SystemExit(main())
