"""Return fingers 1, 2, 3 and 4 to saved initial positions. Default is an offline preview."""
import argparse
import json
import time
from pathlib import Path

from bus import Bus
from core import Fault, states_by_id, validate

ROOT=Path(__file__).resolve().parent
IDS=(1,2,3,4)
AGE=.1
TOLERANCE=10
CURRENT_CAP_MA=1000


def selected_targets(cfg):
    motors={m['id']:m for m in cfg['motors']}
    if set(motors)!={1,2,3,4}:raise Fault('Expected four unique finger IDs')
    targets={i:motors[i]['initial_position_tick'] for i in IDS}
    if any(type(v) is not int or not 0<=v<=4095 for v in targets.values()):
        raise Fault('Invalid saved open positions')
    return targets


class RestoreGuard:
    """Monitor fresh feedback and existing device current/temperature limits."""
    def __init__(self,inventory):
        self.by_id={m['id']:m for m in inventory}
    def check(self,rows):
        by=states_by_id(rows,time.monotonic(),AGE)
        for i in IDS:
            r=by[i];m=self.by_id[i]
            if not 0<=r['position_tick']<=4095:raise Fault(f'ID{i}: invalid position')
            if r.get('hardware_error')!=0:raise Fault(f'ID{i}: hardware error')
            limit=min(CURRENT_CAP_MA,m['current_limit_ma'])
            if abs(r['current_ma'])>limit:raise Fault(f'ID{i}: current limit during return')
            if r['temperature_c']>=m['temperature_limit_c']-1:raise Fault(f'ID{i}: temperature limit')
        return by


def move_to_open(bus,cfg,guard,emit,*,after_goals=None,timeout=15.,clock=time.monotonic,sleep=time.sleep):
    """Move IDs 1/2/3/4 toward their saved open ticks; monitor until settled."""
    targets=selected_targets(cfg)
    rows=bus.read_diagnostic(combined=True);guard.check(rows)
    by={r['id']:r for r in rows}
    # The verified open position may be above or below the present encoder
    # value, including after the springs move a motor while torque is OFF.
    requested=dict(targets)
    emit({'type':'finger_return_attempt','targets':requested,
          'positions':{i:by[i]['position_tick'] for i in IDS}})
    for i in IDS:
        rows=bus.read_diagnostic(combined=True);guard.check(rows)
        bus.write_checked(i,116,requested[i],4)
    if after_goals is not None:after_goals(requested)
    began=clock();settled=None
    while clock()-began<timeout:
        rows=bus.read_diagnostic(combined=True);guard.check(rows)
        by={r['id']:r for r in rows}
        reached=all(abs(by[i]['position_tick']-targets[i])<=TOLERANCE and
                    abs(by[i]['velocity_tick_s'])<=32 for i in IDS)
        now=clock();settled=(now if settled is None else settled) if reached else None
        if settled is not None and now-settled>=.3:
            emit({'type':'finger_return_reached','positions':{i:by[i]['position_tick'] for i in IDS}})
            return rows
        sleep(.02)
    raise Fault('ID1/2/3/4 did not reach saved open positions within 15s')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Command real motors 1, 2, 3 and 4')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.provisional.json').read_text(encoding='utf-8-sig'))
    validate(cfg,policy=True,allow_provisional=True)
    targets=selected_targets(cfg)
    if not args.execute:
        print(f'PREVIEW ONLY: ID1={targets[1]}, ID2={targets[2]}, ID3={targets[3]}, ID4={targets[4]}; no port opened.')
        return 0
    # A fault requires a fresh inventory and explicit operator invocation.
    from apply_policy import preflight
    bus=None;inv=None;restore={};code=0
    try:
        bus=Bus(cfg['port'],cfg['baud'])
        inv=bus.inventory();preflight(inv)
        rows=bus.read_diagnostic(combined=True)
        guard=RestoreGuard(inv);guard.check(rows)
        bus.armed_ids=list(IDS)
        for i in IDS:
            motor=next(m for m in inv if m['id']==i)
            restore[i]=(motor['goal_current_raw'],motor['position_p_gain'],
                        motor['profile_velocity_raw'],motor['profile_acceleration_raw'])
            bus.write_checked(i,102,0,2)
            bus.write_checked(i,84,500,2)
            bus.write_checked(i,112,20,4)
            bus.write_checked(i,108,10,4)
            bus.write_checked(i,98,25,1)
        for i in IDS:
            if bus.read_int(i,102,2)!=0:raise Fault(f'ID{i}: current must be zero before torque ON')
            bus.write_checked(i,64,1,1)
            if bus.read_int(i,102,2)!=0:raise Fault(f'ID{i}: current changed during torque ON')
        def enable_current(goals):
            for i in IDS:
                if bus.read_int(i,64,1)!=1 or bus.read_int(i,102,2)!=0:
                    raise Fault(f'ID{i}: torque/current state changed during return setup')
                if bus.read_int(i,116,4)!=goals[i]:raise Fault(f'ID{i}: open target not confirmed')
                motor=next(m for m in inv if m['id']==i)
                bus.write_checked(i,102,min(CURRENT_CAP_MA,motor['current_limit_ma']),2)
        print('Returning ID1, ID2, ID3 and ID4 to saved open positions...',flush=True)
        def emit(event):print(json.dumps(event,ensure_ascii=False),flush=True)
        move_to_open(bus,cfg,guard,emit,after_goals=enable_current)
        print('ID1, ID2, ID3 and ID4 reached the saved open positions.',flush=True)
    except KeyboardInterrupt:
        code=1;print('Interrupted: stopping motors immediately.',flush=True)
    except Exception as exc:
        code=1;print(f'STOP: {type(exc).__name__}: {exc}',flush=True)
    finally:
        if bus is not None:
            try:
                errors=bus.stop()
                if errors:
                    code=1
                    for error in errors:print(error,flush=True)
                elif bus.armed_ids:
                    print('ID1, ID2, ID3 and ID4 torque OFF confirmed.',flush=True)
                    for i in bus.armed_ids:
                        if i not in restore:continue
                        goal,p_gain,velocity,acceleration=restore[i]
                        for addr,value,size in ((102,goal & 0xffff,2),(84,p_gain,2),
                                                (112,velocity,4),(108,acceleration,4),(98,0,1)):
                            try:bus.write_checked(i,addr,value,size)
                            except Exception as exc:
                                code=1;print(f'ID{i}: restore register {addr} failed: {exc}',flush=True)
            finally:bus.close()
    return code


if __name__=='__main__':raise SystemExit(main())
