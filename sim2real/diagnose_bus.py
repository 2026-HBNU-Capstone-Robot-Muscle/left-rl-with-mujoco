"""Read-only mixed-bus diagnosis. Never writes torque, position or configuration."""
import argparse
import json
import time
from bus import Bus
from wrist_protocol import sts_read


def diagnose(bus, count=30):
    report={'initial':{},'counts':{'finger_ok':0,'wrist_ok':0},'errors':[]}
    for i in range(1,5):
        try:report['initial'][f'id{i}_torque']=bus.read_int(i,64,1)
        except Exception as exc:report['errors'].append(f'ID{i} torque read: {exc}')
    try:report['initial']['id10_torque']=int.from_bytes(sts_read(bus.port,10,40,1),'little')
    except Exception as exc:report['errors'].append(f'ID10 torque read: {exc}')
    if report['errors'] or any(v!=0 for v in report['initial'].values()):
        report['skipped']='Torque OFF was not verified for all five motors; mixed reads skipped.'
        return report
    for cycle in range(count):
        try:
            rows=bus.read_diagnostic(combined=True)
            report['counts']['finger_ok']+=1
            if cycle==0:report['initial']['finger_positions']={r['id']:r['position_tick'] for r in rows}
        except Exception as exc:report['errors'].append(f'cycle {cycle} finger read: {exc}')
        try:
            position=int.from_bytes(sts_read(bus.port,10,56,2),'little')
            report['counts']['wrist_ok']+=1
            if cycle==0:report['initial']['wrist_position']=position
        except Exception as exc:report['errors'].append(f'cycle {cycle} wrist read: {exc}')
        time.sleep(.01)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='COM8');p.add_argument('--baud',type=int,default=1000000)
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--wrist-goal-check',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.wrist_goal_check:p.error('disabled: wrist torque became ON during this diagnostic; use read-only mode')
    if not 1<=args.samples<=1000:p.error('samples must be 1..1000')
    try:
        bus=Bus(args.port,args.baud)
        try:
            report=diagnose(bus,args.samples)
        finally:bus.close()
    except Exception as exc:
        print(json.dumps({'open_error':str(exc)},ensure_ascii=False,indent=2));return 1
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if report['errors'] or 'skipped' in report else 0

if __name__=='__main__':raise SystemExit(main())
