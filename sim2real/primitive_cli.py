"""Interactive standalone left hand primitives; default simulates without a port."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from primitive_worker import Worker, Rejected

def explicit_home_after_stop(worker, hardware, *, fingers_already_open=False):
    """Release the COM port, then run the existing explicit four-finger recovery."""
    if not worker.close():
        print('HOME BLOCKED: previous torque OFF could not be confirmed.',flush=True)
        return False,False
    if not fingers_already_open:
        helper=Path(__file__).resolve().with_name('return_fingers.py')
        if not helper.is_file():
            print(f'HOME FAILED: recovery helper missing: {helper}',flush=True)
            return True,False
        print('Running checked four-finger return to saved initial positions...',flush=True)
        try:
            if subprocess.run([sys.executable,str(helper),'--execute'],check=False).returncode!=0:
                return True,False
        except OSError as exc:
            print(f'HOME FAILED: recovery helper could not start: {exc}',flush=True)
            return True,False
    try:
        hardware.return_wrist_to_reference()
        print('HOME COMPLETE: fingers at saved initial positions, wrist at reference; torque OFF.',flush=True)
        return True,True
    except Exception as exc:
        print(f'HOME FAILED: wrist return: {exc}',flush=True)
        return True,False


class SimulatedHardware:
    """Protocol simulator for exercising request sequencing, not robot physics."""
    def start(self, check): check(); print('SIM: start')
    def tick(self, releasing, dt): return True, True, True
    def rotate(self, turns): print(f'SIM: wrist quarter turns {turns}')
    def close(self): print('SIM: close')
    def return_wrist_to_reference(self): print('SIM: wrist to reference 2400; no motor output')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Enable real hardware')
    parser.add_argument('--calibration', help='Wrist/grasp JSON; required with --execute')
    parser.add_argument('--allow-provisional', action='store_true', help='Explicitly use user-approved provisional calibration')
    parser.add_argument('--timeout', type=float, default=30.)
    args=parser.parse_args()
    if not 0 < args.timeout <= 120: parser.error('timeout must be > 0 and <= 120')
    if args.execute:
        if not args.calibration: parser.error('--execute requires --calibration')
        from primitive_hardware import Hardware
        try:
            hardware=Hardware(args.calibration,allow_provisional=args.allow_provisional)
        except ValueError as exc:
            parser.exit(2, f'실기 시작 불가: {exc}\n')
        if hardware.c.get('verified') is not True:
            print('PROVISIONAL: encoder windows do not prove object contact or holding force',flush=True)
        period=hardware.cfg['period_s']
    else:
        hardware=SimulatedHardware(); period=.03
    worker=Worker(hardware, period=period)
    print('CONTROLLER: diagnostic-20261116-23-home-50deg')
    print('MODE:', 'REAL MOTORS' if args.execute else 'SIMULATED; no motor output')
    print('Commands: grasp | rotate 1 | rotate -1 | rotate 2 | release | home | quit')
    active=None
    quit_release_ok=True
    home_ok=True;worker_closed=False;stopped=False
    try:
        while True:
            try: words=input('left-hand> ').split()
            except EOFError: break
            if not words: continue
            if words==['quit']:
                if worker.state=='HOLDING':
                    print('Returning all four fingers to saved initial positions before exit...',flush=True)
                    try:
                        active=worker.submit('release_upper_layer',0,'standalone',args.timeout)
                        while not active.done.wait(.05): pass
                        print('RETURN RESULT:',active.result,flush=True)
                        quit_release_ok=bool(active.result[0])
                        active=None
                    except Rejected as exc:
                        quit_release_ok=False
                        print('RETURN FAILED:',exc,flush=True)
                break
            if words==['home']:
                if worker.state=='HOLDING':
                    active=worker.submit('release_upper_layer',0,'standalone',args.timeout)
                    while not active.done.wait(.05): pass
                    print('FINGER RETURN RESULT:',active.result,flush=True)
                    fingers_open=bool(active.result[0]);active=None
                    if not fingers_open:
                        home_ok=False;stopped=worker.close()
                    else:
                        stopped,home_ok=explicit_home_after_stop(worker,hardware,fingers_already_open=True)
                elif args.execute:
                    stopped,home_ok=explicit_home_after_stop(worker,hardware)
                else:
                    print('SIM: home; no motor output',flush=True)
                    continue
                worker_closed=True
                break
            if words==['grasp']: name,turns='grasp_upper_layer',0
            elif words==['release']: name,turns='release_upper_layer',0
            elif len(words)==2 and words[0]=='rotate' and words[1] in ('1','-1','2'):
                name,turns='rotate_upper_layer',int(words[1])
            else:
                print('Invalid command'); continue
            try: active=worker.submit(name,turns,'standalone',args.timeout)
            except Rejected as exc:
                print('REJECTED:',exc); continue
            try:
                while not active.done.wait(.05): pass
            except KeyboardInterrupt:
                active.cancel.set()
                print('Cancelling; waiting for stop...')
                while not active.done.wait(.05): pass
            print('RESULT:',active.result)
            active=None
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        if active is not None: active.cancel.set()
        if not worker_closed:
            stopped=worker.close()
        if not stopped:
            print('STOP_UNCONFIRMED or HOME FAILED: check power and connection',flush=True)
    return 0 if stopped and quit_release_ok and home_ok else 1

if __name__=='__main__': raise SystemExit(main())
