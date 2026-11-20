"""Single-owner primitive scheduler; no ROS or hardware dependencies."""
import threading
import time
from dataclasses import dataclass, field


class Rejected(RuntimeError):
    pass


@dataclass
class Job:
    name: str
    turns: int
    owner: str
    deadline: float
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: tuple = (False, 'INTERNAL', 'unfinished')


class Worker:
    def __init__(self, hardware, *, period=.03, settle=.3, report=print):
        self.hw, self.period, self.settle, self.report = hardware, period, settle, report
        self.lock = threading.Lock()
        self.last_close_error = None
        self.state, self.owner, self.job = 'OPEN', None, None
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._run, name='left-hand-io')
        self.thread.start()

    def submit(self, name, turns, owner, timeout):
        with self.lock:
            if self.stop.is_set() or self.state=='FAULT': raise Rejected('FAULT: restart required')
            if self.job: raise Rejected('BUSY')
            if not owner or not 0 < timeout <= 120: raise Rejected('INVALID_REQUEST')
            if name not in ('grasp_upper_layer','release_upper_layer','rotate_upper_layer'):
                raise Rejected('INVALID_REQUEST')
            if turns not in ((-1,1,2) if name=='rotate_upper_layer' else (0,)):
                raise Rejected('INVALID_TURNS')
            if name=='grasp_upper_layer':
                if self.state!='OPEN': raise Rejected('ALREADY_HOLDING')
            elif self.state!='HOLDING' or self.owner!=owner: raise Rejected('NOT_OWNER_OR_NOT_HOLDING')
            self.job=Job(name, turns, owner, time.monotonic()+timeout)
            self.wake.set()
            return self.job

    def close(self):
        was_alive=self.thread.is_alive()
        self.stop.set(); self.wake.set(); self.thread.join()
        # A fault may have already closed the port; verify again when quitting.
        if not was_alive:
            previous_error=self.last_close_error
            try:
                self.hw.close()
                # A new wrist read cannot prove a previous finger-stop or restore succeeded.
                if previous_error:
                    self.report('Previous stop failure still requires checking: '+previous_error)
                self.last_close_error=previous_error
            except Exception as exc:
                self.last_close_error=str(exc)
                self.report('STOP_UNCONFIRMED: '+str(exc))
        return self.last_close_error is None

    def _check(self,j):
        if self.stop.is_set() or (j and j.cancel.is_set()): raise InterruptedError('cancelled')
        if j and time.monotonic()>=j.deadline: raise TimeoutError('primitive deadline exceeded')

    def _run(self):
        started=None; stable=None; last=None; last_grasp_report=0.
        try:
            while not self.stop.is_set():
                with self.lock: j=self.job
                if not j and self.state=='OPEN':
                    self.wake.wait(.05); self.wake.clear(); continue
                t=time.monotonic(); self._check(j)
                begin_rotation=False
                if j is not None and started is not j:
                    started=j; stable=None
                    if j.name=='grasp_upper_layer':
                        self.hw.start(lambda:self._check(j)); last=None
                    elif j.name=='rotate_upper_layer':
                        begin_rotation=True
                releasing=bool(j and j.name=='release_upper_layer')
                now=time.monotonic()
                if last is not None and now-last>.1:
                    raise RuntimeError(f'control deadline exceeded: interval_ms={(now-last)*1000:.2f}, limit_ms=100')
                dt=self.period if last is None else now-last; last=now
                self._check(j)
                grip_ok, open_ok, wrist_ok=self.hw.tick(releasing,dt)
                tick_elapsed=time.monotonic()-now
                if tick_elapsed>.1:
                    raise RuntimeError(f'IO deadline exceeded: elapsed_ms={tick_elapsed*1000:.2f}, limit_ms=100')
                if j and j.name=='grasp_upper_layer' and not grip_ok and hasattr(self.hw,'grasp_diagnostics') and now-last_grasp_report>=2.:
                    self.report('GRASP_WAIT: '+self.hw.grasp_diagnostics())
                    last_grasp_report=now
                self._check(j)
                if not releasing and self.state=='HOLDING' and not grip_ok:
                    raise RuntimeError('grasp telemetry left verified window')
                if begin_rotation:
                    self._check(j)
                    self.hw.rotate(j.turns)
                    wrist_ok=False  # require subsequent measured position, not old target
                if j:
                    ok=open_ok if releasing else grip_ok and (j.name!='rotate_upper_layer' or wrist_ok)
                    stable=(now if stable is None else stable) if ok else None
                    if stable is not None and now-stable>=self.settle:
                        if releasing:
                            self.hw.finish_release(check=lambda: self._check(j))
                        self._check(j)
                        with self.lock:
                            self.state='OPEN' if releasing else 'HOLDING'
                            self.owner=None if releasing else j.owner
                            self.job=None
                            j.result=(True,'','open verified' if releasing else 'encoder posture verified')
                            j.done.set()
                self.stop.wait(max(0,self.period-(time.monotonic()-t)))
        except Exception as exc:
            with self.lock:
                fault_state=self.state
                self.state='FAULT'  # Reject new goals throughout recovery/cleanup.
            detail=str(exc)+f'; primitive={j.name if j else "hold"}; execution_id={j.owner if j else self.owner}'
            if hasattr(self.hw,'timing_diagnostics'):
                try:
                    detail+='; state='+fault_state+'; '+self.hw.timing_diagnostics()
                except Exception:
                    pass  # Diagnostics must never prevent the existing stop path.
            if hasattr(self.hw,'fault_context'):
                try:
                    detail+='; '+self.hw.fault_context()
                except Exception:
                    pass  # Context formatting must not delay or block stop.
            if isinstance(exc,TimeoutError) and hasattr(self.hw,'grasp_diagnostics'):
                detail+='; '+self.hw.grasp_diagnostics()
            can_recover = getattr(self.hw, 'can_recover_after_fault', None)
            recoverable = (
                isinstance(exc, TimeoutError)
                or str(exc).startswith(('IO deadline exceeded', 'control deadline exceeded',
                                        'grasp telemetry left verified window'))
                or (can_recover is not None and can_recover(exc))
            )
            recovery=getattr(self.hw,'recover_fingers_after_fault',None)
            if (recoverable and recovery is not None and fault_state=='HOLDING'
                    and not self.stop.is_set() and not (j and j.cancel.is_set())
                    and not (j and j.name=='release_upper_layer')):
                def recovery_check():
                    if self.stop.is_set() or (j and j.cancel.is_set()):
                        raise InterruptedError('recovery cancelled')
                try:
                    recovery(check=recovery_check)
                    detail+='; finger_recovery=initial pose reached'
                except Exception as recovery_error:
                    detail+='; finger_recovery=skipped/failed: '+str(recovery_error)
            error=('CANCELLED' if isinstance(exc,InterruptedError) else
                   'TIMEOUT' if isinstance(exc,TimeoutError) else 'HARDWARE_FAULT',detail)
        else: error=('CANCELLED','controller shutdown')
        finally:
            with self.lock: self.state='FAULT'
            try:
                self.hw.close()
                self.last_close_error=None
            except Exception as exc:
                self.last_close_error=str(exc)
                # Keep the original operation failure as well as cleanup failure.
                error=('STOP_UNCONFIRMED',f'original {error[0]}: {error[1]}; cleanup: {exc}')
            with self.lock:
                if self.job:
                    self.job.result=(False,*error); self.job.done.set(); self.job=None
            if error!=('CANCELLED','controller shutdown'):
                self.report(': '.join(error))
