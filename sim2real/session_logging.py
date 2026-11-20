"""Asynchronous local session logs. Never reads a motor or changes transport."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import traceback

_KST=dt.timezone(dt.timedelta(hours=9))
_active=None


def stamp():
    return dt.datetime.now(_KST).isoformat(timespec='milliseconds')


class _Tee:
    def __init__(self, original, session, name):
        self.original,self.session,self.name=original,session,name
    def write(self,text):
        self.session.emit('console',stream=self.name,text=text)
        return self.original.write(text)
    def flush(self):
        return self.original.flush()
    def __getattr__(self,name):
        return getattr(self.original,name)


class SessionLog:
    def __init__(self, root=None, *, capture_console=True):
        root=Path(root) if root else Path(__file__).resolve().parent.parent/'logs'/'primitive'
        now=dt.datetime.now(_KST)
        self.directory=root/now.strftime('%Y-%m-%d')/(now.strftime('%H%M%S_%f')+f'_{os.getpid()}')
        self.directory.mkdir(parents=True,exist_ok=False)
        self.events=(self.directory/'events.jsonl').open('a',encoding='utf-8',buffering=1)
        self.console=(self.directory/'console.log').open('a',encoding='utf-8',buffering=1)
        self.queue=queue.SimpleQueue()
        self.closed=False;self.write_error=None
        self.stdout,self.stderr=sys.stdout,sys.stderr
        self.thread=threading.Thread(target=self._write,name='session-log-writer',daemon=True)
        self.thread.start()
        latest=root/f'.latest_{os.getpid()}.tmp'
        latest.write_text(json.dumps({'started_at':stamp(),'directory':str(self.directory.resolve())},ensure_ascii=False,indent=2),encoding='utf-8')
        latest.replace(root/'latest.json')
        manifest={'started_at':stamp(),'pid':os.getpid(),'python':sys.executable,
                  'python_version':sys.version,'argv':sys.argv,
                  'ros_domain_id':os.environ.get('ROS_DOMAIN_ID'),
                  'ros_localhost_only':os.environ.get('ROS_LOCALHOST_ONLY')}
        (self.directory/'session.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        if capture_console:
            sys.stdout=_Tee(self.stdout,self,'stdout')
            sys.stderr=_Tee(self.stderr,self,'stderr')
        self.emit('session_started',**manifest)
        print('SESSION LOG: '+str(self.directory.resolve()),flush=True)

    def emit(self,event,**fields):
        if not self.closed:
            self.queue.put({'time':stamp(),'event':event,**fields})

    def snapshot(self,label,path,*,copy=False):
        path=Path(path)
        if not path.is_file():
            self.emit('file_missing',label=label,path=str(path));return
        data=path.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        self.emit('file_snapshot',label=label,path=str(path.resolve()),sha256=digest)
        if copy:
            (self.directory/(label+path.suffix)).write_bytes(data)

    def exception(self,event):
        self.emit(event,traceback=traceback.format_exc())

    def _write(self):
        try:
            while True:
                record=self.queue.get()
                if record is None:break
                try:
                    self.events.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if record['event']=='console':
                        self.console.write(record['text'])
                    else:
                        details={key:value for key,value in record.items() if key not in ('time','event')}
                        self.console.write(f"[{record['time']}] {record['event']} "+json.dumps(details,ensure_ascii=False)+'\n')
                    self.console.flush()
                except Exception as exc:
                    if self.write_error is None:
                        self.write_error=str(exc)
                        try:self.stderr.write('SESSION LOG WRITE FAILED: '+str(exc)+'\n')
                        except Exception:pass
        finally:
            self.events.close();self.console.close()

    def close(self):
        if self.closed:return
        self.emit('session_closed')
        self.closed=True
        if getattr(sys.stdout,'session',None) is self:sys.stdout=self.stdout
        if getattr(sys.stderr,'session',None) is self:sys.stderr=self.stderr
        self.queue.put(None)
        self.thread.join(timeout=5.)
        if self.thread.is_alive():
            self.stderr.write('SESSION LOG: writer still pending; final records may be incomplete\n')


def get_session():
    global _active
    if _active is None or _active.closed:
        _active=SessionLog()
    return _active
