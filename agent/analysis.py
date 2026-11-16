"""Fail-closed macOS sandbox for model-written numeric analysis; no input-tree copies."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
MAX_OUTPUT=65536


def profile(workspace, readonly):
    def literal(path):return json.dumps(str(Path(path).resolve()))
    return '\n'.join(['(version 1)','(allow default)','(deny network*)',
        '(deny file-write*)',
        '(allow file-write* (subpath '+literal(workspace)+') (literal "/dev/null"))',
        '(deny file-read-data (subpath '+literal(Path.home())+'))',
        *['(allow file-read-data (subpath '+literal(p)+'))' for p in (ROOT/'.venv-agent',Path(sys.prefix),Path(sys.base_prefix),readonly,workspace)],
        '(allow file-read-data (literal '+literal(sys.executable)+'))',
        '(deny file-write* (subpath '+literal(readonly)+'))'])


def bounded_process(args,cwd,timeout=30):
    env={'PATH':'/usr/bin:/bin','HOME':str(cwd),'TMPDIR':str(cwd),
         'MPLCONFIGDIR':str(cwd/'mplconfig'),'MPLBACKEND':'Agg','LANG':'en_US.UTF-8'}
    process=subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    buffers={'stdout':bytearray(),'stderr':bytearray()};selector=selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ,'stdout');selector.register(process.stderr,selectors.EVENT_READ,'stderr')
    deadline=time.monotonic()+timeout;reason=None
    try:
        while selector.get_map():
            if time.monotonic()>deadline:reason='timeout';break
            for key,_ in selector.select(min(.2,max(0,deadline-time.monotonic()))):
                chunk=os.read(key.fd,8192)
                if not chunk:selector.unregister(key.fileobj);continue
                remaining=MAX_OUTPUT-sum(map(len,buffers.values()))
                buffers[key.data].extend(chunk[:remaining])
                if sum(map(len,buffers.values()))>=MAX_OUTPUT:reason='output_limit';break
            if reason:break
        if reason:os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=2)
    finally:
        selector.close();process.stdout.close();process.stderr.close()
        # Also terminate descendants on success; they must not outlive the tool.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    return dict(returncode=process.returncode,output=buffers['stdout'].decode(errors='replace'),
                stderr=buffers['stderr'].decode(errors='replace'),stopped=reason)


def sandbox_self_test():
    with tempfile.TemporaryDirectory(prefix='sandbox-probe-',dir=ROOT/'agent') as tmp:
        outer=Path(tmp);work=outer/'work';work.mkdir();facts=work/'facts';facts.mkdir()
        (facts/'probe.json').write_text('{}')
        policy=work/'policy.sb';policy.write_text(profile(work,facts))
        script=work/'probe.py'
        script.write_text('''import errno,json,os,pathlib,socket
checks={}
checks['allowed_read']=pathlib.Path('facts/probe.json').read_text()=='{}'
try:
 pathlib.Path('facts/probe.json').write_text('bad');checks['facts_readonly']=False
except PermissionError:checks['facts_readonly']=True
try:
 fd=os.open(ENV_PATH,os.O_RDONLY);os.close(fd);checks['env_read_denied']=False
except PermissionError:checks['env_read_denied']=True
except FileNotFoundError:checks['env_read_denied']=False
try:
 pathlib.Path('../outside-write').write_text('bad');checks['outside_write_denied']=False
except PermissionError:checks['outside_write_denied']=True
try:
 s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',9));checks['network_denied']=False
except OSError as e:checks['network_denied']=e.errno in (errno.EPERM,errno.EACCES)
pathlib.Path('allowed-write').write_text('ok');checks['workspace_write']=True
print(json.dumps(checks))
'''.replace('ENV_PATH',repr(str(ROOT/'.env'))))
        try:result=bounded_process(['/usr/bin/sandbox-exec','-f',str(policy),sys.executable,'-I',str(script)],work,10)
        except OSError as error:return dict(available=False,reason='sandbox_launcher_unavailable',stderr=str(error),model_code_executed=False)
        # The trusted probe never reads secret content; launcher stderr is diagnostic.
        diagnostics=dict(returncode=result['returncode'],stderr=result['stderr'],stopped=result['stopped'])
        if result['returncode']!=0:
            reason='sandbox_apply_failed' if result['stderr'].lstrip().startswith('sandbox-exec:') else 'sandboxed_python_failed'
            return dict(available=False,reason=reason,model_code_executed=False,**diagnostics)
        try:checks=json.loads(result['output'])
        except ValueError:return dict(available=False,reason='sandbox_probe_invalid',model_code_executed=False,**diagnostics)
        available=bool(checks) and all(checks.values())
        return dict(available=available,checks=checks,model_code_executed=False,
                    **({} if available else {'reason':'sandbox_permission_checks_failed'}),**diagnostics)


def run_analysis(script: str, workspace: Path, facts: list[dict]) -> dict:
    """Run model-written Python only after enforced no-network/read-only checks pass."""
    if not isinstance(script,str) or len(script)>32000:raise ValueError('Invalid analysis script')
    probe=sandbox_self_test()
    if not probe['available']:
        return dict(status='unavailable',reason=probe['reason'],model_code_executed=False,
                    message='Analysis sandbox unavailable; model-written code was not run.')
    work=Path(tempfile.mkdtemp(prefix='analysis-',dir=workspace));readonly=work/'facts';readonly.mkdir()
    # Derived numeric rows only, not copies of upstream tables or listing text.
    (readonly/'homes.json').write_text(json.dumps(facts))
    source=work/'analysis.py';source.write_text(script)
    runner=work/'runner.py';runner.write_text('''import resource,runpy
resource.setrlimit(resource.RLIMIT_FSIZE,(5*1024*1024,5*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU,(25,25))
runpy.run_path('analysis.py',run_name='__main__')
''')
    policy=work/'policy.sb';policy.write_text(profile(work,readonly))
    result=bounded_process(['/usr/bin/sandbox-exec','-f',str(policy),sys.executable,'-I',str(runner)],work)
    return dict(status='ok' if result['returncode']==0 and not result['stopped'] else 'failed',
                model_code_executed=True,workspace=str(work.relative_to(workspace)),**result)


def sun_chart(rows, destination):
    """Fixed trusted chart code, receiving only ids and numeric durations; no listing text."""
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'agent/state/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    usable=[r for r in rows if isinstance(r.get('minutes'),(int,float))]
    if not usable:return dict(status='unavailable',reason='No reliable computed durations',figure=None)
    fig,ax=plt.subplots(figsize=(8,4));ax.bar([r['sid'] for r in usable],[r['minutes'] for r in usable],color='#505050')
    ax.set_ylabel('Best main-room direct sun (minutes)');ax.set_xlabel('Scan id')
    ax.set_title('Windows-only geometry on March 20, 2026')
    fig.tight_layout();destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=130);plt.close(fig)
    return dict(status='ok',figure=str(destination),rows=usable,method='Fixed trusted chart, not model-written analysis',
                caveats=['Best-room maxima, not sums. Geometric direct sun is not brightness. Placement and source caveats remain in dossiers.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        result=sandbox_self_test();print(json.dumps(result));sys.exit(0 if result['available'] else 2)
    parser.error('Use --self-test')
