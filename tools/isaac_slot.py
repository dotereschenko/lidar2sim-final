"""Serialize managed Isaac jobs and restore explicitly authorized shared sessions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess


def acquire_slot(path):
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o666)
    stream=os.fdopen(fd,'w')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close();raise RuntimeError('Isaac slot occupied by another job') from None
    return stream


def classify_sessions(rows, allowed_pause):
    active=[x['Names'] for x in rows if any(tag in x['Image'].lower() for tag in ('isaac-sim', 'isaac-lab', 'isaaclab')) or x['Names'].startswith('lidar2sim-')]
    return [n for n in active if n in allowed_pause],[n for n in active if n not in allowed_pause]


def execute(job_file):
    job_file=Path(job_file);job=json.loads(job_file.read_text())
    with acquire_slot(job.get('lock_path','/tmp/lidar2sim-isaac-slot.lock')):
        raw=subprocess.check_output(['docker','ps','--format','{{json .}}'],text=True)
        rows=[json.loads(line) for line in raw.splitlines() if line]
        paused,blocked=classify_sessions(rows,job.get('pause_known_sessions',[]))
        if blocked:raise RuntimeError('Isaac sessions require a free slot: '+', '.join(blocked))
        for name in paused:
            auto=subprocess.check_output(['docker','inspect','--format','{{json .HostConfig.AutoRemove}}',name],text=True).strip()
            if auto!='false':raise RuntimeError('Cannot safely restore auto-remove session '+name)
        record={'paused':paused,'blocked':blocked,'container_name':job['container_name'],'restored':[]}
        path=job_file.with_name('slot.json');path.write_text(json.dumps(record,indent=2))
        stopped=[]
        try:
            for name in paused:
                subprocess.run(['docker','stop',name],check=True,timeout=60);stopped.append(name)
            result=subprocess.run(job['command'],timeout=job['timeout_s'],check=False)
            return result.returncode
        finally:
            subprocess.run(['docker','rm','-f',job['container_name']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
            errors=[]
            for name in stopped:
                result=subprocess.run(['docker','start',name],timeout=60,check=False)
                if result.returncode:errors.append(name)
                else:record['restored'].append(name)
            record['restore_errors']=errors;path.write_text(json.dumps(record,indent=2))
            if errors:raise RuntimeError('Failed to restore sessions: '+', '.join(errors))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job');args=p.parse_args()
    try:raise SystemExit(execute(args.job))
    except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
        p.exit(2,str(exc)+'\n')
