"""Run submitted reconstruction with only sanitized A input, no B/GT or network."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def prepare_input(scans, config, target):
    scans,config,target=Path(scans),Path(config),Path(target)
    manifest=json.loads((scans/'manifest.json').read_text())
    if manifest.get('units')!='m' or manifest.get('up_axis')!='Z':raise ValueError('Expected meters/Z up')
    target.mkdir(mode=0o755)
    shutil.copyfile(scans/'A.ply',target/'scene.ply')
    clean={'units':'m','up_axis':'Z'}
    (target/'manifest.json').write_text(json.dumps(clean)+'\n')
    shutil.copyfile(config,target/'config.yaml')
    trajectory=scans/'trajectory_A.csv'
    if trajectory.exists():shutil.copyfile(trajectory,target/'trajectory_A.csv')
    for path in target.iterdir():path.chmod(0o644)
    return trajectory.exists()


def command(image, source, output, with_trajectory, name):
    args=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL',
          '--pids-limit','512','--cpus','4','--memory','16g',
          '--user','65534:65534','--tmpfs','/tmp:rw,nosuid,size=512m',
          '-v',f'{source}:/input:ro','-v',f'{output}:/output:rw',image,
          '-m','lidar2sim','--input','/input/scene.ply','--manifest','/input/manifest.json',
          '--config','/input/config.yaml','--out','/output']
    if with_trajectory:args+=['--trajectory','/input/trajectory_A.csv']
    return args


def run(scans, config, out, image, timeout=1800):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.mkdir(mode=0o777);out.chmod(0o777)
    import uuid
    name='scan2sim-cpu-'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='lidar2sim-input-',dir=out.parent) as temporary:
        source=Path(temporary)/'input';has_trajectory=prepare_input(scans,config,source)
        args=command(image,source,out,has_trajectory,name)
        started=time.monotonic();timed_out=False
        with (out/'reconstruction.log').open('w') as log:
            proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;subprocess.run(['docker','stop','-t','5',name],capture_output=True,timeout=30)
                code=proc.wait(timeout=30)
            finally:
                subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
                if proc.poll() is None:proc.wait(timeout=30)
        record={'isolation':'A-only read-only input; no network; nonroot; no host sockets',
                'input_inventory':sorted(p.name for p in source.iterdir()),'image':image,
                'cpu_limit':4,'memory_limit_gib':16,'timeout_s':timeout,'timed_out':timed_out,
                'exit_code':code,'wall_s':round(time.monotonic()-started,3),
                'required_outputs_present':all((out/n).is_file() for n in ['visual.ply','collision.ply','scene.usd','report.json'])}
    (out/'isolation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['scans','config','out','image']:p.add_argument('--'+n,required=True)
    p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    if a.timeout<1:p.error('positive timeout required')
    result=run(a.scans,a.config,a.out,a.image,a.timeout);print(json.dumps(result))
    raise SystemExit(0 if result['exit_code']==0 and result['required_outputs_present'] else 2)
