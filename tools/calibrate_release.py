"""Operator-only five-repeat A/C calibration, isolated CPU and serialized Isaac."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.jury_reconstruct import run as reconstruct
from tools.isaac_slot import execute as physics_slot


def docker_cpu(image, mounts, arguments, log, timeout=600):
    name='scan2sim-eval-'+uuid.uuid4().hex
    cmd=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL',
         '--cpus','4','--memory','16g','--pids-limit','512',
         '--user',f'{os.getuid()}:{os.getgid()}','--tmpfs','/tmp:rw,nosuid,size=512m']
    for mount in mounts:cmd+=['-v',mount]
    cmd += [image,*arguments]
    try:
        with Path(log).open('w') as stream:
            return subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout).returncode
    finally:subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)


def run(jury,out,image,repeats=5,start_repeat=1):
    jury,out=Path(jury).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,mode=0o700)
    pinned=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
    cases=[{'scene':scene,'variant':variant,'repeat':i,'folder':f'{scene}/{variant}/r{i}'}
           for i in range(start_repeat,start_repeat+repeats) for scene in ['scene_a','scene_c'] for variant in ['baseline','final']]
    (out/'cases.json').write_text(json.dumps(cases,indent=2))
    (out/'environment.json').write_text(json.dumps({'image_id':pinned,'cpus':4,'memory_gib':16,'network':'none'},indent=2))
    failures=[]
    def produce():
        try:
            for case in cases:
                if (out/'.abort').exists():raise RuntimeError('Calibration aborted')
                folder=out/case['folder'];config=ROOT/'configs'/('baseline.yaml' if case['variant']=='baseline' else 'poc-high-detail.yaml')
                record=reconstruct(jury/case['scene']/'scans',config,folder,pinned)
                if record['exit_code'] or not record['required_outputs_present']:raise RuntimeError('CPU reconstruction failed: '+case['folder'])
                mounts=[f'{jury}:/jury:ro',f'{folder}:/output:rw']
                args=['/app/tools/eval_scene.py',f"/jury/{case['scene']}/source",'/output','--mode','b',
                      '--scans',f"/jury/{case['scene']}/scans",'--annotations',f"/jury/{case['scene']}/annotations",
                      '--profile','/app/configs/acceptance/lidar2sim-rules-v2.yaml','--dimensions']
                if docker_cpu(pinned,mounts,args,folder/'geometry.log')!=0:raise RuntimeError('Geometry evaluator failed: '+case['folder'])
                (folder/'.ready').write_text('ready\n')
                print('CPU ready',case['folder'],flush=True)
        except BaseException as exc:
            failures.append(str(exc));(out/'.abort').write_text(str(exc));print('CPU abort',str(exc),flush=True)
    work=out/'physics-work';(work/'tools').mkdir(parents=True);(work/'lidar2sim').mkdir()
    for name in ['tools/isaac_scene_test.py','lidar2sim/acceptance.py','lidar2sim/__init__.py']:
        shutil.copyfile(ROOT/name,work/name)
    shutil.copyfile(ROOT/'configs/acceptance/lidar2sim-rules-v2.yaml',work/'profile.yaml')
    consumer='''import json,os,signal,subprocess,time
from pathlib import Path
root=Path('/cases');cases=json.loads((root/'cases.json').read_text());done=[]
for case in cases:
 folder=root/case['folder']
 while not (folder/'.ready').exists():
  if (root/'.abort').exists():raise RuntimeError((root/'.abort').read_text())
  time.sleep(2)
 command=['/isaac-sim/python.sh','/work/tools/isaac_scene_test.py','--scene',str(folder/'scene.usd'),'--config',f"/jury/{case['scene']}/source/test_config.json",'--out',str(folder/'physics.json'),'--cpu-threads','4','--profile','/work/profile.yaml']
 start=time.monotonic()
 with (folder/'physics.log').open('w') as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=process.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
 done.append({**case,'exit_code':code,'wall_s':round(time.monotonic()-start,3)})
 (root/'physics-progress.json').write_text(json.dumps(done,indent=2))
 print('Physics done',case['folder'],code,flush=True)
 if code:raise RuntimeError('Physics process failed; inspect private log')
'''
    (work/'consumer.py').write_text(consumer)
    name='lidar2sim-calibration-'+uuid.uuid4().hex
    inner='Xvfb :99 -screen 0 1280x720x24 -nolisten tcp >/cases/xvfb.log 2>&1 & export DISPLAY=:99; /isaac-sim/python.sh /work/consumer.py'
    command=['docker','run','--rm','--name',name,'--runtime','nvidia','--network','none','--cpus','4','--memory','16g',
             '-e','ACCEPT_EULA=Y','-e','OMNI_KIT_ALLOW_ROOT=1','--entrypoint','/bin/bash',
             '-v',f'{out}:/cases:rw','-v',f'{jury}:/jury:ro','-v',f'{work}:/work:ro','isaac-sim-xvfb:6.2','-lc',inner]
    job={'command':command,'container_name':name,'timeout_s':7200,'pause_known_sessions':['isaac-stream','isaac-flight'],'lock_path':'/tmp/lidar2sim-isaac-slot.lock'}
    jobfile=work/'job.json';jobfile.write_text(json.dumps(job))
    thread=threading.Thread(target=produce);thread.start()
    try:
        code=physics_slot(jobfile)
        if code:
            failures.append(f'Physics batch exit {code}');(out/'.abort').write_text(f'Physics batch exit {code}')
    except BaseException as exc:
        failures.append(str(exc));(out/'.abort').write_text(str(exc))
    finally:thread.join()
    if failures:
        (out/'failure.json').write_text(json.dumps(failures));raise RuntimeError('; '.join(failures))
    for case in cases:
        folder=out/case['folder']
        args=['-m','lidar2sim.reporting','/output','--geometry','/output/eval_scene.json','--physics','/output/physics.json',
              '--physics-config',f"/jury/{case['scene']}/source/test_config.json",'--profile','/app/configs/acceptance/lidar2sim-rules-v2.yaml']
        if docker_cpu(pinned,[f'{jury}:/jury:ro',f'{folder}:/output:rw'],args,folder/'merge.log') not in (0,1):
            raise RuntimeError('Report merge failed: '+case['folder'])
    (out/'complete.json').write_text(json.dumps({'cases':len(cases),'repeats':repeats,'image_id':pinned,'physics_slot':json.loads((work/'slot.json').read_text())},indent=2))
    print('Calibration complete',len(cases),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jury',required=True);p.add_argument('--out',required=True);p.add_argument('--image',required=True);p.add_argument('--repeats',type=int,default=5);p.add_argument('--start-repeat',type=int,default=1);a=p.parse_args()
    if min(a.repeats,a.start_repeat)<1:p.error('positive repetitions required')
    run(a.jury,a.out,a.image,a.repeats,a.start_repeat)
