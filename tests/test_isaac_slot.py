import fcntl
import json
import subprocess
import sys

import pytest


def test_slot_lock_rejects_concurrent_owner(tmp_path):
    from tools.isaac_slot import acquire_slot
    lock=tmp_path/'isaac.lock'
    with acquire_slot(lock):
        code='from tools.isaac_slot import acquire_slot; import sys; acquire_slot(sys.argv[1])'
        run=subprocess.run([sys.executable,'-c',code,str(lock)],capture_output=True,text=True)
        assert run.returncode != 0
        assert 'occupied' in run.stderr
    with acquire_slot(lock):pass


def test_only_explicit_known_sessions_may_be_paused():
    from tools.isaac_slot import classify_sessions
    rows=[{'Names':'isaac-stream','Image':'isaac-sim-xvfb:6.2'},
          {'Names':'another-team','Image':'isaac-sim-xvfb:6.2'},
          {'Names':'isaac-foxglove','Image':'foxglove:latest'}]
    pauses,blocked=classify_sessions(rows,['isaac-stream'])
    assert pauses==['isaac-stream']
    assert blocked==['another-team']


def test_qualified_runner_stages_imports_and_profile(tmp_path):
    from tools.spark_physics_runner import source_files
    scene=tmp_path/'scene.usd';scene.write_text('test')
    config=tmp_path/'config.json';config.write_text('{}')
    from lidar2sim.acceptance import DEFAULT_PROFILE
    pairs=source_files(scene,config,DEFAULT_PROFILE)
    names={name for _,name in pairs}
    assert {'tools/isaac_scene_test.py','lidar2sim/acceptance.py','lidar2sim/__init__.py','profile.yaml','isaac_slot.py'} <= names
    assert all(p.is_file() for p,_ in pairs)


def test_failed_job_restores_only_explicit_sessions(tmp_path,monkeypatch):
    from tools.isaac_slot import execute
    binary=tmp_path/'docker'
    log=tmp_path/'events';monkeypatch.setenv('EVENTS',str(log))
    binary.write_text('#!/usr/bin/env python3\nimport os,sys,json\na=sys.argv[1:]\nwith open(os.environ["EVENTS"],"a") as f:f.write(json.dumps(a)+"\\n")\nif a[0]=="ps":print(json.dumps({"Names":"isaac-stream","Image":"isaac-sim-xvfb:6.2"}))\nelif a[0]=="inspect":print("false")\nelif a[0]=="run":sys.exit(7)\n')
    binary.chmod(0o755)
    import os
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    job=tmp_path/'job.json';job.write_text(json.dumps({'command':['docker','run'],
        'container_name':'lidar2sim-test','pause_known_sessions':['isaac-stream'],
        'timeout_s':10,'lock_path':str(tmp_path/'slot.lock')}))
    assert execute(job)==7
    result=json.loads((tmp_path/'slot.json').read_text())
    assert result['restored']==['isaac-stream']
    events=[json.loads(x) for x in log.read_text().splitlines()]
    assert events[-1]==['start','isaac-stream']
    assert ['rm','-f','lidar2sim-test'] in events
