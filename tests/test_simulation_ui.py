import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from lidar2sim.web.server import Jobs
from lidar2sim.web.simulation import Simulations


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    monkeypatch.delenv('LIDAR2SIM_ISAAC_COMMAND', raising=False)
    monkeypatch.setenv('LIDAR2SIM_SCENE_A_JOB', 'a'*32)
    jobs = Jobs(tmp_path/'jobs')
    folder = jobs.root/('a'*32)/'result'; folder.mkdir(parents=True)
    (folder/'scene.usd').write_bytes(b'test USD bytes')
    (folder/'report.json').write_text(json.dumps({'input': {'points': 100}, 'solution': 'nksr'}))
    jobs.update('a'*32, status='completed')
    root = tmp_path/'project'; (root/'data/scene_a').mkdir(parents=True)
    (root/'data/scene_a/test_config.json').write_text(json.dumps({
        'routes': [{'name': 'test', 'points': [[0, 0], [1, 0], [2, 0]]}],
        'body': {'size': [.4, .4, .3], 'mass': 20}, 'speed': .2}))
    sim = Simulations(jobs, root)
    yield sim
    jobs.pool.shutdown()


def wait_for(sim, ident):
    for _ in range(100):
        state = sim.read(ident)
        if state['status'] not in ('queued', 'running'): return state
        time.sleep(.02)
    pytest.fail('Simulation worker did not finish')


def test_unconfigured_runner_cannot_report_preview_as_physics(simulation):
    catalog = simulation.catalog()
    assert catalog['scene']['id'] == 'a'*32
    assert not catalog['runner_configured']
    with pytest.raises(ValueError, match='недоступен'):
        simulation.start({'robot': 'velocity_body', 'route': 0, 'start': 0, 'end': 1})
    assert not simulation.directory.exists()


@pytest.mark.parametrize('payload', [
    {'robot': 'other', 'route': 0, 'start': 0, 'end': 1},
    {'robot': 'velocity_body', 'route': -1, 'start': 0, 'end': 1},
    {'robot': 'velocity_body', 'route': 0, 'start': True, 'end': 1},
    {'robot': 'velocity_body', 'route': 0, 'start': 1, 'end': 1},
    {'robot': 'velocity_body', 'route': 0, 'start': 0, 'end': 99},
])
def test_invalid_selection_never_starts_runner(simulation, monkeypatch, payload):
    monkeypatch.setenv('LIDAR2SIM_ISAAC_COMMAND', json.dumps([sys.executable]))
    with pytest.raises(ValueError): simulation.start(payload)
    assert not simulation.directory.exists()


@pytest.mark.parametrize('valid', [True, False])
def test_output_must_match_scene_and_requested_route(simulation, monkeypatch, tmp_path, valid):
    # Contract fixture, not an Isaac simulation or a physics acceptance test.
    runner = tmp_path/'runner.py'
    runner.write_text('''import argparse,pathlib,json,hashlib
p=argparse.ArgumentParser()
for key in ('scene','config','out','route','start-index','end-index'):p.add_argument('--'+key)
a=p.parse_args();out=pathlib.Path(a.out);out.mkdir()
r={'scene_sha256':hashlib.sha256(pathlib.Path(a.scene).read_bytes()).hexdigest(),
'config_sha256':hashlib.sha256(pathlib.Path(a.config).read_bytes()).hexdigest(),
'robot':'velocity_body','route':int(a.route),'start':int(a.start_index),'end':int(a.end_index),'outcome':'reached'}
''' + ('' if valid else "r['route']=123\n") + '''
(out/'recording.json').write_text(json.dumps(r));(out/'simulation.mp4').write_bytes(b'test video fixture')
''')
    monkeypatch.setenv('LIDAR2SIM_ISAAC_COMMAND', json.dumps([sys.executable, str(runner)]))
    job = simulation.start({'robot': 'velocity_body', 'route': 0, 'start': 0, 'end': 1})
    state = wait_for(simulation, job['id'])
    assert state['status'] == ('completed' if valid else 'failed')
    assert ('simulation.mp4' in state['files']) is valid
    with pytest.raises(ValueError): simulation.file(job['id'], '../scene.usd')


def test_single_flight_and_cancel_only_managed_process(simulation, monkeypatch):
    monkeypatch.setenv('LIDAR2SIM_ISAAC_COMMAND', json.dumps([sys.executable, '-c', 'import time;time.sleep(30)']))
    payload = {'robot': 'velocity_body', 'route': 0, 'start': 0, 'end': 1}
    job = simulation.start(payload)
    try:
        with pytest.raises(ValueError, match='Уже выполняется'): simulation.start(payload)
    finally:
        simulation.cancel(job['id'])
    assert wait_for(simulation, job['id'])['status'] == 'cancelled'
    for _ in range(100):
        if not simulation.processes: break
        time.sleep(.02)
    assert not simulation.processes


def test_restart_marks_incomplete_run_failed(simulation):
    ident = 'b'*32
    simulation.folder(ident).mkdir(parents=True)
    simulation.update(ident, status='running')
    Simulations(simulation.jobs, simulation.root)
    assert simulation.read(ident)['status'] == 'failed'
