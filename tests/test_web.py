import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from lidar2sim.web.server import Jobs, handler
from lidar2sim.web.worker import normalize


@pytest.fixture
def server(tmp_path):
    jobs = Jobs(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(jobs))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    yield f'http://127.0.0.1:{server.server_port}', jobs
    server.shutdown(); server.server_close(); jobs.pool.shutdown()


def post(base, path, body=b'', headers=None):
    request = urllib.request.Request(base+path, data=body, headers=headers or {}, method='POST')
    with urllib.request.urlopen(request) as response: return json.load(response)


def test_upload_validation_and_origin(server):
    base, jobs = server
    for path, body, headers, status in [('/api/jobs', b'bad', {}, 400),
            ('/api/jobs', b'ply\n', {'Content-Length':str(513*1048576)}, 400),
            ('/api/jobs?method=other', b'ply\n', {}, 400),
            ('/api/jobs?scale=nan', b'ply\n', {}, 400),
            ('/api/jobs', b'ply\n', {'Origin':'https://example.com'}, 403)]:
        with pytest.raises(urllib.error.HTTPError) as e: post(base,path,body,headers)
        assert e.value.code == status
    assert not list(jobs.root.iterdir())


def test_upload_and_whitelisted_files(server):
    base, jobs = server
    result = post(base, '/api/jobs?scale=0.01&up=Y', b'ply\nend_header\n')
    assert result['settings']['scale'] == .01
    ident = result['id']
    assert (jobs.folder(ident)/'source.ply').read_bytes() == b'ply\nend_header\n'
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base+f'/api/jobs/{ident}/files/settings.json')
    assert e.value.code == 404
    with pytest.raises(ValueError): jobs.folder('../outside')


@pytest.mark.parametrize('method,depth,triangles', [
    ('baseline', 9, 20000), ('improved', 9, 60000), ('high-detail', 10, 200000)])
def test_upload_pins_selected_reconstruction_config(server, method, depth, triangles):
    import yaml
    from lidar2sim.baseline import PARAMS
    base, jobs = server
    result = post(base, '/api/jobs?method='+method, b'ply\nend_header\n')
    params = PARAMS | yaml.safe_load((jobs.folder(result['id'])/'config.yaml').read_text())
    assert result['settings']['method'] == method
    assert params['poisson_depth'] == depth
    assert params['collision_triangles'] == triangles


def test_restart_marks_interrupted_job(tmp_path):
    jobs = Jobs(tmp_path); ident = 'a'*32; (tmp_path/ident).mkdir()
    jobs.update(ident, status='running'); jobs.pool.shutdown()
    recovered = Jobs(tmp_path)
    assert recovered.read(ident)['status'] == 'failed'
    recovered.pool.shutdown()


def test_failed_process_and_duplicate_start(server, monkeypatch):
    base, jobs = server
    job = post(base, '/api/jobs', b'ply\nend_header\n')
    class Process:
        def __init__(self, *args, **kwargs): pass
        def wait(self, **kwargs): return 2
    monkeypatch.setattr('subprocess.Popen', Process)
    jobs.run(job['id'])
    assert jobs.read(job['id'])['status'] == 'failed'
    with pytest.raises(ValueError): jobs.start(job['id'])


def test_cloud_and_trajectory_coordinate_transform():
    result = normalize([[100,200,300]], .01, 'Y')
    np.testing.assert_allclose(result, [[1,-3,2]])


def test_ui_publishes_the_shared_input_contract(server):
    from lidar2sim.input_contract import contract
    base, _ = server
    with urllib.request.urlopen(base+'/api/input-contract') as response:
        assert json.load(response) == contract()


def test_single_worker_keeps_second_job_queued(server, monkeypatch):
    base, jobs = server
    first = post(base, '/api/jobs', b'ply\nend_header\n')['id']
    second = post(base, '/api/jobs', b'ply\nend_header\n')['id']
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    def run(ident):
        if ident == first:
            entered.set()
            release.wait(3)
        else:
            done.set()
    monkeypatch.setattr(jobs, 'run', run)
    jobs.start(first)
    try:
        assert entered.wait(1)
        jobs.start(second)
        assert jobs.read(second)['status'] == 'queued'
        assert not done.is_set()
    finally:
        release.set()
    assert done.wait(1)


def test_validation_error_survives_buffered_native_warning():
    from lidar2sim.web.server import failure_message
    assert failure_message(['WEB_ERROR "Empty cloud"', '\x1b[1;33mwarning\x1b[0;m']) == 'Empty cloud'
    assert failure_message(['\x1b[1;33mwarning\x1b[0;m']) == 'warning'
