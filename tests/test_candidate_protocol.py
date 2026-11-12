import json
from pathlib import Path

import pytest

from tools import calibrate_release, jury_protocol
from tools.jury_compare import sha


def test_calibration_pins_selected_config_outside_execution_environment(tmp_path, monkeypatch):
    config = tmp_path/'candidate.yaml'
    config.write_text('floor_boundary_pad: 0.2\n')
    out = tmp_path/'batch'
    selected = []

    def reconstruct(scans, cfg, folder, image):
        selected.append(Path(cfg))
        folder.mkdir(parents=True)
        return {'exit_code': 0, 'required_outputs_present': True}

    def physics(job):
        job.with_name('slot.json').write_text('{}')
        return 0

    monkeypatch.setattr(calibrate_release, 'reconstruct', reconstruct)
    monkeypatch.setattr(calibrate_release, 'physics_slot', physics)
    monkeypatch.setattr(calibrate_release, 'docker_cpu', lambda *args, **kwargs: 0)
    monkeypatch.setattr(calibrate_release.subprocess, 'check_output', lambda *a, **k: 'image\n')
    calibrate_release.run(tmp_path/'jury', out, 'image', repeats=1, final_config=config)
    assert selected.count(config) == 2
    assert json.loads((out/'configuration.json').read_text())['final']['sha256'] == sha(config)
    assert json.loads((out/'environment.json').read_text()) == {
        'image_id': 'image', 'cpus': 4, 'memory_gib': 16, 'network': 'none'}


def test_protocol_uses_recorded_candidate_hash_and_rejects_mixed_configs(tmp_path):
    jury = tmp_path/'jury'
    for name in ['scene_a', 'scene_c']:
        for relative in ['scans/A.ply', 'scans/B.ply', 'source/gt_mesh.ply',
                         'source/test_config.json', 'annotations/annotations.json']:
            p = jury/name/relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture')
        (jury/name/'scans/manifest.json').write_text(json.dumps({'scans': {'A': {'seed': 1}, 'B': {'seed': 2}}}))
    perf = tmp_path/'performance'
    (perf/'baseline').mkdir(parents=True)
    (perf/'baseline/benchmark.json').write_text(json.dumps({'input_sha256': 'dense'}))
    batches = []
    for number, config in enumerate(['candidate-hash', 'other-hash']):
        batch = tmp_path/f'batch-{number}'
        batch.mkdir()
        for name, value in [('complete.json', {'image_id': 'image'}),
                            ('environment.json', {'image_id': 'image'}),
                            ('configuration.json', {'final': {'sha256': config}}),
                            ('cases.json', [])]:
            (batch/name).write_text(json.dumps(value))
        batches.append(batch)
    out = tmp_path/'protocol'
    jury_protocol.protocol(jury, batches[:1], perf, 1, out)
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['config_sha256']['reference/final'] == 'candidate-hash'
    with pytest.raises(ValueError, match='configuration'):
        jury_protocol.protocol(jury, batches, perf, 1, tmp_path/'mixed')
    (batches[0]/'configuration.json').unlink()
    legacy = tmp_path/'legacy'
    jury_protocol.protocol(jury, batches[:1], perf, 1, legacy)
    assert json.loads((legacy/'manifest.json').read_text())['config_sha256']['reference/final'] == sha(
        jury_protocol.ROOT/'configs/poc-high-detail.yaml')
