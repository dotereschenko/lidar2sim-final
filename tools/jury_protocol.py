"""Build a private baseline/final protocol from completed calibration batches."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.jury_compare import compare, sha


def protocol(jury, batches, performance, repeats, out):
    jury, performance, out = map(lambda p: Path(p).resolve(), (jury, performance, out))
    runs, environments, completions = [], [], []
    for batch in map(Path, batches):
        batch = batch.resolve()
        completion = json.loads((batch / 'complete.json').read_text())
        environment = json.loads((batch / 'environment.json').read_text())
        if completion['image_id'] != environment['image_id']:
            raise ValueError('Batch image mismatch')
        environments.append(environment)
        completions.append({'directory': str(batch), **completion})
        for case in json.loads((batch / 'cases.json').read_text()):
            folder = (batch / case['folder']).resolve()
            if not folder.is_relative_to(batch):
                raise ValueError('Case outside batch')
            runs.append({'team': 'reference', 'variant': case['variant'], 'scene': case['scene'],
                         'repeat': case['repeat'], 'directory': str(folder)})
    if not environments or any(e != environments[0] for e in environments):
        raise ValueError('All repetitions require the same execution environment')
    datasets = {}
    for scene in ['scene_a', 'scene_c']:
        path = jury / scene
        datasets[scene] = {'A_sha256': sha(path/'scans/A.ply'), 'B_sha256': sha(path/'scans/B.ply'),
                           'GT_sha256': sha(path/'source/gt_mesh.ply'),
                           'manifest_sha256': sha(path/'scans/manifest.json'),
                           'physics_config_sha256': sha(path/'source/test_config.json'),
                           'annotations_sha256': sha(path/'annotations/annotations.json'),
                           'acquisition_seeds': {n: json.loads((path/'scans/manifest.json').read_text())['scans'][n]['seed'] for n in ['A','B']}}
    manifest = {'required_repeats': repeats, 'datasets': datasets,
                'profile_sha256': sha(ROOT/'configs/acceptance/lidar2sim-rules-v2.yaml'),
                'dimension_measurer_sha256': sha(ROOT/'lidar2sim/dimensions.py'),
                'evaluator_sha256': sha(ROOT/'tools/eval_scene.py'),
                'config_sha256': {'reference/'+v: sha(ROOT/'configs'/c) for v,c in
                                  [('baseline','baseline.yaml'),('final','poc-high-detail.yaml')]},
                'scoring_environment': environments[0],
                'performance_input_sha256': json.loads((performance/'baseline/benchmark.json').read_text())['input_sha256'],
                'performance': {'reference/'+v: str(performance/v/'benchmark.json') for v in ['baseline','final']},
                'runs': runs}
    result = compare(manifest)
    out.mkdir(parents=True, exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'baseline_report.json').write_text(json.dumps({**result, 'batches': completions}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['groups']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['jury','performance','out']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--batch', action='append', required=True)
    p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args()
    protocol(a.jury, a.batch, a.performance, a.repeats, a.out)
