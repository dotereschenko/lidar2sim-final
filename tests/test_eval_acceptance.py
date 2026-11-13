"""Real small meshes catch order-dependent sampling and misleading partial reports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class EvalAcceptanceTest(unittest.TestCase):
    def test_cli_reports_independent_results_for_same_named_outputs_in_any_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                           o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
            o3d.io.write_triangle_mesh(str(data / 'gt_mesh.ply'), mesh)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                [[x, y, 0] for x in np.linspace(0, 1, 31) for y in np.linspace(0, 1, 31)]))
            o3d.io.write_point_cloud(str(data / 'A.ply'), cloud)
            outs = [root / 'first/result', root / 'second/result']
            for out, offset in zip(outs, [0.0, 0.2]):
                out.mkdir(parents=True)
                shifted = o3d.geometry.TriangleMesh(mesh)
                # A tilted result makes sampled distances vary: missing per-sample seeds
                # must change raw metrics when evaluation order changes.
                np.asarray(shifted.vertices)[:, 2] = offset * np.asarray(shifted.vertices)[:, 0]
                for name in ['visual.ply', 'collision.ply']:
                    o3d.io.write_triangle_mesh(str(out / name), shifted)
            profile = yaml.safe_load((ROOT / 'configs/acceptance/lidar2sim-rules-v1.yaml').read_bytes())
            profile['geometry']['samples'] = 1000
            path = root / 'profile.yaml'
            path.write_text(yaml.safe_dump(profile))
            reports = []
            for order in [outs, outs[::-1]]:
                run = subprocess.run([PYTHON, str(ROOT / 'tools/eval_scene.py'), str(data),
                                      *map(str, order), '--profile', str(path)],
                                     cwd=root, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                reports.append([json.loads((out / 'eval_scene.json').read_text()) for out in outs])
            self.assertEqual(reports[0], reports[1])
            good, bad = reports[0]
            self.assertTrue(all(good['passed'].values()))
            self.assertEqual(good['poc_status'], 'incomplete')
            self.assertEqual(bad['poc_status'], 'failed')
            self.assertEqual(good['evaluation_mode'], 'scene_a_gt_diagnostic')
            self.assertEqual(good['coverage']['collision_completeness'], 'not_run')
            self.assertEqual(good['raw_metrics']['extra_5cm_pct'], 100.0)
            self.assertEqual(good['acceptance_profile']['profile_id'], 'lidar2sim-rules-v1')
            # Missing mesh is an input error, not a passing empty sample.
            (outs[0] / 'visual.ply').unlink()
            run = subprocess.run([PYTHON, str(ROOT / 'tools/eval_scene.py'), str(data), str(outs[0]),
                                  '--profile', str(path)], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)

    def test_invalid_profile_is_rejected_before_isaac_or_mesh_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.yaml'
            path.write_text('schema_version: 99\n')
            for tool, argv in [('isaac_scene_test.py', ['--scene', 'missing.usd', '--config', 'missing.json', '--out', 'physics.json']),
                               ('eval_scene.py', ['missing-data', 'missing-out'])]:
                run = subprocess.run([PYTHON, str(ROOT / 'tools' / tool), *argv, '--profile', str(path)],
                                     cwd=tmp, capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn('profile', run.stderr.lower())
                self.assertNotIn('unrecognized arguments', run.stderr)
                self.assertNotIn('No module named', run.stderr)
                self.assertNotIn('SimulationApp', run.stderr)

    def test_conflicting_physics_config_is_rejected_without_isaac(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = json.loads((ROOT / 'data/scene_a/test_config.json').read_text())
            cfg['speed'] = 99
            path = Path(tmp) / 'test.json'
            path.write_text(json.dumps(cfg))
            run = subprocess.run([PYTHON, str(ROOT / 'tools/isaac_scene_test.py'), '--scene', 'missing.usd',
                                  '--config', str(path), '--out', 'physics.json'],
                                 cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('speed', run.stderr)
            self.assertNotIn('No module named', run.stderr)
