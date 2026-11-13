import hashlib
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d
from pxr import Gf, Usd, UsdGeom
from tools import fetch_gt


class GroundTruthBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.source = self.data / 'source.usda'
        stage = Usd.Stage.CreateNew(str(self.source))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, '/Root')
        root.AddTranslateOp().Set(Gf.Vec3d(2, 3, 0))
        for name in ['Floor', 'Glass', 'SM_Buildings_Background', 'Hidden']:
            mesh = UsdGeom.Mesh.Define(stage, '/Root/' + name)
            mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            if name == 'Hidden':
                mesh.CreateVisibilityAttr('invisible')
        stage.GetRootLayer().Save()
        self.content = self.source.read_bytes()
        self.url = 'https://example.invalid/Assets/Isaac/6.2/Isaac/Test/root.usda'
        self.manifest = {
            'units': 'm', 'up_axis': 'Z', 'bounds_min': [1, 2, -1], 'bounds_max': [4, 5, 1],
            'gt': {'url': self.url, 'triangles': 1, 'area_m2': 0.5,
                   'build': {'max_edge_m': 2, 'transparent_name_pattern': 'glass|window|mirror|transparent|crystal',
                             'exclude_name_substrings': ['SM_Buildings']}}}
        self.write_manifest()

    def write_manifest(self):
        (self.data / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_pair_matches_world_coordinates_and_rebuilds_offline(self):
        with patch.object(fetch_gt.urllib.request, 'urlopen', return_value=io.BytesIO(self.content)):
            fetch_gt.main(self.data)
        usd = self.data / 'gt_scene.usd'
        self.assertTrue(usd.exists(), 'Build must provide the physics/reference USD alongside PLY')
        ply = o3d.io.read_triangle_mesh(str(self.data / 'gt_mesh.ply'))
        expected = np.asarray(ply.vertices)[np.asarray(ply.triangles)]
        np.testing.assert_allclose(fetch_gt.extract(usd), expected, atol=1e-6)
        np.testing.assert_allclose(expected[0], [[2, 3, 0], [3, 3, 0], [2, 4, 0]])
        report = json.loads((self.data / 'gt_build.json').read_text())
        for name in ['gt_scene.usd', 'gt_mesh.ply']:
            self.assertEqual(report['outputs'][name]['sha256'], hashlib.sha256((self.data / name).read_bytes()).hexdigest())
        old = {n: (self.data / n).read_bytes() for n in ['gt_mesh.ply', 'gt_scene.usd', 'gt_build.json']}
        with patch.object(fetch_gt.urllib.request, 'urlopen', side_effect=AssertionError('Unexpected network')):
            fetch_gt.main(self.data)
        for n, contents in old.items():
            self.assertEqual((self.data / n).read_bytes(), contents)
        # A changed cached source must not silently replace the accepted pair.
        self.manifest['gt']['source_layers_sha256'] = report['source_layers_sha256']
        self.write_manifest()
        cached = self.data / 'nvidia/Test/root.usda'
        cached.write_bytes(cached.read_bytes() + b'\n# upstream changed\n')
        with self.assertRaisesRegex(ValueError, 'source.*hash'):
            fetch_gt.main(self.data)
        self.assertEqual((self.data / 'gt_mesh.ply').read_bytes(), old['gt_mesh.ply'])

    def test_transient_download_retries_and_failure_leaves_no_cached_file(self):
        with patch.object(fetch_gt.urllib.request, 'urlopen', side_effect=[urllib.error.URLError('offline'), io.BytesIO(self.content)]) as get:
            with patch.object(fetch_gt.time, 'sleep'):
                path = fetch_gt.download(self.url, self.data / 'retry')
        self.assertEqual(get.call_count, 2)
        self.assertEqual(path.read_bytes(), self.content)
        with patch.object(fetch_gt.urllib.request, 'urlopen', side_effect=urllib.error.URLError('offline')) as get:
            with patch.object(fetch_gt.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, '3 attempts'):
                    fetch_gt.download(self.url, self.data / 'failed')
        self.assertEqual(get.call_count, 3)
        self.assertFalse((self.data / 'failed/Test/root.usda').exists())

    def test_mismatch_does_not_publish_outputs(self):
        self.manifest['gt']['triangles'] = 99
        self.write_manifest()
        with patch.object(fetch_gt.urllib.request, 'urlopen', return_value=io.BytesIO(self.content)):
            with self.assertRaisesRegex(ValueError, 'triangles'):
                fetch_gt.main(self.data)
        self.assertFalse((self.data / 'gt_mesh.ply').exists())
        self.assertFalse((self.data / 'gt_scene.usd').exists())

    def test_interrupted_body_is_not_reused_and_404_is_not_retried(self):
        class Interrupted(io.BytesIO):
            def read(self, *args):
                raise http.client.IncompleteRead(b'partial', 100)
        with patch.object(fetch_gt.urllib.request, 'urlopen', side_effect=[Interrupted(), io.BytesIO(self.content)]):
            with patch.object(fetch_gt.time, 'sleep'):
                path = fetch_gt.download(self.url, self.data / 'interrupted')
        self.assertEqual(path.read_bytes(), self.content)
        error = urllib.error.HTTPError(self.url, 404, 'missing', {}, None)
        with patch.object(fetch_gt.urllib.request, 'urlopen', side_effect=error) as get:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 404'):
                fetch_gt.download(self.url, self.data / 'missing')
        self.assertEqual(get.call_count, 1)

    def test_wrong_output_pin_preserves_previous_outputs(self):
        self.manifest['gt']['sha256'] = {'gt_mesh.ply': '0' * 64}
        self.write_manifest()
        previous = self.data / 'gt_mesh.ply'
        previous.write_bytes(b'previous accepted artifact')
        with patch.object(fetch_gt.urllib.request, 'urlopen', return_value=io.BytesIO(self.content)):
            with self.assertRaisesRegex(ValueError, 'Output hash mismatch'):
                fetch_gt.main(self.data)
        self.assertEqual(previous.read_bytes(), b'previous accepted artifact')


if __name__ == '__main__':
    unittest.main()
