import unittest
import numpy as np
import open3d as o3d


class AnnotationTests(unittest.TestCase):
    def scene(self):
        mesh = o3d.geometry.TriangleMesh.create_box(4, 4, 3)
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        return scene

    def test_region_requires_both_clouds_and_rejects_invalid_input(self):
        from lidar2sim.annotations import observed_by_both
        a = np.array([[0,0,0]]*10 + [[1,0,0]]*10, dtype=float)
        b = np.array([[1,0,0]]*10 + [[2,0,0]]*10, dtype=float)
        mask, ca, cb = observed_by_both(np.array([[0,0,0],[1,0,0],[2,0,0]],dtype=float), a, b)
        self.assertEqual(mask.tolist(), [False, True, False])
        self.assertEqual(ca.tolist(), [10,10,0])
        self.assertEqual(cb.tolist(), [0,10,10])
        for bad in [np.empty((0,3)), np.array([[np.nan,0,0]])]:
            with self.assertRaises(ValueError):
                observed_by_both(bad, a, b)
        with self.assertRaises(ValueError):
            observed_by_both(a, a, b, radius=0)

    def test_dimension_measures_gt_surfaces_and_missing_surface_fails(self):
        from lidar2sim.annotations import dimension
        spec = {'id':'width','label':'Width','category':'room','origin_m':[2,2,1.5], 'axis':[1,0,0]}
        result = dimension(self.scene(), spec)
        self.assertAlmostEqual(result['reference_m'], 4, places=5)
        np.testing.assert_allclose(result['surface_points_m'], [[0,2,1.5],[4,2,1.5]], atol=1e-6)
        with self.assertRaises(ValueError):
            dimension(self.scene(), {**spec,'origin_m':[5,5,5]})
        with self.assertRaises(ValueError):
            dimension(self.scene(), {**spec,'axis':[2,0,0]})

    def test_door_detects_blocked_interior(self):
        from lidar2sim.annotations import door
        # Four panels around a real opening in x=0, width 1 m, height 2 m.
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        for size, pos in [((.2,1,3),(-.1,-1.5,0)),((.2,1,3),(-.1,.5,0)),((.2,1,1),(-.1,-.5,2)),((4,4,.1),(-2,-2,-.1))]:
            mesh = o3d.geometry.TriangleMesh.create_box(*size).translate(pos)
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        spec={'id':'door','origin_m':[0,0,1],'width_axis':[0,1,0],'normal':[1,0,0]}
        result=door(scene,spec,[.4,.4,.3])
        self.assertAlmostEqual(result['width_m'],1,places=5)
        self.assertAlmostEqual(result['height_m'],2,places=5)
        self.assertTrue(result['body_fits'])
        wall=o3d.geometry.TriangleMesh.create_box(.1,1,2).translate((-.05,-.5,0))
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wall))
        with self.assertRaises(ValueError):
            door(scene,spec,[.4,.4,.3])

    def test_builder_rejects_changed_scan_before_writing_outputs(self):
        import json
        import tempfile
        from pathlib import Path
        from tools.annotate_scene import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scans = root/'scans'; scans.mkdir()
            (scans/'A.ply').write_bytes(b'changed')
            (scans/'manifest.json').write_text(json.dumps({'units':'m','up_axis':'Z','sha256':{'A.ply':'wrong'}}))
            spec = root/'definition.json'
            spec.write_text(json.dumps({'units':'m','up_axis':'Z','schema_version':1,
                'dimensions':[{'id':str(i)} for i in range(10)],'doors':[{},{}],
                'samples':1000,'seed':18,'region':{'min_common_samples':1000}}))
            evidence = root/'evidence.json'; evidence.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                build(root,scans,root/'out',spec,evidence)
            self.assertFalse((root/'out').exists())

    def test_dimension_rejects_unobserved_endpoint(self):
        from lidar2sim.annotations import require_observed_dimensions
        items=[{'id':'D01','surface_points_m':[[0,0,0],[1,0,0]]}]
        cloud=np.repeat([[0.,0,0],[1.,0,0]],10,axis=0)
        require_observed_dimensions(items,cloud,cloud,.1,10)
        self.assertEqual(items[0]['surface_neighbors_A'],[10,10])
        with self.assertRaisesRegex(ValueError,'D01'):
            require_observed_dimensions(items,cloud,cloud[:10],.1,10)
