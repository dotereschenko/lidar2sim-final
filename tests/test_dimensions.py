import unittest
import numpy as np
import open3d as o3d
from lidar2sim.annotations import dimension


def scene(mesh):
    s=o3d.t.geometry.RaycastingScene(nthreads=1)
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


class DimensionTests(unittest.TestCase):
    def definition(self,axis=[1,0,0]):
        return dimension(scene(o3d.geometry.TriangleMesh.create_box(4,4,3)),
                         {'id':'D01','label':'box','origin_m':[2,2,1.5],'axis':axis})

    def test_actual_width_and_height_detect_ten_cm_distortion(self):
        from lidar2sim.dimensions import measure_dimension
        for axis,size in [([1,0,0],(4.1,4,3)),([0,0,1],(4,4,3.1))]:
            d=self.definition(axis)
            good=measure_dimension(scene(o3d.geometry.TriangleMesh.create_box(4,4,3)),d,.05)
            self.assertEqual(good['status'],'passed')
            self.assertLess(good['absolute_error_m'],1e-5)
            bad=measure_dimension(scene(o3d.geometry.TriangleMesh.create_box(*size)),d,.05)
            self.assertEqual(bad['status'],'failed')
            self.assertAlmostEqual(bad['absolute_error_m'],.1,places=5)

    def test_missing_and_ambiguous_surface_are_errors(self):
        from lidar2sim.dimensions import measure_dimension
        mesh=o3d.geometry.TriangleMesh.create_box(4,4,3).translate([20,0,0])
        r=measure_dimension(scene(mesh),self.definition(),.05)
        self.assertEqual(r['status'],'error');self.assertIsNone(r['measured_m'])
        # A 10 cm step across the probe bundle admits two incompatible spans.
        left=o3d.geometry.TriangleMesh.create_box(4,2,3)
        right=o3d.geometry.TriangleMesh.create_box(4.1,2,3).translate([0,2,0])
        r=measure_dimension(scene(left+right),self.definition(),.05)
        self.assertEqual(r['status'],'error')
        self.assertEqual(r['reason'],'ambiguous_surface_probe_spread')

    def test_invalid_boxes_and_nonunit_axis_rejected(self):
        from lidar2sim.dimensions import validate_dimensions
        d=self.definition()
        with self.assertRaises(ValueError):validate_dimensions([d],10)
        with self.assertRaises(ValueError):validate_dimensions([{**d,'axis':[2,0,0]}],1)
        d['search_boxes'][0]['min_m']=[float('nan'),0,0]
        with self.assertRaises(ValueError):validate_dimensions([d],1)

    def test_oblique_surface_is_measurable_when_bundle_is_stable(self):
        from lidar2sim.dimensions import measure_dimension
        mesh=o3d.geometry.TriangleMesh.create_box(4,4,3)
        vertices=np.asarray(mesh.vertices)
        vertices[:,0]+=1.2*(vertices[:,2]-1.5)
        # At z=1.5 both real intersections still define a 4 m span. Tilted
        # normals alone must not turn an unambiguous measurement into missing data.
        result=measure_dimension(scene(mesh),self.definition(),.05)
        self.assertEqual(result['status'],'passed')
        self.assertAlmostEqual(result['measured_m'],4,places=5)


def test_endpoint_boxes_measure_room_width_despite_unrelated_foreground_surface():
    from lidar2sim.dimensions import measure_dimension
    room=o3d.geometry.TriangleMesh.create_box(4,4,3)
    definition=dimension(scene(room),{'id':'D01','origin_m':[2,2,1.5],'axis':[1,0,0]})
    furniture=o3d.geometry.TriangleMesh.create_box(.1,1,2).translate([2.8,1.5,.2])
    result=measure_dimension(scene(room+furniture),definition,.05)
    assert result['passed']
    assert abs(result['measured_m']-4)<1e-5
    # Foreground geometry must not substitute for an absent endpoint surface.
    vertices=np.asarray(room.vertices);faces=np.asarray(room.triangles)
    room.remove_triangles_by_mask((vertices[faces][:,:,0] == 4).all(axis=1))
    missing=measure_dimension(scene(room+furniture),definition,.05)
    assert missing['status']=='error' and missing['measured_m'] is None


def test_local_boxes_support_rotated_measurement_axis():
    from lidar2sim.dimensions import measure_dimension
    c,s=np.cos(.6),np.sin(.6)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    room=o3d.geometry.TriangleMesh.create_box(4,4,3).rotate(rotation,center=[2,2,1.5])
    definition=dimension(scene(room),{'id':'D01','origin_m':[2,2,1.5],'axis':[c,s,0]})
    furniture=o3d.geometry.TriangleMesh.create_box(.1,1,2).translate([2.8,1.5,.2]).rotate(rotation,center=[2,2,1.5])
    result=measure_dimension(scene(room+furniture),definition,.05)
    assert result['passed'] and abs(result['measured_m']-4)<1e-5
