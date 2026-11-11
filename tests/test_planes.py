import numpy as np
import open3d as o3d

from lidar2sim.planes import observed_vertical_planes


def wall(x=0., door=False):
    y,z=np.meshgrid(np.arange(0,3.001,.06),np.arange(0,2.401,.06))
    points=np.column_stack([np.full(y.size,x),y.ravel(),z.ravel()])
    if door:points=points[~((points[:,1]>.9)&(points[:,1]<1.9)&(points[:,2]<2.05))]
    return points


def cloud(points,normals=None):
    p=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    p.normals=o3d.utility.Vector3dVector(np.tile([1.,0.,0.],(len(points),1)) if normals is None else normals)
    return p


def hit(mesh,origins,direction):
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays=np.column_stack([origins,np.tile(direction,(len(origins),1))]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()


def test_observed_wall_recovers_local_missing_patch_without_bridging_door():
    points=wall(door=True)
    # A small acquisition hole is recoverable, a doorway is not a supported patch.
    points=points[~((points[:,1]>.3)&(points[:,1]<.48)&(points[:,2]>.9)&(points[:,2]<1.08))]
    mesh,stats=observed_vertical_planes(cloud(points),.012,.3)
    assert stats['planes']
    distances=hit(mesh,[[-1,.39,.99],[-1,1.4,1.0],[-1,2.4,1.0]],[1,0,0])
    assert np.isclose(distances[[0,2]],1,atol=1e-5).all()
    assert np.isinf(distances[1])


def test_nearby_opposing_surfaces_remain_distinct_and_translate_with_input():
    a,b=wall(0),wall(.08)
    points=np.vstack([a,b]);normals=np.vstack([np.tile([-1.,0,0],(len(a),1)),np.tile([1.,0,0],(len(b),1))])
    mesh,stats=observed_vertical_planes(cloud(points,normals),.012,.3)
    assert len(stats['planes'])==2
    assert abs(hit(mesh,[[-1,1,1]],[1,0,0])[0]-1)<1e-5
    assert abs(hit(mesh,[[1,1,1]],[-1,0,0])[0]-.92)<1e-5
    shift=np.array([13.,-9.,4.])
    moved,_=observed_vertical_planes(cloud(points+shift,normals),.012,.3)
    assert abs(hit(moved,np.array([[-1,1,1]])+shift,[1,0,0])[0]-1)<1e-5


def test_sparse_and_horizontal_support_do_not_invent_walls():
    for p in [cloud(wall()[:30]),cloud(wall(),np.tile([0.,0.,1.],(len(wall()),1)))]:
        mesh,stats=observed_vertical_planes(p,.012,.3)
        assert not len(mesh.triangles) and not stats['planes']


def test_boundary_uncertainty_stays_local_and_parameters_reject_invalid_values():
    import pytest
    from lidar2sim.baseline import PARAMS
    from lidar2sim.validation import ContractError, validate_params
    mesh,_=observed_vertical_planes(cloud(wall()),.012,.3)
    vertices=np.asarray(mesh.vertices)
    assert np.max(np.abs(vertices[:,0]))<1e-6
    assert vertices[:,1].min()>=-.00601 and vertices[:,1].max()<=3.00601
    assert vertices[:,2].min()>=-.00601 and vertices[:,2].max()<=2.40601
    for params in [{'plane_distance':-.01},{'plane_distance':float('nan')},{'plane_max_edge':0}]:
        with pytest.raises(ContractError):validate_params(params,PARAMS)


def test_observed_low_wall_is_kept_despite_unreliable_local_normals():
    points = wall()
    normals = np.tile([1., 0., 0.], (len(points), 1))
    normals[points[:, 2] < .35] = [0., 0., 1.]
    mesh, stats = observed_vertical_planes(cloud(points, normals), .012, .3)
    assert stats['planes']
    assert abs(hit(mesh, [[-1, 1.5, .1]], [1, 0, 0])[0] - 1) < 1e-5


def test_noise_tails_do_not_become_parallel_duplicate_walls():
    points = wall()
    points[:, 0] += np.random.default_rng(19).normal(0, .01, len(points))
    mesh, stats = observed_vertical_planes(cloud(points), .012, .3)
    assert len(stats['planes']) == 1
    assert abs(hit(mesh, [[-1, 1.5, 1]], [1, 0, 0])[0] - 1) < .005
