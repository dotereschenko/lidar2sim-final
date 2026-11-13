import numpy as np
import open3d as o3d
import pytest
from lidar2sim.web.routes import route_config


@pytest.fixture
def floor(tmp_path):
    mesh=o3d.geometry.TriangleMesh.create_box(4,4,.1).translate([-2,-2,1.9])
    path=tmp_path/'collision.ply';o3d.io.write_triangle_mesh(str(path),mesh)
    return path


def test_route_uses_real_elevated_floor(floor):
    cfg=route_config(floor, [[-.5,0,2],[.5,0,2]])
    assert cfg['routes'][0]['points']==[[-.5,0,2],[.5,0,2]]
    assert cfg['surface_waypoints']


@pytest.mark.parametrize('points', [[],[[0,0,2]],[[False,0,2],[1,0,2]],[[0,0,float('nan')],[1,0,2]],[[0,0,2],[.01,0,2]],[[0,0,2],[1000,0,2]],[[0,0,0],[1,0,0]],[[1.9,0,2],[1.9,1,2]]])
def test_invalid_or_unsupported_route_rejected(floor,points):
    with pytest.raises(ValueError):route_config(floor,points)


def test_low_ceiling_rejected(floor):
    mesh=o3d.io.read_triangle_mesh(str(floor))
    mesh+=o3d.geometry.TriangleMesh.create_box(4,4,.1).translate([-2,-2,2.25])
    o3d.io.write_triangle_mesh(str(floor),mesh)
    with pytest.raises(ValueError,match='места'):route_config(floor,[[-.5,0,2],[.5,0,2]])
