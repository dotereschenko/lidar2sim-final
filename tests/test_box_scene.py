import numpy as np
import pytest
from tools.build_box_scene import trajectory


def test_waypoint_interpolation_keeps_geometry_speed_and_endpoints():
    path=trajectory([[0,0,1],[1,0,1],[1,2,1]])
    np.testing.assert_allclose(path[[0,-1],1:],[[0,0,1],[1,2,1]])
    distance=np.linalg.norm(np.diff(path[:,1:],axis=0),axis=1)
    assert distance.max()<=.3+1e-12
    np.testing.assert_allclose(distance/np.diff(path[:,0]),.2)
    with pytest.raises(ValueError):trajectory([[0,0,0],[0,0,0]])
    with pytest.raises(ValueError):trajectory([[0,0,0],[float('nan'),0,0]])
