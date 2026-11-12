"""Pinned GT annotations and the shared A/B observation predicate."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def cloud_array(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected nonempty finite Nx3 points')
    return points


def observed_by_both(points, a, b, radius=.10, min_neighbors=10):
    points, a, b = map(cloud_array, (points, a, b))
    if not np.isfinite(radius) or radius <= 0 or type(min_neighbors) is not int or min_neighbors <= 0:
        raise ValueError('Positive finite radius and integer neighbor count required')
    counts = [cKDTree(cloud).query_ball_point(points, radius, return_length=True, workers=1) for cloud in (a, b)]
    return (counts[0] >= min_neighbors) & (counts[1] >= min_neighbors), *counts


def unit_vector(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value), 1, atol=1e-7):
        raise ValueError('Expected a finite unit direction')
    return value


def dimension(scene, definition):
    origin = cloud_array([definition['origin_m']])[0]
    axis = unit_vector(definition['axis'])
    rays = np.array([np.r_[origin, -axis], np.r_[origin, axis]], dtype=np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)
    distances = result['t_hit'].numpy().astype(float)
    normals = result['primitive_normals'].numpy()
    if not np.isfinite(distances).all() or (distances <= .01).any() or (distances > 20).any():
        raise ValueError(f"{definition['id']}: missing or degenerate dimension surface")
    if (np.abs(normals @ axis) < .95).any():
        raise ValueError(f"{definition['id']}: surface not aligned with dimension axis")
    endpoints = origin + np.array([-1,1])[:,None]*distances[:,None]*axis
    return {**definition, 'surface_points_m': endpoints.tolist(), 'reference_m': float(distances.sum()),
            'search_boxes': [{'min_m': (p-.15).tolist(), 'max_m': (p+.15).tolist()} for p in endpoints],
            'search_half_extent_m': .15, 'measurement_contract': 'find reconstructed surfaces in both boxes; do not use GT endpoints as the measured result'}


def door(scene, definition, body_size):
    origin = cloud_array([definition['origin_m']])[0]
    axis, normal = unit_vector(definition['width_axis']), unit_vector(definition['normal'])
    up = np.array([0.,0.,1.])
    if abs(axis @ normal) > 1e-7 or abs(axis @ up) > 1e-7 or abs(normal @ up) > 1e-7:
        raise ValueError('Door axes must be horizontal and orthogonal')
    body = np.asarray(body_size, dtype=float)
    if body.shape != (3,) or not np.isfinite(body).all() or (body <= 0).any():
        raise ValueError('Positive finite body dimensions required')
    width = dimension(scene, {'id': definition['id'], 'origin_m': origin.tolist(), 'axis': axis.tolist()})
    height = dimension(scene, {'id': definition['id'], 'origin_m': origin.tolist(), 'axis': up.tolist()})
    left, right = np.array(width['surface_points_m'])
    bottom, top = np.array(height['surface_points_m'])[:,2]
    corners = [np.r_[p[:2], z].tolist() for p,z in [(left,bottom),(right,bottom),(right,top),(left,top)]]
    # Grid excludes a 5 cm jamb/header margin, tests through 0.50 m wall thickness.
    if width['reference_m'] <= .10 or height['reference_m'] <= .10:
        raise ValueError('Door opening too small')
    offsets = np.linspace(.05, width['reference_m']-.05, 17)
    heights = np.linspace(bottom+.05, top-.05, 25)
    xyz = np.array([left + axis*t for t in offsets for z in heights])
    xyz[:,2] = np.tile(heights, len(offsets))
    rays = np.column_stack([xyz-normal*.25, np.tile(normal,(len(xyz),1))]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
    if (hit < .5).any():
        raise ValueError(f"{definition['id']}: blocked door interior")
    fits = bool(width['reference_m'] >= float(np.abs(axis) @ body) and height['reference_m'] >= body[2])
    if not fits:
        raise ValueError('Test body does not fit opening')
    return {**definition, 'width_m': width['reference_m'], 'height_m': height['reference_m'],
            'corners_m': corners, 'body_fits': fits, 'interior_rays': len(xyz),
            'interior_margin_m': .05, 'tested_wall_thickness_m': .5}


def require_observed_dimensions(dimensions, a, b, radius, min_neighbors):
    points = np.array([p for item in dimensions for p in item['surface_points_m']])
    mask, ca, cb = observed_by_both(points, a, b, radius, min_neighbors)
    for i, item in enumerate(dimensions):
        if not mask[2*i:2*i+2].all():
            raise ValueError(f"{item['id']}: dimension surfaces not observed by both A and B")
        item['surface_neighbors_A'] = ca[2*i:2*i+2].tolist()
        item['surface_neighbors_B'] = cb[2*i:2*i+2].tolist()
