"""Validate user-selected floor waypoints against the actual collision mesh."""
import numpy as np
import open3d as o3d


def route_config(mesh_path, points):
    if (not isinstance(points, list) or not 2 <= len(points) <= 32
        or any(not isinstance(p, list) or len(p) != 3 or
               any(type(v) not in (int, float) or not np.isfinite(v) for v in p) for p in points)):
        raise ValueError('Нужны 2–32 точки XYZ в метрах с конечными координатами')
    xyz = np.asarray(points, dtype=float)
    length = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if (length < .2).any() or length.sum() > 30:
        raise ValueError('Сегменты маршрута: от 20 см, общая длина — до 30 м')
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles() or len(mesh.triangles) > 300000:
        raise ValueError('Для физики нужна collision-сетка до 300000 треугольников')
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    samples = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(d/.1))+1))
                              for a, b, d in zip(xyz[:-1], xyz[1:], length)])
    # Nine support probes cover the 40 cm body footprint. This is a preflight,
    # not a proof of collision-free motion between samples; physics decides that.
    offsets = np.array([[x, y, 0] for x in [-.2, 0, .2] for y in [-.2, 0, .2]])
    probes = (samples[:, None, :] + offsets).reshape(-1, 3)
    origins = probes + [0, 0, .08]
    rays = np.column_stack([origins, np.tile([0, 0, -1], (len(origins), 1))]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    distance = hit['t_hit'].numpy()
    normals = hit['primitive_normals'].numpy()
    if (not np.isfinite(distance).all() or (distance > .16).any()
        or (np.abs(normals[:, 2]) < .85).any()):
        raise ValueError('Нет ровной опоры для корпуса 40 × 40 см вдоль маршрута. Выберите пол и обход препятствий; проверьте единицы и ось вверх.')
    # Check vertical headroom; horizontal wall collisions are resolved by physics.
    up_origins = probes.copy(); up_origins[:, 2] = origins[:, 2] - distance + .04
    up_rays = np.column_stack([up_origins, np.tile([0, 0, 1], (len(probes), 1))]).astype(np.float32)
    overhead = scene.cast_rays(o3d.core.Tensor(up_rays))['t_hit'].numpy()
    if (overhead < .36).any():
        raise ValueError('Над маршрутом недостаточно места для корпуса: выберите свободный участок пола')
    return {'routes': [{'name': 'Маршрут пользователя', 'points': xyz.tolist()}],
            'body': {'size': [.4, .4, .3], 'mass': 20.0}, 'speed': .2,
            'surface_waypoints': True, 'preflight': 'sampled_support_and_headroom'}
