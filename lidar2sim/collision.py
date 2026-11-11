"""Упрощение статического коллайдера с отдельным бюджетом для опорного пола."""

import numpy as np

from .common import clean, decimate


def _simplify(mesh, target, boundary_weight):
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target, boundary_weight=boundary_weight)
    return clean(mesh)


def build_collision(mesh, rest, floor, target, floor_fraction, boundary_weight):
    """Сохранить отдельную сетку пола, упрощая стены и опоры независимо.

    Две компоненты уже разделены при восстановлении пола. Поэтому упрощение
    остальной поверхности не может стянуть край пола через дверной проём.
    При недостаточном бюджете или отсутствии пола используется старый путь.
    """
    if floor is None or rest is None or floor_fraction == 0 or target < 2:
        output = decimate(mesh, target)
        return output, {"mode": "global", "target_triangles": target,
                        "triangles": len(output.triangles)}

    floor_target = min(len(floor.triangles), max(1, round(target * floor_fraction)))
    rest_target = target - floor_target
    if rest_target < 1:
        output = decimate(mesh, target)
        return output, {"mode": "global", "target_triangles": target,
                        "triangles": len(output.triangles), "reason": "budget_too_small"}
    simplified_floor = _simplify(floor, floor_target, boundary_weight)
    simplified_rest = _simplify(rest, rest_target, boundary_weight)

    # Quadric decimation should retain winding on a planar component. Verify it
    # here because the static PhysX triangle collider is approached from above.
    vertices = np.asarray(simplified_floor.vertices)
    faces = np.asarray(simplified_floor.triangles)
    triangles = vertices[faces]
    normals_z = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])[:, 2]
    if np.any(normals_z <= 0):
        raise ValueError("collision: floor simplification inverted an upward face")

    output = clean(simplified_rest + simplified_floor)
    return output, {"mode": "protected_floor", "target_triangles": target,
                    "floor_fraction": floor_fraction,
                    "boundary_weight": boundary_weight,
                    "floor_triangles": len(simplified_floor.triangles),
                    "other_triangles": len(simplified_rest.triangles),
                    "triangles": len(output.triangles),
                    "floor_faces_upward": True}
