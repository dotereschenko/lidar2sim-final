"""Conservative A-only correction of normals facing an occluded sensor pose."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def correct_occluded_normals(pcd, trajectory, voxel):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals).copy()
    info = {"method": "full-visibility-v4", "ambiguous": 0, "flipped": 0}
    if trajectory is None or len(trajectory) < 2:
        return info
    poses = np.asarray(trajectory)
    pose_tree, surface_tree = cKDTree(poses), cKDTree(points)

    def blocked(origins, views, ns):
        length = np.linalg.norm(views, axis=1)
        unit = views / np.maximum(length[:, None], 1e-12)
        cosine = np.abs(np.einsum("ij,ij->i", unit, ns))
        start = 3 * voxel / np.maximum(cosine, 1e-12)
        end = np.minimum(.5, length - 3 * voxel)
        valid = end > start + 2 * voxel
        distances = start[:, None] + (end - start)[:, None] * np.linspace(0, 1, 16)
        probes = origins[:, None, :] + distances[:, :, None] * unit[:, None, :]
        counts = surface_tree.query_ball_point(probes.reshape(-1, 3), 1.25 * voxel,
                                               return_length=True).reshape(-1, 16)
        occupied = counts >= 3
        return valid, np.any(occupied[:, :-1] & occupied[:, 1:], axis=1)

    def full_ray_blocked(origins, views, ns):
        length = np.linalg.norm(views, axis=1)
        unit = views / length[:, None]
        cosine = np.abs(np.einsum("ij,ij->i", unit, ns))
        start, end = 3 * voxel / cosine, length - 3 * voxel
        sizes = np.ceil((end - start) / voxel).astype(int) + 1
        owners = np.repeat(np.arange(len(origins)), sizes)
        distance = np.concatenate([np.linspace(a, b, n) for a, b, n in zip(start, end, sizes)])
        probes = origins[owners] + distance[:, None] * unit[owners]
        occupied = surface_tree.query_ball_point(probes, 1.25 * voxel, return_length=True) >= 3
        pairs = occupied[:-1] & occupied[1:] & (owners[:-1] == owners[1:])
        return np.bincount(owners[:-1][pairs], minlength=len(origins)) > 0

    # shortcut: consider 64 nearest poses; expand for sparse trajectories only
    # when a measured failure demonstrates a visible pose outside this set.
    for offset in range(0, len(points), 4096):
        ps, ns = points[offset:offset + 4096], normals[offset:offset + 4096]
        distance, indices = pose_tree.query(ps, k=min(64, len(poses)))
        views = poses[indices] - ps[:, None, :]
        side = np.einsum("nkj,nj->nk", views, ns)
        cosine = np.abs(side) / np.maximum(distance, 1e-12)
        start = 3 * voxel / np.maximum(cosine, 1e-12)
        usable = np.minimum(.5, distance - 3 * voxel) > start + 2 * voxel
        front_side, back_side = side > 0, (side < 0) & usable
        front = np.where(front_side, distance, np.inf).argmin(axis=1)
        back = np.where(back_side, distance, np.inf).argmin(axis=1)
        rows = np.flatnonzero(np.any(front_side, axis=1) & np.any(back_side, axis=1))
        info["ambiguous"] += len(rows)
        if not len(rows):
            continue
        vf, bf = blocked(ps[rows], views[rows, front[rows]], ns[rows])
        vb, bb = blocked(ps[rows], views[rows, back[rows]], ns[rows])
        candidates = rows[vf & vb & bf & ~bb]
        for first in range(0, len(candidates), 512):
            selected = candidates[first:first + 512]
            occluded = full_ray_blocked(ps[selected], views[selected, back[selected]], ns[selected])
            flip = selected[~occluded]
            ns[flip] *= -1
            info["flipped"] += len(flip)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return info
