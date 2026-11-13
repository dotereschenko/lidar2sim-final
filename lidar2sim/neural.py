"""Optional pretrained NKSR backend; imported only when explicitly selected."""
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .common import announce_stage, Timer, clean, decimate, preprocess, sha256
from .validation import validate_mesh, validate_params

CHECKPOINT_SHA256 = "c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955"
UPSTREAM_COMMIT = "e40336845e67761343a756788e5a98b827d4a143"


def validate_extent(pcd):
    extent = np.ptp(np.asarray(pcd.points), axis=0)
    if (extent > 100).any():
        dimensions = ' × '.join(f'{x:.2f}' for x in extent)
        raise ValueError(f'NKSR: размер облака {dimensions} м превышает предел 100 м по оси '
                         'для текущего режима. Проверьте единицы PLY (мм/см/м), выбросы '
                         'или разделите большую сцену. Масштаб автоматически не меняется.')


def runtime():
    """Check the actual runtime; no download and no silent CPU/Poisson fallback."""
    try:
        import torch
        import nksr
    except (ImportError, OSError) as exc:
        raise RuntimeError("NKSR не установлен в серверном окружении") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("NKSR: CUDA недоступна")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints/ks.pth"
    if not checkpoint.is_file() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("NKSR: отсутствуют закреплённые веса ks.pth или неверный SHA256")
    return torch, nksr


def availability():
    try:
        runtime()
        return {"available": True, "label": "Нейросетевая реконструкция — NKSR"}
    except (RuntimeError, ValueError) as exc:
        return {"available": False, "reason": str(exc)}


def run(pcd, meta, traj, params=None):
    from .baseline import PARAMS
    from .collision import build_collision

    p = validate_params({} if params is None else params, PARAMS)
    validate_extent(pcd)
    torch, nksr = runtime()
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        if not np.isfinite(normals).all() or (np.linalg.norm(normals, axis=1) < 1e-8).any():
            raise ValueError("NKSR: входные нормали содержат нули или неконечные значения")
    t, orientation = Timer(), {}
    announce_stage("normals")
    with t("preprocess"):
        pcd = preprocess(pcd, meta, traj, p, orientation, preserve_normals=True)
    with t("nksr"):
        announce_stage("nksr")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model = nksr.Reconstructor(torch.device("cuda:0"), config="ks")
            xyz = torch.as_tensor(np.asarray(pcd.points).copy(), dtype=torch.float32, device="cuda")
            normals = torch.as_tensor(np.asarray(pcd.normals).copy(), dtype=torch.float32, device="cuda")
            field = model.reconstruct(xyz, normals, detail_level=0.0)
            output = field.extract_dual_mesh(mise_iter=1)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(output.v.cpu().numpy()),
                o3d.utility.Vector3iVector(output.f.cpu().numpy()))
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        mesh = clean(mesh.crop(pcd.get_axis_aligned_bounding_box()))
        validate_mesh(mesh, "NKSR mesh")
    with t("decimate"):
        announce_stage("collision")
        visual = decimate(mesh, p["visual_triangles"])
        collision, collision_info = build_collision(mesh, None, None, p["collision_triangles"], 0, 1)
        if pcd.has_colors():
            _, indices = cKDTree(np.asarray(pcd.points)).query(np.asarray(visual.vertices))
            visual.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices])
    return visual, collision, {
        "solution": "nksr", "params": p, "points_after_voxel": len(pcd.points),
        "timings_s": t.steps, "normal_orientation": orientation,
        "collision_build": collision_info, "quality_status": "not_verified",
        "neural_backend": {"name": "nksr", "model": "ks", "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256, "torch": torch.__version__,
            "compatibility_patches": ["explicit_thrust_tuple_include_cuda13", "rename_unused_lerp_cpp20",
                                      "cusparse_cublas_exceptions_instead_of_malformed_printf_abort",
                                      "nanovdb_vec3_pybind_input_caster"],
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "image_id": os.environ.get("LIDAR2SIM_IMAGE_ID"),
            "detail_level": 0.0, "mise_iter": 1, "chunk_size": None,
            "peak_cuda_allocated_bytes": peak_memory,
            "code_license": "NVIDIA Source Code License (research/evaluation)",
            "weights_license": "CC-BY-SA-4.0"},
    }
