"""GPU smoke test for the isolated NKSR runtime, not scene qualification.

Run with a pinned ks.pth in TORCH_HOME/hub/checkpoints and --network none.
Only torch/nksr from the experimental container are used; no project dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "ks.pth"
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != args.weights_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runtime smoke test")
    import nksr

    torch.manual_seed(0)
    # Fixed Fibonacci sphere avoids a private room and supplies exact normals.
    i = torch.arange(8192, device="cuda", dtype=torch.float32)
    z = 1 - 2 * (i + 0.5) / len(i)
    angle = i * 2.399963229728653
    radius = torch.sqrt(1 - z * z)
    normals = torch.stack((radius * angle.cos(), radius * angle.sin(), z), dim=1)
    xyz = normals.clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        reconstructor = nksr.Reconstructor(torch.device("cuda:0"), config="ks")
        field = reconstructor.reconstruct(xyz, normals, detail_level=0.0)
        mesh = field.extract_dual_mesh(mise_iter=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    v, f = mesh.v, mesh.f
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0 or not torch.isfinite(v).all():
        raise ValueError("Invalid or empty vertices")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("Invalid or empty triangles")
    if f.is_floating_point() or f.min() < 0 or f.max() >= len(v):
        raise ValueError("Invalid triangle indices")
    if (f[:, 0] == f[:, 1]).any() or (f[:, 1] == f[:, 2]).any() or (f[:, 0] == f[:, 2]).any():
        raise ValueError("Repeated triangle indices")
    args.output.mkdir(parents=True, exist_ok=False)
    obj = args.output / "sphere.obj"
    with obj.open("w") as stream:
        for vertex in v.cpu().tolist():
            stream.write("v " + " ".join(map(str, vertex)) + "\n")
        for face in f.cpu().tolist():
            stream.write("f " + " ".join(str(index + 1) for index in face) + "\n")
    report = {
        "status": "runtime_smoke_passed", "quality_status": "not_verified",
        "fixture": "unit_fibonacci_sphere_8192", "checkpoint_sha256": actual,
        "upstream_commit": os.environ.get("NKSR_COMMIT"),
        "machine": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "vertices": len(v), "triangles": len(f), "seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "mesh_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
