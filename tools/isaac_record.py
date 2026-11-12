"""Record a short PhysX demonstration; this is not the acceptance test suite.

Run using Isaac's python.sh with --scene, --config and a new --out directory.
Only the presentation mesh is cut at 1.6 m; collision geometry stays intact.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
scene = Path(args.scene).resolve()
cfg = json.loads(Path(args.config).read_text())
assert scene.is_file()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": False, "width": 1280, "height": 720, "renderer": "RaytracedLighting",
    # This script uses the legacy World/Replicator stepping API.
    "extra_args": ["--/rtx/hydra/supportMultiTickRate=false", "--/rtx/rendering/perSensorTickTlas=false"],
}, experience="/isaac-sim/apps/isaacsim.exp.full.kit")

try:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, Vt
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage

    world = World(physics_dt=1 / 120, rendering_dt=1 / 24, stage_units_in_meters=1)
    add_reference_to_stage(str(scene), "/World/Scene")
    stage = omni.usd.get_context().get_stage()
    physics = stage.GetPrimAtPath("/World/Scene/PhysicsScene")
    if physics.IsValid():
        physics.SetActive(False)
    visual = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Scene/Visual"))
    vertices = np.asarray(visual.GetPointsAttr().Get())
    counts = np.asarray(visual.GetFaceVertexCountsAttr().Get())
    assert (counts == 3).all(), "The recording cutaway expects triangles"
    triangles = np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    triangles = triangles[vertices[triangles, 2].max(axis=1) <= 1.6]
    assert len(triangles), "Empty cutaway"
    visual.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(triangles)))
    visual.GetFaceVertexIndicesAttr().Set(Vt.IntArray(triangles.flatten().tolist()))
    visual.CreateDisplayColorAttr([(0.48, 0.64, 0.70)])
    visual.CreateDoubleSidedAttr(True)
    light = UsdLux.DomeLight.Define(stage, "/World/RecordingLight")
    light.CreateIntensityAttr(1200)

    start = cfg["routes"][0]["points"][0]
    body = world.scene.add(DynamicCuboid(
        prim_path="/World/DemoBody", name="demo_body",
        position=np.array([*start, 0.65]), scale=np.array(cfg["body"]["size"]),
        mass=cfg["body"]["mass"], size=1, color=np.array([1.0, 0.24, 0.04])))
    camera = UsdGeom.Camera.Define(stage, "/World/RecordingCamera")
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 500))
    camera.GetFocalLengthAttr().Set(24)
    transform = UsdGeom.Xformable(camera).AddTransformOp()

    def look_at(eye, target):
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1))
        transform.Set(view.GetInverse())

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) / 2
    look_at((center[0] + 12, center[1] - 14, 24), (center[0], center[1], 0))
    product = rep.create.render_product(str(camera.GetPath()), (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    world.reset()
    for _ in range(30):
        RenderingManager.render()

    def save_frame(path):
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        data = rgb.get_data()
        assert data is not None and data.shape[:2] == (720, 1280), "No camera image"
        frame = np.ascontiguousarray(data[:, :, :3])
        assert frame.std() > 2, "Blank camera image"
        Image.fromarray(frame).save(path)

    save_frame(out / "overview.png")
    look_at((start[0] + 3, start[1] - 3, 3.8), (start[0], start[1] + 0.8, 0.2))
    for _ in range(15):
        RenderingManager.render()
    frames = out / "frames"
    frames.mkdir()
    poses = []
    for frame in range(240):
        for _ in range(5):
            if frame >= 48:
                velocity = body.get_linear_velocity()
                body.set_linear_velocity(np.array([0.0, cfg["speed"], velocity[2]]))
                body.set_angular_velocity(np.zeros(3))
            world.step(render=False)
        RenderingManager.render()
        position = body.get_world_pose()[0]
        assert np.isfinite(position).all(), "Non-finite body position"
        poses.append({"time_s": (frame + 1) / 24, "position_m": position.tolist()})
        save_frame(frames / f"{frame:04d}.png")
        if frame in (0, 47, 239):
            save_frame(out / f"simulation-{frame:04d}.png")
        if frame % 24 == 0:
            print(f"RECORD frame={frame}/240 position={position.tolist()}", flush=True)
    report = {
        "mode": "demonstration_not_acceptance", "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "physics_dt_s": 1 / 120, "fps": 24, "duration_s": 10, "frames": len(poses),
        "body_size_m": cfg["body"]["size"], "body_mass_kg": cfg["body"]["mass"],
        "visual_cutaway_height_m": 1.6, "collision_modified": False, "trajectory": poses,
    }
    (out / "recording.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"RECORD complete: {out}", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    app.close()
