"""Тесты физики в Isaac Sim для тестовой комнаты: пол, препятствия, проём.

Запускать питоном Isaac Sim (python.sh), headless:
    python.sh tools/isaac_physics_test.py --scene out/baseline/scene.usd \
        --manifest data/test_room/manifest.json --out out/baseline/physics.json

Корпус робота — коробка 0,4×0,4×0,3 м, 20 кг.
- Пол: 10 позиций по 30 с, две из них в пятне пола без точек. Провал — низ корпуса ниже −5 см.
- Препятствия: 10 подходов на 0,2 м/с с упором 3 с. Проникновение считается по эталонной
  геометрии из manifest.json, а не по collision mesh. Порог — 2 см.
- Проём: проезд через проём на 0,2 м/с. Успех — корпус оказался за перегородкой и не провалился.
- Во всех тестах корпус, ушедший ниже пола, — провал, даже если проникновения в стену нет.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402

try:                                    # Isaac Sim 4.5+
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
except ImportError:                     # Isaac Sim 4.0–4.2
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage

DT = 1 / 120
BODY = np.array([0.4, 0.4, 0.3])
t = json.load(open(args.manifest))["test"]
L, W, H = t["size"]
(bx0, by0, _), (bx1, by1, bz1) = t["box"]
(cx, cy), cr = t["column"]
door = t["door"]
WALL_HALF = 0.05

world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=args.scene, prim_path="/World/Scene")
stage = omni.usd.get_context().get_stage()
ps = stage.GetPrimAtPath("/World/Scene/PhysicsScene")
if ps.IsValid():
    ps.SetActive(False)                 # своя PhysicsScene есть у World
body = world.scene.add(DynamicCuboid(prim_path="/World/Body", name="body",
                                     position=np.array([1.0, 3.0, 1.0]),
                                     scale=BODY, size=1.0, mass=20.0))
world.reset()


def place(x, y, z=None):
    z = BODY[2] / 2 + 0.02 if z is None else z
    body.set_world_pose(position=np.array([x, y, z]), orientation=np.array([1.0, 0, 0, 0]))
    body.set_linear_velocity(np.zeros(3))
    body.set_angular_velocity(np.zeros(3))


def run(seconds, vel=None, track=None):
    """Шаги физики. vel — горизонтальная скорость, поддерживается каждый шаг."""
    worst = 0.0
    for _ in range(int(seconds / DT)):
        if vel is not None:
            v = body.get_linear_velocity()
            body.set_linear_velocity(np.array([vel[0], vel[1], v[2]]))
            body.set_angular_velocity(np.zeros(3))
        world.step(render=False)
        if track:
            worst = max(worst, track(body.get_world_pose()[0]))
    return body.get_world_pose()[0], worst


res = {"floor": [], "obstacles": [], "door": None}

# пол: 8 свободных точек + 2 в пятне без точек
(hx0, hy0), (hx1, hy1) = t["floor_hole"]
floor_pts = [(0.6, 0.6), (0.6, 3.4), (2.6, 0.6), (3.3, 0.6), (1.0, 2.6),
             (2.0, 2.2), (5.0, 1.0), (5.2, 3.2), ((hx0 + hx1) / 2, (hy0 + hy1) / 2),
             (hx0 + 0.15, hy1 - 0.15)]
for x, y in floor_pts:
    place(x, y)
    lowest = [np.inf]

    def low(p):
        lowest[0] = min(lowest[0], p[2] - BODY[2] / 2)
        return 0.0
    run(30, track=low)
    res["floor"].append({"xy": [x, y], "min_bottom_z": round(float(lowest[0]), 4),
                         "ok": bool(lowest[0] > -0.05)})

# препятствия: (старт, направление, функция проникновения по эталону)
hx, hy = BODY[0] / 2, BODY[1] / 2
approaches = [
    ("стена x=0", (0.9, 3.4), (-1, 0), lambda p: max(0.0, hx - p[0])),
    ("стена y=0", (0.9, 0.9), (0, -1), lambda p: max(0.0, hy - p[1])),
    ("стена y=W", (2.6, W - 0.9), (0, 1), lambda p: max(0.0, p[1] + hy - W)),
    ("стена x=L", (L - 0.9, 0.8), (1, 0), lambda p: max(0.0, p[0] + hx - L)),
    ("перегородка, слева", (door["x"] - 0.9, 0.7), (1, 0),
     lambda p: max(0.0, p[0] + hx - (door["x"] - WALL_HALF))),
    ("перегородка, справа", (door["x"] + 0.9, 3.3), (-1, 0),
     lambda p: max(0.0, (door["x"] + WALL_HALF) - (p[0] - hx))),
    ("ящик, сбоку", (bx1 + 0.9, (by0 + by1) / 2), (-1, 0), lambda p: max(0.0, bx1 - (p[0] - hx))),
    ("ящик, спереди", ((bx0 + bx1) / 2, by1 + 0.9), (0, -1), lambda p: max(0.0, by1 - (p[1] - hy))),
    ("колонна, по x", (cx - 1.0, cy), (1, 0), lambda p: max(0.0, (p[0] + hx) - (cx - cr))),
    ("колонна, по y", (cx, cy - 1.0), (0, 1), lambda p: max(0.0, (p[1] + hy) - (cy - cr))),
]
for name, (x, y), (dx, dy), pen in approaches:
    place(x, y)
    run(1.0)
    end, worst = run(8.0, vel=(0.2 * dx, 0.2 * dy), track=pen)
    fell = bool(end[2] < BODY[2] / 2 - 0.05)
    res["obstacles"].append({"name": name, "penetration_m": round(float(worst), 4),
                             "fell": fell, "ok": bool(worst <= 0.02 and not fell)})

# проём. Старт вне пятна пола без точек: иначе в простой версии корпус падает сквозь пол
# и «проезжает» проём под полом. Скорость поддерживается каждый шаг, но трение её срезает,
# поэтому едем до 20 с, пока корпус не окажется за перегородкой.
place(door["x"] - 0.4, sum(door["y"]) / 2)
run(1.0)
track = []
for _ in range(20):                     # положение раз в секунду: видно, где корпус застревает
    end, _ = run(1.0, vel=(0.2, 0.0))
    track.append([round(float(c), 3) for c in end])
    if end[0] > door["x"] + 0.4 or end[2] < 0:
        break
res["door"] = {"end": track[-1], "track": track,
               "ok": bool(end[0] > door["x"] + 0.4 and end[2] > BODY[2] / 2 - 0.05)}

res["summary"] = {
    "floor_ok": f'{sum(r["ok"] for r in res["floor"])}/{len(res["floor"])}',
    "obstacles_ok": f'{sum(r["ok"] for r in res["obstacles"])}/{len(res["obstacles"])}',
    "door_ok": res["door"]["ok"],
}
json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
print(json.dumps(res["summary"], ensure_ascii=False))
app.close()
