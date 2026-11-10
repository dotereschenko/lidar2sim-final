"""Тесты физики сцены в Isaac Sim по test_config.json: пол, препятствия, маршруты.

    python.sh tools/isaac_scene_test.py --scene out/scene.usd --config test_config.json --out physics.json

- Пол: корпус ставится на пол на 30 с. Провал — низ корпуса ниже пола на floor_drop_m.
- Препятствия: корпус едет по dir со скоростью speed 8 с. free_m — свободный путь передней грани
  до препятствия по эталону; проникновение = наибольший пройденный путь вдоль dir − free_m.
- Маршруты: корпус едет по точкам маршрута. Провал — застрял (за 5 с сдвинулся меньше 5 см),
  не доехал за отведённое время или провалился.

Пороги — в разделе pass файла test_config.json: проникновение, провал пола и сколько тестов
каждого вида нужно пройти для зачёта.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cpu-threads", type=int, default=4, help="число рабочих потоков Isaac/PhysX")
args = ap.parse_args()
if args.cpu_threads < 1:
    ap.error("--cpu-threads должен быть ≥ 1")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "disable_viewport_updates": True,
                     "limit_cpu_threads": args.cpu_threads,
                     "extra_args": [f"--/persistent/physics/numThreads={args.cpu_threads}"]})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage

cfg = json.load(open(args.config))
DT = cfg["dt"]
BODY = np.array(cfg["body"]["size"])
V = cfg["speed"]
P = cfg["pass"]

world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=args.scene, prim_path="/World/Scene")
ps = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Scene/PhysicsScene")
if ps.IsValid():
    ps.SetActive(False)
body = world.scene.add(DynamicCuboid(prim_path="/World/Body", name="body", position=np.array([0, 0, 50.0]),
                                     scale=BODY, size=1.0, mass=cfg["body"]["mass"]))
world.reset()


def place(xy, z_floor):
    body.set_world_pose(position=np.array([xy[0], xy[1], z_floor + BODY[2] / 2 + 0.02]),
                        orientation=np.array([1.0, 0, 0, 0]))
    body.set_linear_velocity(np.zeros(3))
    body.set_angular_velocity(np.zeros(3))


def step(vel=None):
    if vel is not None:
        v = body.get_linear_velocity()
        body.set_linear_velocity(np.array([vel[0], vel[1], v[2]]))
        body.set_angular_velocity(np.zeros(3))
    world.step(render=False)
    return body.get_world_pose()[0]


res = {"floor": [], "obstacles": [], "routes": []}

for f in cfg["floor"]:
    place(f["xy"], f["floor_z"])
    low = min(step()[2] for _ in range(int(30 / DT))) - BODY[2] / 2
    drop = f["floor_z"] - low
    res["floor"].append({**f, "drop_m": round(float(drop), 4), "ok": bool(drop < P["floor_drop_m"])})
    print(f"floor {len(res['floor'])}/{len(cfg['floor'])}", flush=True)

for o in cfg["obstacles"]:
    place(o["start"], o["floor_z"])
    for _ in range(int(1 / DT)):
        step()
    d = np.array(o["dir"])
    p0 = body.get_world_pose()[0][:2]
    best, low = 0.0, np.inf
    for _ in range(int(8 / DT)):
        p = step(V * d)
        best = max(best, float(np.dot(p[:2] - p0, d)))
        low = min(low, p[2] - BODY[2] / 2)
    pen = max(0.0, best - o["free_m"])
    fell = o["floor_z"] - low > P["floor_drop_m"]
    res["obstacles"].append({**o, "travel_m": round(best, 3), "penetration_m": round(pen, 4),
                             "fell": bool(fell), "ok": bool(pen <= P["penetration_m"] and not fell)})
    print(f"obstacles {len(res['obstacles'])}/{len(cfg['obstacles'])}", flush=True)

for r in cfg["routes"]:
    pts = np.array(r["points"])
    place(pts[0], 0.0)
    for _ in range(int(1 / DT)):
        step()
    k, t, status = 1, 0.0, "ok"
    limit = 3 * r["length_m"] / V + 10
    hist = []
    while k < len(pts):
        p = body.get_world_pose()[0]
        if p[2] - BODY[2] / 2 < -P["floor_drop_m"]:
            status = "провалился"
            break
        to = pts[k] - p[:2]
        if np.linalg.norm(to) < 0.15:
            k += 1
            continue
        step(V * to / np.linalg.norm(to))
        t += DT
        hist.append(p[:2].copy())
        if len(hist) > int(5 / DT) and np.linalg.norm(hist[-1] - hist[-int(5 / DT)]) < 0.05:
            status = "застрял"
            break
        if t > limit:
            status = "не успел"
            break
    p = body.get_world_pose()[0]
    res["routes"].append({"name": r["name"], "length_m": r["length_m"], "status": status,
                          "reached_point": int(k), "of": len(pts), "at": p[:2].round(2).tolist(),
                          "time_s": round(t, 1), "ok": status == "ok"})
    print(f"routes {len(res['routes'])}/{len(cfg['routes'])}", flush=True)

res["summary"] = {key: f'{sum(x["ok"] for x in res[key])}/{len(res[key])}'
                  for key in ("floor", "obstacles", "routes")}
# зачёт: не меньше заданного числа пройденных тестов каждого вида
res["passed"] = all(sum(x["ok"] for x in res[key]) >= P.get(f"{key}_min", len(res[key]))
                    for key in ("floor", "obstacles", "routes"))
res["summary"]["passed"] = res["passed"]
json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res["summary"], ensure_ascii=False))
app.close()
