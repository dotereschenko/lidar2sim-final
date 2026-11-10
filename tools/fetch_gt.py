"""Точная геометрия сцены (эталон) — скачать сцену у NVIDIA и собрать gt_mesh.ply.

    python tools/fetch_gt.py data/scene_a

Эталон в репозитории не лежит: сцена — ассет Isaac Sim, его условия не разрешают нам его
распространять. Скрипт берёт сцену с открытого сервера ассетов NVIDIA, вы пользуетесь ею
на условиях NVIDIA Isaac Sim Additional Software and Materials License:
https://docs.isaacsim.omniverse.nvidia.com/6.2.0/common/license-isaac-sim-additional.html

Шаги:
1. Скачать USD-файлы сцены и всё, на что они ссылаются (без текстур, ~10 МБ) в <data>/nvidia/.
2. Собрать всю геометрию в один меш в метрах, без стекла и городского фона.
3. Обрезать по границам из manifest.json и сверить с контрольными числами оттуда же.
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils

TRANSPARENT = re.compile(r"glass|window|mirror|transparent|crystal", re.I)
GEOMETRY = (".usd", ".usda", ".usdc")


def download(root_url, local_root):
    """Скачивает слой и все его зависимости-слои. Возвращает путь к локальной копии корня."""
    base = root_url.rsplit("/Isaac/", 1)[0] + "/Isaac/"
    rel = lambda url: urllib.parse.unquote(url[len(base):])
    seen, frontier = set(), [root_url]

    def fetch(url):
        dst = local_root / rel(url)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
        subs, refs, payloads = UsdUtils.ExtractExternalReferences(str(dst))
        out = []
        for p in [*subs, *refs, *payloads]:
            p = p.split("?")[0]
            if not p.lower().endswith(GEOMETRY):
                continue
            u = p if p.startswith("http") else urllib.parse.urljoin(url, p)
            if u.startswith(base):
                out.append(u)
        return out

    with ThreadPoolExecutor(16) as pool:
        while frontier:
            frontier = [u for u in dict.fromkeys(frontier) if u not in seen]
            seen.update(frontier)
            frontier = [u for deps in pool.map(fetch, frontier) for u in deps]
            print(f"\rскачано слоёв: {len(seen)}", end="", flush=True)
    print()
    return local_root / rel(root_url)


def extract(path):
    """Все видимые меши сцены → вершины и треугольники в метрах, ось Z вверх."""
    stage = Usd.Stage.Open(str(path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    y_up = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    cache = UsdGeom.XformCache()
    tris = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        name = (str(mat.GetPath()) if mat else "") + " " + str(prim.GetPath())
        if TRANSPARENT.search(name) or "SM_Buildings" in name:     # стекло и городской фон
            continue
        mesh = UsdGeom.Mesh(prim)
        pts, counts, idx = (mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(),
                            mesh.GetFaceVertexIndicesAttr().Get())
        if not pts or not counts or not idx:
            continue
        P = np.asarray(pts, dtype=np.float64)
        M = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)   # строки — векторы
        P = (np.hstack([P, np.ones((len(P), 1))]) @ M)[:, :3] * mpu
        if y_up:
            P = P[:, [0, 2, 1]] * [1, -1, 1]
        counts, idx = np.asarray(counts), np.asarray(idx)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        T = [np.stack([idx[s] * np.ones(c - 2, int), idx[s + 1:s + c - 1], idx[s + 2:s + c]], 1)
             for s, c in zip(starts, counts) if c >= 3]                        # веер
        if T:
            # через float32, как у организаторов: иначе обрезка по границе даёт другие треугольники
            tris.append(P.astype(np.float32).astype(np.float64)[np.concatenate(T)])
    return np.concatenate(tris)


def clip(tv, lo, hi, max_edge=0.2):
    """Обрезка по зоне. Большие треугольники сначала дробятся, иначе обрезка их выкидывает."""
    tv = tv[np.all((tv.max(1) >= lo) & (tv.min(1) <= hi), axis=1)]
    while True:
        big = np.linalg.norm(tv - np.roll(tv, -1, axis=1), axis=2).max(1) > max_edge
        if not big.any():
            break
        t = tv[big]
        m01, m12, m20 = (t[:, 0] + t[:, 1]) / 2, (t[:, 1] + t[:, 2]) / 2, (t[:, 2] + t[:, 0]) / 2
        tv = np.concatenate([tv[~big], np.stack([t[:, 0], m01, m20], 1), np.stack([m01, t[:, 1], m12], 1),
                             np.stack([m20, m12, t[:, 2]], 1), np.stack([m01, m12, m20], 1)])
    return tv[np.all((tv.mean(1) >= lo) & (tv.mean(1) <= hi), axis=1)]


def main(data):
    data = Path(data)
    man = json.loads((data / "manifest.json").read_text())
    src = man["gt"]
    root = download(src["url"], data / "nvidia")
    tv = clip(extract(root), np.array(man["bounds_min"]), np.array(man["bounds_max"]))
    gt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tv.reshape(-1, 3)),
                                   o3d.utility.Vector3iVector(np.arange(len(tv) * 3).reshape(-1, 3)))
    gt.remove_duplicated_vertices()
    o3d.io.write_triangle_mesh(str(data / "gt_mesh.ply"), gt)

    area, n = gt.get_surface_area(), len(gt.triangles)
    ok = n == src["triangles"] and abs(area - src["area_m2"]) < 0.01 * src["area_m2"]
    print(f"{data / 'gt_mesh.ply'}: {n} треугольников, площадь {area:.1f} м² — "
          + ("совпадает с эталоном организаторов" if ok else
             f"НЕ совпадает: ожидалось {src['triangles']} треугольников, {src['area_m2']} м²"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
