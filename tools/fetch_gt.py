"""Скачать эталон NVIDIA и собрать gt_mesh.ply, gt_scene.usd и gt_build.json.

    python tools/fetch_gt.py data/scene_a

Эталон в репозитории не лежит: сцена — ассет Isaac Sim, его условия не разрешают нам его
распространять. Скрипт берёт сцену с открытого сервера ассетов NVIDIA, вы пользуетесь ею
на условиях NVIDIA Isaac Sim Additional Software and Materials License:
https://docs.isaacsim.omniverse.nvidia.com/6.2.0/common/license-isaac-sim-additional.html

Шаги:
1. Скачать USD-файлы сцены и всё, на что они ссылаются (без текстур, ~10 МБ) в <data>/nvidia/.
2. Собрать всю геометрию в один меш в метрах, без стекла и городского фона.
3. Обрезать по manifest.json, сверить числа/хеши, сохранить согласованные PLY/USD и отчёт.
"""
import json
import hashlib
import http.client
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom, UsdShade, UsdUtils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar2sim.common import export_usd

TRANSPARENT = re.compile(r"glass|window|mirror|transparent|crystal", re.I)
GEOMETRY = (".usd", ".usda", ".usdc")
BUILD_DEFAULTS = {"max_edge_m": 0.2, "transparent_name_pattern": TRANSPARENT.pattern,
                  "exclude_name_substrings": ["SM_Buildings"]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(root_url, local_root):
    """Скачивает слой и все его зависимости-слои. Возвращает путь к локальной копии корня."""
    base = root_url.rsplit("/Isaac/", 1)[0] + "/Isaac/"
    def rel(url):
        if not url.startswith(base):
            raise ValueError(f"Asset outside source root: {url}")
        path = Path(urllib.parse.unquote(url[len(base):]))
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f"Unsafe asset path: {url}")
        return path
    seen, frontier = set(), [root_url]

    def fetch(url):
        dst = local_root / rel(url)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        content = r.read()
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError,
                        http.client.HTTPException) as exc:
                    if isinstance(exc, urllib.error.HTTPError) and exc.code != 429 and exc.code < 500:
                        raise RuntimeError(f"Download failed: {url}: HTTP {exc.code}") from exc
                    if attempt == 2:
                        raise RuntimeError(f"Download failed after 3 attempts: {url}: {exc}. Rerun to resume from cache.") from exc
                    time.sleep(2 ** attempt)
            # Never expose a partial download as a cached layer.
            with tempfile.NamedTemporaryFile(dir=dst.parent, suffix=dst.suffix, delete=False) as stream:
                temp = Path(stream.name)
                stream.write(content)
            try:
                UsdUtils.ExtractExternalReferences(str(temp))
                temp.replace(dst)
            finally:
                temp.unlink(missing_ok=True)
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
    sources = [{"path": str(rel(url)), "sha256": sha256(local_root / rel(url))} for url in sorted(seen)]
    (local_root / "layers.json").write_text(json.dumps(sources, indent=2) + "\n")
    return local_root / rel(root_url)


def extract(path, transparent_name_pattern=TRANSPARENT.pattern, exclude_name_substrings=("SM_Buildings",)):
    """Все видимые меши сцены → вершины и треугольники в метрах, ось Z вверх."""
    stage = Usd.Stage.Open(str(path))
    transparent = re.compile(transparent_name_pattern, re.I)
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
        if transparent.search(name) or any(part in name for part in exclude_name_substrings):
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
    if not tris:
        raise ValueError(f"No visible geometry after filters: {path}")
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
    build = {**BUILD_DEFAULTS, **src.get("build", {})}
    lo, hi = np.array(man["bounds_min"]), np.array(man["bounds_max"])
    if (man.get("units") != "m" or man.get("up_axis") != "Z" or lo.shape != (3,) or hi.shape != (3,)
            or not np.isfinite([lo, hi]).all() or not (lo < hi).all()
            or not np.isfinite(build["max_edge_m"]) or build["max_edge_m"] <= 0):
        raise ValueError("Expected finite ordered bounds in meters/Z-up and positive max_edge_m")
    root = download(src["url"], data / "nvidia")
    sources = json.loads((data / "nvidia/layers.json").read_text())
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if src.get("source_layers_sha256", source_hash) != source_hash:
        raise ValueError("source layers hash mismatch; cached/upstream assets differ from manifest")
    tv = clip(extract(root, build["transparent_name_pattern"], build["exclude_name_substrings"]),
              lo, hi, max_edge=build["max_edge_m"])
    gt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tv.reshape(-1, 3)),
                                   o3d.utility.Vector3iVector(np.arange(len(tv) * 3).reshape(-1, 3)))
    gt.remove_duplicated_vertices()
    area, n = gt.get_surface_area(), len(gt.triangles)
    ok = n == src["triangles"] and abs(area - src["area_m2"]) < 0.01 * src["area_m2"]
    if not ok:
        raise ValueError(f"GT mismatch: triangles={n}, area_m2={area}; expected {src['triangles']}, {src['area_m2']}")
    # Validate both artifacts before replacing an existing accepted build.
    with tempfile.TemporaryDirectory(dir=data) as staging:
        staging = Path(staging)
        if not o3d.io.write_triangle_mesh(str(staging / "gt_mesh.ply"), gt):
            raise OSError("Failed to write gt_mesh.ply")
        export_usd(staging / "gt_scene.usd", gt, gt)
        usd_triangles = extract(staging / "gt_scene.usd")
        ply_triangles = np.asarray(gt.vertices)[np.asarray(gt.triangles)]
        delta = float(np.max(np.abs(usd_triangles - ply_triangles)))
        if delta > 1e-6:
            raise ValueError(f"USD/PLY coordinate mismatch: {delta} m")
        outputs = {name: {"sha256": sha256(staging / name)} for name in ['gt_mesh.ply', 'gt_scene.usd']}
        for name, expected in src.get('sha256', {}).items():
            if name not in outputs or outputs[name]['sha256'] != expected:
                raise ValueError(f"Output hash mismatch: {name}")
        report = {"schema_version": 1, "source_url": src['url'], "source_layers": sources,
                  "source_layers_sha256": source_hash, "build": build, "units": "m", "up_axis": "Z",
                  "crop": {"bounds_min": lo.tolist(), "bounds_max": hi.tolist(), "rule": "triangle_centroid"},
                  "triangles": n, "area_m2": area, "vertices": len(gt.vertices),
                  "bounds_min": gt.get_min_bound().tolist(), "bounds_max": gt.get_max_bound().tolist(),
                  "usd_ply_max_coordinate_delta_m": delta, "outputs": outputs}
        (staging / "gt_build.json").write_text(json.dumps(report, indent=2) + "\n")
        for name in [*outputs, 'gt_build.json']:
            (staging / name).replace(data / name)
    print(f"{data / 'gt_mesh.ply'}: {n} треугольников, площадь {area:.1f} м² — "
          "совпадает с контрольными числами manifest; USD/PLY и хеши проверены")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        main(sys.argv[1])
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(f"GT build failed: {exc}")
