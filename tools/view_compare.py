"""Интерактивный просмотр: baseline слева, восстановленный пол справа (оранжевый)."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_parts(before, after, kind, cut_height, errors=False):
    filename = f"{kind}_errors.ply" if errors else f"{kind}.ply"
    meshes = [o3d.io.read_triangle_mesh(str(folder / filename)) for folder in (before, after)]
    if any(not mesh.has_triangles() for mesh in meshes):
        raise ValueError("Сначала запустите реконструкцию для обеих папок")
    report = json.loads((after / "report.json").read_text())
    added = report.get("floor", {}).get(kind, {}).get("added_triangles", 0)
    if not 0 <= added <= len(meshes[1].triangles):
        raise ValueError("Число заплаток в report.json не соответствует сетке")
    floor_z = report.get("floor", {}).get(kind, {}).get("height_m", 0)
    width = max(mesh.get_max_bound()[0] - mesh.get_min_bound()[0] for mesh in meshes)
    parts = []
    for side, mesh in enumerate(meshes):
        triangles = np.asarray(mesh.triangles)
        split = len(triangles) - (added if side and not errors else 0)
        for faces, color in [(triangles[:split], [0.65, 0.72, 0.78]),
                             (triangles[split:], [1.0, 0.42, 0.06])]:
            if not len(faces):
                continue
            part = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                                            o3d.utility.Vector3iVector(faces))
            if errors:
                if not mesh.has_vertex_colors():
                    raise ValueError("Сначала создайте цветные карты: tools/color_errors.py")
                part.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
            high = (np.asarray(part.vertices)[faces, 2] > floor_z + cut_height).any(axis=1)
            part.remove_triangles_by_mask(high)
            part.remove_unreferenced_vertices()
            if not part.has_triangles():
                continue
            part.compute_vertex_normals()
            if not errors:
                part.paint_uniform_color(color)
            part.translate([side * (width + 2), 0, 0])
            parts.append(part)
    if not parts:
        raise ValueError("Срез убрал всю геометрию; увеличьте --cut-height")
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, default=Path("out/baseline"))
    ap.add_argument("--after", type=Path, default=Path("out/floor"))
    ap.add_argument("--kind", choices=["visual", "collision"], default="collision")
    ap.add_argument("--cut-height", type=float, default=1.6, help="срез над полом, м")
    ap.add_argument("--screenshot", type=Path, help="сохранить начальный вид в PNG")
    ap.add_argument("--errors", action="store_true", help="показать карты ошибок от color_errors.py")
    args = ap.parse_args()
    parts = load_parts(args.before, args.after, args.kind, args.cut_height, args.errors)
    vis = o3d.visualization.Visualizer()
    legend = "Red: error >5cm" if args.errors else "Orange: repaired floor"
    if not vis.create_window(f"BEFORE (left) | AFTER (right) | {legend}", 1400, 900):
        raise RuntimeError("Не удалось открыть окно Open3D; нужен графический сеанс")
    for part in parts:
        vis.add_geometry(part)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.10, 0.13])
    opt.mesh_show_back_face = True
    view = vis.get_view_control()
    view.set_front([0, -0.35, 1])
    view.set_up([0, 1, 0])
    view.set_zoom(0.75)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(args.screenshot), do_render=True)
    legend_ru = "Красный — ошибка >5 см." if args.errors else "Оранжевый — заплатки пола."
    print(f"Слева — было, справа — стало. {legend_ru} Потолок скрыт.", flush=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
