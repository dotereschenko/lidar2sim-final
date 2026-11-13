"""Render a reproducible top-down input/baseline/final overview without a GUI."""
import argparse
import base64
import html
import json
from pathlib import Path
import tempfile

import numpy as np
import open3d as o3d

from lidar2sim.reporting import file_record


def render(cloud, baseline, final, output):
    sources = [Path(cloud), Path(baseline)/'collision.ply',
               Path(final)/'visual.ply', Path(final)/'collision.ply']
    labels = ['Input A', 'Baseline collision', 'Final visual', 'Final collision']
    points = [np.asarray(o3d.io.read_point_cloud(str(sources[0])).points)]
    o3d.utility.random.seed(29)
    for path in sources[1:]:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_triangles():
            raise ValueError(f'Empty mesh: {path}')
        points.append(np.asarray(mesh.sample_points_uniformly(300000).points))
    if any(len(p) == 0 or not np.isfinite(p).all() for p in points):
        raise ValueError('All panels require finite nonempty point sets')
    low, high = points[0][:, :2].min(axis=0), points[0][:, :2].max(axis=0)
    scale = min(420 / max(high[0]-low[0], .001), 760 / max(high[1]-low[1], .001))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="910" viewBox="0 0 1920 910">',
           '<rect width="1920" height="910" fill="#101820"/>',
           '<text x="28" y="36" fill="white" font-family="sans-serif" font-size="24">Same A input • XY projection • height slice 0.1–1.6 m (floor omitted)</text>']
    with tempfile.TemporaryDirectory() as temp:
        for i, (label, xyz) in enumerate(zip(labels, points)):
            image = np.full((800, 460, 3), [20, 31, 42], dtype=np.uint8)
            slab = xyz[(xyz[:, 2] >= .1) & (xyz[:, 2] <= 1.6)]
            if len(slab) == 0:
                raise ValueError(f'Empty height slice: {label}')
            pixels = ((slab[:, :2]-low)*scale).astype(int)+20
            valid = (pixels[:, 0] >= 0)&(pixels[:, 0] < 460)&(pixels[:, 1] >= 0)&(pixels[:, 1] < 800)
            pixels = pixels[valid]
            image[799-pixels[:, 1], pixels[:, 0]] = [104, 210, 225]
            path = Path(temp)/f'{i}.png'
            if not o3d.io.write_image(str(path), o3d.geometry.Image(image)):
                raise OSError('Cannot write panel')
            encoded = base64.b64encode(path.read_bytes()).decode()
            x = 20+i*480
            svg += [f'<text x="{x}" y="76" fill="white" font-family="sans-serif" font-size="22">{html.escape(label)}</text>',
                    f'<image x="{x}" y="90" width="460" height="800" href="data:image/png;base64,{encoded}"/>']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(svg+['</svg>']))
    output.with_suffix('.json').write_text(json.dumps({'sources': [file_record(p) for p in sources],
        'projection': 'XY', 'height_slice_m': [.1, 1.6], 'mesh_surface_samples': 300000,
        'seed': 29, 'artifact': file_record(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cloud', 'baseline', 'final', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    render(args.cloud, args.baseline, args.final, args.out)
