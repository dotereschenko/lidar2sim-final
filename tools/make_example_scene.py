"""Create the small, asset-free Isaac example used for participant preflight.

    python tools/make_example_scene.py examples/example_scene.usd
"""

import argparse
from pathlib import Path

import open3d as o3d

from lidar2sim.common import export_usd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path,
                        default=Path("examples/example_scene.usd"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The top face of the floor is z=0; the box is a simple visible obstacle.
    floor = o3d.geometry.TriangleMesh.create_box(3.0, 2.0, 0.10)
    floor.translate((0, 0, -0.10))
    obstacle = o3d.geometry.TriangleMesh.create_box(0.30, 0.60, 0.80)
    obstacle.translate((1.40, 0.70, 0))
    mesh = floor + obstacle
    export_usd(args.out, mesh, mesh)
    print(f"{len(mesh.triangles)} triangles -> {args.out}")


if __name__ == "__main__":
    main()
