"""Geometry evaluation: legacy A/GT diagnostic, or pinned common-region B/GT.

Default CLI remains backward compatible. Explicit --mode b/gt requires
--annotations and --scans. Mesh precision uses GT triangles; completeness
uses uniform-area GT anchors or their nearest real B returns, separately
for visual and collision. Other PoC groups remain unmeasured.
"""
import argparse
import json
import sys
from pathlib import Path

# Source-checkout scripts also run by absolute path outside the repository cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar2sim.acceptance import geometry_checks, load_profile, overall_status

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def read_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if (not len(vertices) or not len(triangles) or not np.isfinite(vertices).all()
            or triangles.min() < 0 or triangles.max() >= len(vertices)
            or not np.isfinite(mesh.get_surface_area()) or mesh.get_surface_area() <= 0):
        raise ValueError(f'empty or invalid mesh: {path}')
    return mesh


def raycasting(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def sample(mesh, profile):
    g = profile['geometry']
    o3d.utility.random.seed(g['seed'])
    return np.asarray(mesh.sample_points_uniformly(g['samples']).points)


def region(data, profile):
    """GT sample in the observed A region; the coverage rule is diagnostic only."""
    gt = read_mesh(data / 'gt_mesh.ply')
    points_a = np.asarray(o3d.io.read_point_cloud(str(data / 'A.ply')).points)
    if not len(points_a) or not np.isfinite(points_a).all():
        raise ValueError('A point cloud must be nonempty and finite')
    pts = sample(gt, profile)
    seen = cKDTree(points_a).query_ball_point(pts, 0.10, return_length=True) >= 10
    if not seen.any():
        raise ValueError('empty reference region observed by A')
    return gt, pts[seen], float(seen.mean())


def extra_within(mesh, gt_scene, profile):
    points = sample(mesh, profile)
    distances = gt_scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
    if not np.isfinite(distances).all():
        raise ValueError('nonfinite precision distances')
    return float((distances <= profile['geometry']['precision_distance_m']).mean()) * 100


def evaluate(out, gt_scene, ref, profile, provenance):
    if not len(ref) or not np.isfinite(ref).all():
        raise ValueError('reference must be nonempty and finite')
    vis = read_mesh(out / 'visual.ply')
    col = read_mesh(out / 'collision.ply')
    distances = raycasting(vis).compute_distance(o3d.core.Tensor(ref.astype(np.float32))).numpy()
    if not np.isfinite(distances).all():
        raise ValueError('nonfinite completeness distances')
    raw = {
        'extra_5cm_pct': extra_within(vis, gt_scene, profile),
        'extra_5cm_pct_collision': extra_within(col, gt_scene, profile),
        'median_cm': float(np.median(distances)) * 100,
        'p95_cm': float(np.percentile(distances, 95, method='linear')) * 100,
    }
    passed = geometry_checks(raw, profile)
    status = 'passed' if all(passed.values()) else 'failed'
    return {
        **{key: round(value, 1 if key.startswith('extra_') else 2) for key, value in raw.items()},
        'passed': passed, 'raw_metrics': raw, 'acceptance_profile': provenance,
        'evaluation_mode': 'scene_a_gt_diagnostic',
        'coverage': {'collision_completeness': 'not_run'},
        'geometry_status': status,
        'poc_status': overall_status({'implemented_geometry': status, 'remaining_groups': 'not_run'}),
    }


def load_common_reference(data, scans, annotations, mode, profile):
    from lidar2sim.annotations import cloud_array
    from tools.make_dense_scan import digest
    data, scans, annotations = map(Path, (data, scans, annotations))
    if mode not in ('b', 'gt'):
        raise ValueError('Common evaluation mode must be b or gt')
    metadata = json.loads((annotations/'annotations.json').read_text())
    if metadata.get('schema_version') != 1 or metadata.get('units') != 'm' or metadata.get('up_axis') != 'Z':
        raise ValueError('Invalid annotation schema or coordinates')
    for path, expected in [(data/'gt_mesh.ply', metadata['inputs_sha256']['gt_mesh.ply']),
                           *[(scans/f'{n}.ply',metadata['inputs_sha256'][f'{n}.ply']) for n in ('A','B')],
                           (annotations/'eval_region.npz',metadata['artifacts_sha256']['eval_region.npz'])]:
        if digest(path) != expected:
            raise ValueError(f'Annotation input hash mismatch: {path.name}')
    region = metadata['region']
    minimum = region['min_common_samples']
    if type(minimum) is not int or minimum < 1000:
        raise ValueError('Minimum common sample count must be at least 1000')
    a,b = [cloud_array(np.asarray(o3d.io.read_point_cloud(str(scans/f'{n}.ply')).points)) for n in ('A','B')]
    with np.load(annotations/'eval_region.npz', allow_pickle=False) as saved:
        points = cloud_array(saved['points_m'])
        mask = saved['common_mask']
        ca,cb = saved['a_neighbors'],saved['b_neighbors']
        if (mask.dtype != np.bool_ or mask.shape != (len(points),) or
                ca.shape != mask.shape or cb.shape != mask.shape or
                not np.isfinite(ca).all() or not np.isfinite(cb).all() or
                (ca<0).any() or (cb<0).any() or
                not np.array_equal(mask,(ca>=region['min_neighbors']) & (cb>=region['min_neighbors']))):
            raise ValueError('Invalid saved common region arrays')
        if len(points) != region['samples'] or int(mask.sum()) != region['common_samples'] or mask.sum() < minimum:
            raise ValueError('Insufficient or inconsistent common region samples')
        reference = points[mask]
    eligible = len(reference)
    count = min(eligible,profile['geometry']['samples'])
    if count < minimum:
        raise ValueError('Insufficient eligible reference samples')
    indices = np.random.default_rng(profile['geometry']['seed']).choice(eligible,count,replace=False)
    reference = reference[indices].astype(np.float32)
    mapping = None
    if mode == 'b':
        distance, nearest = cKDTree(b).query(reference,workers=1)
        if not np.isfinite(distance).all() or (distance > region['radius_m']).any():
            raise ValueError('Saved common region lacks nearby B support')
        reference = b[nearest].astype(np.float32)
        mapping = {'anchor_to_B_max_m':float(distance.max()),
                   'anchor_to_B_median_m':float(np.median(distance)),
                   'unique_B_returns':int(len(np.unique(nearest))),
                   'repeated_returns':'retain repeated nearest returns with their area-anchor weights'}
    return {'reference':reference,'a':a,'b':b,'region':region,'minimum':minimum,'mode':mode,
            'provenance':{'annotations_sha256':digest(annotations/'annotations.json'),
                          'region_sha256':digest(annotations/'eval_region.npz'),
                          'inputs_sha256':metadata['inputs_sha256'],
                          'annotation_status':metadata.get('status','unknown'),
                          'reference_type':'B_returns' if mode=='b' else 'GT_surface',
                          'reference_weighting':'uniform_GT_area_anchors_nearest_B_return' if mode=='b' else 'uniform_surface_area',
                          'B_anchor_mapping':mapping,
                          'eligible_reference_samples':eligible,'reference_samples':count,
                          'region_rule':region}}


def evaluate_common(out, gt_scene, context, profile, provenance, dimensions=None):
    from lidar2sim.annotations import observed_by_both
    from tools.make_dense_scan import digest
    out = Path(out)
    g, region = profile['geometry'], context['region']
    meshes = {}
    for name in g['meshes']:
        path = out/f'{name}.ply'
        mesh = read_mesh(path)
        points = sample(mesh,profile)
        observed,_,_ = observed_by_both(points,context['a'],context['b'],region['radius_m'],region['min_neighbors'])
        if int(observed.sum()) < context['minimum']:
            raise ValueError(f'{name}: insufficient common-region mesh samples')
        precision = gt_scene.compute_distance(o3d.core.Tensor(points[observed].astype(np.float32))).numpy()
        mesh_scene = raycasting(mesh)
        completeness = mesh_scene.compute_distance(o3d.core.Tensor(context['reference'])).numpy()
        if not np.isfinite(precision).all() or not np.isfinite(completeness).all():
            raise ValueError('Nonfinite geometry distances')
        raw = {'precision_fraction':float((precision<=g['precision_distance_m']).mean()),
               'median_m':float(np.median(completeness)),
               'p95_m':float(np.percentile(completeness,95,method='linear'))}
        checks = {'precision':raw['precision_fraction']>=g['precision_min_fraction'],
                  'median':raw['median_m']<=g['completeness_median_max_m'],
                  'p95':raw['p95_m']<=g['completeness_p95_max_m']}
        meshes[name] = {'raw_metrics':raw,'passed':checks,'sha256':digest(path),
                        'precision_reference':'GT_triangles','precision_direction':'mesh_surface_to_GT',
                        'completeness_reference':context['provenance']['reference_type'],
                        'completeness_direction':'reference_to_mesh_triangles',
                        'surface_samples':len(points),'common_surface_samples':int(observed.sum()),
                        'excluded_surface_fraction':float((~observed).mean())}
        if dimensions is not None:
            from lidar2sim.dimensions import measure_dimensions
            meshes[name]['dimensions'] = measure_dimensions(mesh_scene,dimensions,profile['dimensions'])
    status = 'passed' if all(all(m['passed'].values()) for m in meshes.values()) else 'failed'
    dimension_status = ('passed' if all(m['dimensions']['status']=='passed' for m in meshes.values()) else 'failed') if dimensions is not None else 'not_run'
    return {'evaluation_mode':'independent_b_common_region' if context['mode']=='b' else 'gt_common_region',
            'meshes':meshes,'acceptance_profile':provenance,'reference_provenance':context['provenance'],
            'evaluator_sha256':digest(__file__),'geometry_status':status,'dimensions_status':dimension_status,
            'dimension_measurer_sha256':digest(Path(__file__).resolve().parents[1]/'lidar2sim/dimensions.py') if dimensions is not None else None,
            'coverage':{'visual_completeness':'measured','collision_completeness':'measured',
                        'dimensions':'measured' if dimensions is not None else 'not_run','physics':'not_run','performance':'not_run'},
            'poc_status':overall_status({'geometry':status,'dimensions':dimension_status,'remaining_groups':'not_run'})}


def main(data, *outs, profile=None, mode="diagnostic", annotations=None, scans=None, dimensions=False):
    rules, provenance = load_profile(profile)
    if mode == 'diagnostic':
        if annotations is not None or scans is not None or dimensions:
            raise ValueError('--annotations/--scans/--dimensions require explicit --mode b or gt')
        gt, ref, share = region(Path(data), rules)
        gt_scene = raycasting(gt)
        print(f'диагностика A/GT: видимая область {share * 100:.0f}% поверхности сцены')
        rows = [(Path(out), evaluate(Path(out), gt_scene, ref, rules, provenance)) for out in outs]
    else:
        if annotations is None or scans is None:
            raise ValueError('--mode b/gt requires --annotations and --scans')
        definitions = None
        if dimensions:
            from lidar2sim.dimensions import validate_dimensions
            definitions = json.loads((Path(annotations)/'annotations.json').read_text())['dimensions']
            validate_dimensions(definitions,rules['dimensions']['required_count'])
        context = load_common_reference(data,scans,annotations,mode,rules)
        gt_scene = raycasting(read_mesh(Path(data)/'gt_mesh.ply'))
        rows = [(Path(out),evaluate_common(out,gt_scene,context,rules,provenance,definitions)) for out in outs]
    for out, result in rows:
        (out / 'eval_scene.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if mode != 'diagnostic':
            print(f'{out}: {result["evaluation_mode"]}; geometry={result["geometry_status"]}; PoC={result["poc_status"]}')
            continue
        print(f'{out}: {result["extra_5cm_pct"]}% visual, {result["extra_5cm_pct_collision"]}% collision; '
              f'медиана {result["median_cm"]} см, p95 {result["p95_cm"]} см; '
              f'четыре реализованные проверки: {result["geometry_status"]}; PoC: {result["poc_status"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data')
    ap.add_argument('outs', nargs='+')
    ap.add_argument('--profile')
    ap.add_argument('--mode', choices=['diagnostic','b','gt'], default='diagnostic')
    ap.add_argument('--annotations')
    ap.add_argument('--scans')
    ap.add_argument('--dimensions', action='store_true', help='Measure all annotated dimensions on both meshes')
    args = ap.parse_args()
    try:
        main(args.data, *args.outs, profile=args.profile, mode=args.mode, annotations=args.annotations, scans=args.scans, dimensions=args.dimensions)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        ap.exit(1, f'evaluation error: {exc}\n')
