"""Shared acceptance contract; thresholds live only in the YAML profile."""
import hashlib
import json
import math
from pathlib import Path

import yaml

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / 'configs/acceptance/lidar2sim-rules-v1.yaml'


def release_decision(provenance):
    """Approval is tied to exact adopted profile bytes, never just its name."""
    path = DEFAULT_PROFILE.with_name('release-decision-v2.json')
    if not path.is_file():
        return None
    decision = json.loads(path.read_text())
    if (decision.get('status') == 'adopted' and decision.get('profile_sha256') == provenance['sha256']
            and decision.get('profile_id') == provenance['profile_id']):
        return decision
    return None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _positive(value):
    return _finite(value) and value > 0


def _count(value):
    return type(value) is int and value > 0


def _body(value):
    return type(value) is list and len(value) == 3 and all(_positive(x) for x in value)


def _fields(value, fields, where):
    if not isinstance(value, dict) or value.keys() != fields.keys():
        raise ValueError(f'{where}: expected exactly fields {", ".join(fields)}')
    for key, valid in fields.items():
        if not valid(value[key]):
            raise ValueError(f'{where}.{key}: invalid value {value[key]!r}')


def load_profile(path=None):
    """Read once, validate strictly, and hash the exact bytes consumed."""
    path = DEFAULT_PROFILE if path is None else Path(path)
    try:
        data = path.read_bytes()
        profile = yaml.safe_load(data)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f'acceptance profile {path}: {exc}') from exc
    _fields(profile, {
        'schema_version': lambda x: type(x) is int and x in (1, 2),
        'profile_id': lambda x: x in ('lidar2sim-rules-v1', 'lidar2sim-rules-v2'),
        'units': lambda x: x == 'm', 'up_axis': lambda x: x == 'Z',
        **{name: lambda x: isinstance(x, dict)
           for name in ('geometry', 'dimensions', 'physics', 'performance')},
    }, 'profile')
    _fields(profile['geometry'], {
        'meshes': lambda x: x == ['visual', 'collision'],
        'precision_reference': lambda x: x == ('b_cloud' if profile['schema_version'] == 2 else 'gt_mesh'),
        'precision_distance_m': _positive,
        'precision_min_fraction': lambda x: _finite(x) and 0 <= x <= 1,
        'precision_comparison': lambda x: x == 'le',
        'completeness_median_max_m': _positive, 'completeness_p95_max_m': _positive,
        'samples': _count, 'seed': lambda x: type(x) is int and 0 <= x <= 2**31 - 1,
    }, 'profile.geometry')
    _fields(profile['dimensions'], {
        'required_count': _count, 'absolute_error_max_m': _positive,
    }, 'profile.dimensions')
    _fields(profile['physics'], {
        'body_size_m': _body,
        **{name: _positive for name in ('body_mass_kg', 'speed_m_s', 'dt_s', 'floor_duration_s',
                                       'floor_drop_max_m', 'penetration_max_m')},
        **{f'{group}_{suffix}': _count for group in ('floor', 'obstacles', 'routes')
           for suffix in ('pass_min', 'count')},
    }, 'profile.physics')
    _fields(profile['performance'], {
        'input_points': _count, 'reconstruction_wall_max_s': _positive,
    }, 'profile.performance')
    g, p = profile['geometry'], profile['physics']
    if profile['profile_id'] != f"lidar2sim-rules-v{profile['schema_version']}":
        raise ValueError('Profile identifier and schema version disagree')
    if g['completeness_median_max_m'] > g['completeness_p95_max_m']:
        raise ValueError('profile.geometry: median must not exceed p95')
    for group in ('floor', 'obstacles', 'routes'):
        if p[f'{group}_pass_min'] > p[f'{group}_count']:
            raise ValueError(f'profile.physics: {group}_pass_min exceeds count')
    return profile, {key: profile[key] for key in ('profile_id', 'schema_version')} | {
        'sha256': hashlib.sha256(data).hexdigest()}


def validate_physics_config(config, profile):
    """Reject conflicting legacy settings and incomplete scene test sets before Isaac."""
    p = profile['physics']
    if not isinstance(config, dict):
        raise ValueError('physics config must be a mapping')
    body = config.get('body')
    if not isinstance(body, dict) or not _body(body.get('size')) or body['size'] != p['body_size_m']:
        raise ValueError('physics config body.size conflicts with profile')
    pairs = [('body.mass', body.get('mass'), p['body_mass_kg']),
             ('dt', config.get('dt'), p['dt_s']), ('speed', config.get('speed'), p['speed_m_s'])]
    legacy = config.get('pass', {})
    fields = {'floor_drop_m': 'floor_drop_max_m', 'penetration_m': 'penetration_max_m',
              **{f'{group}_min': f'{group}_pass_min' for group in ('floor', 'obstacles', 'routes')}}
    if not isinstance(legacy, dict) or legacy.keys() - fields.keys():
        raise ValueError('physics config pass has unknown fields')
    pairs.extend((f'pass.{key}', value, p[fields[key]]) for key, value in legacy.items())
    for name, value, expected in pairs:
        if not _finite(value) or value != expected:
            raise ValueError(f'physics config {name} conflicts with profile')
    for group in ('floor', 'obstacles', 'routes'):
        tests = config.get(group)
        if not isinstance(tests, list) or len(tests) != p[f'{group}_count']:
            raise ValueError(f'physics config {group}: expected {p[f"{group}_count"]} tests')

    for obstacle in config['obstacles']:
        if not isinstance(obstacle, dict):
            raise ValueError('physics config obstacle must be a mapping')
        for key in ('floor_z', 'free_m'):
            if not _finite(obstacle.get(key)):
                raise ValueError(f'physics config obstacle.{key} must be finite')
        for key in ('start', 'dir'):
            vector = obstacle.get(key)
            if not isinstance(vector, list) or len(vector) != 2 or not all(_finite(x) for x in vector):
                raise ValueError(f'physics config obstacle.{key} must be a finite XY vector')
        if obstacle['free_m'] < 0 or abs(sum(x * x for x in obstacle['dir']) - 1) > 1e-6:
            raise ValueError('physics config obstacle needs nonnegative free_m and unit direction')

    for floor in config['floor']:
        if not isinstance(floor, dict) or not _finite(floor.get('floor_z')):
            raise ValueError('physics config floor.floor_z must be finite')
        xy = floor.get('xy')
        if not isinstance(xy, list) or len(xy) != 2 or not all(_finite(x) for x in xy):
            raise ValueError('physics config floor.xy must be a finite XY vector')
    for route in config['routes']:
        if not isinstance(route, dict) or not _finite(route.get('floor_z', 0.0)):
            raise ValueError('physics config route.floor_z must be finite')
        if not _finite(route.get('length_m')) or route['length_m'] <= 0:
            raise ValueError('physics config route.length_m must be positive')
        points = route.get('points')
        if not isinstance(points, list) or len(points) < 2 or any(
            not isinstance(p, list) or len(p) != 2 or not all(_finite(x) for x in p) for p in points
        ):
            raise ValueError('physics config route.points must contain at least two finite XY vectors')


def upper_bound_passed(value, limit):
    """An inclusive upper bound; invalid runtime numbers can never pass."""
    return _finite(value) and _finite(limit) and value <= limit


def geometry_checks(raw_metrics, profile):
    """Legacy v1 metric names (percent and cm); decide before display rounding."""
    g = profile['geometry']
    precision = g['precision_min_fraction'] * 100
    return {
        'лишняя геометрия, visual': upper_bound_passed(precision, raw_metrics['extra_5cm_pct']),
        'лишняя геометрия, collision': upper_bound_passed(precision, raw_metrics['extra_5cm_pct_collision']),
        'полнота, медиана': upper_bound_passed(raw_metrics['median_cm'], g['completeness_median_max_m'] * 100),
        'полнота, p95': upper_bound_passed(raw_metrics['p95_cm'], g['completeness_p95_max_m'] * 100),
    }


def overall_status(checks):
    values = set(checks.values())
    if values - {'passed', 'failed', 'not_run', 'error'}:
        raise ValueError(f'unknown acceptance status: {values}')
    if values & {'failed', 'error'}:
        return 'failed'
    if not values or 'not_run' in values:
        return 'incomplete'
    return 'passed'
