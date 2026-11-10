"""Shared acceptance contract; thresholds live only in the YAML profile."""
import hashlib
import math
from pathlib import Path

import yaml

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / 'configs/acceptance/lidar2sim-rules-v1.yaml'


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
        'schema_version': lambda x: type(x) is int and x == 1,
        'profile_id': lambda x: x == 'lidar2sim-rules-v1',
        'units': lambda x: x == 'm', 'up_axis': lambda x: x == 'Z',
        **{name: lambda x: isinstance(x, dict)
           for name in ('geometry', 'dimensions', 'physics', 'performance')},
    }, 'profile')
    _fields(profile['geometry'], {
        'meshes': lambda x: x == ['visual', 'collision'],
        'precision_reference': lambda x: x == 'gt_mesh',
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
