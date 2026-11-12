"""Shared technical limits and declared scope; never certifies scene quality."""
import argparse
import hashlib
from importlib import resources
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def contract():
    raw = resources.files('lidar2sim.schemas').joinpath('input-contract-v1.json').read_bytes()
    return {**json.loads(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


LIMITS = contract()['limits']


def validate_limits(*, file_bytes=None, point_count=None):
    for value, low, high, label in (
        (file_bytes, 1, LIMITS['max_file_bytes'], 'PLY bytes'),
        (point_count, LIMITS['min_points'], LIMITS['max_points'], 'PLY points'),
    ):
        if value is not None and (type(value) is not int or not low <= value <= high):
            raise ValueError(f'{label}: требуется целое число от {low} до {high}')


def assess_declaration(declaration):
    schema = json.loads(resources.files('lidar2sim.schemas').joinpath(
        'input-declaration-v1.schema.json').read_text())
    errors = list(Draft202012Validator(schema).iter_errors(declaration))
    if errors:
        raise ValueError('input declaration: ' + errors[0].message)
    policy = contract()
    missing = [k for k in ('units', 'up_axis', 'scene_kind') if declaration[k] is None]
    unsupported = (declaration['scene_kind'] is not None
                   and declaration['scene_kind'] not in policy['supported_scope'])
    state = 'unsupported' if unsupported else 'needs_metadata' if missing else 'supported'
    return {'contract_id': policy['contract_id'], 'contract_sha256': policy['sha256'],
            'input_status': state, 'missing_metadata': missing,
            'declaration': declaration, 'evidence': 'operator_declared_only',
            'quality_status': 'not_verified', 'quality_envelope': policy['quality_envelope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('declaration', help='JSON декларации; null означает неизвестное значение')
    args = parser.parse_args()
    try:
        result = assess_declaration(json.loads(Path(args.declaration).read_text()))
    except (OSError, ValueError) as exc:
        print(json.dumps({'input_status': 'invalid', 'error': str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['input_status'] == 'supported' else 1


if __name__ == '__main__':
    raise SystemExit(main())
