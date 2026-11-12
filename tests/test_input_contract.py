import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lidar2sim.input_contract import assess_declaration, contract, validate_limits


DECLARATION = {'schema_version': 1, 'units': 'm', 'up_axis': 'Z',
               'scene_kind': 'static_single_level_indoor'}


def test_declared_supported_scope_never_certifies_quality():
    result = assess_declaration(DECLARATION)
    assert result['input_status'] == 'supported'
    assert result['quality_status'] == 'not_verified'
    assert result['quality_envelope']['status'] == 'uncalibrated'
    assert result['contract_sha256'] == contract()['sha256']
    assert 'accepted' not in result.values()


@pytest.mark.parametrize('key', ['units', 'up_axis', 'scene_kind'])
def test_missing_metadata_is_not_guessed(key):
    declaration = DECLARATION | {key: None}
    result = assess_declaration(declaration)
    assert result['input_status'] == 'needs_metadata'
    assert result['missing_metadata'] == [key]
    assert declaration[key] is None


@pytest.mark.parametrize('kind', ['dynamic', 'outdoor', 'multi_level_indoor', 'object'])
def test_unsupported_scope_cannot_pass(kind):
    result = assess_declaration(DECLARATION | {'scene_kind': kind})
    assert result['input_status'] == 'unsupported'
    assert result['quality_status'] == 'not_verified'


@pytest.mark.parametrize('update', [{'schema_version': 2}, {'schema_version': True},
    {'units': 'unknown'}, {'up_axis': 'X'}, {'passed': True}])
def test_invalid_schema_fails(update):
    with pytest.raises(ValueError):
        assess_declaration(DECLARATION | update)


def test_resource_boundaries_without_allocating_large_clouds():
    limits = contract()['limits']
    validate_limits(file_bytes=limits['max_file_bytes'], point_count=limits['max_points'])
    validate_limits(file_bytes=1, point_count=limits['min_points'])
    for kwargs in [{'file_bytes': 0}, {'file_bytes': limits['max_file_bytes'] + 1},
                   {'point_count': 3}, {'point_count': limits['max_points'] + 1},
                   {'point_count': True}, {'point_count': 4.5}]:
        with pytest.raises(ValueError):
            validate_limits(**kwargs)


def test_contract_copy_does_not_mutate_shared_policy():
    policy = copy.deepcopy(contract())
    policy['limits']['max_points'] = 1
    assert contract()['limits']['max_points'] == 20_000_000


def test_cli_reports_unknown_units_without_reconstruction(tmp_path):
    path = tmp_path/'declaration.json'
    path.write_text(json.dumps(DECLARATION | {'units': None}))
    result = subprocess.run([sys.executable, '-m', 'lidar2sim.input_contract', str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)['input_status'] == 'needs_metadata'
