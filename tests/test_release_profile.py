from pathlib import Path
import copy
import json
import pytest
from lidar2sim.acceptance import load_profile, release_decision, validate_physics_config

ROOT = Path(__file__).resolve().parents[1]


def test_release_decision_is_bound_to_exact_profile():
    profile, provenance = load_profile(ROOT/'configs/acceptance/lidar2sim-rules-v2.yaml')
    assert profile['geometry']['precision_reference'] == 'b_cloud'
    assert release_decision(provenance)['A_points'] == 750000
    assert release_decision(provenance | {'sha256':'0'*64}) is None
    _, legacy = load_profile()
    assert release_decision(legacy) is None


def test_release_physics_rejects_legacy_limits():
    profile, _ = load_profile(ROOT/'configs/acceptance/lidar2sim-rules-v2.yaml')
    config = json.loads((ROOT/'configs/physics/scene-a-v2.json').read_text())
    validate_physics_config(config, profile)
    altered = copy.deepcopy(config); altered['pass']['penetration_m'] = .035
    with pytest.raises(ValueError, match='conflicts'):
        validate_physics_config(altered, profile)
