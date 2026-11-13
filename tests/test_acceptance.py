"""Contract regressions: malformed profiles, inclusive raw decisions, and preflight."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/acceptance/lidar2sim-rules-v1.yaml'


class AcceptanceTest(unittest.TestCase):
    def setUp(self):
        from lidar2sim import acceptance
        self.a = acceptance
        self.profile, self.provenance = acceptance.load_profile()

    def load_changed(self, profile):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'profile.yaml'
            path.write_text(yaml.safe_dump(profile))
            return self.a.load_profile(path)

    def test_hash_tracks_read_bytes_and_default_is_independent_of_cwd(self):
        self.assertEqual(self.provenance, {
            'profile_id': 'lidar2sim-rules-v1', 'schema_version': 1,
            'sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest()})
        old = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                self.assertEqual(self.a.load_profile(), (self.profile, self.provenance))
        finally:
            os.chdir(old)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p.yaml'
            path.write_bytes(PROFILE.read_bytes() + b'\n# formatting\n')
            changed, provenance = self.a.load_profile(path)
            self.assertEqual(changed, self.profile)
            self.assertNotEqual(provenance['sha256'], self.provenance['sha256'])

    def test_rejects_unknown_and_missing_fields_at_every_mapping_level(self):
        for section in [None, 'geometry', 'dimensions', 'physics', 'performance']:
            base = self.profile if section is None else self.profile[section]
            for field in [*base, 'typo']:
                with self.subTest(section=section, field=field):
                    p = copy.deepcopy(self.profile)
                    target = p if section is None else p[section]
                    if field == 'typo':
                        target[field] = 1
                    else:
                        del target[field]
                    with self.assertRaises(ValueError):
                        self.load_changed(p)

    def test_rejects_invalid_types_ranges_and_relations(self):
        changes = [
            (None, 'schema_version', 2), (None, 'schema_version', True),
            (None, 'profile_id', 'other'), (None, 'units', 'cm'), (None, 'up_axis', 'Y'),
            ('geometry', 'meshes', ['visual']), ('geometry', 'meshes', ['collision', 'visual']),
            ('geometry', 'precision_reference', 'B'), ('geometry', 'precision_comparison', 'lt'),
            ('geometry', 'precision_min_fraction', -0.01), ('geometry', 'precision_min_fraction', 1.01),
            ('geometry', 'samples', 1.5), ('geometry', 'samples', 0), ('geometry', 'seed', -1),
            ('geometry', 'completeness_median_max_m', 0.11),
            ('physics', 'body_size_m', [0.4, 0.4]), ('physics', 'body_size_m', [0.4, True, 0.3]),
            ('physics', 'body_size_m', [0.4, float('nan'), 0.3]),
            ('physics', 'body_size_m', [0.4, float('inf'), 0.3]),
            ('physics', 'body_size_m', [0.4, 0, 0.3]), ('physics', 'floor_pass_min', 11),
            ('physics', 'obstacles_pass_min', 11), ('physics', 'routes_pass_min', 6),
        ]
        for section in ['geometry', 'dimensions', 'physics', 'performance']:
            for key, value in self.profile[section].items():
                if isinstance(value, (float, int)):
                    changes.extend((section, key, bad) for bad in [True, float('nan'), float('inf'), -1])
        for section, key, value in changes:
            with self.subTest(section=section, key=key, value=value):
                p = copy.deepcopy(self.profile)
                (p if section is None else p[section])[key] = value
                with self.assertRaises(ValueError):
                    self.load_changed(p)

    def test_bad_yaml_or_nonmapping_profile_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'profile.yaml'
            for content in ['geometry: [', '[]', 'null', '']:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'profile'):
                    self.a.load_profile(path)
            with self.assertRaisesRegex(ValueError, 'profile'):
                self.a.load_profile(Path(tmp) / 'missing.yaml')

    def test_raw_geometry_boundaries_precede_display_rounding(self):
        raw = dict(extra_5cm_pct=92.0, extra_5cm_pct_collision=92.0, median_cm=3.0, p95_cm=10.0)
        self.assertTrue(all(self.a.geometry_checks(raw, self.profile).values()))
        for key, bad in [('extra_5cm_pct', 91.96), ('extra_5cm_pct', math.nextafter(92.0, 0.0)), ('extra_5cm_pct_collision', 91.96),
                         ('median_cm', 3.00001), ('p95_cm', 10.00001)]:
            with self.subTest(key=key):
                self.assertFalse(all(self.a.geometry_checks({**raw, key: bad}, self.profile).values()))
        for key in raw:
            for bad in [float('nan'), float('inf'), -float('inf')]:
                self.assertFalse(all(self.a.geometry_checks({**raw, key: bad}, self.profile).values()))

    def test_upper_bounds_are_inclusive_and_reject_nonfinite(self):
        for limit in [0.05, 0.035, 0.03, 0.10]:
            self.assertTrue(self.a.upper_bound_passed(limit, limit))
            self.assertTrue(self.a.upper_bound_passed(limit - 0.00001, limit))
            self.assertFalse(self.a.upper_bound_passed(limit + 0.00001, limit))
        for bad in [float('nan'), float('inf'), -float('inf'), True]:
            self.assertFalse(self.a.upper_bound_passed(bad, 0.05))
            self.assertFalse(self.a.upper_bound_passed(0.05, bad))

    def test_overall_status_never_hides_missing_or_failed_groups(self):
        for checks, want in [({}, 'incomplete'), ({'geometry': 'passed'}, 'passed'),
                             ({'geometry': 'passed', 'physics': 'not_run'}, 'incomplete'),
                             ({'physics': 'failed', 'B': 'not_run'}, 'failed'),
                             ({'physics': 'error'}, 'failed')]:
            self.assertEqual(self.a.overall_status(checks), want)
        with self.assertRaises(ValueError):
            self.a.overall_status({'geometry': 'success'})

    def test_physics_config_rejects_conflicts_and_incomplete_test_sets(self):
        cfg = json.loads((ROOT / 'data/scene_a/test_config.json').read_text())
        self.a.validate_physics_config(cfg, self.profile)
        for section, key, value in [('body', 'size', [1, 1, 1]), ('body', 'mass', True),
                                    ('pass', 'floor_drop_m', 0.051), ('pass', 'penetration_m', 0.034),
                                    ('pass', 'floor_min', 8), ('pass', 'obstacles_min', 9),
                                    ('pass', 'routes_min', 3), (None, 'dt', float('nan')),
                                    (None, 'speed', 0.3), (None, 'floor', cfg['floor'][:-1]),
                                    (None, 'obstacles', []), (None, 'routes', [])]:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(cfg)
                (changed if section is None else changed[section])[key] = value
                with self.assertRaises(ValueError):
                    self.a.validate_physics_config(changed, self.profile)
        for key in ['body', 'dt', 'speed', 'floor', 'obstacles', 'routes']:
            changed = copy.deepcopy(cfg)
            del changed[key]
            with self.assertRaises(ValueError):
                self.a.validate_physics_config(changed, self.profile)
        # Transitional duplicated thresholds may be absent; profile supplies them.
        del cfg['pass']
        self.a.validate_physics_config(cfg, self.profile)

    def test_nonfinite_obstacle_inputs_cannot_be_hidden_by_min_max(self):
        cfg = json.loads((ROOT / 'data/scene_a/test_config.json').read_text())
        for field, value in [('free_m', float('nan')), ('free_m', float('inf')),
                             ('floor_z', float('nan')), ('dir', [float('nan'), 1]),
                             ('start', [0, float('inf')])]:
            changed = copy.deepcopy(cfg)
            changed['obstacles'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.a.validate_physics_config(changed, self.profile)

    def test_physics_floor_routes_and_direction_are_validated_before_isaac(self):
        cfg = json.loads((ROOT / 'data/scene_a/test_config.json').read_text())
        cases = [('floor', 'floor_z', float('nan')), ('floor', 'xy', [0]),
                 ('routes', 'floor_z', float('inf')), ('routes', 'points', []),
                 ('routes', 'points', [[0, 0], [1, float('nan')]]),
                 ('routes', 'length_m', 0), ('obstacles', 'dir', [0, 2]),
                 ('obstacles', 'dir', [0, 0]), ('obstacles', 'free_m', -1)]
        for group, field, value in cases:
            changed = copy.deepcopy(cfg)
            changed[group][0][field] = value
            with self.subTest(group=group, field=field), self.assertRaises(ValueError):
                self.a.validate_physics_config(changed, self.profile)
        cfg['routes'][0]['floor_z'] = 2.5
        self.a.validate_physics_config(cfg, self.profile)
