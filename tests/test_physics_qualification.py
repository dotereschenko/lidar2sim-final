import copy
import unittest
from tools.check_physics_qualification import check


class QualificationTests(unittest.TestCase):
    def reports(self):
        base = {'inputs': {'scene_sha256': 'gt-usd', 'config_sha256': 'config', 'runner_sha256': 'runner'},
                'acceptance_profile': {'sha256': 'profile'},
                'floor': [{'ok': True, 'drop_m': 0} for _ in range(10)],
                'obstacles': [{'ok': True, 'penetration_m': 0, 'fell': False} for _ in range(10)],
                'routes': [{'ok': True, 'status': 'ok'} for _ in range(5)]}
        reports = {'gt': base}
        manifest = {'source_gt_scene_sha256': 'gt-usd', 'cases': []}
        for name, group in [('missing_floor', 'floor'), ('missing_obstacle', 'obstacles'), ('blocked_passage', 'routes')]:
            report = copy.deepcopy(base)
            report['inputs']['scene_sha256'] = name
            report[group][0]['ok'] = False
            reports[name] = report
            manifest['cases'].append({'scene': name+'.usd', 'sha256': name, 'expected_failure': {'group': group, 'index': 0}})
        reports['missing_floor']['floor'][0]['drop_m'] = 1
        reports['missing_obstacle']['obstacles'][0]['penetration_m'] = .5
        reports['blocked_passage']['routes'][0]['status'] = 'застрял'
        return reports, manifest

    def test_requires_perfect_gt_and_expected_physical_failures(self):
        reports, manifest = self.reports()
        self.assertTrue(check(reports, manifest, .05, .035)['passed'])
        reports['gt']['routes'][0]['ok'] = False
        self.assertFalse(check(reports, manifest, .05, .035)['passed'])
        reports, manifest = self.reports()
        reports['missing_obstacle']['obstacles'][0]['penetration_m'] = float('nan')
        self.assertFalse(check(reports, manifest, .05, .035)['passed'])

    def test_rejects_unrelated_failure_and_mismatched_provenance(self):
        for mutate in [lambda r: r['blocked_passage']['routes'][0].update(status='провалился'),
                       lambda r: r['missing_obstacle']['inputs'].update(runner_sha256='other'),
                       lambda r: r['missing_floor']['inputs'].update(scene_sha256='wrong')]:
            reports, manifest = self.reports()
            mutate(reports)
            self.assertFalse(check(reports, manifest, .05, .035)['passed'])
