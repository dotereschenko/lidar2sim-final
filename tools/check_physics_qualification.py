"""Check perfect GT results and targeted failures; CLI exits nonzero if not qualified."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar2sim.acceptance import load_profile


def check(reports, manifest, floor_limit, penetration_limit):
    gt = reports['gt']
    checks = {}
    for group, count in [('floor', 10), ('obstacles', 10), ('routes', 5)]:
        rows = gt.get(group, [])
        checks['gt_' + group] = len(rows) == count and all(row.get('ok') is True for row in rows)
    checks['gt_hash'] = gt['inputs']['scene_sha256'] == manifest['source_gt_scene_sha256']
    checks['case_set'] = {Path(c['scene']).stem for c in manifest['cases']} == {
        'missing_floor', 'missing_obstacle', 'blocked_passage'}
    for case in manifest['cases']:
        name = Path(case['scene']).stem
        report = reports[name]
        target = case['expected_failure']
        row = report[target['group']][target['index']]
        checks[name + '_provenance'] = (
            report['inputs']['scene_sha256'] == case['sha256'] and
            report['acceptance_profile'] == gt['acceptance_profile'] and
            all(report['inputs'][k] == gt['inputs'][k] for k in ['config_sha256', 'runner_sha256']))
        if name == 'missing_floor':
            value = row.get('drop_m')
            detected = isinstance(value, (int, float)) and math.isfinite(value) and value > floor_limit
        elif name == 'missing_obstacle':
            value = row.get('penetration_m')
            detected = (isinstance(value, (int, float)) and math.isfinite(value) and
                        value > penetration_limit and row.get('fell') is False)
        else:
            detected = row.get('status') == 'застрял'
        checks[name + '_detected'] = row.get('ok') is False and detected
    return {'passed': all(checks.values()), 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results')
    parser.add_argument('cases')
    parser.add_argument('--out', required=True)
    parser.add_argument('--profile')
    args = parser.parse_args()
    profile, provenance = load_profile(args.profile)
    manifest = json.loads(Path(args.cases).read_text())
    names = ['gt', *[Path(c['scene']).stem for c in manifest['cases']]]
    reports = {name: json.loads((Path(args.results) / (name+'.json')).read_text()) for name in names}
    result = check(reports, manifest, profile['physics']['floor_drop_max_m'], profile['physics']['penetration_max_m'])
    result['checks']['expected_profile'] = reports['gt']['acceptance_profile'] == provenance
    result['passed'] = all(result['checks'].values())
    result['acceptance_profile'] = provenance
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
