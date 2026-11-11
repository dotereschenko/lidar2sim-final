"""Produce jury comparison and ranking from freshly evaluated, pinned reports."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def number(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


def row(run, manifest):
    result={k:run[k] for k in ['team','variant','scene','repeat']}
    try:
        folder=Path(run['directory']);report=json.loads((folder/'report.json').read_text())
        if report['sources']['config']['sha256']!=manifest['config_sha256'][run['team']+'/'+run['variant']]:raise ValueError('Configuration mismatch')
        if report['threshold_profile']['source']['sha256']!=manifest['profile_sha256']:raise ValueError('Profile mismatch')
        if report['sources']['cloud']['sha256']!=manifest['datasets'][run['scene']]['A_sha256']:raise ValueError('A hash mismatch')
        for file in ['visual.ply','collision.ply','scene.usd']:
            if report['artifacts'][file]['sha256']!=sha(folder/file):raise ValueError('Changed output '+file)
        checks=report['checks'];geometry=checks['geometry']['details'];physics=checks['physics']['details']
        refs=geometry['reference_provenance']['inputs_sha256'];dataset=manifest['datasets'][run['scene']]
        if any(refs[n]!=dataset[key] for n,key in [('A.ply','A_sha256'),('B.ply','B_sha256'),('gt_mesh.ply','GT_sha256')]):raise ValueError('Evaluation dataset mismatch')
        if physics['inputs']['scene_sha256']!=sha(folder/'scene.usd'):raise ValueError('Stale physics')
        if any(geometry['meshes'][n]['sha256']!=sha(folder/f'{n}.ply') for n in ['visual','collision']):raise ValueError('Stale geometry')
        p95=max(m['raw_metrics']['p95_m'] for m in geometry['meshes'].values())
        errors=[d['absolute_error_m'] for m in geometry['meshes'].values() for d in m['dimensions']['items']]
        maximum=max(errors) if all(number(x) for x in errors) else None
        isolation=json.loads((folder/'isolation.json').read_text())
        environment=manifest['scoring_environment']
        if (isolation['exit_code']!=0 or isolation['timed_out'] or not isolation['required_outputs_present']
                or isolation['image']!=environment['image_id']
                or isolation['cpu_limit']!=environment['cpus']
                or isolation['memory_limit_gib']!=environment['memory_gib']):
            raise ValueError('Reconstruction execution mismatch or failure')
        # Ranking uses the operator's process clock, never a submitted report time.
        wall=isolation['wall_s']
        if wall>1800:raise ValueError('Reconstruction exceeded time limit')
        if not number(p95) or not number(wall):raise ValueError('Invalid measured values')
        passed=all(checks[k]['status']=='passed' for k in ['contract','geometry','dimensions','physics'])
        if passed and not number(maximum):raise ValueError('Passed dimensions lack measured errors')
        result.update(valid=True,scene_passed=passed,p95_m=p95,max_dimension_error_m=maximum,wall_s=wall,
                      dimensions={n:m['dimensions']['passed_count'] for n,m in geometry['meshes'].items()},
                      physics={n:sum(x['ok'] for x in physics[n]) for n in ['floor','obstacles','routes']},
                      code_sha256=report['code']['source_sha256'],report_sha256=sha(folder/'report.json'))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result.update(valid=False,scene_passed=False,error=str(exc))
    return result


def compare(manifest):
    repeats=manifest['required_repeats']
    if type(repeats) is not int or repeats<1:raise ValueError('Positive repetition count required')
    keys=[tuple(r[k] for k in ['team','variant','scene','repeat']) for r in manifest['runs']]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate run')
    rows=[row(r,manifest) for r in manifest['runs']];groups=[]
    for team,variant in sorted({(r['team'],r['variant']) for r in rows}):
        subset=[r for r in rows if (r['team'],r['variant'])==(team,variant)]
        expected={(scene,i) for scene in manifest['datasets'] for i in range(1,repeats+1)}
        complete={(r['scene'],r['repeat']) for r in subset}==expected
        performance_ok=False
        try:
            perf=json.loads(Path(manifest['performance'][team+'/'+variant]).read_text())
            performance_ok=(perf['exit_code']==0 and not perf['timed_out'] and perf['point_count']==10000000
                            and perf['input_sha256']==manifest['performance_input_sha256'] and number(perf['wall_s'])
                            and perf['wall_s']<=1800 and number(perf['peak_rss_bytes']) and perf['peak_rss_bytes']>0
                            and perf['config_sha256']==manifest['config_sha256'][team+'/'+variant]
                            and perf['execution']==manifest['scoring_environment'])
        except (KeyError,OSError,ValueError,TypeError):pass
        eligible=complete and performance_ok and all(r['scene_passed'] for r in subset)
        group={'team':team,'variant':variant,'complete_repetitions':complete,'performance_passed':performance_ok,'eligible':eligible,'rank':None}
        if eligible:group['score']=[max(r['p95_m'] for r in subset),max(r['max_dimension_error_m'] for r in subset),max(r['wall_s'] for r in subset)]
        groups.append(group)
    for rank,group in enumerate(sorted((g for g in groups if g['eligible'] and g['variant']=='final'),key=lambda g:(g['score'],g['team'],g['variant'])),1):group['rank']=rank
    return {'schema_version':1,'profile_sha256':manifest['profile_sha256'],'required_repeats':repeats,
            'ranking':'Eligible only if every scene/repeat and 10M pass; ascending worst p95, worst dimension error, worst reconstruction time; lexical IDs break exact ties',
            'rows':rows,'groups':groups}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--out',required=True);a=p.parse_args()
    result=compare(json.loads(Path(a.manifest).read_text()));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Jury comparison','','| Team | Variant | Scene | Repeat | Valid | Scene passed | p95, m | Time, s |','|---|---|---|---:|---|---|---:|---:|']
    for r in result['rows']:
        values=[r.get(k,'—') for k in ['team','variant','scene','repeat','valid','scene_passed','p95_m','wall_s']]
        lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in values)+' |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['groups']))
