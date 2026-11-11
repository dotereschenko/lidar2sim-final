import pytest
from tools.jury_compare import compare


def test_missing_reports_and_repeats_never_receive_rank(tmp_path):
    manifest={'profile_sha256':'expected','required_repeats':5,'datasets':{'A':{},'C':{}},'performance':{},'performance_input_sha256':'dense',
              'runs':[{'team':'test','variant':'final','scene':'A','repeat':1,'directory':str(tmp_path/'missing')}]}
    r=compare(manifest)
    assert not r['rows'][0]['valid']
    assert r['groups'][0]['rank'] is None
    assert not r['groups'][0]['complete_repetitions']
    manifest['runs']*=2
    with pytest.raises(ValueError,match='Duplicate'):compare(manifest)


def test_only_complete_fresh_runs_in_same_environment_rank(tmp_path):
    import json
    from tools.jury_compare import sha
    folder=tmp_path/'result';folder.mkdir()
    hashes={}
    for name in ['visual.ply','collision.ply','scene.usd']:
        (folder/name).write_text(name);hashes[name]=sha(folder/name)
    (folder/'isolation.json').write_text(json.dumps({'exit_code':0,'timed_out':False,
        'required_outputs_present':True,'image':'linux','cpu_limit':4,'memory_limit_gib':16,'wall_s':12}))
    profile='profile';config='config'
    meshes={n:{'sha256':hashes[n+'.ply'],'raw_metrics':{'p95_m':.02},
               'dimensions':{'passed_count':10,'items':[{'absolute_error_m':.01}]*10}}
            for n in ['visual','collision']}
    checks={k:{'status':'passed'} for k in ['contract','geometry','dimensions','physics']}
    checks['geometry']['details']={'dimension_measurer_sha256':'method','evaluator_sha256':'evaluator','meshes':meshes,'reference_provenance':{'inputs_sha256':{'A.ply':'a','B.ply':'b','gt_mesh.ply':'gt'}}}
    checks['physics']['details']={'inputs':{'scene_sha256':hashes['scene.usd']},'floor':[{'ok':True}]*10,'obstacles':[{'ok':True}]*10,'routes':[{'ok':True}]*5}
    report={'sources':{'config':{'sha256':config},'cloud':{'sha256':'a'}},'threshold_profile':{'source':{'sha256':profile}},
            'artifacts':{n:{'sha256':h} for n,h in hashes.items()},'checks':checks,'total_s':10,'code':{'source_sha256':'code'}}
    (folder/'report.json').write_text(json.dumps(report))
    perf={'exit_code':0,'timed_out':False,'point_count':10000000,'input_sha256':'dense','wall_s':50,
          'peak_rss_bytes':1000,'config_sha256':config,'execution':{'image_id':'linux','cpus':4,'memory_gib':16}}
    performance=tmp_path/'performance.json';performance.write_text(json.dumps(perf))
    manifest={'dimension_measurer_sha256':'method','evaluator_sha256':'evaluator','required_repeats':1,'profile_sha256':profile,'config_sha256':{'team/final':config},
              'datasets':{'A':{'A_sha256':'a','B_sha256':'b','GT_sha256':'gt'}},
              'performance':{'team/final':str(performance)},'performance_input_sha256':'dense',
              'scoring_environment':{'image_id':'linux','cpus':4,'memory_gib':16},
              'runs':[{'team':'team','variant':'final','scene':'A','repeat':1,'directory':str(folder)}]}
    result=compare(manifest)
    assert result['groups'][0]['rank']==1
    assert result['rows'][0]['wall_s']==12  # Ignore submitted total_s=10.
    manifest['dimension_measurer_sha256']='changed-method'
    assert not compare(manifest)['rows'][0]['valid']
    manifest['dimension_measurer_sha256']='method'
    perf['execution']={'image_id':'other-machine'};performance.write_text(json.dumps(perf))
    assert compare(manifest)['groups'][0]['rank'] is None
    (folder/'collision.ply').write_text('tampered')
    result=compare(manifest)
    assert not result['rows'][0]['valid']
    assert 'Changed output' in result['rows'][0]['error']
