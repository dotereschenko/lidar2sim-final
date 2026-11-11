import json
from tools.jury_reconstruct import prepare_input,command


def test_submission_cannot_read_jury_files_or_b_metadata(tmp_path):
    scans=tmp_path/'scans';scans.mkdir()
    (scans/'manifest.json').write_text(json.dumps({'units':'m','up_axis':'Z','secret_B_positions':[1,2,3]}))
    for n in ['A.ply','B.ply','gt_mesh.ply','trajectory_A.csv','trajectory_B.csv']:(scans/n).write_text(n)
    config=tmp_path/'config.yaml';config.write_text('voxel: 0.03\n')
    target=tmp_path/'input';assert prepare_input(scans,config,target)
    assert set(p.name for p in target.iterdir())=={'scene.ply','manifest.json','config.yaml','trajectory_A.csv'}
    assert json.loads((target/'manifest.json').read_text())=={'units':'m','up_axis':'Z'}
    args=command('sha256:fixture',target,tmp_path/'out',True,'test')
    assert args[args.index('--network')+1]=='none'
    assert args[args.index('--user')+1]=='65534:65534'
    assert '--read-only' in args
    mounts=[args[i+1] for i,x in enumerate(args) if x=='-v']
    assert len(mounts)==2 and all(str(scans) not in m for m in mounts)
