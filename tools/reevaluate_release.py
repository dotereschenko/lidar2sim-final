"""Re-evaluate copied frozen outputs; never count this as new reconstruction/physics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.calibrate_release import docker_cpu


def run(sources,jury,out,image):
    jury,out=Path(jury).resolve(),Path(out).resolve()
    batches=[Path(p).resolve() for p in sources]
    completions=[json.loads((p/'complete.json').read_text()) for p in batches]
    environments=[json.loads((p/'environment.json').read_text()) for p in batches]
    if not environments or any(e!=environments[0] for e in environments):
        raise ValueError('Source runs must share a reconstruction environment')
    cases=[case for batch in batches for case in json.loads((batch/'cases.json').read_text())]
    if len({(c['scene'],c['variant'],c['repeat']) for c in cases})!=len(cases):
        raise ValueError('Duplicate source run')
    out.mkdir(parents=True,mode=0o700)
    image=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
    (out/'environment.json').write_text(json.dumps(environments[0],indent=2)+'\n')
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    for batch in batches:
        for case in json.loads((batch/'cases.json').read_text()):
            source=(batch/case['folder']).resolve()
            target=(out/case['folder']).resolve()
            if not source.is_relative_to(batch) or not target.is_relative_to(out):
                raise ValueError('Case outside its batch')
            target.mkdir(parents=True)
            for name in ['visual.ply','collision.ply','scene.usd','report.json','physics.json','isolation.json']:
                shutil.copyfile(source/name,target/name)
            (target/'reevaluation.json').write_text(json.dumps({'source':str(source),'new_reconstruction_or_physics_runs':0,
                'fresh_geometry_and_dimensions':True,'evaluator_image':image,'physics_reused_only_after_provenance_validation':True},indent=2)+'\n')
            mounts=[f'{jury}:/jury:ro',f'{target}:/output:rw']
            args=['/app/tools/eval_scene.py',f"/jury/{case['scene']}/source",'/output','--mode','b',
                  '--scans',f"/jury/{case['scene']}/scans",'--annotations',f"/jury/{case['scene']}/annotations",
                  '--profile','/app/configs/acceptance/lidar2sim-rules-v2.yaml','--dimensions']
            if docker_cpu(image,mounts,args,target/'geometry.log'):
                raise RuntimeError('Re-evaluation failed: '+case['folder'])
            args=['-m','lidar2sim.reporting','/output','--geometry','/output/eval_scene.json','--physics','/output/physics.json',
                  '--physics-config',f"/jury/{case['scene']}/source/test_config.json",'--profile','/app/configs/acceptance/lidar2sim-rules-v2.yaml']
            if docker_cpu(image,mounts,args,target/'merge.log') not in (0,1):
                raise RuntimeError('Provenance validation/merge failed: '+case['folder'])
            print('Re-evaluated',case['folder'],flush=True)
    result={'kind':'reevaluation_of_existing_runs','cases':len(cases),'image_id':environments[0]['image_id'],
            'evaluation_image_id':image,'new_reconstruction_or_physics_runs':0,
            'source_batches':[str(p) for p in batches],'source_completion_records':completions}
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',action='append',required=True)
    for name in ['jury','out','image']:p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.source,a.jury,a.out,a.image)
