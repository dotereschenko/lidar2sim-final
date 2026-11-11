"""Deterministic, allowlisted participant archive; private jury files stay separate."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest_bytes(data):return hashlib.sha256(data).hexdigest()


def archive(out, entries):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    hashes={}
    for name,value in entries.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not name or '\\' in name:
            raise ValueError('Unsafe archive path')
        if isinstance(value,Path):
            if not value.is_file() or value.is_symlink():raise ValueError('Only regular source files allowed')
            with value.open('rb') as f:hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
        else:hashes[name]=digest_bytes(value)
    checksums=''.join(f'{sha}  {name}\n' for name,sha in sorted(hashes.items())).encode()
    entries=entries|{'SHA256SUMS':checksums}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as zipped, tarfile.open(fileobj=zipped,mode='w') as tar:
        for name,value in sorted(entries.items()):
            info=tarfile.TarInfo(name);info.mode=0o644;info.mtime=0;info.uid=info.gid=0
            if isinstance(value,Path):
                info.size=value.stat().st_size
                with value.open('rb') as f:tar.addfile(info,f)
            else:info.size=len(value);tar.addfile(info,io.BytesIO(value))
    with out.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'archive':out.name,'sha256':sha,'bytes':out.stat().st_size,'files':hashes}


def public_entries(scans,dense,annotations):
    scans,dense,annotations=map(Path,(scans,dense,annotations))
    m=json.loads((scans/'manifest.json').read_text());a=json.loads((annotations/'annotations.json').read_text())
    # This inventory is explicit: never recurse into the scan/jury directories.
    files={'scene_a/scene.ply':scans/'A.ply','scene_a/trajectory_A.csv':scans/'trajectory_A.csv',
           'scene_a/test_config.json':ROOT/'configs/physics/scene-a-v2.json',
           'performance/scene_10M.ply':dense/'scene_10M.ply','performance/trajectory.csv':dense/'trajectory.csv',
           'performance/manifest.json':dense/'manifest.json','selfcheck/manifest.json':ROOT/'data/scene_a/manifest.json'}
    public={'scene':'scene_a','release':'scene-a-v2','units':'m','up_axis':'Z','point_count':m['scans']['A']['point_count'],
            'sensor':m['scans']['A']['sensor'],'seed':m['scans']['A']['seed'],
            'sha256':{'scene.ply':m['sha256']['A.ply'],'trajectory_A.csv':m['sha256']['trajectory_A.csv']},
            'acquisition_manifest_sha256':digest_bytes((scans/'manifest.json').read_bytes()),'B_usage':'jury_only'}
    files['scene_a/manifest.json']=(json.dumps(public,indent=2)+'\n').encode()
    for name,data in [('control_dims.yaml',a['dimensions']),('doors.yaml',a['doors']),
                      ('routes.yaml',json.loads((ROOT/'configs/physics/scene-a-v2.json').read_text())['routes'])]:
        files['scene_a/'+name]=yaml.safe_dump(data,sort_keys=False).encode()
    region={k:a['region'][k] for k in ['radius_m','min_neighbors','min_common_samples','predicate','query_contract']}
    files['scene_a/eval_region.yaml']=yaml.safe_dump(region,sort_keys=False).encode()
    for name in ['pyproject.toml','uv.lock','.python-version']:
        files['starter/'+name]=ROOT/name
    for p in (ROOT/'lidar2sim').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc']:
            files['starter/'+str(p.relative_to(ROOT))]=p
    for name in ['fetch_gt.py','make_example_scene.py','benchmark_10m.py']:
        files['starter/tools/'+name]=ROOT/'tools'/name
    for name in ['baseline.yaml','poc-detailed.yaml','poc-high-detail.yaml','acceptance/lidar2sim-rules-v1.yaml',
                 'acceptance/lidar2sim-rules-v2.yaml','acceptance/release-decision-v2.json','physics/scene-a-v2.json']:
        files['starter/configs/'+name]=ROOT/'configs'/name
    files['starter/deployment/Dockerfile.cpu']=ROOT/'deployment/Dockerfile.cpu'
    files['RULES.md']=(ROOT/'docs/rules.md').read_text().replace('../configs/','starter/configs/').encode()
    files['README.md']=ROOT/'docs/participant-release-v2.md'
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scans',required=True);p.add_argument('--dense',required=True);p.add_argument('--annotations',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=archive(a.out,public_entries(a.scans,a.dense,a.annotations));Path(str(a.out)+'.manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}))
