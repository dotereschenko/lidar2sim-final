from pathlib import Path
import tarfile
import pytest
from tools.package_release import archive


def test_archive_repeats_bytes_and_rejects_paths_and_symlinks(tmp_path):
    source=tmp_path/'input';source.write_bytes(b'point cloud fixture')
    entries={'scene_a/scene.ply':source,'README.md':b'instructions'}
    a=archive(tmp_path/'one.tar.gz',entries);b=archive(tmp_path/'two.tar.gz',entries)
    assert a['sha256']==b['sha256']
    with tarfile.open(tmp_path/'one.tar.gz') as t:
        assert set(t.getnames())=={'scene_a/scene.ply','README.md','SHA256SUMS'}
        assert b'scene_a/scene.ply' in t.extractfile('SHA256SUMS').read()
    with pytest.raises(ValueError):archive(tmp_path/'bad.tar.gz',{'../secret':b'x'})
    link=tmp_path/'link';link.symlink_to(source)
    with pytest.raises(ValueError):archive(tmp_path/'bad.tar.gz',{'safe':link})
    assert not (tmp_path/'bad.tar.gz').exists()


def test_public_inventory_never_walks_jury_data(tmp_path):
    import json
    from tools.package_release import public_entries
    scans=tmp_path/'scans';scans.mkdir();ann=tmp_path/'ann';ann.mkdir()
    (scans/'manifest.json').write_text(json.dumps({'scans':{'A':{'point_count':750000,'sensor':{},'seed':1}},'sha256':{'A.ply':'a','trajectory_A.csv':'t'}}))
    (ann/'annotations.json').write_text(json.dumps({'dimensions':[],'doors':[],'region':{'radius_m':.1,'min_neighbors':10,'min_common_samples':1000,'predicate':'both','query_contract':'pinned'}}))
    for name in ['B.ply','gt_mesh.ply','hidden-secret.json']:(scans/name).write_text('private')
    entries=public_entries(scans,tmp_path/'dense',ann)
    assert 'scene_a/scene.ply' in entries
    assert not any(Path(n).name in ['B.ply','gt_mesh.ply','gt_scene.usd','hidden-secret.json'] for n in entries)
    assert not any('annotations.json' in n or 'eval_region.npz' in n for n in entries)
