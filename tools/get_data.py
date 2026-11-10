"""Скачать и распаковать данные сцены в data/, проверив контрольную сумму.

    python tools/get_data.py                   # scene_a по ссылке из configs/data.json
    python tools/get_data.py --archive scene_a_v1.tar.gz   # уже скачанный архив
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("scene", nargs="?", default="scene_a")
ap.add_argument("--archive", help="локальный архив вместо скачивания")
ap.add_argument("--out", default=str(ROOT / "data"))
a = ap.parse_args()

spec = json.loads((ROOT / "configs" / "data.json").read_text())[a.scene]
with tempfile.TemporaryDirectory() as tmp:
    arc = Path(a.archive) if a.archive else Path(tmp) / spec["file"]
    if not a.archive:
        if not spec.get("url"):
            raise SystemExit(f"{a.scene}: ссылки пока нет — возьмите архив у организаторов и передайте --archive")
        print(f"скачиваю {spec['url']}")
        with urllib.request.urlopen(spec["url"]) as r, open(arc, "wb") as f:
            shutil.copyfileobj(r, f)
    h = hashlib.sha256(arc.read_bytes()).hexdigest()
    if h != spec["sha256"]:
        raise SystemExit(f"контрольная сумма не совпала: {h}, ожидалась {spec['sha256']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arc) as t:
        names = t.getnames()
        if any(n.startswith("/") or ".." in Path(n).parts for n in names):
            raise SystemExit("в архиве недопустимые пути")
        t.extractall(out)
print(f"{a.scene} → {out / a.scene}: " + ", ".join(sorted(p.name for p in (out / a.scene).iterdir())))
