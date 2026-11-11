"""Measure both reference configurations on a pinned, isolated CPU image."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calibrate_release import docker_cpu


def run(data, out, image):
    data, out = Path(data).resolve(), Path(out).resolve()
    out.mkdir(mode=0o700)
    image = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
    execution = {'image_id': image, 'cpus': 4, 'memory_gib': 16, 'network': 'none'}
    for variant, config in [('baseline', 'baseline.yaml'), ('final', 'poc-high-detail.yaml')]:
        folder = out / variant
        folder.mkdir()
        code = docker_cpu(image, [f'{data}:/input:ro', f'{folder}:/output:rw'],
                          ['/app/tools/benchmark_10m.py', '--data', '/input', '--config',
                           '/app/configs/' + config, '--out', '/output'], folder / 'driver.log', 1900)
        path = folder / 'benchmark.json'
        if not path.exists():
            raise RuntimeError(f'{variant}: no measurement, exit {code}; inspect driver.log')
        result = json.loads(path.read_text())
        result['execution'] = execution
        result['host'] = {'uname': platform.uname()._asdict(),
                          'os_release': Path('/etc/os-release').read_text(),
                          'sharing': 'Shared host; cgroup limits are recorded, no dedicated-host claim'}
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(variant, {k: result[k] for k in ['exit_code', 'wall_s', 'peak_rss_bytes', 'under_1800_s']}, flush=True)
        if code:
            raise RuntimeError(f'{variant}: process exit {code}; measurement retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'out', 'image']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.data, args.out, args.image)
