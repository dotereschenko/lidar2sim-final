"""Local-only HTTP workbench; run python -m lidar2sim.web.server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlsplit
import uuid
from lidar2sim.input_contract import LIMITS, contract

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent/'static'
ARTIFACTS = {'input.ply': 'preview.ply', 'visual.ply': 'result/visual.ply',
             'collision.ply': 'result/collision.ply', 'scene.usd': 'result/scene.usd',
             'report.json': 'result/report.json', 'run.log': 'run.log'}
STAGES = {'loading', 'reconstructing', 'normals', 'nksr', 'collision', 'exporting'}


def failure_message(lines):
    for line in reversed(lines):
        if line.startswith('WEB_ERROR '):
            return json.loads(line[len('WEB_ERROR '):])[:600]
        if line.startswith('{'):
            try:
                record = json.loads(line)
                if record.get('status') == 'failed' and isinstance(record.get('error'), str):
                    return record['error'][:600]
            except (ValueError, AttributeError):
                pass
    message = lines[-1] if lines else 'Ошибка реконструкции'
    return re.sub(r'\x1b\[[0-9;]*m', '', message)[:600]


class Jobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self.timeout_s = int(os.environ.get('LIDAR2SIM_JOB_TIMEOUT_S', '1800'))
        self.min_free_bytes = int(os.environ.get('LIDAR2SIM_MIN_FREE_BYTES', '0'))
        if self.timeout_s <= 0 or self.min_free_bytes < 0:
            raise ValueError('Некорректные лимиты сервера')
        self.pool = ThreadPoolExecutor(max_workers=1)
        from lidar2sim.neural import availability
        self.backends = {'nksr': availability()}
        for path in self.root.glob('*/job.json'):
            job = json.loads(path.read_text())
            if job['status'] in ('queued', 'running'):
                self.update(path.parent.name, status='failed', error='Сервер перезапущен. Создайте новое задание.')

    def folder(self, ident):
        if not re.fullmatch('[0-9a-f]{32}', ident):
            raise ValueError('Некорректный ID задания')
        path = self.root/ident
        if not (path/'job.json').is_file():
            raise FileNotFoundError('Задание не найдено')
        return path

    def update(self, ident, **fields):
        with self.lock:
            path = self.root/ident/'job.json'
            data = json.loads(path.read_text()) if path.exists() else {'id': ident}
            data.update(fields)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, ensure_ascii=False))
            temporary.replace(path)
            return data

    def read(self, ident):
        folder = self.folder(ident)
        data = json.loads((folder/'job.json').read_text())
        data['files'] = [name for name, relative in ARTIFACTS.items() if (folder/relative).is_file() and (data['status'] == 'completed' or name in ('input.ply', 'run.log') or (data['status'] == 'failed' and name == 'report.json'))]
        log = folder/'run.log'
        if data['status'] == 'running' and log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                lines = stream.read().decode(errors='replace').splitlines()
            stages = [line.removeprefix('LIDAR2SIM_STAGE ') for line in lines
                      if line.startswith('LIDAR2SIM_STAGE ')]
            if stages and stages[-1] in STAGES:
                data['stage'] = stages[-1]
        if data['status'] == 'completed':
            report = json.loads((folder/'result/report.json').read_text())
            data['result'] = {key: report.get(key) for key in ('solution', 'total_s', 'outputs', 'input', 'acceptance', 'timings_s', 'normal_orientation')}
        return data

    def start(self, ident):
        with self.lock:
            self.ensure_free_space()
            job = self.read(ident)
            if job['status'] != 'uploaded':
                raise ValueError('Это задание уже запущено')
            active = sum(json.loads(p.read_text())['status'] in ('queued', 'running') for p in self.root.glob('*/job.json'))
            if active >= 8:
                raise ValueError('Очередь заполнена. Дождитесь завершения заданий.')
            self.update(ident, status='queued')
            self.pool.submit(self.run, ident)

    def ensure_free_space(self, incoming_bytes=0):
        if shutil.disk_usage(self.root).free < self.min_free_bytes + incoming_bytes:
            raise ValueError('Недостаточно свободного места: сервер сохраняет резерв диска')

    def run(self, ident):
        folder = self.folder(ident)
        with self.lock:
            if self.read(ident)['status'] == 'cancelled':
                return
            self.update(ident, status='running')
        try:
            self.ensure_free_space()
            with (folder/'run.log').open('w') as log:
                # Worker and reconstruction share a process group for timeout cleanup.
                process = subprocess.Popen([sys.executable, '-m', 'lidar2sim.web.worker', str(folder)],
                    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
                with self.lock:
                    self.processes[ident] = process
                    if self.read(ident)['status'] == 'cancelled':
                        self.stop_process(ident)
                try:
                    code = process.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RuntimeError(f'Превышен лимит обработки: {self.timeout_s} секунд')
            if self.read(ident)['status'] == 'cancelled':
                return
            if code:
                lines = (folder/'run.log').read_text(errors='replace').strip().splitlines()
                raise RuntimeError(failure_message(lines))
            from lidar2sim.validation import validate_outputs
            validate_outputs(folder/'result')
            with self.lock:
                if self.read(ident)['status'] != 'cancelled':
                    self.update(ident, status='completed', error=None)
        except Exception as exc:
            with self.lock:
                if self.read(ident)['status'] != 'cancelled':
                    self.update(ident, status='failed', error=str(exc))
        finally:
            with self.lock:
                self.processes.pop(ident, None)

    def stop_process(self, ident):
        process = self.processes.get(ident)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cancel(self, ident):
        with self.lock:
            if self.read(ident)['status'] not in ('uploaded', 'queued', 'running'):
                raise ValueError('Это задание уже завершено')
            self.update(ident, status='cancelled', error=None)
            self.stop_process(ident)


def handler(jobs):
    class Handler(BaseHTTPRequestHandler):
        def json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def safe(self):
            host = self.headers.get('Host', '')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if host not in allowed or self.headers.get('Origin', 'http://'+host) != 'http://'+host:
                raise PermissionError('Разрешён только локальный доступ с того же origin')

        def receive(self, path, maximum):
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > maximum:
                raise ValueError(f'Размер файла должен быть от 1 до {maximum//1048576} МБ')
            jobs.ensure_free_space(length)
            self.connection.settimeout(60)
            try:
                with path.open('wb') as stream:
                    while length:
                        chunk = self.rfile.read(min(length, 1024*1024))
                        if not chunk:
                            raise ValueError('Загрузка прервана')
                        stream.write(chunk); length -= len(chunk)
            except Exception:
                path.unlink(missing_ok=True)
                raise

        def do_POST(self):
            try:
                self.safe()
                url = urlsplit(self.path); parts = url.path.strip('/').split('/')
                if parts == ['api', 'jobs']:
                    query = parse_qs(url.query)
                    method = query.get('method', ['improved'])[0]
                    scale = float(query.get('scale', ['1'])[0]); up = query.get('up', ['Z'])[0]
                    if method not in ('baseline', 'improved', 'high-detail', 'nksr') or scale not in (1, .01, .001) or up not in ('Y', 'Z'):
                        raise ValueError('Некорректные параметры обработки')
                    if method == 'nksr' and not jobs.backends['nksr']['available']:
                        raise ValueError(jobs.backends['nksr'].get('reason', 'NKSR недоступен'))
                    ident = uuid.uuid4().hex; folder = jobs.root/ident; folder.mkdir()
                    try:
                        self.receive(folder/'source.ply', LIMITS['max_file_bytes'])
                        with (folder/'source.ply').open('rb') as stream:
                            if stream.read(4) not in (b'ply\n', b'ply\r'):
                                raise ValueError('Нужен PLY-файл с заголовком ply')
                        settings = {'method': method, 'scale': scale, 'up': up}
                        (folder/'settings.json').write_text(json.dumps(settings))
                        config = {'baseline': 'baseline.yaml', 'improved': 'poc-detailed.yaml',
                                  'high-detail': 'poc-high-detail.yaml', 'nksr': 'nksr.yaml'}[method]
                        shutil.copyfile(ROOT/'configs'/config, folder/'config.yaml')
                        jobs.update(ident, status='uploaded', settings=settings,
                                    created_at=datetime.now(timezone.utc).isoformat())
                    except Exception:
                        shutil.rmtree(folder); raise
                    return self.json(jobs.read(ident), 201)
                if len(parts) == 4 and parts[:2] == ['api', 'jobs']:
                    ident, action = parts[2:]; folder = jobs.folder(ident)
                    if action == 'start':
                        jobs.start(ident)
                    elif action == 'cancel':
                        jobs.cancel(ident)
                    elif action == 'trajectory':
                        with jobs.lock:
                            if jobs.read(ident)['status'] != 'uploaded':
                                raise ValueError('Нельзя менять запущенное задание')
                            self.receive(folder/'trajectory.csv', 10*1048576)
                    else:
                        raise FileNotFoundError()
                    return self.json(jobs.read(ident))
                raise FileNotFoundError()
            except PermissionError as exc: self.json({'error': str(exc)}, 403)
            except FileNotFoundError: self.json({'error': 'Не найдено'}, 404)
            except (ValueError, OSError) as exc: self.json({'error': str(exc)}, 400)

        def do_GET(self):
            try:
                self.safe()
                path = urlsplit(self.path).path; parts = path.strip('/').split('/')
                if parts == ['api', 'backends']:
                    return self.json(jobs.backends)
                if parts == ['api', 'input-contract']:
                    return self.json(contract())
                if parts == ['api', 'jobs']:
                    data = [jobs.read(p.parent.name) for p in sorted(jobs.root.glob('*/job.json'), key=lambda p:p.stat().st_mtime, reverse=True)[:30]]
                    return self.json(data)
                if len(parts) == 3 and parts[:2] == ['api', 'jobs']:
                    return self.json(jobs.read(parts[2]))
                if len(parts) == 5 and parts[:2] == ['api', 'jobs'] and parts[3] == 'files':
                    if parts[4] not in ARTIFACTS: raise FileNotFoundError()
                    file = jobs.folder(parts[2])/ARTIFACTS[parts[4]]
                else:
                    file = (STATIC/('index.html' if path == '/' else path.lstrip('/'))).resolve()
                    if not file.is_relative_to(STATIC.resolve()): raise FileNotFoundError()
                if not file.is_file(): raise FileNotFoundError()
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(file.stat().st_size))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                with file.open('rb') as stream: shutil.copyfileobj(stream, self.wfile)
            except PermissionError as exc: self.json({'error': str(exc)}, 403)
            except (FileNotFoundError, ValueError): self.json({'error': 'Не найдено'}, 404)
            except (BrokenPipeError, ConnectionResetError): pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--jobs', type=Path, default=ROOT/'out/web-jobs')
    args = parser.parse_args()
    jobs = Jobs(args.jobs)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(jobs))
    print(f'Lidar2Sim: http://127.0.0.1:{server.server_port}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        jobs.pool.shutdown(wait=True)


if __name__ == '__main__': main()
