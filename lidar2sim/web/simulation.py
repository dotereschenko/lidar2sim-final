"""Scene A catalog and opt-in physics runners; never substitute preview for physics."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import uuid


class Simulations:
    def __init__(self, jobs, root):
        self.jobs, self.root = jobs, root
        self.directory = jobs.root/'simulations'
        self.lock = jobs.lock
        self.processes = {}
        for path in self.directory.glob('*/state.json'):
            state = json.loads(path.read_text())
            if state['status'] in ('queued', 'running'):
                self.update(path.parent.name, status='failed', error='Сервер перезапущен')

    def folder(self, ident):
        if not re.fullmatch('[0-9a-f]{32}', ident):
            raise ValueError('Некорректный ID симуляции')
        return self.directory/ident

    def update(self, ident, **fields):
        with self.lock:
            path = self.folder(ident)/'state.json'
            data = json.loads(path.read_text()) if path.exists() else {'id': ident}
            data.update(fields)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, ensure_ascii=False))
            temporary.replace(path)
            return data

    def read(self, ident):
        folder = self.folder(ident)
        state = json.loads((folder/'state.json').read_text())
        state['files'] = [name for name in ('simulation.mp4', 'overview.png', 'preview.jpg', 'recording.json', 'run.log')
                          if self.file(ident, name).is_file() and
                          (state['status'] == 'completed' or name in ('overview.png', 'preview.jpg', 'run.log'))]
        log = folder/'run.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-8192))
                state['log_tail'] = stream.read().decode(errors='replace')
        return state

    def file(self, ident, name):
        if name not in ('simulation.mp4', 'overview.png', 'preview.jpg', 'recording.json', 'run.log'):
            raise ValueError('Неизвестный файл симуляции')
        folder = self.folder(ident)
        return folder/name if name == 'run.log' else folder/'result'/name

    def command(self):
        command = json.loads(os.environ.get('LIDAR2SIM_NEWTON_COMMAND') or os.environ.get('LIDAR2SIM_ISAAC_COMMAND', '[]'))
        if not isinstance(command, list) or not all(isinstance(x, str) and x for x in command):
            raise ValueError('Некорректная настройка физического исполнителя')
        return command

    def backend(self):
        return 'newton' if os.environ.get('LIDAR2SIM_NEWTON_COMMAND') else 'isaac'

    def active(self):
        return any(json.loads(p.read_text())['status'] in ('queued', 'running')
                   for p in self.directory.glob('*/state.json'))

    def catalog(self):
        config = json.loads((self.root/'data/scene_a/test_config.json').read_text())
        marker = self.root/'scene-a-job.txt'
        ident = os.environ.get('LIDAR2SIM_SCENE_A_JOB', '') or (marker.read_text().strip() if marker.is_file() else '')
        scene = None
        if ident:
            try:
                job = self.jobs.read(ident)
                if job['status'] == 'completed':
                    scene = {'id': ident, 'points': job['result']['input']['points'],
                             'solution': job['result']['solution']}
            except (FileNotFoundError, ValueError):
                pass
        return {'scene': scene, 'routes': config['routes'], 'body': config['body'],
                'backend': self.backend(),
                'speed': config['speed'], 'robot': 'velocity_body',
                'runner_configured': bool(self.command()),
                'reason': 'Isaac-сервер не подключён. Spark недоступен; A100 не поддерживается Isaac Sim.',
                'archive': (self.root/'docs/img/release-v2/simulation.mp4').is_file()}

    def start(self, payload):
        with self.lock:
            command = self.command()
            if not command:
                raise ValueError('Физический запуск недоступен: подключите Isaac-сервер')
            catalog = self.catalog()
            if not catalog['scene']:
                raise ValueError('Реконструкция сцены A ещё не подготовлена')
            if payload.get('scene_job') != catalog['scene']['id']:
                raise ValueError('Маршруты настроены только для сцены A. Откройте её явно перед запуском.')
            if payload.get('robot') != 'velocity_body':
                raise ValueError('Неизвестная модель робота')
            route, start, end = (payload.get(k) for k in ('route', 'start', 'end'))
            if any(type(x) is not int for x in (route, start, end)):
                raise ValueError('Маршрут, старт и цель должны быть целыми индексами')
            if not 0 <= route < len(catalog['routes']):
                raise ValueError('Неизвестный маршрут')
            if not 0 <= start < end < len(catalog['routes'][route]['points']):
                raise ValueError('Цель должна находиться после старта на выбранном маршруте')
            if self.active():
                raise ValueError('Уже выполняется симуляция')
            if any(json.loads(p.read_text())['status'] in ('queued', 'running')
                   for p in self.jobs.root.glob('*/job.json')):
                raise ValueError('Сначала дождитесь завершения реконструкции')
            self.jobs.ensure_free_space()
            ident = uuid.uuid4().hex
            folder = self.folder(ident); folder.mkdir(parents=True)
            scene = self.jobs.folder(catalog['scene']['id'])/'result/scene.usd'
            # Snapshot inputs so a subsequent reconstruction cannot change this run.
            import shutil
            shutil.copyfile(scene, folder/'scene.usd')
            shutil.copyfile(self.root/'data/scene_a/test_config.json', folder/'config.json')
            state = self.update(ident, status='queued', error=None, scene_job=catalog['scene']['id'],
                                route=route, start=start, end=end, robot='velocity_body', backend=self.backend())
            thread = threading.Thread(target=self.run, args=(ident, command), daemon=True)
            thread.start()
            return state

    def cancel(self, ident):
        with self.lock:
            state = self.read(ident)
            if state['status'] not in ('queued', 'running'):
                raise ValueError('Симуляция уже завершена')
            self.update(ident, status='cancelled')
            process = self.processes.get(ident)
            if process and process.poll() is None:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        return self.read(ident)

    def run(self, ident, command):
        folder = self.folder(ident)
        try:
            with self.lock:
                state = self.read(ident)
                if state['status'] == 'cancelled': return
                self.update(ident, status='running')
                with (folder/'run.log').open('w') as log:
                    process = subprocess.Popen(command+['--scene', str(folder/'scene.usd'),
                        '--config', str(folder/'config.json'), '--out', str(folder/'result'),
                        '--route', str(state['route']), '--start-index', str(state['start']),
                        '--end-index', str(state['end'])], stdout=log, stderr=subprocess.STDOUT,
                        cwd=self.root, start_new_session=True)
                self.processes[ident] = process
            try:
                code = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise RuntimeError('Превышен лимит симуляции: 15 минут')
            with self.lock:
                if self.read(ident)['status'] == 'cancelled': return
                if code: raise RuntimeError('Физический исполнитель завершился с ошибкой. См. журнал запуска.')
                record = json.loads(self.file(ident, 'recording.json').read_text())
                digest = hashlib.sha256((folder/'scene.usd').read_bytes()).hexdigest()
                config_digest = hashlib.sha256((folder/'config.json').read_bytes()).hexdigest()
                video = self.file(ident, 'simulation.mp4')
                if (record.get('scene_sha256') != digest or record.get('config_sha256') != config_digest
                    or any(record.get(k) != state[k] for k in ('robot', 'route', 'start', 'end'))
                    or record.get('outcome') not in ('reached', 'blocked', 'fell', 'timeout')):
                    raise RuntimeError('Запись не соответствует сцене или выбранному маршруту')
                if state.get('backend') == 'newton':
                    validate_newton_record(record)
                elif not video.is_file() or video.stat().st_size == 0:
                    raise RuntimeError('Нет видео физического прогона')
                self.update(ident, status='completed', result=record, error=None)
        except Exception as exc:
            with self.lock:
                if self.read(ident)['status'] != 'cancelled':
                    self.update(ident, status='failed', error=str(exc))
        finally:
            with self.lock: self.processes.pop(ident, None)


def validate_newton_record(record):
    if record.get('backend') != 'newton' or record.get('device') != 'cuda:0':
        raise ValueError('Не подтверждён физический расчёт Newton на CUDA')
    size = record.get('body_size_m', [])
    if len(size) != 3 or not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in size):
        raise ValueError('Некорректный размер корпуса')
    poses = record.get('trajectory')
    if not isinstance(poses, list) or not 1 <= len(poses) <= 3600:
        raise ValueError('Нет измеренной траектории')
    previous = -1
    for pose in poses:
        position, quaternion, timestamp = pose.get('position', []), pose.get('quaternion_xyzw', []), pose.get('t')
        if (len(position) != 3 or len(quaternion) != 4
            or not all(type(x) in (int, float) and math.isfinite(x) for x in [timestamp, *position, *quaternion])
            or not previous < timestamp <= 180 or abs(sum(x*x for x in quaternion)-1) > .01):
            raise ValueError('Некорректная измеренная траектория')
        previous = timestamp
    if record.get('duration_s') != previous:
        raise ValueError('Длительность не соответствует траектории')
