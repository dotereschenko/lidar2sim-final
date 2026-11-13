# Развёртывание на vkteacher

Сервер: `ssh vkteacher@172.20.21.105`, каталог `/home/vkteacher/lidar2sim-denis`.
Ubuntu 20.04, x86_64, четыре A100 40 GB, CUDA toolkit 12.8, driver 570.86.10.
GPU 1 занят сторонним расчётом. Перед запуском выбирать свободный GPU по UUID.

Docker недоступен текущему пользователю. Приложение устанавливается в отдельную
`.venv` с системным Python 3.11. Системные пакеты, драйверы и чужие службы не меняются.
Open3D 0.20 из основного lock-файла требует glibc 2.35, поэтому для этого хоста
используется Open3D 0.19. Это отдельное окружение совместимости, не установка из lock.
Полный набор тестов и сквозной HTTP-прогон на 8192 точках прошли;
артефакты PLY/USD скачаны и сверены с SHA256 в отчёте.
Результат: [cpu-api.json](https://github.com/dotereschenko/lidar2sim/blob/15766f2fbc7666d757d7ea57dcca0be8bb7f2373/benchmarks/vkteacher/cpu-api.json).

Целевой метод в UI — **NKSR на GPU**, он выбран по умолчанию. При недоступном
NKSR запуск блокируется; CPU-метод можно выбрать явно. Нормали, упрощение сеток
и экспорт в текущем конвейере остаются на CPU. Это не целиком GPU-конвейер.

Проверено 12 ноября 2026: NKSR 1.0.3 работает на A100 №2
(`GPU-c4306a6e-f03f-55fc-3f12-1c427935c5e4`). Изолированный тест 8192 точек:
3994 треугольника, 0.895 секунды, 115831808 байт пиковой памяти CUDA.
Полный HTTP-прогон через UI API: 4.648 секунды, 3920 треугольников после
предобработки; visual/collision PLY и USD скачаны, SHA256 совпадают с отчётом,
в USD присутствует CollisionAPI. Сфера проверяет работоспособность, но не
качество реконструкции комнаты и не физику.
Доказательства: [GPU runtime](https://github.com/dotereschenko/lidar2sim/blob/15766f2fbc7666d757d7ea57dcca0be8bb7f2373/benchmarks/vkteacher/gpu-runtime.json),
[GPU API](https://github.com/dotereschenko/lidar2sim/blob/15766f2fbc7666d757d7ea57dcca0be8bb7f2373/benchmarks/vkteacher/gpu-api.json),
[версии пакетов](https://github.com/dotereschenko/lidar2sim/blob/15766f2fbc7666d757d7ea57dcca0be8bb7f2373/benchmarks/vkteacher/installed-packages.json).
Полный `python -m pytest -q` повторно прошёл в установленном GPU-окружении.
Проверены API и логика выбора метода; интерактивная браузерная проверка
загрузки и вращения модели отдельно не выполнялась.

Реальный скан кухни (976350 точек) также прошёл HTTP-конвейер: 23.93 с всего,
2.12 с NKSR на A100, 125024 визуальных треугольника и 60000 коллизионных.
В UI открыть «Последние задания», результат `3d29e7614cb5449d9e6cf02d5544bca4`.
Время подготовки 17.12 с; отдельный профиль показал 15.12 с на ориентацию нормалей.
Масштаб m/Z не подтверждён, качество комнаты и физика не квалифицированы.
[Отчёт без геометрии скана](https://github.com/dotereschenko/lidar2sim/blob/15766f2fbc7666d757d7ea57dcca0be8bb7f2373/benchmarks/vkteacher/kitchen-runtime.json),
[спецификация GPU-ориентации](https://github.com/dotereschenko/lidar2sim/blob/15766f2fbc7666d757d7ea57dcca0be8bb7f2373/docs/superpowers/specs/2026-11-12-gpu-normal-orientation.md).
Результат UI показывает записанное в отчёте устройство и время GPU/CPU-этапов;
для старых отчётов без имени устройства используется общее обозначение CUDA.

Для установки нужны Python 3.11, uv 0.8.14 и CUDA toolkit 12.8. В каталог
копируются исходники, configs, tools, tests, docs, examples и
`data/scene_a/test_config.json`; приватные облака в дистрибутив не входят.
Базовые зависимости устанавливаются в `.venv` отдельно от основного lock:

```sh
uv venv --python /usr/bin/python3.11
uv pip install open3d==0.19.0 usd-core==26.11 pyyaml 'jsonschema>=4,<5' 'pytest>=8,<10'
git clone https://github.com/nv-tlabs/NKSR.git nksr-source
git -C nksr-source checkout e40336845e67761343a756788e5a98b827d4a143
bash deployment/vkteacher-install-nksr.sh
```

Скрипт ожидает uv в `uv-x86_64-unknown-linux-gnu/uv` внутри каталога проекта.
PyTorch закреплён на 2.7.0+cu128, расширения собираются под `sm_80`.
Готовый wheel torch-scatter требует glibc 2.32, поэтому на этом хосте он
также собирается из исходников. Скрипт применяет те же совместимые правки NKSR,
что описаны в [nksr-runtime.md](nksr-runtime.md).
Веса `ks.pth` размещаются в `weights/hub/checkpoints/`; ожидаемый SHA256:
`c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955`.

Запуск после установки и проверки зависимостей:

```sh
sh deployment/vkteacher-run.sh "$PWD" GPU-UUID
journalctl --user -u lidar2sim-ui -n 50
systemctl --user stop lidar2sim-ui
```

Последняя команда останавливает только эту службу. Автозапуск после перезагрузки
не настраивается. Процесс закреплён за CPU 0–3, 4 потока BLAS/OpenMP,
таймаут задания 600 секунд. Жёсткий лимит RAM для пользовательской службы
на этом хосте пока не подтверждён; лимит виртуальной памяти не применяется,
так как он мешает резервированию адресного пространства CUDA.
Проверка диска перед загрузкой и запуском требует
резерв 10 GiB; это не дисковая квота и не непрерывный контроль свободного места.

Доступ с ноутбука:

```sh
ssh -N -L 8768:127.0.0.1:8768 vkteacher@172.20.21.105
```

Открыть `http://127.0.0.1:8768`. Сервер слушает только loopback.
NKSR включается только после успешной проверки CUDA, импорта расширения и SHA256 весов.

Isaac Sim на A100 официально не поддерживается: отсутствуют RT cores.
Здесь можно подготовить USD и запустить Newton CUDA (см. [маршруты](scene-a-simulation.md)).
Для симуляции именно в Isaac нужна совместимая RTX-машина: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
