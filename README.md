# lidar2sim

Задача хакатона «Сцены для симулятора из лидарного скана». Нужен конвейер одной командой: на входе облако точек помещения, на выходе сцена Isaac Sim с коллизиями и отчёт проверки.

![Сцена в симуляторе, облако точек и восстановленная сцена](docs/img/triptych.jpg)

*Слева направо: офис в Isaac Sim, облако лидара по этой сцене и сцена, восстановленная базовым решением. Потолок срезан, чтобы было видно помещения.*

В репозитории — рабочее базовое решение. Конвейер уже проходит от облака до сцены, которая загружается в Isaac Sim. Ваша задача — сделать сцену точной и физически корректной. Как именно — решаете вы.

Правила и пороги — [docs/rules.md](docs/rules.md).

## Данные

Сцена A — западное крыло офиса из Isaac Sim: кабинет с мебелью и Г-образный коридор. Облако снято лидаром на высоте 1,3 м по маршруту через сцену. Оно грязное, как настоящее SLAM-облако: шум дальности, выпадения, точки-призраки в воздухе и дрейф совмещения — стены местами двоятся.

| Файл | Содержимое |
|---|---|
| `A.ply` | Облако точек, 249 тыс. точек: XYZ в метрах, ось Z вверх |
| `trajectory_A.csv` | Положения лидара во время съёмки: `t,x,y,z` |
| `manifest.json` | Границы сцены, оси, описание файлов |
| `gt_mesh.ply` | Точная геометрия сцены — только для самопроверки, на вход конвейеру её подавать нельзя. Собирается скриптом, см. ниже |
| `test_config.json` | Тесты физики сцены A для `tools/isaac_scene_test.py` |

Облако, траектория и тесты лежат в репозитории, в `data/scene_a/`. Точную геометрию сцены соберите один раз сами:

```bash
.venv/bin/python tools/fetch_gt.py data/scene_a
```

Сцена — ассет NVIDIA Isaac Sim, и выкладывать её мы не можем. Скрипт скачивает её с открытого сервера ассетов NVIDIA (~10 МБ, без текстур), собирает `gt_mesh.ply` и сверяет с нашим эталоном. Пользуетесь вы ею на условиях [NVIDIA Isaac Sim Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/common/license-isaac-sim-additional.html).

## Запуск

Рабочая версия Python — **3.12.13**, она записана в `.python-version`. Зависимости
с хешами закреплены в `uv.lock`; Open3D — 0.20.0, usd-core — 26.11.
Поддерживаемый диапазон Python — 3.11–3.12: закреплённый usd-core требует Python ≥3.11.

```bash
uv sync --locked --no-dev
.venv/bin/python tools/fetch_gt.py data/scene_a

.venv/bin/python -m lidar2sim --input data/scene_a/A.ply \
    --manifest data/scene_a/manifest.json --trajectory data/scene_a/trajectory_A.csv \
    --config configs/baseline.yaml --out out/baseline
.venv/bin/python tools/eval_scene.py data/scene_a out/baseline
```

`--locked` проверяет актуальность lock-файла и не меняет его при установке
([документация uv](https://docs.astral.sh/uv/concepts/projects/sync/)).
Установка через обычный `pip install -e .` подходит для разработки, но не фиксирует
все транзитивные зависимости: для зачётного прогона используйте команды выше.

На Mac с Apple Silicon для Open3D 0.20 нужна системная `libusb`:

```bash
brew install libusb
uv sync --locked --no-dev
```

CPU-конвейер проверяется на Ubuntu `x86_64`. Для физики используется отдельное
окружение Isaac Sim 6.2 на Spark, существующий образ `isaac-sim-xvfb:6.2`.
Библиотеки Isaac не устанавливаются через этот lock-файл.
Проверенные версии ОС, системных библиотек и ID существующего образа записаны
в `configs/runtime.json`.

`tools/eval_scene.py` принимает несколько папок с результатами и выводит их рядом — удобно сравнивать свои версии.

Для первой проверки Isaac без офисных данных используйте маленький
[example_scene.usd](examples/example_scene.usd) (пол и ящик, без внешних ассетов):

```bash
<isaac-sim>/python.sh tools/isaac_scene_test.py \
    --scene examples/example_scene.usd --config examples/test_config.json \
    --out out/example-physics.json --cpu-threads 4
```

Сцену можно открыть в Isaac Sim через **File → Open**; единицы — метры, ось Z
направлена вверх, `/World/Collision` является статическим коллайдером.
Воспроизвести файл можно командой `.venv/bin/python tools/make_example_scene.py`.
В Isaac Sim 6.2 на Spark контрольный прогон прошёл пол 2/2, препятствие 1/1 и
свободный маршрут 1/1; результат сохранён в
[benchmarks/example/validation.json](benchmarks/example/validation.json).

Для версии с отсечением неподдержанной геометрии и локальным восстановлением пола
используйте `configs/poc.yaml`:

```bash
.venv/bin/python -m lidar2sim --input data/scene_a/A.ply \
    --manifest data/scene_a/manifest.json --trajectory data/scene_a/trajectory_A.csv \
    --config configs/poc.yaml --out out/poc
.venv/bin/python tools/eval_scene.py data/scene_a out/baseline out/poc
```

Конфигурация бейзлайна сохраняет исходный алгоритм. Новые этапы и результаты
сравнения описаны в [docs/baseline.md](docs/baseline.md#отсечение-и-восстановление-пола).
В `report.json.refinement` записываются число удалённых граней, модель и площадь
пола, объём его наблюдений и число треугольников, отклонённых из-за перепада высоты.
В PoC коллайдер упрощается по двум компонентам: восстановленный пол получает
25% бюджета в 20 тыс. треугольников, остальные поверхности — оставшиеся 75%.
Это сохраняет локальные участки пола и границы проёмов при упрощении.
`report.json.collision_build` фиксирует режим, бюджет, вес границ и число
треугольников каждой компоненты. `configs/baseline.yaml` по-прежнему использует
исходное глобальное упрощение. Сравнение вариантов воспроизводится командой:

```bash
.venv/bin/python tools/compare_collision.py --data data/scene_a \
    --out out/collision-comparison
```

Физика проверяется в Isaac Sim 6.2:

```bash
<isaac-sim>/python.sh tools/isaac_scene_test.py --scene out/baseline/scene.usd \
    --config data/scene_a/test_config.json --out out/baseline/physics.json --cpu-threads 4
```

`--cpu-threads` задаёт рабочие потоки Isaac и PhysX (по умолчанию 4).
В контейнере согласуйте это число с выделенным CPU-бюджетом. Обновление viewport
отключено; шаг времени, длительность и критерии физических тестов сохраняются.
Это соответствует настройкам из
[рекомендаций NVIDIA по производительности](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/reference_material/sim_performance_optimization_handbook.html#cpu-thread-count-optimizations).
Скрипт выводит ход проверки по позициям пола, препятствиям и маршрутам.

Как получить доступ к Isaac Sim на нашей машине, расскажем отдельно.

### Проверка файлового контракта

Конвейер отклоняет некорректные параметры, единицы/оси manifest, нечисловые
координаты и повреждённую траекторию до запуска Poisson. Экспортированные PLY и USD
повторно читаются и сверяются: коллайдер должен быть статическим и находиться на
`/World/Collision`, а геометрия USD — совпадать с PLY.

Проверить готовые файлы отдельно:

```bash
.venv/bin/python -m lidar2sim.validation out/baseline
```

Это проверка структуры и согласованности файлов. Качество геометрии и физику
проверяют отдельные скрипты выше. При ошибке конвейер возвращает ненулевой код
и записывает `report.json` со статусом `failed`, включая причину ошибки.

Проверки разработчика:

```bash
uv sync --locked
uv run --locked pytest
```

### Итоговый отчёт

`report.json` версии 1 содержит SHA256 облака, manifest, траектории и конфигурации,
версию и отпечаток исходников, версии библиотек, эффективные параметры и хеши
трёх геометрических артефактов. `total_s` включает загрузку модулей, чтение входов,
реконструкцию, экспорт, обратную проверку файлов и первичную запись отчёта.
Схема поставляется в `lidar2sim/schemas/report.schema.json`.

После расчёта метрик для полученной сцены можно присоединить их к отчёту:

```bash
.venv/bin/python tools/eval_scene.py data/scene_a out/baseline
.venv/bin/python -m lidar2sim.reporting out/baseline \
    --geometry out/baseline/eval_scene.json

# После проверки той же сцены в Isaac:
.venv/bin/python -m lidar2sim.reporting out/baseline \
    --physics out/baseline/physics.json
```

### Единый прогон PoC

`tools/run_poc.py` одной командой строит четыре контрактных файла из A,
запускает геометрический оценщик, вызывает физический runner и объединяет
результаты в `report.json`. CPU-этапы выполняются локально; физику можно
выполнить на Spark через `tools/spark_physics_runner.py`. Скопируйте
`configs/spark.example.json` в собственный файл настроек и укажите доступные
SSH-адрес, порт и при необходимости `ssh_control_path` (путь к существующему
SSH master-соединению). Пароли и ключи в JSON не записываются.

```bash
.venv/bin/python tools/run_poc.py --data data/scene_a --config configs/poc.yaml \
    --out out/poc --physics-runner .venv/bin/python \
    --runner-arg tools/spark_physics_runner.py \
    --runner-arg=--settings=path/to/spark.json
```

Runner передаёт USD, конфигурацию физических тестов и оценщик в отдельный
каталог Spark, сверяет SHA256 каждого файла, запускает изолированный Docker
с 4 CPU, 16 GiB и без сети, возвращает `physics.json`. Логи `run.log` хранятся
рядом с результатом; `physics.log` и `xvfb.log` остаются в каталоге запуска
на Spark. Для повторной оценки готовой сцены добавьте `--reuse`. Вместо
запуска Isaac можно присоединить уже полученный файл через
`--physics-result path/to/physics.json`; он должен относиться к тому же USD.
Без runner физика получает явный статус `not_run` и команда возвращает код 1.
Код 2 означает ошибку этапа и фиксирует её в `report.json` (когда отчёт уже
создан). Без окончательного профиля порогов даже успешные проверки дают общий
статус `incomplete`; после его получения передайте `--profile`. Проверка
сценария, включая реальный Spark, записана в
[benchmarks/orchestration/validation.json](benchmarks/orchestration/validation.json).

Результаты оценщиков сохраняются целиком. Незапущенная физика имеет статус
`not_run`; итоговая приёмка становится `passed` после прохождения файлового
контракта, геометрии и физики с указанным профилем порогов. После фиксации профиля
передайте его командой `--profile путь/к/профилю.yaml`; содержимое и хеш сохранятся
в отчёте. Команда объединения возвращает 0 при полном
прохождении, 1 при провале или незавершённой оценке и 2 при ошибке данных.
Время внешнего оценщика сохраняется как неизвестное (`null`), если сам оценщик
его не записал. Исходное время реконструкции при объединении сохраняется.

## Контракт

**Вход**
| Файл | Содержимое |
|---|---|
| `scene.ply` | Облако точек: XYZ в метрах, ось Z вверх, единая система координат |
| `manifest.json` | Границы сцены (`bounds_min`, `bounds_max`), оси, источник, версия данных |
| `trajectory.csv` | Положения лидара во время съёмки: `t,x,y,z` |
| `config.yaml` | Параметры конвейера |

**Выход** — в папке `--out`:
| Файл | Содержимое |
|---|---|
| `visual.ply` | Сетка для отображения |
| `collision.ply` | Сетка для физики |
| `scene.usd` | Сцена Isaac Sim: метры, ось Z вверх, `/World/Visual` и `/World/Collision` со статическим коллайдером |
| `report.json` | Хеш входа, параметры, время каждого шага, размер сеток |

Названия файлов и структура `scene.usd` — интерфейс, по которому проверяется результат. Внутри конвейера можно менять что угодно.

## Ориентиры на сцене A

Базовое решение против нашего внутреннего. Его цифры — желаемый уровень. Для него взят худший из пяти прогонов: цифры от прогона к прогону немного гуляют. Как оно устроено, не раскрываем: подход выбираете вы.

| Проверка | Порог | Базовое | Желаемый уровень |
|---|---|---|---|
| Сетка в пределах 5 см от эталона, visual | ≥ 92% | 88,2% | ≥ 94,7% |
| То же, collision | ≥ 92% | 88,6% | ≥ 93,1% |
| Эталон → сетка в области, которую видел A, медиана / p95 | ≤ 3 / ≤ 10 см | 1,7 / 4,8 см | ≤ 1,6 / ≤ 4,2 см |
| Isaac Sim: пол, позиций без провала | ≥ 9 из 10 | 8 из 10 | ≥ 9 из 10 |
| Isaac Sim: препятствия, проникновение ≤ 3,5 см | 10 из 10 | 10 из 10 | 10 из 10 |
| Isaac Sim: маршруты пройдены | ≥ 4 из 5 | 5 из 5 | ≥ 4 из 5 |
| Время | — | ~12 с | ~30 с |

Базовое решение не проходит два порога: лишняя геометрия и пол.

## Тестовая комната

Синтетическая комната 6×4×2,7 м с перегородкой, проёмом, ящиком и колонной — дымовой тест окружения, данные хакатона для неё не нужны. Подробно — [docs/baseline.md](docs/baseline.md).

```bash
.venv/bin/python tools/make_test_room.py data/test_room
.venv/bin/python -m lidar2sim --input data/test_room/scene.ply \
    --manifest data/test_room/manifest.json --trajectory data/test_room/trajectory.csv \
    --config configs/baseline.yaml --out out/test_room
.venv/bin/python tools/quick_eval.py data/test_room out/test_room
```

## Файлы

| Путь | Что |
|---|---|
| `lidar2sim/baseline.py` | Базовое решение |
| `lidar2sim/common.py` | Чтение входа, подготовка облака, экспорт PLY/USD, отчёт |
| `lidar2sim/refinement.py` | Отсечение неподдержанной геометрии, восстановление локальных разрывов пола |
| `configs/baseline.yaml` | Параметры базового решения |
| `configs/poc.yaml` | Параметры улучшенного Poisson |
| `data/scene_a/` | Данные сцены A |
| `tools/fetch_gt.py` | Точная геометрия сцены A: скачать у NVIDIA и собрать |
| `tools/eval_scene.py` | Проверки геометрии на сцене хакатона |
| `tools/compare_reconstruction.py` | Сравнение отсечения, достройки пола и их сочетания на одной исходной сетке |
| `tools/isaac_scene_test.py` | Тесты физики в Isaac Sim по `test_config.json` |
| `tools/make_test_room.py`, `tools/quick_eval.py`, `tools/isaac_physics_test.py` | Тестовая комната: данные, быстрые проверки, физика |
