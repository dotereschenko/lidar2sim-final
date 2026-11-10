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
| `gt_mesh.ply` | Точная геометрия сцены — только для самопроверки, на вход конвейеру её подавать нельзя |
| `test_config.json` | Тесты физики сцены A для `tools/isaac_scene_test.py` |

```bash
python tools/get_data.py                                  # скачать и проверить контрольную сумму
python tools/get_data.py --archive scene_a_v1.tar.gz      # или распаковать уже полученный архив
```

Данные лягут в `data/scene_a/`. Пока ссылки в `configs/data.json` нет, архив выдают организаторы.

## Запуск

Нужен Python 3.10–3.12: под 3.13 и новее нет пакетов Open3D.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python tools/get_data.py --archive scene_a_v1.tar.gz

.venv/bin/python -m lidar2sim --input data/scene_a/A.ply \
    --manifest data/scene_a/manifest.json --trajectory data/scene_a/trajectory_A.csv \
    --config configs/baseline.yaml --out out/baseline
.venv/bin/python tools/eval_scene.py data/scene_a out/baseline
```

Без `uv` подойдёт обычный `python -m venv .venv && .venv/bin/pip install -e .`.

`tools/eval_scene.py` принимает несколько папок с результатами и выводит их рядом — удобно сравнивать свои версии.

Физика проверяется в Isaac Sim 6.2:

```bash
<isaac-sim>/python.sh tools/isaac_scene_test.py --scene out/baseline/scene.usd \
    --config data/scene_a/test_config.json --out out/baseline/physics.json
```

Как получить доступ к Isaac Sim на нашей машине, расскажем отдельно.

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
| `configs/baseline.yaml` | Параметры базового решения |
| `configs/data.json` | Где брать данные и их контрольные суммы |
| `tools/get_data.py` | Загрузка данных |
| `tools/eval_scene.py` | Проверки геометрии на сцене хакатона |
| `tools/isaac_scene_test.py` | Тесты физики в Isaac Sim по `test_config.json` |
| `tools/make_test_room.py`, `tools/quick_eval.py`, `tools/isaac_physics_test.py` | Тестовая комната: данные, быстрые проверки, физика |
