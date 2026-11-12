# HACKATHON-3 Acceptance Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Один профиль приёмки для существующих оценщиков с явной неполнотой PoC.

**Architecture:** YAML задаёт пороги; чистый модуль валидирует его, проверяет
совместимость конфигурации сцены и принимает решения по неокруглённым числам.
Геометрический и физический скрипты сохраняют id/hash профиля и область проверки.

**Tech Stack:** Python 3.10–3.12, unittest, PyYAML, существующие NumPy/Open3D/Isaac.

**Spec:** `docs/superpowers/specs/2026-11-10-hackathon-3-acceptance-contract-design.md`

## Global Constraints

- Профиль `lidar2sim-rules-v1`, schema_version 1; точное содержимое YAML — раздел 6 SDD.
- Метры, Z-up; visual/collision сохраняют прежние названия и USD-контракт.
- Решение по сырым значениям: ≤ для верхних границ, ≥ для нижних.
- Профиль не поступает в реконструкцию; GT/B не становятся её входами.
- Не реализовывать B, контрольные размеры, новый алгоритм реконструкции или общий удалённый runner.
- Не запускать и не объявлять выполненной физику без Isaac; не менять Plane.
- Существующие локальные изменения сохранять; не коммитить чужие/предыдущие изменения.

## Task 1: Единый профиль и два потребителя

**Files:**
- Create: `configs/acceptance/lidar2sim-rules-v1.yaml`
- Create: `lidar2sim/acceptance.py`
- Modify: `tools/eval_scene.py`, `tools/isaac_scene_test.py`
- Create: `tests/test_acceptance.py`, `tests/test_eval_acceptance.py`

**Interfaces:**
- `load_profile(path=None) -> (dict, dict)`: значения и provenance
  `{profile_id, schema_version, sha256}` по фактически прочитанным байтам.
- `validate_physics_config(config, profile) -> None`: сверка существующих
  pass/body/dt/speed и количества floor/obstacles/routes; несовпадение — ValueError.
- `geometry_checks(raw_metrics, profile) -> dict[str, bool]`: прежние четыре
  ключа pass на основе неокруглённых extra_5cm_pct, extra_5cm_pct_collision,
  median_cm, p95_cm. Эти имена остаются для совместимости v1.
- `upper_bound_passed(value, limit) -> bool`: конечное значение ≤ конечного лимита.
- `overall_status(checks: dict[str, str]) -> str`: failed при failed/error,
  иначе incomplete при not_run или пустом наборе, иначе passed. Неизвестный статус — ValueError.

- [x] **Step 1: Написать и запустить проваливающиеся тесты.**

Реальные временные YAML, без mock конфигурационного загрузчика. Проверить
неизвестные/отсутствующие поля на каждом уровне, bool вместо чисел, NaN/Inf,
неизвестные schema/id, доли вне [0,1], count/min и median/p95, неверный список
meshes/body. Проверить отсутствие зависимости default path от cwd.
Пример регрессии округления (не должно пройти при выводе 92,0%):

```python
profile, provenance = load_profile()
raw = {"extra_5cm_pct": 91.96, "extra_5cm_pct_collision": 94.0,
       "median_cm": 1.0, "p95_cm": 4.0}
assert not geometry_checks(raw, profile)["лишняя геометрия, visual"]
assert upper_bound_passed(0.05, 0.05)
assert not upper_bound_passed(0.05001, 0.05)
assert overall_status({"geometry": "passed", "physics": "not_run"}) == "incomplete"
```

Run: `.venv/bin/python -m unittest discover -s tests -p 'test*acceptance*.py' -v`
Expected: FAIL из-за отсутствующего поведения; зафиксировать причину в отчёте.

- [x] **Step 2: Реализовать профиль и чистые функции.**

Считать байты один раз, разобрать safe_load, проверить точный набор полей,
типы, конечность и диапазоны, вернуть hash тех же байтов. Не хранить вторую
копию всех порогов в Python. Список допустимых полей/типов — схема, не источник
чисел. Поддерживаемые schema/id заданы явно. Runtime checks не допускают NaN.
Для физики сравнить дублирующие значения test_config с профилем и отклонить
неполные наборы тестов. Пороговые значения для работы брать из профиля.

- [x] **Step 3: Подключить геометрию.**

Сохранить старые позиционные CLI-аргументы; добавить `--profile`. Снять
hardcoded THRESHOLDS/SAMPLES, читать профиль до mesh IO. Сбрасывать seed перед
каждой независимой выборкой. Вычислять сырые метрики, затем passes, затем
округлённые поля. Использовать ≤ для distance, валидировать непустые ref/mesh.
В JSON сохранить raw_metrics, acceptance_profile, evaluation_mode
`scene_a_gt_diagnostic`, coverage collision completeness=not_run и
poc_status=incomplete (либо failed при провале геометрии).
Не писать «все проверки пройдены» без уточнения области. Отдельный файл для
каждого out, даже если разные пути имеют одинаковое имя каталога.
При успешном выполнении оценки сохранить прежний exit code 0, даже если
пороги не пройдены; ошибки входа/профиля дают ненулевой код. Полный runner №23
будет отдельно определять код приёмки.

- [x] **Step 4: Подключить физику.**

Добавить `--profile`, проверять профиль и конфиг до импорта/создания
SimulationApp. Поддержать запуск `python.sh /absolute/path/tools/...` без
установленного lidar2sim (корень проекта в sys.path). Метаданные профиля и
частичный статус включить в physics.json. Пол использует общий helper ≤,
препятствия не пропускают NaN/Inf, числа успешных тестов берутся из профиля.
Параметры тела, dt, speed и длительность теста пола также брать из профиля.
Настройки контактов и исправление floor_z маршрутов оставить №12.

- [x] **Step 5: Проверить интеграцию и весь набор.**

Run: `.venv/bin/python -m unittest discover -s tests -v`.
Проверить subprocess: некорректный профиль физического скрипта отклоняется
до необходимости Isaac. Геометрия — реальные маленькие сетки, проверка
идентичного результата при перестановке out, отсутствие ложного полного pass.
Синтаксис Isaac проверить compile, без утверждения о физической симуляции.
Run: `.venv/bin/python tools/eval_scene.py data/scene_a out/baseline out/floor out/filtered`.
Сохранить существующие артефакты; ожидаются провал baseline и успех четырёх
реализованных проверок filtered, но не полный PoC passed.

## Task 2: Документация и сверка SDD

**Files:** `README.md`, `docs/rules.md`, SDD и этот план.
**Consumes:** фактические интерфейсы Task 1 и результаты тестов.
**Produces:** согласованное описание профиля, режимов, статусов и незапущенных проверок.

- [x] Добавить команды `--profile`, версии, inclusive-границы и ссылку на YAML.
- [x] Разделить текущую A-диагностику и полную приёмку; размеры/B/физика/10M не выполнены.
- [x] Проверить все требования SDD по изменённым файлам и тестам.
- [x] Запустить `git diff --check`; проверить полный diff только текущей задачи.

## Execution ledger

- Ruling: продолжаем текущую feature-ветку с разрешёнными пользователем
  локальными изменениями; новый worktree потерял бы незакоммиченный контекст.
- Ruling: Task 1 объединяет тесно связанные интерфейсы в одну интеграционную
  единицу, чтобы два агента не расходились по формату профиля и статусов.
- Preflight: Task 1 производит CLI/metadata, Task 2 только документирует их;
  общих редактируемых runtime-файлов нет. Spec ограничивает полноту collision
  будущей №19; текущие отчёты должны указывать этот пробел.
- До начала: 11 существующих тестов; результат запуска проверяется контроллером.
- Изменение кода не означает закрытия Plane или утверждения правил организатором.


## Результат выполнения 10.11.2026

- Task 1: complete — профиль, строгий загрузчик, сырые сравнения, общий helper
  статусов, preflight физики и интеграция обоих скриптов реализованы.
- Task 2: complete — README, правила и SDD согласованы с фактическими полями
  частичных отчётов; неподтверждённая физическая квалификация отмечена отдельно.
- TDD: исходные 11 тестов прошли; новые тесты сначала воспроизвели отсутствие
  профиля и неправильный порядок preflight, затем весь набор — 23/23.
  Дополнительно удаление установки seed ломает тест порядка, восстановление
  возвращает PASS. Compile и git diff --check успешны.
- Контрольный запуск: `python tools/eval_scene.py data/scene_a out/baseline
  out/floor out/filtered --profile configs/acceptance/lidar2sim-rules-v1.yaml`
  завершился успешно. Baseline 88,2/88,6%, floor 88,8/89,1%, filtered 93,6/94,7%.
  Filtered: median=1,69 см, p95=4,74 см; geometry_status=passed,
  poc_status=incomplete, collision_completeness=not_run.
- SHA256 профиля в трёх отчётах совпал с YAML:
  `f96101b5a83bcaccc44e8df9249477e90530aa7468e20a2c0be990d105bd294b`.
- Независимое ревью: исправлены историческое утверждение о пяти прогонах и
  неточная формулировка README об отдельных статусах. Повторное ревью:
  Critical/Important замечаний нет.
- Проверка реального окружения: `ssh spark` → `10.10.0.2`, существующий
  `isaac-stream`, образ `isaac-sim-xvfb:6.2`, Python 3.12.14, PyYAML 6.0.3.
  В этом Python выполнены загрузка профиля, проверка scene_a/test_config и
  отклонение намеренно неверной скорости. SimulationApp не импортировался.
- Реальная физика не запущена: на сервере уже работают `isaac-stream` и
  `isaac-flight`; отдельный слот/разрешение на остановку запрошены у пользователя.
- Ruling: сериализация нечисловых runtime-измерений физики как NaN/Infinity
  остаётся существующим ограничением до HACKATHON-9; новые guards не позволяют
  таким измерениям пройти проверку, но строгий JSON-потребитель может их отклонить.
- Изменения сохранены локально без commit/merge/push. Задачи Plane не менялись.
  Публикация правил организаторами и квалификация стенда не объявлены завершёнными.
