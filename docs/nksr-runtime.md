# NKSR runtime: эксперимент H42

SDD и распределение: [2026-11-12-nksr-pretrained.md](superpowers/specs/2026-11-12-nksr-pretrained.md).
Состояние: [nksr-runtime-spike.json](../benchmarks/nksr-runtime-spike.json).
Все тяжёлые операции выполняются на Spark. Runtime подключён к отдельному UI на
`http://127.0.0.1:8767` через SSH-туннель.
Образ собирается для исследования и оценки; учитывать лицензию NKSR при распространении.

## Подготовка контекста (на Spark)

Рабочий каталог текущей попытки: `/home/unknown641/lidar2sim-nksr-spike`.
Он содержит upstream `NKSR/`, `Dockerfile.nksr-spike`, `patch_nksr_error_checks.py`,
`nksr_smoke.py`, `weights/`.
Исходники закреплены на `e40336845e67761343a756788e5a98b827d4a143`.
Dockerfile/patch script и smoke script копируются из `deployment/` и `tools/` этого репозитория.

```sh
cd /home/unknown641/lidar2sim-nksr-spike
docker build --network=host -f Dockerfile.nksr-spike -t lidar2sim-nksr:spike-e403368 .
```

Этот образ содержит native-расширения. Для запуска нужен полный образ
`lidar2sim-nksr:ui` с Open3D и системными библиотеками: собрать его по
[nksr-ui.md](nksr-ui.md). Отдельно проверено, что без libgfortran5 Open3D не загружается.

Веса уже скачаны из официального checkpoint repository в
`weights/hub/checkpoints/ks.pth`. SHA256:
`c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955`.
Torch hub использует этот локальный cache; smoke проверяет SHA перед загрузкой модели.
Не запускать параллельную сборку: сначала проверить `build.pid` и `build.log`.

## Проверка после успешной сборки

Два запуска ниже успешно выполнены 12 ноября 2026 года, без доступа к сети.
Выходной подкаталог не должен существовать. Повтор выполнить с именем `smoke-2`.

```sh
mkdir -p results
docker run --rm --network none --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all --cpus 4 --memory 16g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/weights:/weights:ro" -v "$PWD/results:/results:rw" \
  --entrypoint python lidar2sim-nksr:ui /opt/nksr_smoke.py \
  --weights-sha256 c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955 \
  --output /results/smoke-1
```

Готовый image ID:
`sha256:186a032dde5f98e32f7f25ea0053e2d60674f52da45db8284f67fa4214b7a7b0`.
Полный список установленных версий сохранён в `benchmarks/nksr/installed-packages.json`,
результаты — в `benchmarks/nksr/repeat-summary.json`.
Оба прогона: 2000 вершин / 3994 треугольника, 1.26 и 1.12 с inference.
Байты mesh различаются: максимальное смещение соответствующих вершин 0.119 мм,
индексы треугольников совпадают. Побитовая детерминированность не заявляется.
Непустая сфера проверяет runtime, но не качество комнаты или физику.

Первоначальный запуск с `--gpus all` отклонён NVIDIA hook; явный `--runtime=nvidia`
исправил доступ без изменения хоста. Для toolchain Spark применены четыре совместимых
правки upstream: явный Thrust tuple include, переименование неиспользуемого lerp,
исключения вместо некорректного printf/abort, NanoVDB coordinate caster для pybind11.
Логи неудачных сборок сохранены на Spark как `build-attempt1.log`–`build-attempt5.log`.
Dockerfile остаётся рецептом сборки с диапазонами Python-зависимостей; для повторения
проверенного окружения использовать указанный готовый image ID. Это не release lock
для будущих установок из изменившихся package registry.
