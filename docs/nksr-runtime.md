# NKSR runtime: эксперимент H42

SDD и распределение: [2026-11-12-nksr-pretrained.md](superpowers/specs/2026-11-12-nksr-pretrained.md).
Состояние: [nksr-runtime-spike.json](../benchmarks/nksr-runtime-spike.json).
Все тяжёлые операции выполняются на Spark. Это отдельный runtime, не backend UI.
Образ собирается для исследования и оценки; учитывать лицензию NKSR при распространении.

## Подготовка контекста (на Spark)

Рабочий каталог текущей попытки: `/home/unknown641/lidar2sim-nksr-spike`.
Он содержит upstream `NKSR/`, `Dockerfile.nksr-spike`, `nksr_smoke.py`, `weights/`.
Исходники закреплены на `e40336845e67761343a756788e5a98b827d4a143`.
Dockerfile и smoke script копируются из `deployment/` и `tools/` этого репозитория.

```sh
cd /home/unknown641/lidar2sim-nksr-spike
docker build --network=host -f Dockerfile.nksr-spike -t lidar2sim-nksr:spike-e403368 .
```

Веса уже скачаны из официального checkpoint repository в
`weights/hub/checkpoints/ks.pth`. SHA256:
`c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955`.
Torch hub использует этот локальный cache; smoke проверяет SHA перед загрузкой модели.
Не запускать параллельную сборку: сначала проверить `build.pid` и `build.log`.

## Проверка после успешной сборки

Ниже команда для будущего запуска; успешный NKSR inference пока не заявлен.
Выходной подкаталог не должен существовать. Повтор выполнить с именем `smoke-2`.

```sh
mkdir -p results
docker run --rm --network none --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all --cpus 4 --memory 16g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/weights:/weights:ro" -v "$PWD/results:/results:rw" \
  lidar2sim-nksr:spike-e403368 \
  --weights-sha256 c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955 \
  --output /results/smoke-1
```

После двух успешных запусков сохранить image ID/digest, `pip freeze`, версии
external openvdb/eigen из сборки, команды и логи. Переводить H42 в Done только
после воспроизводимой сборки и offline inference. Непустая сфера проверяет runtime,
но не качество комнаты или физику. Следующий шаг после H42 — H43/H44/H45.

Первоначальный запуск с `--gpus all` отклонён NVIDIA hook; явный `--runtime=nvidia`
исправил доступ без изменения хоста. GPU tensor probe прошёл. Зависимости Dockerfile
пока не зафиксированы полностью: этот файл — совместимость, не финальный release lock.
