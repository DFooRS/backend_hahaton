# Backend: прогноз вероятности банкротства

FastAPI-обёртка над ML-моделью `tabular-baseline-v1`. Модель загружается один раз при запуске API. Поиск признаков по `index` выполняется в PostgreSQL. Публичное S3-совместимое хранилище используется для скачивания архива модели и исходных CSV/ZIP без ключей доступа.

Эта версия использует только 1367 табличных признаков. `word_count` предусмотрен для будущей модели и пока не влияет на прогноз. `probability` — предсказанная вероятность положительного класса от 0 до 1; отдельной калибровки процентов нет. Публичный ROC-AUC первой версии: 0.65005.

## Быстрый запуск

Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DBNAME'
.venv/bin/python -m scripts.migrate
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`DATABASE_URL` храните в окружении или Kubernetes Secret, не в Git. Документация API: `http://localhost:8000/docs`. `GET /health` сообщает `model_ready` и `lookup_ready`. Пока база не импортирована, запрос по ID отвечает `503`, а прогноз по JSON/CSV работает.

Если задан `MODEL_BUNDLE_URL`, API скачивает публичный HTTPS ZIP при запуске, извлекает `model.cbm`, `features.json`, `model_info.json` и проверяет SHA-256 модели. Подходит исходный `ml_backend_ready_v1.zip`. Без переменной используются артефакты из репозитория. При обновлении модели меняйте архив и импортируйте lookup заново, если изменились признаки.

## PostgreSQL: миграция и импорт

Миграции находятся в `migrations/`. Запускайте их перед импортом:

```bash
.venv/bin/python -m scripts.migrate
```

Команда безопасна для повторного запуска и фиксирует применённые версии в `schema_migrations`. После миграции импортируйте данные одним из способов:

```bash
export DATA_ZIP_URL='https://PUBLIC-BUCKET/path/Hackathon-task.zip'
export DATA_ZIP_PASSWORD='***'
.venv/bin/python -m scripts.import_lookup
```

Либо укажите `TRAIN_CSV_URL` и `TEST_CSV_URL` для отдельных публичных CSV. Для локальной проверки можно указать `DATA_DIR` с `TRAIN.csv` и `TEST.csv`. Импорт читает CSV порциями, сохраняет только `index` и бинарный вектор признаков `float32`; `target` и текстовые данные в PostgreSQL не попадают. Ожидается 354246 строк. Импорт идёт в одной транзакции: при ошибке изменения откатываются. Пока выполняется повторный импорт, запросы по ID могут ждать блокировку таблицы. Для базы и WAL нужен запас диска в PostgreSQL, ориентировочно от 5 ГБ; при использовании ZIP Job также скачивает архив во временный каталог контейнера.

### Kubernetes Jobs

Оба манифеста используют один и тот же образ из [Dockerfile](Dockerfile). В `k8s/migrate-job.yaml` и `k8s/import-lookup-job.yaml` замените `REPLACE_WITH_BACKEND_IMAGE` на опубликованный образ. Создайте Secret `bankruptcy-backend-secrets` с ключом `database-url`; если ZIP зашифрован, добавьте `data-zip-password`. Создайте ConfigMap `bankruptcy-backend-config` с `DATA_ZIP_URL` **или** с обоими `TRAIN_CSV_URL` и `TEST_CSV_URL`. `MODEL_BUNDLE_URL` тоже можно положить в ConfigMap.

Сначала примените `k8s/migrate-job.yaml` и дождитесь завершения, затем `k8s/import-lookup-job.yaml`. Пример:

```bash
kubectl apply -f k8s/migrate-job.yaml
kubectl wait --for=condition=complete job/bankruptcy-db-migrate --timeout=300s
kubectl apply -f k8s/import-lookup-job.yaml
kubectl wait --for=condition=complete job/bankruptcy-lookup-import --timeout=3600s
```

Передайте контейнеру API тот же `DATABASE_URL` и, если модель хранится в S3, `MODEL_BUNDLE_URL`. Бакет должен отдавать объекты по публичным HTTPS URL. Ни AWS-ключи, ни SDK для чтения не нужны.

## Контракт для фронтенда

### Прогноз по ID

```http
POST /api/v1/predict/by-id
Content-Type: application/json

{"index":"194357"}
```

```json
{"index":"194357","probability":0.00887221822283649}
```

Неизвестный ID: `404`. База ещё не готова: `503`.

### Загрузка CSV

```http
POST /api/v1/predict/csv
Content-Type: multipart/form-data

file=<CSV-файл>
```

CSV в UTF-8: все столбцы `feature_0` … `feature_1366`, дополнительно допускаются `index` и `word_count`. Пустые значения признаков превращаются в `null`. Не более 1000 строк и 32 МиБ за запрос. Ответ:

```json
{"predictions":[{"index":"new-1","probability":0.12}]}
```

Неверный ввод: `422`; слишком большой файл: `413`.

### Программный вызов по признакам

`POST /api/v1/predict/records` принимает `{"records":[<плоский объект со всеми 1367 признаками>]}` и возвращает тот же массив `predictions`. Полный образец строк: `ml_backend/example_request.json`. `target` во входе запрещён.

Фронт умножает `probability` на 100 для отображения процентов. Пороги категорий риска нужно согласовать с ML-специалистом.

## Настройки

- `DATABASE_URL` — строка подключения к PostgreSQL.
- `MODEL_BUNDLE_URL` — публичный HTTPS URL архива модели; необязателен при использовании модели из образа.
- `DATA_ZIP_URL` + `DATA_ZIP_PASSWORD` — публичный ZIP с TRAIN/TEST; только для импортирующей Job.
- `TRAIN_CSV_URL` + `TEST_CSV_URL` — альтернатива ZIP; только для импортирующей Job.
- `EXPECTED_LOOKUP_ROWS` — ожидаемое число строк, по умолчанию `354246`.
- `CORS_ORIGINS` — разрешённые origin через запятую; по умолчанию `https://web.team-frontend.hack.week-infra.ru`.
- `MODEL_THREADS` — число потоков CatBoost на вызов, по умолчанию `2`.

Тесты: `.venv/bin/python -m unittest discover -s tests -v` (для TestClient установите также `requirements-dev.txt`).
