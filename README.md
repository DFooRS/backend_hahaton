# Backend: прогноз вероятности банкротства

FastAPI-обёртка над переданной ML-моделью `tabular-baseline-v1`. Модель загружается один раз при запуске. Переобучения при запросах нет. `word_count` в этой версии моделью не используется. Число `probability` — предсказанная вероятность положительного класса от 0 до 1; отдельная калибровка процентов не проводилась. Публичный ROC-AUC первой версии — 0.65005.

## Запуск

Нужен Python 3.12. Из корня репозитория:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Документация API: `http://localhost:8000/docs`. `GET /health` возвращает `model_ready` и `lookup_ready`. До создания базы поиск по ID отвечает `503`, а прогноз по признакам уже работает.

Для Docker:

```bash
docker build -t bankruptcy-backend .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data:ro" bankruptcy-backend
```

## База для поиска по ID

Исходные CSV и готовая SQLite-база не коммитятся. Если CSV лежат в каталоге, выполните:

```bash
.venv/bin/python ml_backend/build_lookup.py --data-dir /path/to/csv --output data/lookup.sqlite
```

Для исходного зашифрованного ZIP можно собрать базу напрямую, без распаковки CSV:

```bash
.venv/bin/python ml_backend/build_lookup.py --data-zip /path/to/Hackathon-task.zip --output data/lookup.sqlite
```

Скрипт спросит пароль в терминале. Дождитесь завершения и перезапустите API. Сборка требует около 2 ГБ для базы. `TRAIN.csv` и `TEST.csv` индексируются вместе; колонка `target` в базу не попадает. Если сборка прервалась, удалите только `data/lookup.building.sqlite` перед повтором. Путь к готовой базе можно изменить переменной `LOOKUP_PATH`.

## Контракт для фронтенда

### По ID

```http
POST /api/v1/predict/by-id
Content-Type: application/json

{"index": "194357"}
```

```json
{"index":"194357","probability":0.00887221822283649}
```

`index` — строка или целое. Неизвестный ID: `404`. База ещё не готова: `503`.

### По признакам нового объекта

```http
POST /api/v1/predict/records
Content-Type: application/json

{"records":[{"index":"new-1","feature_0":0,"...":"...","feature_1366":null}]}
```

Здесь `"..."` — только сокращение для README: фактически нужны **все 1367** полей `feature_0` ... `feature_1366`, каждое число или `null`. Поля `index` и `word_count` необязательны. `target` и любые другие поля запрещены. Готовый полный пример: `ml_backend/example_request.json`; при вызове оберните его массив в `{"records": ...}`. Ответ:

```json
{"predictions":[{"index":"new-1","probability":0.12}]}
```

### Загрузка CSV

```http
POST /api/v1/predict/csv
Content-Type: multipart/form-data

file=<CSV-файл>
```

CSV в UTF-8: заголовок со всеми 1367 `feature_*`, дополнительно допускаются `index` и `word_count`. Пустое значение признака превращается в `null`. До 1000 строк и 32 МиБ за запрос. Ответ совпадает с `/records`: массив `predictions`. Фронт может показать результаты или сформировать CSV для скачивания. Неверный ввод: `422`; слишком большой файл: `413`.

Число для шкалы фронта: `probability * 100`. Не задавайте категории «низкая/высокая» без согласованных порогов с ML-специалистом.

## Настройки

- `CORS_ORIGINS` — список разрешённых origin через запятую. По умолчанию разрешён `https://web.team-frontend.hack.week-infra.ru`.
- `LOOKUP_PATH` — абсолютный или относительный путь к SQLite-базе.
- `MODEL_THREADS` — число потоков CatBoost на вызов, по умолчанию `2`.

Артефакты модели и исходный интерфейс ML находятся в `ml_backend/`. При обновлении модели заменяйте код, артефакты и зависимости целым комплектом.
