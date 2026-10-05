# Carpet Pattern Studio — контейнер для хостинга (Render / Fly.io / Cloud Run и т.п.)
#
# Диск контейнера эфемерный: при каждом старте restore.py подтягивает
# последний бэкап данных из ветки cps-data репозитория carpevo и сканы —
# из zip-архива на Google Drive (см. переменные окружения ниже).
FROM python:3.11-slim

# git — нужен backup.py (пушит правки в cps-data) и restore.py (тянет их обратно);
# libgl1/libglib2.0-0 — рантайм-зависимости opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# v10.15: torch (только процессор) — для свёрточной сети клеток (cnn.py); без него студия работает как раньше, на порогах и маленькой модели
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

ENV PORT=8000
EXPOSE 8000

# Переменные окружения задаются на хостинге (см. render.yaml / README):
#   CPS_GITHUB_TOKEN   — fine-grained токен, Contents: Read and write на carpevo
#   CPS_SCANS_ZIP_ID   — id zip-архива сканов на Google Drive (доступ "у кого есть ссылка")
#   CPS_PASSWORD       — пароль входа (иначе студия открыта всем, у кого есть URL)
#   CPS_BACKUP_MIN     — как часто пушить правки в GitHub (на хостинге лучше 2, не 10)
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT}"]
