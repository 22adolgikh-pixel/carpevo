# Деплой CPS на свой VPS (Hostkey и т.п.)

Один раз на чистом Ubuntu/Debian-сервере:

```bash
# 1. Docker (если ещё не стоит)
curl -fsSL https://get.docker.com | sh

# 2. Код студии
git clone --branch cps-v5 --single-branch https://github.com/22adolgikh-pixel/carpevo.git
cd carpevo/cps/deploy

# 3. Секреты
cp .env.example .env
nano .env   # вписать CPS_GITHUB_TOKEN и CPS_PASSWORD (CPS_SCANS_ZIP_ID уже вписан)

# 4. Домен (необязательно, но с ним будет https)
nano Caddyfile   # заменить :80 на свой домен, если он есть и уже указывает A-записью на этот VPS

# 5. Запуск
docker compose up -d --build
```

Открыть `http://IP-адрес-VPS` (или `https://ваш-домен`, если настроен в Caddyfile) — попросит пароль (`CPS_PASSWORD`).

## Обновление кода после правок на Mac
На Mac, в `~/Desktop/cps/`, закоммитить и запушить в `cps-v5` (или попросить это сделать Claude), затем на VPS:
```bash
cd carpevo && git pull && cd cps/deploy && docker compose up -d --build
```

## Проверка бэкапа
`curl http://localhost:8000/api/backup` (с VPS) или открыть `/api/backup` через домен — покажет `last_ok`/`last_error`. Первая ручная проверка: `curl -X POST http://localhost:8000/api/backup`.

## Если что-то не поднялось
```bash
docker compose logs -f cps
```
Частые причины: пустой `CPS_GITHUB_TOKEN` (бэкап и восстановление работы не будут ходить в GitHub, сама студия при этом всё равно откроется), неверный `CPS_SCANS_ZIP_ID` или файл на Drive закрыт от чужого доступа (проверить: открыть `https://drive.google.com/uc?export=download&id=<ID>` в браузере в приватном окне — должен начаться скачивание zip, а не запрос входа).
