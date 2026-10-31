# Исправление номеров и подписей схем Керимова т. I (сверка со сканами 2026-10-31)

- `book_map.json` — для каждой страницы скана: номер таблицы в студии → по книге, перестановки рисунков на листе, кадры к удалению (разделители 249/284/339, дубль 323).
- `legend_book.csv` — легенда, переписанная заново со сканов стр. 85–117 (табл. 1–196 + подписи 197–239), с проверенной латиницей `name_az`. Заменяет `data/legend.csv`.
- `apply_kerimov_fix.py DATA [--apply]` — переписывает meta схем по книге (с бэкапом), переносит привязки к сайту на схему с правильным номером, пишет `remap.json`.
- `patch_schemes_db.py` — переносит то же во вход сайта `site/proto/inputs/schemes_db_full.json.gz`.
- `remap.json` — старая схема → схема с тем же книжным номером (для `site/model/apply_decisions.py --remap`).

Студия (`cps/kerimov_book.py`) при нарезке листа Керимова теперь сама ставит `book_table`/`book_fig` и берёт подпись по книжному номеру; выгрузка (`/api/bundle`) отдаёт книжные номера.

## На VPS (один раз)
```
cd ~/carpevo && git pull && cd cps/deploy && docker compose -f docker-compose.existing-nginx.yml up -d --build
docker exec deploy-cps-1 python3 tools/kerimov_fix/apply_kerimov_fix.py /app/data           # сухой прогон
docker exec deploy-cps-1 python3 tools/kerimov_fix/apply_kerimov_fix.py /app/data --apply
```
