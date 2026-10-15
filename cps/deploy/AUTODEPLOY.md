# Автодеплой CPS (GitHub Actions → VPS)

После каждого пуша в `cps-v5` GitHub прогоняет тесты («CPS tests»), и если они зелёные — workflow «CPS deploy» заходит на сервер
по SSH, делает `git pull --ff-only` и `docker compose -f docker-compose.existing-nginx.yml up -d --build`, затем ждёт, пока студия
ответит (200 или 401 на `/api/health`). Если не ответила — workflow красный, в логе последние строки контейнера.

## Один раз настроить

1. **Ключ только для деплоя** (на любом компьютере):
   `ssh-keygen -t ed25519 -f cps_deploy -N "" -C cps-deploy`
2. **На сервере** добавьте публичную часть тому пользователю, у которого лежит `~/carpevo` и который состоит в группе `docker`
   (ограничим ключ: он сможет выполнить только вход, без проброса портов):
   `echo 'no-port-forwarding,no-agent-forwarding,no-X11-forwarding '"$(cat cps_deploy.pub)" >> ~/.ssh/authorized_keys`
3. **В GitHub** (репозиторий `carpevo` → Settings → Secrets and variables → Actions → Secrets) создайте:
   - `CPS_SSH_HOST` — адрес сервера,
   - `CPS_SSH_USER` — пользователь (например, `root`, если репозиторий в `/root/carpevo`),
   - `CPS_SSH_KEY` — **приватный** ключ (весь файл `cps_deploy`, с первой до последней строки).
   Если папка не `~/carpevo`: Variables → `CPS_SERVER_DIR`.
4. Проверка: Actions → «CPS deploy» → Run workflow.

Файл `.env` на сервере (`cps/deploy/.env`) не в git и не затрагивается. `--ff-only` не перезапишет правки, сделанные на сервере вручную:
если они мешают, деплой остановится с ошибкой, и их надо будет закоммитить или убрать.

## Что не входит
Перезагрузка nginx не нужна: конфиг узнаёт адрес контейнера заново (см. NGINX_EXISTING.md).
Откат: на сервере `git checkout <прежний коммит> && docker compose -f docker-compose.existing-nginx.yml up -d --build`.
