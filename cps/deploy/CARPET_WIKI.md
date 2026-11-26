# Переезд CPS на carpet.wiki

Цель:
- `carpet.wiki` (и `www.`) — лендинг-заглушка (`cps/deploy/site/index.html`, контейнер `site`);
- `studio.carpet.wiki` — CPS (тот же контейнер `cps`, данные и сканы остаются в томах `cps_data` / `cps_scans`);
- `cps.jeyran.az` — постоянный редирект на `studio.carpet.wiki` с сохранением пути.

В коде студии домен нигде не зашит, менять ничего не нужно. Cookie входа привязана к хосту, поэтому на новом адресе пароль (`CPS_PASSWORD`) нужно один раз ввести заново.

Шаги 1–2 делает владелец домена (Cloudflare), шаги 3–6 — тот, у кого есть SSH на VPS 66.151.34.206. Все серверные команды выполняются на VPS.

## 1. Cloudflare: добавить домен

1. Cloudflare → Add a domain → `carpet.wiki` → тариф Free.
2. Cloudflare покажет два nameserver'а (вида `xxx.ns.cloudflare.com`). Прописать их у регистратора carpet.wiki вместо текущих. Активация занимает от нескольких минут до нескольких часов: Cloudflare пришлёт письмо, статус домена станет Active.

## 2. Cloudflare: DNS-записи

DNS → Records → Add record, все записи типа **A** на `66.151.34.206`:

| Name | Content | Proxy status |
|---|---|---|
| `@` (carpet.wiki) | 66.151.34.206 | DNS only (серое облако) |
| `www` | 66.151.34.206 | DNS only |
| `studio` | 66.151.34.206 | DNS only |

Пока выпускается сертификат, оставьте **DNS only**. Когда всё заработает, облако можно включить (Proxied), предварительно поставив SSL/TLS → Overview → **Full (strict)**. С режимом Flexible будет бесконечный редирект.

Проверка (с любого компьютера): `dig +short carpet.wiki` и `dig +short studio.carpet.wiki` должны вернуть `66.151.34.206`.

## 3. VPS: обновить код и поднять контейнер лендинга

Если автодеплой уже прошёл после коммита с этим файлом, контейнер `site` поднят и этот шаг можно пропустить (проверка ниже).

```bash
cd ~/carpevo && git pull
cd cps/deploy && docker compose -f docker-compose.existing-nginx.yml up -d --build
docker exec nginx getent hosts carpetwiki-site     # должен вернуть внутренний IP
docker exec nginx getent hosts cps                 # и этот тоже
```

## 4. VPS: временный http-конфиг и сертификат

Сертификат один на три имени. Сначала нужен конфиг только с портом 80 (полный конфиг ссылается на сертификат, которого ещё нет, и `nginx -t` его не пропустит).

```bash
C=/home/vallefor/upriver-server/data/nginx/conf.d
cat > $C/carpet.wiki.conf <<'EOF'
server {
  listen 80;
  server_name carpet.wiki www.carpet.wiki studio.carpet.wiki;
  location /.well-known/acme-challenge/ { root /var/www/certbot; }
  location / { return 404; }
}
EOF
docker exec nginx nginx -t && docker exec nginx nginx -s reload

docker run --rm \
  -v /home/vallefor/upriver-server/data/certbot/conf:/etc/letsencrypt \
  -v /home/vallefor/upriver-server/data/certbot/www:/var/www/certbot \
  certbot/certbot certonly --webroot -w /var/www/certbot \
  --cert-name carpet.wiki -d carpet.wiki -d www.carpet.wiki -d studio.carpet.wiki
```

Ошибка вида `DNS problem` или `Timeout during connect` значит, что DNS ещё не обновился (шаг 2) или включено оранжевое облако.

## 5. VPS: боевые конфиги

```bash
C=/home/vallefor/upriver-server/data/nginx/conf.d
cp ~/carpevo/cps/deploy/nginx/carpet.wiki.conf        $C/carpet.wiki.conf
cp ~/carpevo/cps/deploy/nginx/studio.carpet.wiki.conf $C/studio.carpet.wiki.conf
docker exec nginx nginx -t && docker exec nginx nginx -s reload
```

Проверка:
- `https://carpet.wiki` — лендинг; `https://www.carpet.wiki` уводит на `carpet.wiki`;
- `https://studio.carpet.wiki` — студия, спросит пароль; `/m` и `/api/health` работают;
- `https://studio.carpet.wiki/api/vlm/status` — `key: true`.

Старый `cps.jeyran.az` на этом шаге продолжает работать как раньше.

## 6. VPS: редирект со старого адреса

Только после того, как шаг 5 проверен:

```bash
C=/home/vallefor/upriver-server/data/nginx/conf.d
cp $C/cps.jeyran.az.conf ~/cps.jeyran.az.conf.bak      # на случай отката
cp ~/carpevo/cps/deploy/nginx/cps.jeyran.az.redirect.conf $C/cps.jeyran.az.conf
docker exec nginx nginx -t && docker exec nginx nginx -s reload
```

Откат: вернуть `~/cps.jeyran.az.conf.bak` на место и снова `nginx -s reload`.

## Продление сертификата

Как и у `cps.jeyran.az`: если контейнер `upriver-server-certbot` продлевает всё из `/etc/letsencrypt/live`, новый сертификат `carpet.wiki` продлится сам. Если нет — раз в ~80 дней `docker run ... certbot/certbot renew` с теми же томами, затем `nginx -s reload`.

## Новые поддомены потом

Для `wiki.`, `azerbaijan.`, `shop.` и т.д.: A-запись в Cloudflare, затем `certbot certonly ... --cert-name carpet.wiki --expand` со всеми именами (старые + новое) и отдельный `server`-блок в nginx. Если поддоменов станет много, удобнее перейти на wildcard `*.carpet.wiki`: Cloudflare Origin Certificate (15 лет, при включённом проксировании) или certbot с DNS-проверкой через API Cloudflare.
