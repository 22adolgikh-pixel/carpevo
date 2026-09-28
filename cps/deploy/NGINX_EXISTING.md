# Подключение CPS к уже существующему nginx+certbot (upriver-server), а не к
# своему Caddy. Ситуация 2026-09-29: на VPS jeyran.az уже стоит боевой
# docker-стек (nginx, certbot, n8n, postgres, ...), порты 80/443 заняты им.

## 1. Запустить CPS в сети существующего nginx
```bash
cd ~/carpevo/cps/deploy
docker compose down                       # снести неудачную попытку с Caddy
docker compose -f docker-compose.existing-nginx.yml up -d --build
```
Проверить, что контейнер `cps` виден в сети nginx:
```bash
docker exec nginx getent hosts cps        # должен вернуть внутренний IP
```

## 2. Добавить сайт в nginx (шаг A — только http, для проверки домена)
Файл `/home/vallefor/upriver-server/data/nginx/conf.d/cps.jeyran.az.conf`:
```nginx
server {
  listen 80;
  server_name cps.jeyran.az;
  location /.well-known/acme-challenge/ {
    root /var/www/certbot;
  }
  location / {
    return 404;   # временно, до получения сертификата
  }
}
```
```bash
docker exec nginx nginx -t && docker exec nginx nginx -s reload
```

## 3. Получить сертификат
```bash
docker run --rm \
  -v /home/vallefor/upriver-server/data/certbot/conf:/etc/letsencrypt \
  -v /home/vallefor/upriver-server/data/certbot/www:/var/www/certbot \
  certbot/certbot certonly --webroot -w /var/www/certbot -d cps.jeyran.az
```
Если ругается на DNS — проверить `dig +short cps.jeyran.az` указывает на `66.151.34.206` (напрямую либо через Cloudflare-проксирование, которое у вас уже и так работает для остальных доменов на этом сервере).

## 4. Дописать в тот же файл https-блок (шаг B, по образцу n8n.jeyran.az.conf)
Заменить файл `cps.jeyran.az.conf` на:
```nginx
server {
  listen 80;
  server_name cps.jeyran.az;
  location /.well-known/acme-challenge/ {
    root /var/www/certbot;
  }
  location / {
    return 301 https://$host$request_uri;
  }
}

server {
  listen 443 ssl;
  listen [::]:443 ssl;
  server_name cps.jeyran.az;
  ssl_certificate /etc/letsencrypt/live/cps.jeyran.az/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/cps.jeyran.az/privkey.pem;

  client_max_body_size 100m;   # сканы/загрузки могут быть крупными

  location / {
    proxy_pass http://cps:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_buffering off;
  }
}
```
```bash
docker exec nginx nginx -t && docker exec nginx nginx -s reload
```

Открыть `https://cps.jeyran.az` — попросит пароль (`CPS_PASSWORD` из `.env`).

## Обновление кода после правок
```bash
cd ~/carpevo && git pull
cd cps/deploy && docker compose -f docker-compose.existing-nginx.yml up -d --build
```

## Продление сертификата
Скорее всего у существующего контейнера `upriver-server-certbot` уже настроено автопродление всех доменов из `conf.d` разом (по крону/циклу внутри контейнера) — отдельно ничего делать не нужно, если новый домен подхватывается автоматически так же, как остальные. Если нет — тот же `docker run ... certbot renew` вручную раз в ~80 дней.
