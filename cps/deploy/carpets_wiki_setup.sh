#!/usr/bin/env bash
# Подключение carpets.wiki (лендинг) и studio.carpets.wiki (CPS) к существующему nginx на VPS.
#
# Запуск на VPS (под root, как обычный ручной деплой):
#   bash carpets_wiki_setup.sh              # шаги 1–5: проверка DNS, сертификат, конфиги
#   bash carpets_wiki_setup.sh --redirect   # шаг 6: cps.jeyran.az → studio.carpets.wiki (когда новое проверено)
#
# Скрипт можно запускать повторно: готовый сертификат не перевыпускается, конфиги перед заменой
# сохраняются в ~/nginx-backup-<время>/, при ошибке nginx -t всё возвращается как было.
set -euo pipefail

IP=66.151.34.206
REPO=~/carpevo
UPR=/home/vallefor/upriver-server/data
C=$UPR/nginx/conf.d
CB=$UPR/certbot
NAMES=(carpets.wiki www.carpets.wiki studio.carpets.wiki)
BK=~/nginx-backup-$(date +%Y%m%d-%H%M%S)

say(){ printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
die(){ printf '\n\033[1;31mОШИБКА: %s\033[0m\n' "$*"; exit 1; }
ok(){ printf '\033[1;32mOK\033[0m %s\n' "$*"; }

[ -d "$C" ] || die "нет папки $C — скрипт запущен не на том сервере?"
docker ps --format '{{.Names}}' | grep -qx nginx || die "не найден контейнер nginx"

reload_or_restore(){
  if docker exec nginx nginx -t; then
    docker exec nginx nginx -s reload; ok "nginx перечитал конфиги"
  else
    say "nginx -t не прошёл — возвращаю прежние конфиги из $BK"
    for f in "$@"; do
      if [ -f "$BK/$f" ]; then cp "$BK/$f" "$C/$f"; else rm -f "$C/$f"; fi
    done
    docker exec nginx nginx -t && docker exec nginx nginx -s reload
    die "конфиг не принят nginx, изменения откатаны"
  fi
}
backup(){ mkdir -p "$BK"; for f in "$@"; do [ -f "$C/$f" ] && cp "$C/$f" "$BK/$f" || true; done; }

# ---------------------------------------------------------------- шаг 6 отдельно
if [ "${1:-}" = "--redirect" ]; then
  say "Шаг 6: редирект cps.jeyran.az → studio.carpets.wiki"
  code=$(curl -sk -o /dev/null -w '%{http_code}' --resolve studio.carpets.wiki:443:127.0.0.1 https://studio.carpets.wiki/api/health || true)
  [ "$code" = "200" ] || [ "$code" = "401" ] || die "studio.carpets.wiki ещё не отвечает (код $code) — сначала запустите скрипт без --redirect"
  backup cps.jeyran.az.conf
  cp "$REPO/cps/deploy/nginx/cps.jeyran.az.redirect.conf" "$C/cps.jeyran.az.conf"
  reload_or_restore cps.jeyran.az.conf
  ok "старый конфиг сохранён: $BK/cps.jeyran.az.conf (откат: скопировать обратно и docker exec nginx nginx -s reload)"
  exit 0
fi

# ---------------------------------------------------------------- 1. DNS
say "1. DNS"
for n in "${NAMES[@]}"; do
  got=$(getent ahostsv4 "$n" | awk '{print $1; exit}' || true)
  if [ "$got" = "$IP" ]; then ok "$n → $got"
  elif [ -z "$got" ]; then die "$n пока не резолвится. Проверьте A-запись в Cloudflare и NS у регистратора, подождите и запустите снова."
  else die "$n → $got, а нужно $IP. Если это адрес Cloudflare — переключите запись в «DNS only» (серое облако)."
  fi
done

# ---------------------------------------------------------------- 2. код и контейнер лендинга
say "2. Код и контейнер лендинга"
cd "$REPO" && git pull -q --ff-only origin cps-v5 && echo "код: $(git log --oneline -1)"
[ -f "$REPO/cps/deploy/nginx/carpets.wiki.conf" ] || die "в $REPO нет cps/deploy/nginx/carpets.wiki.conf — не та ветка?"
cd "$REPO/cps/deploy"
docker compose -f docker-compose.existing-nginx.yml up -d site
docker exec nginx getent hosts carpetwiki-site >/dev/null || die "nginx не видит контейнер лендинга carpetwiki-site"
docker exec nginx getent hosts cps >/dev/null || die "nginx не видит контейнер cps"
ok "nginx видит carpetwiki-site и cps"

# ---------------------------------------------------------------- 3. сертификат
say "3. Сертификат Let's Encrypt (carpets.wiki, www, studio)"
if [ -f "$CB/conf/live/carpets.wiki/fullchain.pem" ]; then
  ok "сертификат уже есть, пропускаю"
else
  backup carpets.wiki.conf studio.carpets.wiki.conf
  rm -f "$C/studio.carpets.wiki.conf"
  cat > "$C/carpets.wiki.conf" <<'EOF'
server {
  listen 80;
  server_name carpets.wiki www.carpets.wiki studio.carpets.wiki;
  location /.well-known/acme-challenge/ { root /var/www/certbot; }
  location / { return 404; }
}
EOF
  reload_or_restore carpets.wiki.conf studio.carpets.wiki.conf
  docker run --rm \
    -v "$CB/conf:/etc/letsencrypt" \
    -v "$CB/www:/var/www/certbot" \
    certbot/certbot certonly --webroot -w /var/www/certbot \
    --cert-name carpets.wiki -d carpets.wiki -d www.carpets.wiki -d studio.carpets.wiki \
    --non-interactive --agree-tos --register-unsafely-without-email \
    || die "certbot не выпустил сертификат (см. сообщение выше). Обычно это DNS или оранжевое облако."
  ok "сертификат выпущен"
fi

# ---------------------------------------------------------------- 4. боевые конфиги
say "4. Конфиги nginx"
backup carpets.wiki.conf studio.carpets.wiki.conf
cp "$REPO/cps/deploy/nginx/carpets.wiki.conf"        "$C/carpets.wiki.conf"
cp "$REPO/cps/deploy/nginx/studio.carpets.wiki.conf" "$C/studio.carpets.wiki.conf"
reload_or_restore carpets.wiki.conf studio.carpets.wiki.conf

# ---------------------------------------------------------------- 5. проверка
say "5. Проверка"
chk(){ curl -sk -o /dev/null -w '%{http_code}' --resolve "$1:443:127.0.0.1" "https://$1$2" || echo 000; }
a=$(chk carpets.wiki /); b=$(chk www.carpets.wiki /); s=$(chk studio.carpets.wiki /api/health)
echo "carpets.wiki/                 → $a (ждём 200)"
echo "www.carpets.wiki/             → $b (ждём 301)"
echo "studio.carpets.wiki/api/health→ $s (ждём 200 или 401)"
if [ "$a" = "200" ] && { [ "$s" = "200" ] || [ "$s" = "401" ]; }; then
  ok "всё работает. Откройте https://carpets.wiki и https://studio.carpets.wiki"
  echo "Когда студия проверена в браузере, включите редирект со старого адреса:  bash $0 --redirect"
else
  echo "Что-то не так: docker logs --tail 30 nginx ; бэкап конфигов: $BK"; exit 1
fi
