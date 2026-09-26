#!/bin/bash
# Двойной клик на Mac: находит рабочий Python, ставит зависимости в изолированное
# окружение (.venv) и открывает студию в браузере.
cd "$(dirname "$0")"

find_python() {
  local candidates=(python3.13 python3.12 python3.11 python3.10 python3)
  local dirs=(/opt/homebrew/bin /usr/local/bin /usr/bin)
  for d in "${dirs[@]}"; do
    for name in "${candidates[@]}"; do
      local p="$d/$name"
      [ -x "$p" ] || continue
      case "$p" in */CommandLineTools/*) continue ;; esac
      if "$p" -c "import venv, ensurepip" >/dev/null 2>&1; then
        echo "$p"; return 0
      fi
    done
  done
  # запасной вариант: что бы ни было первым в PATH, кроме заглушки CLT
  local p
  p="$(command -v python3 2>/dev/null)"
  if [ -n "$p" ]; then
    case "$p" in
      */CommandLineTools/*) ;;
      *) if "$p" -c "import venv, ensurepip" >/dev/null 2>&1; then echo "$p"; return 0; fi ;;
    esac
  fi
  return 1
}

echo "Ищу рабочий Python 3…"
PY="$(find_python)"
if [ -z "$PY" ]; then
  echo
  echo "⚠️  Не нашёл рабочий Python 3."
  echo "По умолчанию на Mac иногда стоит только заглушка от Xcode Command Line Tools —"
  echo "она не умеет ставить пакеты (именно это и произошло: PermissionError / No module named uvicorn)."
  echo
  echo "Установите настоящий Python 3, любым способом:"
  echo "  • с сайта https://www.python.org/downloads/macos/  (обычная установка .pkg)"
  echo "  • или через Homebrew:  brew install python"
  echo
  echo "После установки запустите этот файл ещё раз."
  read -n1 -r -p "Нажмите любую клавишу для выхода..."
  exit 1
fi
echo "Использую: $PY ($("$PY" --version 2>&1))"

if [ ! -d .venv ]; then
  echo "Создаю изолированное окружение (.venv)…"
  "$PY" -m venv .venv || { echo "⚠️ Не удалось создать .venv"; read -n1 -r -p "Нажмите любую клавишу..."; exit 1; }
fi

VPY=".venv/bin/python3"
echo "Устанавливаю зависимости (один раз, может занять минуту)…"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q -r requirements.txt || { echo "⚠️ Не удалось поставить зависимости — см. ошибку выше."; read -n1 -r -p "Нажмите любую клавишу..."; exit 1; }

# Папка со сканами: по умолчанию ./scans. Для Google Drive раскомментируйте и поправьте путь:
# export CPS_SCANS="$HOME/Library/CloudStorage/GoogleDrive-ИМЯ@gmail.com/My Drive/scans"

# ---- порт 8000 уже занят? ----
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  if curl -s -m 3 http://localhost:8000/api/config 2>/dev/null | grep -q '"scans"'; then
    if [ "$CPS_SHARE" != "1" ]; then
      echo "Студия уже запущена (в другом окне) — просто открываю её."
      open http://localhost:8000
      exit 0
    fi
    # для ссылки нужен сервер с паролем — перезапускаем уже открытую студию
    echo "Студия уже запущена в другом окне — перезапускаю её с паролем для ссылки…"
    lsof -nP -tiTCP:8000 -sTCP:LISTEN | xargs kill 2>/dev/null
    for i in $(seq 1 10); do lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
  else
    echo
    echo "⚠️ Порт 8000 занят другой программой:"
    lsof -nP -iTCP:8000 -sTCP:LISTEN | tail -n +2
    echo "Закройте её и запустите ещё раз."
    read -n1 -r -p "Нажмите любую клавишу для выхода..."
    exit 1
  fi
fi

RUN=()
if [ "$CPS_SHARE" = "1" ]; then
  # ---- доступ по ссылке (share.command): пароль + туннель Cloudflare ----
  [ -f .env ] && set -a && . ./.env && set +a
  if [ -z "$CPS_PASSWORD" ]; then
    echo
    read -r -s -p "Придумайте пароль для входа по ссылке (сохранится в .env): " CPS_PASSWORD; echo
    [ -z "$CPS_PASSWORD" ] && { echo "⚠️ Пароль пустой — отмена."; read -n1 -r -p "Нажмите любую клавишу..."; exit 1; }
    printf 'CPS_PASSWORD=%q\n' "$CPS_PASSWORD" >> .env; chmod 600 .env
  fi
  export CPS_PASSWORD

  CF="$(command -v cloudflared || true)"
  if [ -z "$CF" ]; then
    CF=".bin/cloudflared"
    if [ ! -x "$CF" ]; then
      case "$(uname -m)" in arm64) A=arm64 ;; *) A=amd64 ;; esac
      echo "Скачиваю cloudflared (один раз)…"
      mkdir -p .bin
      curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-$A.tgz" | tar -xz -C .bin \
        && chmod +x "$CF" || { echo "⚠️ Не удалось скачать cloudflared. Можно поставить вручную: brew install cloudflared"; read -n1 -r -p "Нажмите любую клавишу..."; exit 1; }
    fi
  fi

  echo "Открываю туннель…"
  "$CF" tunnel --no-autoupdate --url http://localhost:8000 > .tunnel.log 2>&1 &
  TPID=$!
  trap 'kill $TPID 2>/dev/null' EXIT
  URL=""
  for i in $(seq 1 40); do
    URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' .tunnel.log | head -1)"
    [ -n "$URL" ] && break
    kill -0 $TPID 2>/dev/null || break
    sleep 1
  done
  if [ -z "$URL" ]; then
    echo "⚠️ Туннель не открылся. Последние строки лога:"; tail -5 .tunnel.log
    read -n1 -r -p "Нажмите любую клавишу..."; exit 1
  fi
  printf '%s' "$URL" | pbcopy 2>/dev/null
  echo
  echo "============================================================"
  echo "  Ссылка на студию (уже скопирована):"
  echo "     $URL"
  echo "  Вход — по паролю из .env. На этом Mac пароль не нужен."
  echo "  Ссылка работает, пока открыто это окно и Mac не спит."
  echo "============================================================"
  echo
  RUN=(caffeinate -i)   # не давать Mac уснуть, пока идёт раздача
fi

(sleep 2; open http://localhost:8000) &
echo "Запускаю сервер на http://localhost:8000 (это окно должно остаться открытым)…"
"${RUN[@]}" "$VPY" -m uvicorn backend:app --port 8000
read -n1 -r -p "Сервер остановлен. Нажмите любую клавишу для выхода..."
