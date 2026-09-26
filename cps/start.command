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

(sleep 2; open http://localhost:8000) &
echo "Запускаю сервер на http://localhost:8000 (это окно должно остаться открытым)…"
"$VPY" -m uvicorn backend:app --port 8000
read -n1 -r -p "Сервер остановлен. Нажмите любую клавишу для выхода..."
