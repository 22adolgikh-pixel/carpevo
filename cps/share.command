#!/bin/bash
# Двойной клик на Mac: запускает студию и даёт на неё ссылку через бесплатный туннель
# Cloudflare (адрес вида https://….trycloudflare.com, новый при каждом запуске).
# Вход по ссылке — по паролю (спросит при первом запуске, хранится в .env).
cd "$(dirname "$0")"
CPS_SHARE=1 exec bash ./start.command
