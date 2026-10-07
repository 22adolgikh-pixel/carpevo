#!/bin/sh
# Пересборка прототипа из файлов репозитория. Запуск из корня репозитория: sh site/proto/rebuild.sh "прототип v14" /путь/к/результату.html
set -e
T=$(mktemp -d)
gunzip -c site/proto/inputs/site_data.json.gz > "$T/data.json"
gunzip -c site/proto/inputs/schemes_db_full.json.gz > "$T/db_full.json"
THUMBS_DIRS=site/model/sources/museum_photos/thumbs python3 site/proto/build_proto.py "$T/data.json" "$T/db_full.json" "${2:-/tmp/carpet_dna_proto.html}" "${1:-прототип}"
