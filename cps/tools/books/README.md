# Книги-каталоги → CPS и Carpet DNA (v11.48, 2026-11-11)

1. **Нарезка** (Colab): `carpet-dna/output/drive_audit/book_extract_colab_v5.py` (копия скрипта на Диске; `BOOK=часть имени PDF`, `STEP=all`) →
   архив `<slug>.zip` (`<slug>/images/pNNNN_n.jpg`, `items.jsonl`, `text.jsonl`, контакт-листы).
2. **Подписи и паспорта** (здесь, без моделей): положить в `BOOKS_WORK` файлы `enc_items_full.json` (items.jsonl энциклопедии одним JSON-списком)
   и `items_passport.jsonl` (Мурадов), затем
   `BOOKS_WORK=… python3 parse_enc.py && python3 build_catalog.py && python3 build_dna.py`.
   - `books_catalog.json` — каталог для студии (формат элемента как в `rug_catalog_colab.add`, `extra: true`, группы двойников «книга ↔ сайт музея» по DDK);
   - `site/model/sources/<источник>/plates.json`, `glossary_map.json`, `school_check.json` — для Carpet DNA (прототип подхватывает `plates.json` и `glossary_map.json` сам).
3. **Пакет для студии** (Colab): `books_catalog.json` → `carpet-dna/output/books/`, затем `tools/colab/books_bundle_colab.py` →
   `rug_catalog/books_bundle.zip` + id; в студии «⟳ из Диска» с этим id.
4. **OCR словника** — тот же скрипт, шаг `ocr` → `output/books/ensiklopediya_ii/ocr_pages.jsonl` (дальше: статьи словника → описания понятий).
