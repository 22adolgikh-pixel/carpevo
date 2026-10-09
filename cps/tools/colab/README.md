# Сеть для клеток схем в Google Colab (v10.15)

Запуск (≈15–25 минут на бесплатной видеокарте T4):
1. colab.research.google.com → «Создать блокнот».
2. Меню «Среда выполнения» → «Сменить тип среды выполнения» → **T4 GPU** → «Сохранить».
3. Вставить в первую ячейку и запустить (▶):

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%run /content/carpevo/cps/tools/colab/cnn_colab.py
```

Скрипт сам скачает данные из ветки `cps-data` (берёт то, что ушло в бэкап — свежее лучше запускать после очередного автобэкапа), учит сеть по 3 фолдам листов, сравнивает с порогами и нынешней маленькой моделью и в конце печатает блок «ИТОГ». Просит доступ к Google Drive, чтобы положить `summary.json` и `net_all.pt` в `carpet-dna/cps_cnn/` (можно отказаться — тогда файлы в `/content/cnn_result/`).

Параметры (необязательно, перед `%run`): `%env CPS_ITERS=4000` — шагов на фолд (по умолчанию 4000; 8000 — дольше, чуть точнее).
Токены и пароли в скрипте не нужны: репозиторий публичный.

# Книга с Drive → страницы (текст + превью) — book_pages_colab.py

Для новых источников (сканы PDF на Google Drive). Видеокарта не нужна; ≈2–4 с на страницу с OCR.

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%env BOOK=Caucasian_carpets
%run /content/carpevo/cps/tools/colab/book_pages_colab.py
```

Ищет PDF по части имени на всём Drive, берёт текстовый слой, если он есть, иначе распознаёт (tesseract, eng+deu+fra+rus),
и кладёт в `carpet-dna/output/raw_pages_<имя>/` файлы `page_NNN.json` (текст) и `page_NNN.jpg` (превью 110 dpi) + `_summary.json`.
Прерванный запуск продолжает с места остановки.

# Зарисовки узоров из книги → CPS — drawings_colab.py

Для Gans-Ruedin «Caucasian Carpets»: вырезает рисунки пером со страниц-комментариев (300 dpi), убирая текст по рамкам слов
из текстового слоя. Результат: `carpet-dna/output/gans_ruedin_drawings/` (PNG, manifest.json с плотностью узла ковра,
_contact_sheet.jpg — обзор всех вырезок, drawings_for_cps.zip с доступом по ссылке; id архива печатается в ИТОГ).

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%env BOOK=Caucasian_carpets
%run /content/carpevo/cps/tools/colab/drawings_colab.py
```

# Фото ковров azerbaijanrugs → каталог и школы — azrugs_school.py

Папка `carpet-dna/photos/azerbaijan_rugs_guide` (фото + одноимённые .txt). Нужна видеокарта T4 (15–40 мин).

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%run /content/carpevo/cps/tools/colab/azrugs_school.py
```

Результат в `carpet-dna/photos/azrugs_school/`: catalog.csv (школа, век, размер, описание), report.json, predictions.csv, disagreements.csv, model.npz (веса для CPS), features.npz (кэш; повторный запуск быстрый).
Проверка групповая (почти одинаковые фото не попадают одновременно в обучение и проверку).

# Открытые музеи (V&A + Метрополитен) → фото с музейными подписями — museum_open_colab.py

Видеокарта не нужна, 10–30 мин. Результат: `carpet-dna/photos/museum_open/` (img/, catalog.csv). Met — только CC0; V&A — некоммерческая лицензия.

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%run /content/carpevo/cps/tools/colab/museum_open_colab.py
```

# Своя модель для ковров, v1 — carpet_model_colab.py

DINOv2 ViT-S/14, доучиваются последние 4 блока: «искусственные ковры» из принятых схем CPS ↔ чистые схемы + самообучение на всех фото ковров.
Проверка до/после: узнавание узора на отложенных 15% схем (top-1/top-5), школы на azerbaijanrugs, контактные листы «музейное фото → 5 похожих узоров».
T4 GPU, ~1–2 ч. Результат: `carpet-dna/carpet_model/` (report.json, motif_suggestions.csv, sheets/, synth_examples.jpg, carpet_vits14_v1.pt).

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%run /content/carpevo/cps/tools/colab/carpet_model_colab.py
```
Быстрая проверка без GPU: `%env CM_TEST=1`. Больше шагов: `%env CM_STEPS=6000`.
