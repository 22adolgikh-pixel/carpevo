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

# Каталог ковров («паспорта») + двойники — rug_catalog_colab.py

Собирает с Диска все фото ковров: Керимов т. I–III (страницы + подписи из raw_pages_v3), photos/full (Wikimedia/Met/AIC/Cleveland),
museum_open (V&A/Met), azerbaijan_rugs_guide. Паспорт: названия, источник, музей/номер, дата, техника, фрагмент, школа + почему.
Двойники (один и тот же ковёр в разных источниках): DINOv2 → кандидаты → проверка точками SIFT + RANSAC. T4, 40–70 мин.
Результат: `carpet-dna/rug_catalog/` (catalog.json, groups.json, dup_sheets/*.jpg) и rugs_bundle.zip с доступом по ссылке — студия забирает его кнопкой «⟳ из Диска».

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%run /content/carpevo/cps/tools/colab/rug_catalog_colab.py
```
В конце Colab попросит разрешение на Google Drive (чтобы открыть доступ к пакету по ссылке) и напечатает «id пакета для студии».

# Отпечатки фото для поиска двойников — twin_embed_colab.py (v11.86)

T4 GPU, ≈5–10 минут. Отпечатки DINOv2 (ViT-B/14) всех фото каталога: фото целиком и центр, в 4 поворотах; у каждого ковра 12 ближайших.
Проверяет себя на парах, решённых человеком (`tools/twins/photos.json`): на каком месте стоит настоящий двойник, какое сходство у «тот же» и «разные».

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%run /content/carpevo/cps/tools/colab/twin_embed_colab.py
```

Итог — `carpet-dna/models/twin_nn_v1.json` на Диске. Загрузить в студии: «Ковры» → строка «двойники» → «⬆ отпечатки» (сервер сразу проверит новых кандидатов).


## Сеть углов v4 — `corner_refine_colab.py` (19.11)
1. В репозитории обновлены метки: `cps/tools/corners/labels.json` — 1 201 ковёр с углами, подтверждёнными людьми (v3 училась на 610).
2. Colab → среда выполнения GPU (T4) → клонировать репозиторий в `/content/carpevo` (ветка `cps-v5`) → запустить скрипт (25–40 мин). Фото берутся из пакетов `carpet-dna/rug_catalog/*bundle*.zip` на Диске.
3. В конце строка `ИТОГ:` — прислать Claude. Главные цифры: `worst_corner_le_1pct_after` (доля ковров, где ни один угол не ошибся больше чем на 1 %) и `fringe_out_after` (сколько раз угол ушёл на бахрому).
4. Скачать с Диска `carpet-dna/models/corner_refine_v4.pt` и `.json` → студия: «Ковры» → «＋ Загрузить» → «Веса сети углов» → выбрать оба файла. Сервер проверит и подключит; старые углы сети пересчитаются при показе.

# Худшие углы: человек против сети — corners_worst_colab.py (v12.00)

Видеокарта не нужна, 3–6 минут. Одна ячейка:

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%run /content/carpevo/cps/tools/colab/corners_worst_colab.py
```

Результат в `carpet-dna/models/`: `corners_worst_v1.jpg` — 36 худших ковров (слева целиком, справа крупно худший угол; зелёный — человек, красный — сеть; FRINGE — угол сети вынесен наружу по длинной оси, то есть бахрома принята за край) и `corners_worst_v1.json` — по всем коврам. Claude читает оба файла с Диска.
Оговорка: углы сети, которые человек принял без правки, дают ошибку 0 — доля «≤ 1 %» здесь завышена; смотреть надо на худших.

## Сеть углов v6 (27.11)

Метки обновлены: 1 856 ковров (v5 училась на 986 + 247 проверочных). Запуск и загрузка весов — как у v4 (выше), `VER = 6`, GPU T4, 25–35 мин; в логе `фото найдено …` — сколько меток нашлось в пакетах на Диске (ковры, добавленные после пакетов от 13.11, пропускаются). Сравнивать с v5 нужно по проверочной выборке из того же отчёта (`worst_corner_le_1pct_after` v5 = 0,83 на 247 коврах).

## Что показал лист худших углов (21.11) и версия 5

431 ковёр с углами сети; худший угол ≤ 1 % у 87,2 %, ≤ 2 % у 88,9 %. Из 48 ковров с ошибкой > 2 % у **37 стартовые углы были рамкой всего фото** (`find_quad` не нашёл ковёр — ковёр на деревянном полу, чёрном фоне, стене), и верные углы оказались дальше окна сети (ошибка до 53 %). «Бахрома наружу» — 7, «внутрь» — 3. Значит, главная беда — не бахрома, а плохой старт. Всего с рамкой на старте 362 ковра, из них 37 плохих (тех, где ковёр не заполняет кадр).
v5 (`corner_refine_colab.py`, VER = 5): первая ступень окна 50 % (ступени 0,5 / 0,3 / 0,16 / 0,08 / 0,04 / 0,02), обучение до окна 55 % и 25 % примеров стартуют с угла рамки фото, в отчёте отдельная строка `frame` (старт — рамка фото). Запуск — как у v4 (ячейка в README выше), GPU T4, ≈ 25–35 мин.

# Ночной поиск элементов по всем коврам — xfind_night_colab.py (v12.36)

Для каждой группы подтверждённых рамок ищет такие же места на всех остальных коврах (сетка DINOv3-S → перепроверка овальной вырезкой).
Нужна видеокарта T4; ≈ 30–60 мин. Итог сам уходит на сервер (через файл в Drive `carpet-dna/cps_xfind/xfind_night.json`, сервер забирает раз в час)
и появляется в /m карточкой «на других коврах».

```
!rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
%env CPS_KEY=<ключ из студии: пользователи → «ключ»>
%run /content/carpevo/cps/tools/colab/xfind_night_colab.py
```

Фото кэшируются в `carpet-dna/cps_xfind/raw` (≈ 2–3 ГБ; следующий запуск качает только новые). Параметры: `XF_NR` (сколько ковров перепроверять, 150), `XF_TOP` (находок на группу, 12).
