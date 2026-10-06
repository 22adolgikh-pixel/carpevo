# azerbaijanrugs.com — классификация Азербайджан Ругс (Баку)

## Что это за сайт

**Azerbaijan Rugs** (https://www.azerbaijanrugs.com) — сайт бакинской фирмы-дилера антикварных ковров, основана в 2004 г., руководитель — **Вугар Дадашов**. Раздел `/guide/` — иллюстрированный справочник по азербайджанским коврам: регионы (Куба, Баку, Хызы, Ширван, Мугань, Шахсевены, Карабах, Загатала, Казах, Борчалы, Гянджа, Талыш, Нахчыван, Тебриз и центры Южного Азербайджана), внутри них категории по композициям, мотивам, технике и формату. Индекс справочника последний раз изменён в 2025-07; многие страницы категорий датированы 2011–2014 гг. (заголовок `Last-Modified`, есть и страницы 2015–2026). Часть категорий пуста — заглушки 2011 г. с `n = 0`.

Классификация — дилерская: она опирается на западные рыночные названия (Seychour, Chelaberd, Kasim Ushak, Fachralo и т.п.) и переводит их в азербайджанские. Это ближе к Керимову, чем западная традиция, но совпадает с ним не везде (см. колонку «Конфликт»).

## Авторское право

На сайте сказано: **«Advance authorization is necessary»**, т.е. перепечатка требует предварительного разрешения. Большинство фотографий взято из аукционных каталогов (Sotheby's, Christie's, Rippon Boswell, Skinner, Nagel и др.) и из книг, так что права на них сайту часто не принадлежат. **Фото и тексты не публикуем.** Здесь хранятся только метаданные: названия, счётчики, ссылки и разобранные поля подписей. Их используем как внутренний источник для словаря.

## Что здесь собрано

| Файл | Содержимое |
|---|---|
| `taxonomy.json` | дерево сайта: регион → категории; у каждого узла есть `url`, `last_modified`, число изображений (`n` / `n_direct`) |
| `captions.csv` (не в git: чужой текст; пересобирается `tools/crawl_in_browser.js`) | 7024 подписи к изображениям: `region, category, caption, century, year, size_cm, size_ft, sources, places, page, thumb`. Век, год, размер и тип источника (auction / museum / collection / published) выделены разбором подписи |
| `category_map.json` | сопоставление каждого региона и каждой категории с нашим словарём. Поля: `kind`, `concepts` (id из `out/concepts.json`), `new_name` (что добавить, если понятия нет), `kerimov_type` / `kerimov_group` (по `carpet_schools.json` и Керимову), `conflict`, `note_ru` |

**Правила сопоставления.**
- В `concepts` попадают только понятия, которые совпадают по имени или однозначно по смыслу. Для описательных категорий («various», «floral») список пуст.
- Строка может одновременно иметь `concepts` и `new_name`. Так бывает, когда совпадает лишь часть: формат или мотив есть в словаре, а рыночного названия нет (например, `Tekye` → `namazliq` + `shebeke`).
- `kerimov_type` / `kerimov_group` взяты из нашего `carpet_schools.json`. Если понятия нет, тип выведен из региона сайта. Группы mugan, talysh и nakhchivan в перечень полей не входят, поэтому для них `kerimov_group = null`, а пояснение дано в `note_ru`.
- Регион и одноимённая сводная категория дают одну строку на каждый URL. Например, «Antique Azerbaijani Jajims» и «Shahsavan Jajims» — одна страница.


**Итог:** 168 строк (31 регион + 137 категорий). Сопоставлено с понятиями: 136. С новым именем: 34 строки (23 разных имён). С конфликтом: 29.


## Таблица: регион → категория → наше понятие / новое имя / конфликт

| Регион | Категория | n | Вид | Наши понятия | Новое имя | Керимов | Конфликт |
|---|---|---:|---|---|---|---|---|
| Early Azerbaijan Rugs | *(страница региона)* | 319 | period | `ejdahali` (əjdahalı) | — | guba-shirvan/guba | сайт называет ранние драконовые ковры «Early Karabagh»; у нас ejdahali — guba-shirvan/guba (medium), а каталог «Азер-Ильме» (Мурадов m2-114) — Карабах |
| Historical Dragon Rugs | *(страница региона)* | 85 | composition | `ejdahali` (əjdahalı), `drakon` (əjdaha) | — | guba-shirvan/guba | сайт называет ранние драконовые ковры «Early Karabagh»; у нас ejdahali — guba-shirvan/guba (medium), а каталог «Азер-Ильме» (Мурадов m2-114) — Карабах |
| Azerbaijan Textiles | *(страница региона)* | 51 | object | `gullebduz` (güləbətin) | — | — | — |
| Antique Kuba rugs | *(страница региона)* | 0 | region-only | `quba` (quba) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | PRE-19TH CENTURY HISTORICAL KUBA RUGS AND CARPETS | 85 | period | `quba` (quba) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Zeykhur / Seychour rugs | 186 | composition | — | Zeykhur (Seychour) | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Alpan rugs | 103 | composition | `alpan` (alpan) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Eastern Caucasian Soumak / Sumakh rugs | 130 | technique | `sumax-object` (sumax), `sumax` (sumax) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Dragon Soumak rugs | 39 | technique | `sumax-object` (sumax), `ejdahali` (əjdahalı) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Karagashli rugs and carpets | 59 | composition | `qaraqashli` (qaraqaşlı) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Caucasian Medallion Gubpa rugs | 184 | motif | `qubba` (qubba), `qubbe` (qübbə) | — | guba-shirvan | сайт кладёт категорию в Кубу (URL kuba_shirvan), подписи — Ширван |
| Antique Kuba rugs | Antique Kuba Sunburst Zejwa rugs | 45 | composition | `zeyve` (zeyvə) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Molla Kamalli rugs | 9 | composition | `mollakamalli` (mollakamallı) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Perepedil rugs and carpets | 72 | composition | `pirebedil` (pirəbədil) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Chichi (Chi Chi) rugs | 113 | composition | `chichi` (çiçi), `xirdagul-chichi` (xırdagül çiçi) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Gollu Chichi rugs and carpets | 31 | composition | `qollu-chichi` (qollu çiçi), `golluchichi` (göllüçiçi) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba rugs | 146 | region-only | `quba` (quba) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Tekye rugs | 113 | composition | `namazliq` (namazlıq), `shebeke` (şəbəkə) | Tekye | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Zeyve (so called Lesghi Star) rugs | 169 | composition | `zeyve` (zeyvə) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Biliji & Lajadi rugs | 32 | composition | `bilici` (bilici), `lecedi` (ləcədi) | — | guba-shirvan/guba | у Керимова в тексте главы «Ляджади» — ширванская группа, в таблице — кубинская (см. carpet_schools lecedi); сайт — Куба |
| Antique Kuba rugs | Antique Kuba Shield rugs | 83 | motif | — | Shield | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba 'Ugah' rugs and carpets | 32 | composition | `ugax` (uğax) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Eastern Azerbaijan Afshan rugs | 123 | composition | `efshan-carpet` (əfşan), `xileafshan` (xiləəfşan) | — | tabriz | у нас efshan-carpet привязан к tabriz (medium, «композиция широкого применения»); сайт держит «Afshan» в Кубе и Баку — ближе к бакинскому «Хиля-Афшан» (xileafshan, guba-shirvan/baku) у Керимова |
| Antique Kuba rugs | Antique Kuba Afurja rugs and carpets | 11 | composition | `afurca` (afurca) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Haji Gaib rugs | 18 | composition | `haciqayib` (hacıqayıb) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Gyryz and Jek village rugs | 16 | composition | `giriz` (qırız), `cek` (cek) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Konagkend rugs and carpets | 83 | composition | `qonaqkend` (qonaqkənd) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Yerfi rugs | 6 | composition | `yerfi` (yerfi) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Salma Söyüd rugs | 13 | composition | `salmasoyud` (salmasöyüd), `selmesoyud` (səlməsöyüd) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Kilims | 22 | technique | `kilim` (kilim) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Kuba Khorjuns / Saddle Bags | 17 | object | `xurcun` (Xurcun) | — | guba-shirvan/guba | — |
| Antique Kuba rugs | Antique Gymyl rugs and carpets | 34 | composition | `qimil` (qımıl) | — | guba-shirvan/guba | — |
| Antique Derbend and Daghestan rugs | *(страница региона)* | 157 | region-only | — | Derbend / Daghestan | — | — |
| Antique Baku rugs | *(страница региона)* | 0 | region-only | `baki` (bakı) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Baku "Khila Buta/Boteh" rugs | 9 | composition | `xilebuta` (xiləbuta), `xile` (xilə) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Surakhani rugs | 2 | composition | `suraxani` (suraxanı), `xile-suraxani` (xilə suraxanı) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Baku Shield rugs | 7 | motif | — | Shield | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Early Baku rugs | 18 | period | `baki` (bakı) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Baku village rugs | 6 | region-only | `baki` (bakı) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Baku rugs with various designs | 4 | region-only | `baki` (bakı) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Baku rugs with octagonal medallions | 5 | motif | `gol` (göl) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Baku flatweaves | 3 | technique | `zili` (zili), `palaz` (palaz), `kilim` (kilim) | — | guba-shirvan/baku | — |
| Antique Baku rugs | Antique Baku textiles | 2 | object | `gullebduz` (güləbətin) | — | — | — |
| Antique Baku rugs | Antique Baku Afshan rugs | 47 | composition | `xileafshan` (xiləəfşan), `efshan-carpet` (əfşan) | — | guba-shirvan/baku | у нас efshan-carpet привязан к tabriz (medium, «композиция широкого применения»); сайт держит «Afshan» в Кубе и Баку — ближе к бакинскому «Хиля-Афшан» (xileafshan, guba-shirvan/baku) у Керимова |
| Antique Khyzy rugs | *(страница региона)* | 0 | region-only | — | Khyzy | guba-shirvan/baku | сайт выделяет Хызы в отдельный регион; у Керимова/у нас ковры Хызы (Фындыган, Гади) входят в бакинскую группу Губа-Ширванского типа |
| Antique Khyzy rugs | Antique Khyzy rugs | 6 | region-only | — | Khyzy | guba-shirvan/baku | сайт выделяет Хызы в отдельный регион; у Керимова/у нас ковры Хызы (Фындыган, Гади) входят в бакинскую группу Губа-Ширванского типа |
| Antique Khyzy rugs | Antique Khyzy Findighan rugs | 18 | composition | `findiqan` (fındığan) | — | guba-shirvan/baku | сайт выделяет Хызы в отдельный регион; у Керимова/у нас ковры Хызы (Фындыган, Гади) входят в бакинскую группу Губа-Ширванского типа |
| Antique Khyzy rugs | Antique Khyzy Bags (Khorjun, Heybe) | 43 | object | `xurcun` (Xurcun), `heybe` (Heybə) | — | — | — |
| Antique Khyzy rugs | Antique Khyzy & Baku Bedding Bags (Mafrash) | 8 | object | `mefresh-carpet` (Məfrəş) | — | — | — |
| Antique Khyzy rugs | Antique Khyzy & Baku Horse Covers | 14 | object | `chul` (çul) | — | — | — |
| Antique Khyzy rugs | Antique Khyzy Palaz & Kilims | 8 | technique | `palaz` (palaz), `kilim` (kilim) | — | — | — |
| Antique Khyzy rugs | Antique Baku / Khyzy Zili (Sileh, Verneh) Flatwoven rugs | 4 | technique | `zili` (zili), `verni` (vərni) | — | — | сайт отождествляет «Sileh/Verneh» с зили («correct name: zili»); у нас vərni — отдельный предмет (большой халы полуторными узлами, Карабах/Газах по Керимову) |
| Antique Shirvan rugs | *(страница региона)* | 0 | region-only | `shirvan` (şirvan) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Salyan (Saliani) / "Akstafa" rugs | 75 | composition | `salyan` (salyan) | Akstafa | guba-shirvan/shirvan | «Akstafa» (Ağstafa) — район Газахской группы (ganja-gazakh/gazakh); здесь это рыночное название сальянских ковров с «птицами»/бута, и сайт ставит их в Ширван/Сальян. У нас понятия Ağstafa нет; Керимов сальянские ковры относит к ширванской группе |
| Antique Shirvan rugs | Antique Salyan ("Chajli", "Moghan", "Saliani") rugs | 45 | region-only | `salyan` (salyan), `chayli` (çaylı), `mugan` (muğan) | — | guba-shirvan/shirvan | «Chajli» в carpet_schools — ganja-gazakh/ganja (у Ганса-Рюдена — Карабах; у Керимова есть и село Чайлы под Шемахой); «Moghan» (muğan) у нас — karabakh/mugan; сайт оба названия считает рыночными синонимами сальянских (ширванских) ковров |
| Antique Shirvan rugs | Antique "Akstafa" (Salyan) prayer rugs | 75 | format | `salyan` (salyan), `namazliq` (namazlıq) | Akstafa | guba-shirvan/shirvan | «Akstafa» (Ağstafa) — район Газахской группы (ganja-gazakh/gazakh); здесь это рыночное название сальянских ковров с «птицами»/бута, и сайт ставит их в Ширван/Сальян. У нас понятия Ağstafa нет; Керимов сальянские ковры относит к ширванской группе |
| Antique Shirvan rugs | Antique Shirvan rugs with Star motif | 16 | motif | `ulduz` (ulduz) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan bags, saddle bags, bedding bags | 43 | object | `xurcun` (Xurcun), `heybe` (Heybə), `mefresh-carpet` (Məfrəş) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Early Shirvan rugs | 31 | period | `shirvan` (şirvan), `palmet` (palmet) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan Shield rugs | 35 | motif | `orduc` (orduc) | Shield | guba-shirvan/shirvan | подписи называют медальон «orduch shield», т.е. сайт помещает щитовые ковры в Ширван (и Кубу/Баку); у нас orduc — guba-shirvan/guba |
| Antique Shirvan rugs | Antique Shirvan Kilims | 159 | technique | `kilim` (kilim) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan Kabistan ("Marasali") Prayer rugs | 137 | format | `qobustan` (qobustan), `mereze-carpet` (mərəzə), `namazliq` (namazlıq) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan Marasali rugs | 162 | composition | `mereze-carpet` (mərəzə) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan Prayer rugs with various designs (~130 rugs displayed) | 132 | format | `namazliq` (namazlıq) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan rugs with various motives | 116 | region-only | `shirvan` (şirvan) | — | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan Pictorial rugs | 43 | composition | — | Pictorial | guba-shirvan/shirvan | — |
| Antique Shirvan rugs | Antique Shirvan Bijo rugs | 28 | composition | `bico` (bico) | — | guba-shirvan/shirvan | — |
| Antique Mughan rugs | *(страница региона)* | 125 | region-only | `mugan` (muğan) | — | karabakh | у нас muğan — karabakh/mugan (medium; в таблицах Керимова также guba-shirvan); сайт выделяет Мугань в отдельный регион и одновременно относит «Moghan» к ширванскому Сальяну |
| Antique Shahsavan rugs | *(страница региона)* | 0 | region-only | `shahsever` (Şahsevər) | — | guba-shirvan | у нас Şahsevər — guba-shirvan/mugan (low, только каталог); сайт ставит шахсевенов вместе с Северо-Западным Ираном и Муганью, Керимовской привязки нет |
| Antique Shahsavan rugs | Antique Shahsavan Jajims | 10 | technique | `cecim` (cecim) | — | — | — |
| Antique Shahsavan rugs | Antique Shahsavan Khorjuns (Double Saddle Bags) | 0 | object | `xurcun` (Xurcun), `shahsever` (Şahsevər) | — | guba-shirvan | у нас Şahsevər — guba-shirvan/mugan (low, только каталог); сайт ставит шахсевенов вместе с Северо-Западным Ираном и Муганью, Керимовской привязки нет |
| Antique Shahsavan rugs | Antique Shahsavan Heybes (Shoulder Bags) | 0 | object | `heybe` (Heybə), `shahsever` (Şahsevər) | — | guba-shirvan | у нас Şahsevər — guba-shirvan/mugan (low, только каталог); сайт ставит шахсевенов вместе с Северо-Западным Ираном и Муганью, Керимовской привязки нет |
| Antique Shahsavan rugs | Antique Shahsavan Mafrash Bedding Bags | 0 | object | `mefresh-carpet` (Məfrəş), `shahsever` (Şahsevər) | — | guba-shirvan | у нас Şahsevər — guba-shirvan/mugan (low, только каталог); сайт ставит шахсевенов вместе с Северо-Западным Ираном и Муганью, Керимовской привязки нет |
| Antique Shahsavan rugs | Antique Shahsavan Pile rugs | 7 | region-only | `shahsever` (Şahsevər) | — | guba-shirvan | у нас Şahsevər — guba-shirvan/mugan (low, только каталог); сайт ставит шахсевенов вместе с Северо-Западным Ираном и Муганью, Керимовской привязки нет |
| Antique Shahsavan rugs | Antique Shahsavan Kilims | 0 | technique | `kilim` (kilim), `shahsever` (Şahsevər) | — | guba-shirvan | у нас Şahsevər — guba-shirvan/mugan (low, только каталог); сайт ставит шахсевенов вместе с Северо-Западным Ираном и Муганью, Керимовской привязки нет |
| Antique Karabagh rugs | *(страница региона)* | 0 | region-only | `qarabag` (qarabağ) | — | karabakh | — |
| Antique Karabagh rugs | Antique Sunburst (Adler Kazak, Chelaberd) Karabagh Chelebi Rugs | 79 | composition | `chelebi` (Çələbi) | — | karabakh | западное «Adler Kazak» относит его к Казаху; сайт и Керимов — Карабах |
| Antique Karabagh rugs | Antique "Cloudband" Karabagh Rugs (Malybeyli) | 54 | composition | `malibeyli` (malıbəyli), `bulud-carpet` (bulud) | — | karabakh | аукционы иногда называют их «Kazak»; сайт и Керимов — Карабах |
| Antique Karabagh rugs | Antique Karabagh Kasim Ushak rugs | 74 | composition | `qasimushagi` (qasımuşağı) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh rugs with Moghan / Memling Guls | 31 | motif | — | Memling gul (Moghan gul) | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Lampa Rugs | 37 | composition | `lampa` (lampa) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Pictorial Rugs | 26 | composition | — | Pictorial | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Goradis & "Buynuz" (Horn motif) Rugs | 15 | composition | `buynuz-carpet` (buynuz) | Goradis (Horadiz) | karabakh | не расхождение с сайтом (Карабах), но у нас buynuz — medium: в таблицах Керимова мотив встречается в guba-shirvan, karabakh и ganja-gazakh |
| Antique Karabagh rugs | Antique Karabagh Prayer Rugs | 19 | format | `namazliq` (namazlıq) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Horse Covers | 5 | object | `chul` (çul) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Zili rugs | 0 | technique | `zili` (zili) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Vernehs | 0 | technique | `verni` (vərni), `qarabag-vernisi` (qarabağ vərnisi) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Kilims | 0 | technique | `kilim` (kilim) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Rugs with various designs | 5 | region-only | `qarabag` (qarabağ) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Achma Yumma Rugs | 0 | composition | `achma-yumma` (açma-yumma) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Rugs with Herati design | 0 | motif | `herati` (Heratı), `baliq-carpet` (balıq) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Floral Rugs with European designs | 0 | composition | — | — | karabakh | — |
| Antique Karabagh rugs | 19th/early 20th century Karabagh Floral Rugs | 0 | composition | — | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Rugs with boteh motif | 0 | motif | `buta` (buta) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Turtle design Rugs | 0 | motif | `tisbaga` (tısbağa), `chanaqli-baga` (çanaqlı-bağa) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Flatweaves | 8 | technique | `shedde` (şəddə), `qarabag-sheddesi` (qarabağ şəddəsi) | — | karabakh | — |
| Antique Karabagh rugs | Antique Zangezur rugs | 9 | region-only | — | Zangezur | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Khanlyg Rugs | 16 | composition | `xanliq` (xanlıq) | — | karabakh | — |
| Antique Karabagh rugs | Antique Karabagh Mafrashes, Khorjuns (Saddle Bags) | 0 | object | `mefresh-carpet` (Məfrəş), `xurcun` (Xurcun) | — | karabakh | — |
| Antique Zakatala rugs | *(страница региона)* | 108 | region-only | — | Zakatala | — | — |
| Antique Kazak rugs | *(страница региона)* | 0 | region-only | `qazax` (qazax) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Early Kazak rugs | 19 | period | `qazax` (qazax) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Sykhly / Star Kazak rugs | 33 | composition | `shixli` (şıxlı), `ulduz` (ulduz) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Karachop rugs | 118 | composition | `qarachop` (qaraçöp) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Kazak Prayer rugs | 46 | format | `namazliq` (namazlıq) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Kazak Turtle rugs | 21 | motif | `tisbaga` (tısbağa), `chanaqli-baga` (çanaqlı-bağa) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Shield Kazak rugs | 106 | motif | — | Shield Kazak | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Demirchiler / Chajli rugs | 16 | composition | `demirchiler` (dəmirçilər), `chayli` (çaylı) | — | ganja-gazakh/gazakh | сайт объединяет «Demirchiler» (у нас gazakh) и «Chajli» (у нас ganja; Skinner — «Southeast Caucasus»; сайт же в Ширване ставит «Chajli» как сальянский) |
| Antique Kazak rugs | Antique Kazak rugs with Moghan Gul motif | 77 | motif | — | Memling gul (Moghan gul) | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Kazak Kilims | 25 | technique | `kilim` (kilim) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Kazak rugs with "Rams Horn" motif | 26 | motif | `qoch` (qoç), `buynuz` (buynuz) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Shamshaddil-Karvansaray-Tovuzqala rugs | 3 | region-only | — | Shamshaddil / Karvansaray / Tovuzqala | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Kazak Akstafa Rugs | 10 | composition | — | Akstafa | ganja-gazakh/gazakh | сайт употребляет «Akstafa» для двух разных групп: здесь — газахская (по топониму), а в Ширване — сальянские «птичьи» ковры |
| Antique Kazak rugs | Antique Kazak Flatweaves & Other Textiles | 0 | technique | — | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Kazak Bags / Saddle Bags / Bedding bags | 0 | object | `xurcun` (Xurcun), `heybe` (Heybə), `mefresh-carpet` (Məfrəş) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Kazak rugs with various designs | 375 | region-only | `qazax` (qazax) | — | ganja-gazakh/gazakh | — |
| Antique Kazak rugs | Antique Göyche rugs | 6 | composition | `goycheli` (göyçəli) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | *(страница региона)* | 0 | region-only | `borchali` (borçalı) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Kazak "Sewan" / Kilseli rugs | 180 | composition | — | Kilseli (Sewan) | ganja-gazakh/gazakh | «Sewan» — Севан, т.е. оз. Гёйча (наш göyçəli, gazakh); сайт держит «Göyche» отдельно в Казахе, а «Sewan/Kilseli» — в Борчалы |
| Antique Borchaly (Bordjalou) rugs | Antique Borchalou rugs with hooked motives | 3 | motif | `qarmaqli` (qarmaqlı) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Fachralo (Fakhraly) rugs | 110 | composition | `faxrali` (fəxralı) | — | ganja-gazakh/ganja | у нас faxrali — ganja-gazakh/ganja (medium); сайт ставит «Fachralo» в Борчалы (Газах), так же как каталог «Азер-Ильме» и Ганс-Рюден; при этом в разделе Гянджи у сайта тоже есть «Fakhraly» |
| Antique Borchaly (Bordjalou) rugs | Antique Lambalo (Lambali) rugs | 0 | region-only | — | Lambalo (Lambali) | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Bordjalou rugs with Rams Horn motives | 0 | motif | `qoch` (qoç), `buynuz` (buynuz) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Kazak Pinwheel / Swastika rugs | 14 | motif | — | Pinwheel / Swastika | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Lori Pambak rugs | 8 | region-only | — | Lori Pambak | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Lori Pambak Frog rugs | 8 | motif | `qurbaga` (qurbağa) | Lori Pambak | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Kazak/Borchalou rugs with "Tree of Life" motif | 27 | motif | `agac` (ağac) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Bordjalou Flatweaves | 8 | technique | — | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Bordjalou rugs with Moghan motif | 10 | motif | — | Memling gul (Moghan gul) | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Bordjalou Bags | 0 | object | `xurcun` (Xurcun), `heybe` (Heybə), `mefresh-carpet` (Məfrəş) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Bordjalou rugs with various designs | 53 | region-only | `borchali` (borçalı) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Bordjalou Kilims | 0 | technique | `kilim` (kilim) | — | ganja-gazakh/gazakh | — |
| Antique Borchaly (Bordjalou) rugs | Antique Kazak Shulaver rugs | 0 | region-only | — | Shulaver | ganja-gazakh/gazakh | — |
| Antique Gendje rugs | *(страница региона)* | 0 | region-only | `gence-carpet` (gəncə) | — | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Genje rugs with a diagonal striped field | 35 | composition | `qedim-gence` (qədim gəncə) | — | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Gendje rugs with various designs | 0 | region-only | `gence-carpet` (gəncə) | — | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Gendje Prayer rugs | 2 | format | `namazliq` (namazlıq) | — | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Gendje Fakhraly rugs | 0 | composition | `faxrali` (fəxralı) | — | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Gendje rugs with Moghan design | 0 | motif | — | Memling gul (Moghan gul) | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Gendje "Chiragli" rugs | 0 | composition | `chiraqli` (çıraqlı) | — | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Gedebey rugs | 0 | composition | `gedebey` (gədəbəy) | — | ganja-gazakh/ganja | — |
| Antique Gendje rugs | Antique Genje Fakhrali Prayer rugs | 4 | format | `faxrali` (fəxralı), `namazliq` (namazlıq) | — | ganja-gazakh/ganja | — |
| Antique Talish rugs | *(страница региона)* | 4 | region-only | `talish` (talış) | — | karabakh | сайт: отдельный регион «Talish» (подписи «Lenkoran», Mughan region); у нас talış — karabakh/talysh (Мурадов), у Керимова описан в ширванском скане f36 |
| Antique Nakhchivan rugs | *(страница региона)* | 51 | region-only | `naxchivan` (naxçıvan) | — | karabakh | — |
| Antique Azerbaijan Bags | *(страница региона)* | 0 | object | `xurcun` (Xurcun), `heybe` (Heybə), `mefresh-carpet` (Məfrəş) | — | — | — |
| Antique Azerbaijani Jajims | *(страница региона)* | 10 | technique | `cecim` (cecim) | — | — | — |
| Antique Tabriz rugs | *(страница региона)* | 0 | region-only | `tebriz` (təbriz) | — | tabriz | — |
| Antique Tabriz rugs | Antique Haji Jalili Tabriz rugs and carpets | 51 | composition | — | Haji Jalili | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz medallion rugs and carpets | 75 | composition | `lechekturunc` (ləçəkturunc) | — | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz pictorial rugs and carpets | 36 | composition | `dordfesil` (dördfəsil) | Pictorial | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz rugs with Islimi (Arabesque) motif | 16 | motif | `islimi` (islimi) | — | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz Garden rugs in a compartment format | 10 | composition | — | Garden (compartment) | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz rugs & carpets with boteh motives | 6 | motif | `buta` (buta) | — | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz rugs with palmette & vinescroll designs | 50 | motif | `palmet` (palmet) | — | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz rugs & carpets with Herati motif | 4 | motif | `herati` (Heratı) | — | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz rugs and carpets with various floral designs | 43 | region-only | `tebriz` (təbriz) | — | tabriz | — |
| Antique Tabriz rugs | Antique Tabriz Mihrab (Prayer rug format) rugs | 24 | format | `mehrab` (mehrab), `namazliq` (namazlıq), `mehrabli` (mehrablı) | — | tabriz | — |
| Antique Serapi rugs | *(страница региона)* | 100 | region-only | `heris` (heris) | Serapi | tabriz | — |
| guide_index_antique_heriz_rugs_carpets.htm | *(страница региона)* | 186 | region-only | `heris` (heris) | — | tabriz | — |
| Antique Bakhshaish rugs | *(страница региона)* | 175 | region-only | `baxshayish` (baxşayış) | — | tabriz | — |
| Antique Ardabil rugs | *(страница региона)* | 8 | region-only | `erdebil` (ərdəbil) | — | tabriz | — |
| Early NW Iran/Azerbaijan carpets | *(страница региона)* | 99 | period | — | — | tabriz | — |
| Antique Karaja rugs | *(страница региона)* | 53 | region-only | `qaraca` (qaraca) | — | tabriz | — |
| Antique Meshkin rugs | *(страница региона)* | 24 | region-only | — | Meshkin | tabriz | — |
| Antique Sarab rugs | *(страница региона)* | 0 | region-only | `sarabi` (sərabi) | — | tabriz | — |
| Antique Zanjan rugs | *(страница региона)* | 10 | region-only | `zencan` (zəncan) | — | tabriz | — |
| Antique Karadagh / Arazbaran rugs | *(страница региона)* | 8 | region-only | — | Karadagh / Arasbaran | tabriz | — |
| Antique Iranian Azerbaijani rugs (unclassified) | *(страница региона)* | 123 | region-only | — | — | tabriz | — |

## Новые имена (кандидаты в словарь)

- **Akstafa** — Antique Shirvan rugs / Antique Salyan (Saliani) / "Akstafa" rugs; Antique Shirvan rugs / Antique "Akstafa" (Salyan) prayer rugs; Antique Kazak rugs / Antique Kazak Akstafa Rugs
- **Derbend / Daghestan** — Antique Derbend and Daghestan rugs / регион
- **Garden (compartment)** — Antique Tabriz rugs / Antique Tabriz Garden rugs in a compartment format
- **Goradis (Horadiz)** — Antique Karabagh rugs / Antique Karabagh Goradis & "Buynuz" (Horn motif) Rugs
- **Haji Jalili** — Antique Tabriz rugs / Antique Haji Jalili Tabriz rugs and carpets
- **Karadagh / Arasbaran** — Antique Karadagh / Arazbaran rugs / регион
- **Khyzy** — Antique Khyzy rugs / регион; Antique Khyzy rugs / Antique Khyzy rugs
- **Kilseli (Sewan)** — Antique Borchaly (Bordjalou) rugs / Antique Kazak "Sewan" / Kilseli rugs
- **Lambalo (Lambali)** — Antique Borchaly (Bordjalou) rugs / Antique Lambalo (Lambali) rugs
- **Lori Pambak** — Antique Borchaly (Bordjalou) rugs / Antique Lori Pambak rugs; Antique Borchaly (Bordjalou) rugs / Antique Lori Pambak Frog rugs
- **Memling gul (Moghan gul)** — Antique Karabagh rugs / Antique Karabagh rugs with Moghan / Memling Guls; Antique Kazak rugs / Antique Kazak rugs with Moghan Gul motif; Antique Borchaly (Bordjalou) rugs / Antique Bordjalou rugs with Moghan motif; Antique Gendje rugs / Antique Gendje rugs with Moghan design
- **Meshkin** — Antique Meshkin rugs / регион
- **Pictorial** — Antique Shirvan rugs / Antique Shirvan Pictorial rugs; Antique Karabagh rugs / Antique Karabagh Pictorial Rugs; Antique Tabriz rugs / Antique Tabriz pictorial rugs and carpets
- **Pinwheel / Swastika** — Antique Borchaly (Bordjalou) rugs / Antique Kazak Pinwheel / Swastika rugs
- **Serapi** — Antique Serapi rugs / регион
- **Shamshaddil / Karvansaray / Tovuzqala** — Antique Kazak rugs / Antique Shamshaddil-Karvansaray-Tovuzqala rugs
- **Shield** — Antique Kuba rugs / Antique Kuba Shield rugs; Antique Baku rugs / Antique Baku Shield rugs; Antique Shirvan rugs / Antique Shirvan Shield rugs
- **Shield Kazak** — Antique Kazak rugs / Antique Shield Kazak rugs
- **Shulaver** — Antique Borchaly (Bordjalou) rugs / Antique Kazak Shulaver rugs
- **Tekye** — Antique Kuba rugs / Antique Kuba Tekye rugs
- **Zakatala** — Antique Zakatala rugs / регион
- **Zangezur** — Antique Karabagh rugs / Antique Zangezur rugs
- **Zeykhur (Seychour)** — Antique Kuba rugs / Antique Kuba Zeykhur / Seychour rugs
