"""Карточки ковров из книг-альбомов по результатам VLM (vlm_plates.jsonl, Gemini через OpenRouter, 2026-11-12) → books_catalog.json + plates_<slug>.json.
Запуск: BOOKS_WORK=/путь/с/books_catalog.json VLM_WORK=/путь/с/vlm_plates.jsonl python3 build_series.py
Старые карточки этих книг в каталоге заменяются, карточки энциклопедии и Мурадова-2010 не трогаются."""
import os, json, re, collections, time

WORK = os.environ.get('BOOKS_WORK', '.')
VLM = os.environ.get('VLM_WORK', WORK)
BOOKS = {  # slug -> (подпись книги для school_why, доверие)
    'intangible_2849': 'Мурадов, «Азербайджанские ковры» (2008), ч. 1',
    'intangible_2850': 'Мурадов, «Азербайджанские ковры» (2008), ч. 2',
    'intangible_2851': 'Мурадов, «Azərbaycan xalça sənəti» (2013)',
    'intangible_2854': 'книга об Иреванской группе (файл 2854)',
    'azerbaycan_xalchasi': 'сборник к 100-летию Л. Керимова (Музей ковра)',
    'intangible_2885': 'Музей истории Азербайджана, «Ковры и ковровые изделия» (АН АзССР)',
}
KEEP = {'rug': 'ковёр', 'prayer_rug': 'молитвенный ковёр', 'flatweave': 'безворсовый', 'bag': 'мешок / сумка'}
TYPE_OF = {'guba': 'guba-shirvan', 'shirvan': 'guba-shirvan', 'baku': 'guba-shirvan', 'shaki': 'guba-shirvan', 'ganja': 'ganja-gazakh', 'gazakh': 'ganja-gazakh',
           'karabakh': 'karabakh', 'tabriz': 'tabriz'}
PLACE_RU = {'guba': 'Губа', 'shirvan': 'Ширван', 'karabakh': 'Карабах', 'gazakh': 'Газах', 'ganja': 'Гянджа', 'baku': 'Баку', 'tabriz': 'Тебриз',
            'nakhchivan': 'Нахичевань', 'irevan': 'Иреван', 'shaki': 'Шеки', 'mugan': 'Муган', 'lankaran': 'Ленкорань', 'talish': 'Талыш'}
MAT_RU = {'wool': 'шерсть', 'silk': 'шёлк', 'cotton': 'хлопок', 'mixed': 'смешанный'}
PART_RU = {'early': 'начало', 'mid': 'середина', 'late': 'конец'}
ROM = {'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20, 'XXI': 21}


def century(c):
    m = re.search(r'[XIV]+', (c or '').upper())
    return ROM.get(m.group(0)) if m else None


rows = []
for l in open(os.path.join(VLM, 'vlm_plates.jsonl'), encoding='utf-8'):
    if l.strip(): rows.append(json.loads(l))
cat = json.load(open(os.path.join(WORK, 'books_catalog.json'), encoding='utf-8'))
old = [i for i in cat['items'] if i.get('book') not in BOOKS]
items, stats = [], collections.Counter()
plates = collections.defaultdict(list)
for r in rows:
    slug = r['slug']
    if slug not in BOOKS: continue
    for it in (r.get('data') or {}).get('items', []):
        k = it.get('kind')
        if k not in KEEP: stats[(slug, 'skip:' + str(k))] += 1; continue
        sch, cen, size = it.get('school') or '', century(it.get('century')), it.get('size_cm')
        name = (it.get('name') or '').strip(); name_en = (it.get('name_en') or '').strip()
        if not (sch or cen or size or name): stats[(slug, 'skip:empty')] += 1; continue   # титул, виньетка и т.п. с пустой подписью
        iid = it.get('id') or ''
        if not iid.startswith(slug): stats[(slug, 'skip:badid')] += 1; continue
        date = (f'{cen} в.' + (f", {PART_RU[it['part']]}" if it.get('part') in PART_RU else '')) if cen else (str(it['year']) if it.get('year') else '')
        sz = ''
        if isinstance(size, list) and len(size) >= 2 and all(isinstance(v, (int, float)) for v in size[:2]):
            sz = 'x'.join(str(int(v)) if float(v).is_integer() else str(v) for v in size[:2]) + ' см'
        place = PLACE_RU.get(sch, '')
        title = ' '.join(x for x in [KEEP[k] if k != 'rug' else 'Ковёр', f'«{name}»' if name else '', place] if x)
        d = dict(id=iid, src='book:' + slug, img=f'img/{iid}.jpg', title=title,
                 names=[x for x in (name, name_en) if x], composition=name, date=date, school=TYPE_OF.get(sch, ''),
                 school_why=(f'подпись: школа «{place or sch}» — {BOOKS[slug]}' if sch else ''),
                 technique=('ворсовый' if k in ('rug', 'prayer_rug') else ('безворсовый' if k == 'flatweave' else '')),
                 fragment=False, text=(it.get('caption') or '').strip(), museum=(it.get('museum') or '').strip(), inv=(it.get('inv') or '').strip(),
                 place=place, url='', license='книга: только анализ/обучение, не публиковать', size=sz,
                 page=r['page'], book=slug)
        if it.get('material') in MAT_RU: d['material'] = MAT_RU[it['material']]
        if k == 'prayer_rug': d['object'] = 'молитвенный ковёр'
        elif k == 'bag': d['object'] = 'мешок / сумка'
        if cen and cen >= 21:   # XXI в. в альбомах Мурадова — новые ковры «Азер-ильме»/современные копии: слабое свидетельство о школе
            d['object'] = 'современная копия (XXI в.)'; d['school_why'] += '; XXI в. — современная работа, слабое свидетельство'
        if sch in ('nakhchivan', 'irevan', 'mugan', 'lankaran', 'talish', 'other'): d['school_raw'] = sch   # школы без своей ветки типологии
        items.append(d); stats[(slug, 'ok')] += 1
        plates[slug].append({k2: it.get(k2) for k2 in ('id', 'kind', 'caption', 'name', 'name_en', 'name_ru', 'school', 'century', 'part', 'year', 'material', 'size_cm', 'museum', 'inv', 'confidence')} | {'page': r['page']})
ids = collections.Counter(i['id'] for i in items)
items = list({i['id']: i for i in items}.values())
cat['items'] = old + items
cat['groups'] = [g for g in cat.get('groups', []) if g['members'][0] in {i['id'] for i in cat['items']}]
cat['built'] = time.strftime('%Y-%m-%d %H:%M')
cat['about'] = 'Книги: Энциклопедия т. II; Мурадов «Qarabağ qrupu» (2010); Мурадов 2008 (2 ч.) и 2013, Иреванская группа, сборник к 100-летию Л. Керимова, альбом Музея истории Азербайджана (подписи прочитаны VLM). Фото только для анализа.'
json.dump(cat, open(os.path.join(WORK, 'books_catalog.json'), 'w', encoding='utf-8'), ensure_ascii=False)
for s, p in plates.items(): json.dump(p, open(os.path.join(WORK, f'plates_{s}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('всего карточек', len(cat['items']), '| новых', len(items), '| дубликатов id', sum(v - 1 for v in ids.values()))
for s in BOOKS:
    n = [i for i in items if i['book'] == s]
    print(f'{s}: {len(n)}  школа {sum(1 for i in n if i["place"])}  век {sum(1 for i in n if i["date"])}  размер {sum(1 for i in n if i["size"])}  имя {sum(1 for i in n if i["composition"])}',
          {k[1]: v for k, v in stats.items() if k[0] == s and k[1] != 'ok'})
