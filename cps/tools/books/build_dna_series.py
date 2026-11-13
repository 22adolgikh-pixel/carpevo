"""Источники Carpet DNA из книг, прочитанных VLM (2026-11-13): plates.json, glossary_map.json, school_check.json по каждой книге.
Вход: BOOKS_WORK/plates_<slug>.json (пишет build_series.py). Запуск: BOOKS_WORK=… python3 build_dna_series.py
Мурадов 2008 (файлы 2849 = ч. 1, 2850 = ч. 2) дописывается в уже существующий источник muradov_2008 (там ASSESSMENT.md и описания школ)."""
import os, json, re, collections, sys, difflib
REPO = os.environ.get('REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
WORK = os.environ.get('BOOKS_WORK', '.')
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_enc import fold, NAME, TYPE_OF
OUTROOT = os.environ.get('OUT', REPO + '/site/model/sources')
CS = json.load(open(REPO + '/site/model/carpet_schools.json'))['items']
CON = {c['id']: c for c in json.load(open(REPO + '/site/model/out/concepts.json'))}
SCH_RU = {'karabakh': 'Карабах', 'shirvan': 'Ширван', 'gazakh': 'Газах', 'baku': 'Баку', 'guba': 'Губа', 'tabriz': 'Тебриз', 'nakhchivan': 'Нахчыван',
          'irevan': 'Иреван', 'shaki': 'Шеки', 'ganja': 'Гянджа', 'mugan': 'Муган', 'lankaran': 'Ленкорань', 'talish': 'Талыш', 'other': 'другое'}
KIND_RU = {'rug': 'ворсовый ковёр', 'prayer_rug': 'молитвенный ковёр', 'flatweave': 'безворсовый', 'bag': 'мешок / сумка'}
KEYS = list(NAME)

SOURCES = {
    'muradov_2008': dict(slugs={'intangible_2849': 'ч. 1', 'intangible_2850': 'ч. 2'}, prefix='mur08',
        src={'id': 'muradov_2008', 'short': 'Мурадов, «Азербайджанские ковры» (2008)', 'author': 'Видади Мурадов (текст — Х. Асадова, науч. ред. К. Алиева)', 'year': 2008,
             'title': {'az': 'Azərbaycan xalçaları', 'en': 'Azerbaijan carpets', 'ru': 'Азербайджанские ковры'},
             'publisher': 'Bakı: Elm, 2008, 436 с. ISBN 978-9952-809-71-8 (сверено по выходным данным, стр. 8 файла 2849)', 'type': 'album',
             'file': 'Drive: scans/intangible/ — файл 2849 (ч. 1: вводная, Губа, Ширван, Баку, Гянджа, Газах), 2850 (ч. 2: Карабах, Нахчыван, Тебриз, «Азер-Илмэ»)',
             'note': {'ru': 'Подарочный альбом фирмы «Азер-Илмэ»; оценка книги — ASSESSMENT.md. Подписи к фото прочитаны моделью зрения (Gemini через OpenRouter, 12.11.2026) '
                            'со страницы целиком; века брать из англ./азерб. строки подписи (рус. OCR путал XIX/XX — здесь не применяется, т. к. читалась картинка).'}},
        about='Ковры из подписей альбома Мурадова 2008 (обе части). Ковры XXI в. — новые работы «Азер-Илмэ», свидетельство о школе слабое.'),
    'muradov_2013': dict(slugs={'intangible_2851': ''}, prefix='mur13',
        src={'id': 'muradov_2013', 'short': 'Мурадов 2013, «Azərbaycan xalça sənəti»', 'author': 'Видади Мурадов (ред. А. Ширалиева)', 'year': 2013,
             'title': {'az': 'Azərbaycan xalça sənəti: tarixi, məktəbləri, texnologiyası (illüstrasiyalı nəşr)', 'ru': 'Азербайджанское ковровое искусство: история, школы, технология'},
             'publisher': 'Bakı: Elm, 2013. ISBN 978-9952-453-19-5 (по OCR выходных данных; первая группа цифр распознана как 3952)', 'type': 'textbook',
             'file': 'Drive: scans/intangible/ — файл 2851; текст: carpet-dna/output/books/intangible_2851/ocr_pages.jsonl (tesseract aze+rus+eng)',
             'note': {'ru': 'Иллюстрированное пособие Учебного центра ковроделия: история, классификация (ворсовые и безворсовые), группы ковров по школам, '
                            'первые шаги ткача, инструменты, станок, нити, красители. Фото — в основном безворсовые изделия, детали и процесс; подписи без размеров.'}},
        about='Названия из подписей пособия Мурадова 2013 (в основном безворсовые изделия и элементы).'),
    'irevan_qrupu': dict(slugs={'intangible_2854': ''}, prefix='irv',
        src={'id': 'irevan_qrupu', 'short': '«İrəvan qrupu» (Иреванская группа)', 'author': None, 'year': None,
             'title': {'az': 'İrəvan qrupu', 'ru': 'Иреванская группа'},
             'publisher': None, 'type': 'album',
             'file': 'Drive: scans/intangible/ — файл 2854 (титул и выходные данные в скане не распознаны); текст: carpet-dna/output/books/intangible_2854/ocr_pages.jsonl',
             'note': {'ru': 'Книга об Иреванской группе ковров: история Иреванского ханства и губернии, затем ковры группы. Автор и год по скану не установлены; '
                            'в SOURCES_REVIEW 2026-10-06 записана как «İrəvan qrupu» (2015), вероятно из той же серии Мурадова «Azərbaycan xalçaları bütün məktəbləri ilə» — не сверено. '
                            'В модели нет школы «Иреван»: книга — основной кандидат для её описания.'}},
        about='Ковры Иреванской группы из подписей (у 20 из 22 подписей есть век и размер).'),
    'symposium_2007_paris': dict(slugs={'azerbaycan_xalchasi': ''}, prefix='sym07',
        src={'id': 'symposium_2007_paris', 'short': 'IV Международный симпозиум (Париж, 2007), к 100-летию Л. Керимова', 'author': 'сборник; ред./дизайн Р. Тагиева', 'year': 2006,
             'title': {'az': 'Azərbaycan xalça sənəti. IV Beynəlxalq Simpoziumun materialları', 'en': 'Proceedings of the 4th International Symposium on Azerbaijanian Carpet Art (Paris, February 2007)'},
             'publisher': 'Bakı: Prezident kitabxanası, ©2006. ISBN 5-8066-1758-0', 'type': 'proceedings',
             'file': 'Drive: scans/intangible/Azerbaycan_xalchasi.pdf; текст: carpet-dna/output/books/azerbaycan_xalchasi/text.jsonl (текстовый слой PDF, 77 стр.)',
             'note': {'ru': 'Доклады симпозиума ЮНЕСКО (Париж, 2007) и избранные статьи самого Л. Керимова. Иллюстрации — знаменитые музейные ковры мира '
                            '(Берлин, V&A, Эрмитаж и др.) с подписью «название. Школа. Век. Музей». Доверие к подписям — среднее (атрибуции азербайджанской традиции, '
                            'без инв. номеров); к докладам — по авторам.'}},
        about='Ковры из подписей сборника симпозиума 2007: музейные ковры мира с указанием собрания.'),
    'ahm_kovry': dict(slugs={'intangible_2885': ''}, prefix='ahm',
        src={'id': 'ahm_kovry', 'short': 'Музей истории Азербайджана, «Ковры и ковровые изделия»', 'author': 'под общей ред. акад. П. А. Азизбековой', 'year': None,
             'title': {'ru': 'Ковры и ковровые изделия (серия «Памятники материальной культуры Азербайджана»)'},
             'publisher': 'Баку: АН АзССР, Музей истории Азербайджана (год по скану не установлен)', 'type': 'museum_catalogue',
             'file': 'Drive: scans/intangible/ — файл 2885 (21 стр.)',
             'note': {'ru': 'Советский альбом фондов Музея истории Азербайджана АН АзССР. Подписи русские: «Ворсовый ковёр „Биджо“, XIX в. (Ширван)». '
                            'Доверие — среднее-высокое (музейные предметы), но подписей мало и они короткие.'}},
        about='Ковры из подписей альбома Музея истории Азербайджана.'),
}


def match_concept(*cands):
    for c in cands:
        if not c: continue
        f = fold(c.replace('ii', 'll'))
        if not f or f in ('azerbaycan', 'xalca', 'carpet', 'kovyor', 'kover'): continue
        if f in NAME: return NAME[f], 'exact'
        m = difflib.get_close_matches(f, KEYS, 1, 0.88)
        if m and len(f) >= 4: return NAME[m[0]], 'fuzzy:' + m[0]
    return None, None


def check_school(cid, sch, page):
    t = TYPE_OF.get(sch)
    if not cid or not t: return None
    k = CS.get(cid) or {}; ks = CON.get(cid, {}).get('schools') or {}
    st = 'нет данных в модели'
    if k.get('type'): st = 'совпадает' if k['type'] == t else 'расходится'
    elif ks: st = 'совпадает' if any(x == t or TYPE_OF.get(x) == t or str(x).startswith(t) for x in ks) else 'расходится'
    return {'concept': cid, 'book_school': SCH_RU.get(sch, sch), 'book_type': t, 'page': page,
            'model_carpet_schools': ({'type': k.get('type'), 'group': k.get('group'), 'confidence': k.get('confidence')} if k else None),
            'model_tables_schools': ks or None, 'status': st}


ROM = {'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20, 'XXI': 21}
PART = {'early': 'начало', 'mid': 'середина', 'late': 'конец'}
summary = {}
for sid, S in SOURCES.items():
    plates, chk, terms = [], [], collections.OrderedDict()
    for slug, part in S['slugs'].items():
        fn = os.path.join(WORK, f'plates_{slug}.json')
        if not os.path.exists(fn): print('нет', fn); continue
        for p in json.load(open(fn, encoding='utf-8')):
            sch = p.get('school') or ''
            if not (sch or p.get('century') or p.get('size_cm') or p.get('name')): continue
            cid, how = match_concept(p.get('name'), p.get('name_en'))
            cen = ROM.get((p.get('century') or '').upper().strip())
            date = (f"{p['century']} в." + (f", {PART[p['part']]}" if p.get('part') in PART else '')) if p.get('century') else (str(p['year']) if p.get('year') else None)
            sz = p.get('size_cm')
            n = p['id'].split('_')
            plates.append({'id': f"{S['prefix']}_{n[-2]}_{n[-1]}" if len(S['slugs']) == 1 else f"{S['prefix']}{part[-1]}_{n[-2]}_{n[-1]}",
                           'pdf_page': p['page'], 'part': part or None, 'kind': KIND_RU.get(p.get('kind'), p.get('kind')),
                           'title_raw': p.get('caption'), 'name': p.get('name') or None, 'name_en': p.get('name_en') or None, 'name_ru': p.get('name_ru') or None,
                           'group': SCH_RU.get(sch) if sch else None, 'school': {'type': TYPE_OF.get(sch), 'raw': sch} if sch else None,
                           'date': date, 'modern': bool(cen and cen >= 21), 'material': p.get('material') or None,
                           'dimensions_cm': ' x '.join(str(int(v)) if float(v).is_integer() else str(v) for v in sz[:2]) if isinstance(sz, list) and len(sz) >= 2 and all(isinstance(v, (int, float)) for v in sz[:2]) else None,
                           'collection': p.get('museum') or None, 'inv_no': p.get('inv') or None,
                           'concept': cid, 'link_status': ('auto' if how == 'exact' else ('proposed' if cid else None)),
                           'read_by': 'VLM (google/gemini-3.7-flash), уверенность %.2f' % (p.get('confidence') or 0), 'cps_id': p['id']})
            if not (cen and cen >= 21):
                c = check_school(cid, sch, p['page'])
                if c: chk.append(c)
            nm = (p.get('name') or '').strip()
            if nm and fold(nm) not in ('xalca', 'azerbaycan') and len(nm.split()) <= 4 and not re.search(r'məntəqə|kompozisiya|nümunə', nm):
                e = terms.setdefault(nm, {'term': nm, 'term_en': p.get('name_en') or None, 'term_ru': p.get('name_ru') or None, 'concept': cid,
                                          'definition_ru': 'название по подписи к фото (' + KIND_RU.get(p.get('kind'), '') + ')',
                                          'confidence': 'high' if how == 'exact' else 'medium', 'new_candidate': not cid, 'pages': [], 'schools': collections.Counter()})
                e['pages'].append(p['page'])
                if sch: e['schools'][SCH_RU.get(sch, sch)] += 1
    d = os.path.join(OUTROOT, sid); os.makedirs(d, exist_ok=True)
    tl = []
    for e in terms.values():
        e = dict(e); e['schools'] = dict(e['schools']); e['n'] = len(e['pages']); e['pages'] = sorted(set(e['pages'])); tl.append(e)
    json.dump({'source': S['src'], 'plates': plates}, open(f'{d}/plates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({'_about': S['about'] + ' concept — понятие модели (auto — точное совпадение имени, proposed — нечёткое); new_candidate — понятия в модели нет.',
               'terms': tl, 'borders': [], 'type_candidates': [e['term'] for e in tl if e['new_candidate']]},
              open(f'{d}/glossary_map.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({'_about': 'Сверка школ: подпись книги против модели (carpet_schools.json; concepts.schools — таблицы Керимова). Ковры XXI в. не сверяются. '
                         'Правило 4: расхождение фиксируется, не выбирается.', 'items': chk},
              open(f'{d}/school_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    summary[sid] = {'plates': len(plates), 'with_concept': sum(1 for p in plates if p['concept']), 'terms': len(tl), 'new': sum(e['new_candidate'] for e in tl),
                    'checks': dict(collections.Counter(c['status'] for c in chk)), 'modern': sum(p['modern'] for p in plates),
                    'new_terms': [e['term'] for e in tl if e['new_candidate']][:40],
                    'diverge': [(c['concept'], c['book_school'], (c.get('model_carpet_schools') or {}).get('type') or c.get('model_tables_schools')) for c in chk if c['status'] == 'расходится'][:20]}
json.dump(summary, open(os.path.join(WORK, 'dna_series_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in summary.items(): print(k, {x: v[x] for x in ('plates', 'with_concept', 'terms', 'new', 'checks', 'modern')})
