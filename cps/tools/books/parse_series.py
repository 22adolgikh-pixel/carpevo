"""Разбор подписей альбомов Мурадова (серия «Azərbaycan xalçaları / Azerbaijan Carpets / Азербайджанские ковры», Azər-İlmə)
и сборника симпозиума 2007 по полному OCR страниц. Один ковёр на странице; подпись трёхъязычная (az / en / ru).

Вход (BOOKS_WORK): <slug>_ocr.jsonl ({"page","text"}) и <slug>_items.jsonl (book_extract_colab v5).
Выход: <slug>_parsed.json — список {id, page, name, name_en, name_ru, obj, school_az, school_ru, century, part, year, hijri, material, size_cm, line}.
"""
import os, re, json, sys, collections
WORK = os.environ.get('BOOKS_WORK', '.')
SIZE = re.compile(r'(\d{2,3})\s*[xхXХ×*]\s*(\d{2,3})\s*(?:sm|cm|см|sm\.)?', re.I)
ROMAN = r'(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|İX|IX|XİX|XVİİİ|XVİİ|XVİ|XİV|XİII|XİI|Xİ|Xİİİ)'
CENT = re.compile(r'\b(' + ROMAN + r')\s*(?:əsr|ə[sş]r|century|век|в\.)', re.I)
PART_AZ = [('əvvəli', 'начало'), ('ortası', 'середина'), ('sonu', 'конец'), ('birinci yarısı', '1-я половина'), ('ikinci yarısı', '2-я половина')]
PART_EN = [('early', 'начало'), ('middle', 'середина'), ('late', 'конец'), ('first half', '1-я половина'), ('second half', '2-я половина')]
PART_RU = [('начало', 'начало'), ('середина', 'середина'), ('конец', 'конец'), ('первая половина', '1-я половина'), ('вторая половина', '2-я половина')]
OBJ_AZ = {'xalçası': 'Xalça', 'xalça': 'Xalça', 'palaz': 'Palaz', 'palazı': 'Palaz', 'kilim': 'Kilim', 'kilimi': 'Kilim', 'xurcun': 'Xurcun', 'xurcunu': 'Xurcun',
          'heybə': 'Heybə', 'heybəsi': 'Heybə', 'məfrəş': 'Məfrəş', 'məfrəşi': 'Məfrəş', 'çul': 'Çul', 'çulu': 'Çul', 'zili': 'Zili', 'sumax': 'Sumax',
          'cecim': 'Cecim', 'cecimi': 'Cecim', 'şəddə': 'Şəddə', 'şəddəsi': 'Şəddə', 'vərni': 'Vərni', 'namazlıq': 'Namazlıq', 'torba': 'Torba', 'torbası': 'Torba',
          'xovlu': 'Xalça', 'sandıq': 'Sandıq'}
SCHOOLS = {'quba': 'guba', 'şirvan': 'shirvan', 'qarabağ': 'karabakh', 'gəncə': 'ganja', 'qazax': 'gazakh', 'naxçıvan': 'nakhchivan', 'bakı': 'baku',
           'təbriz': 'tabriz', 'irəvan': 'irevan', 'şəki': 'shaki', 'muğan': 'mugan', 'lənkəran': 'lankaran', 'talış': 'talish', 'cəbrayıl': 'karabakh',
           'şuşa': 'karabakh', 'qobustan': 'shirvan', 'ərdəbil': 'tabriz', 'borçalı': 'gazakh', 'dərbənd': 'guba', 'zəngəzur': 'karabakh'}


def clean(s):
    return re.sub(r'\s+', ' ', s or '').strip()


def parse_line_az(ln):
    """«“Zeyvə” xalçası. Quba məktəbi. Azərbaycan. XIX əsrin əvvəli. Yun, 192х120 sm» → dict или None."""
    m = SIZE.search(ln)
    if not m or not re.search(r'ə[sş]r|hicri|\b1[5-9]\d\d\b', ln, re.I): return None
    head = ln[:m.start()]
    segs = [clean(x) for x in re.split(r'[.]\s*', head) if clean(x)]
    d = {'size_cm': [int(m.group(1)), int(m.group(2))]}
    mat = re.search(r'(yun|ipək|pambıq|yun-ipək|keçə|qarışıq)\b', head[::-1][:0] + head, re.I)
    d['material'] = {'yun': 'шерсть', 'ipək': 'шёлк', 'pambıq': 'хлопок', 'keçə': 'войлок'}.get((mat.group(1).lower() if mat else ''), '') if mat else ''
    cm = CENT.search(head)
    if cm: d['century'] = cm.group(1).upper().replace('İ', 'I')
    for k, v in PART_AZ:
        if re.search(r'ə[sş]rin\s+' + k, head, re.I): d['part'] = v; break
    hy = re.search(r'(\d{3,4})\s*hicri', head, re.I)
    if hy: d['hijri'] = int(hy.group(1))
    yr = re.search(r'\b(1[6-9]\d\d)\b', head)
    if yr and not hy: d['year'] = int(yr.group(1))
    elif yr and hy:
        ys = re.findall(r'\b(1[89]\d\d)\b', head)
        if ys: d['year'] = int(ys[-1])
    title = segs[0] if segs else ''
    nm = re.search(r'[“"„«”]\s*([^”"»“]{2,40}?)\s*[”"»“]', title)
    d['name'] = clean(nm.group(1)) if nm else ''
    low = title.lower()
    obj = ''
    for k, v in OBJ_AZ.items():
        if re.search(r'\b' + k + r'\b', low): obj = v; break
    d['obj'] = obj
    if not d['name'] and not obj:
        t = clean(re.sub(r'^[^\w]+|[^\w]+$', '', title))
        if t and len(t) < 30 and not re.search(r'\d', t): d['name'] = t
    sch = ''
    for sg in segs[1:4]:
        w = re.sub(r'\b(məktəbi|qrupu|xalça|məktəb)\b', '', sg, flags=re.I).strip(' ,')
        if w.lower() in SCHOOLS: sch = w; break
    d['school_az'] = sch
    d['school'] = SCHOOLS.get(sch.lower(), '')
    d['line'] = clean(ln)
    return d


def parse_line_en(ln):
    m = re.search(r'[“"„]([^”"]{2,40})[”"]\s*(carpet|prayer rug|rug|kilim|palas|khurjin)', ln, re.I)
    if not m: m = re.search(r'^\s*([A-Z][\w\' -]{2,30}?)\.\s*[A-Z]', ln)
    return clean(m.group(1)) if m else ''


def parse_line_ru(ln):
    m = re.search(r'[«"“]([^»"”]{2,40})[»"”]', ln)
    return clean(m.group(1)) if m else ''


RU_SCHOOL = [('губин', 'guba'), ('ширван', 'shirvan'), ('гарабах', 'karabakh'), ('карабах', 'karabakh'), ('тебриз', 'tabriz'), ('нахчыван', 'nakhchivan'),
             ('нахичеван', 'nakhchivan'), ('баки', 'baku'), ('бакин', 'baku'), ('газах', 'gazakh'), ('казах', 'gazakh'), ('гянджин', 'ganja'), ('гандж', 'ganja'),
             ('шеки', 'shaki'), ('муган', 'mugan'), ('ленкоран', 'lankaran'), ('ирван', 'irevan'), ('иреван', 'irevan')]
EN_SCHOOL = {'guba': 'guba', 'quba': 'guba', 'shirvan': 'shirvan', 'garabagh': 'karabakh', 'karabakh': 'karabakh', 'tabriz': 'tabriz', 'nakhchivan': 'nakhchivan',
             'baku': 'baku', 'gazakh': 'gazakh', 'kazakh': 'gazakh', 'ganja': 'ganja', 'shaki': 'shaki', 'mugan': 'mugan', 'lankaran': 'lankaran', 'irevan': 'irevan', 'erivan': 'irevan'}
ROM = {'XIX': 'XIX', 'XVIII': 'XVIII', 'XVII': 'XVII', 'XVI': 'XVI', 'XX': 'XX', 'XV': 'XV', 'XIV': 'XIV'}


def lang_of(ln):
    if re.search(r'[А-Яа-яЁё]{3,}', ln): return 'ru'
    if re.search(r'\b(carpet|rug|school|century|wool|cm|kilim|silk)\b', ln, re.I) and not re.search(r'[əğıöüşç]', ln, re.I): return 'en'
    return 'az'


def parse_other(ln, lang):
    """en/ru строка: размер, век, школа (для добора, когда az-строка испорчена OCR)."""
    m = SIZE.search(ln)
    if not m: return None
    d = {'size_cm': [int(m.group(1)), int(m.group(2))], 'line': clean(ln)}
    cm = re.search(r'\b(XX|XIX|XVIII|XVII|XVI|XV|XIV)\b', ln)
    if cm: d['century'] = cm.group(1)
    hy = re.search(r'(\d{3,4})\s*(?:Hijra|хиджр|хид)', ln, re.I)
    if hy: d['hijri'] = int(hy.group(1))
    low = ln.lower()
    if lang == 'ru':
        for k, v in RU_SCHOOL:
            if k in low: d['school'] = v; break
        n = re.search(r'[«"“]([^»"”]{2,40})[»"”]', ln)
        d['name_ru'] = clean(n.group(1)) if n else ''
        for k, v in PART_RU:
            if re.search(k + r'\s+[XVI]+\s*века', ln, re.I): d['part'] = v; break
    else:
        for k, v in EN_SCHOOL.items():
            if re.search(r'\b' + k + r'\b', low): d['school'] = v; break
        n = re.search(r'[“"„]([^”"]{2,40})[”"]\s*(?:carpet|prayer rug|rug|kilim)', ln, re.I)
        d['name_en'] = clean(n.group(1)) if n else ''
        for k, v in PART_EN:
            if re.search(k + r'\s+[XVI]+\s*century', ln, re.I): d['part'] = v; break
    if re.search(r'wool|шерсть', ln, re.I): d['material'] = 'шерсть'
    if re.search(r'silk|шёлк|шелк', ln, re.I): d['material'] = 'шёлк'
    return d


import difflib
HDR_AZ = ['quba', 'şirvan', 'qarabağ', 'naxçıvan', 'bakı', 'qazax', 'gəncə', 'təbriz', 'irəvan', 'şəki', 'muğan', 'lənkəran', 'talış', 'qaradağ']
HDR_EN = {'guba': 'guba', 'shirvan': 'shirvan', 'garabagh': 'karabakh', 'nakhchivan': 'nakhchivan', 'baku': 'baku', 'gazakh': 'gazakh', 'ganja': 'ganja',
          'tabriz': 'tabriz', 'irevan': 'irevan', 'shaki': 'shaki', 'mugan': 'mugan', 'lankaran': 'lankaran', 'talysh': 'talish'}
HDR_RU = {'губинская': 'guba', 'ширванская': 'shirvan', 'гарабахская': 'karabakh', 'карабахская': 'karabakh', 'нахчыванская': 'nakhchivan', 'бакинская': 'baku',
          'газахская': 'gazakh', 'казахская': 'gazakh', 'гянджинская': 'ganja', 'тебризская': 'tabriz', 'ирванская': 'irevan', 'иреванская': 'irevan', 'шекинская': 'shaki'}


def header_school(txt):
    """Школа по колонтитулу страницы «Quba xalça məktəbi / Guba carpet school / Губинская ковровая школа» (устойчив к шуму OCR)."""
    for ln in txt.split('\n'):
        l = ln.lower()
        if '/' not in l or not re.search(r'm[əa]kt|school|школ', l): continue
        for part in re.split(r'/', l):
            toks = re.findall(r'[a-zəğıöüşçа-я]{4,}', part)
            for t in toks[:3]:
                m = difflib.get_close_matches(t, HDR_AZ, 1, 0.72)
                if m: return SCHOOLS[m[0]]
                m = difflib.get_close_matches(t, list(HDR_EN), 1, 0.75)
                if m: return HDR_EN[m[0]]
                m = difflib.get_close_matches(t, list(HDR_RU), 1, 0.78)
                if m: return HDR_RU[m[0]]
    return ''


def salvage_size(ln):
    """«Yun, 180135 sm» (потерян знак ×) → [180, 135]"""
    m = re.search(r'(\d{3})\s?(\d{2,3})\s*(?:sm|cm|см|sin|zm|om|em)\b', ln)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 40 <= a <= 700 and 30 <= b <= 400: return [a, b]
    return None


def name_guess(ln):
    m = re.search(r'[“"„”]\s*([^”"“]{3,30}?)\s*[”"“]?\s*xal[çcӧ]', ln, re.I)
    return clean(m.group(1)) if m else ''


def build(slug):
    ocr = {}
    for l in open(os.path.join(WORK, f'{slug}_ocr.jsonl'), encoding='utf8'):
        r = json.loads(l); ocr[r['page']] = r['text']
    items = [json.loads(l) for l in open(os.path.join(WORK, f'{slug}_items.jsonl'), encoding='utf8')]
    by_page = collections.defaultdict(list)
    for it in items: by_page[it['page']].append(it)
    out, multi, nopic = [], [], []
    for pg, txt in sorted(ocr.items()):
        lines = [l for l in txt.split('\n') if l.strip()]
        rec = {'az': [], 'en': [], 'ru': []}
        for ln in lines:
            lg = lang_of(ln)
            if lg == 'az':
                d = parse_line_az(ln)
                if d: rec['az'].append(d)
            else:
                d = parse_other(ln, lg)
                if d: rec[lg].append(d)
        n = max(len(v) for v in rec.values())
        hs = header_school(txt)
        if n == 0:
            if not hs: continue
            if not any('/' not in l and re.search(r'azerb|wool|centur|hijra|hi[jy]ra|азербайдж|шерст', l, re.I) for l in lines): continue   # колонтитул без подписи — не табличка ковра
            # колонтитул есть, подписи испорчены: карточка только со школой
            az_l = [l for l in lines if lang_of(l) == 'az' and re.search(r'xal[çcӧа]|m[əa]kt', l, re.I) and '/' not in l]
            d = {'name': name_guess(az_l[0]) if az_l else '', 'obj': '', 'school_az': '', 'school': hs, 'material': '', 'size_cm': salvage_size(' '.join(lines)) or [],
                 'name_en': '', 'name_ru': '', 'line': clean(az_l[0]) if az_l else '', 'from': 'h', 'recheck': True}
            rec = {'az': [d], 'en': [], 'ru': []}
            n = 1
        caps = []
        for k in range(n):
            a = rec['az'][k] if k < len(rec['az']) else {}
            e = rec['en'][k] if k < len(rec['en']) else {}
            r_ = rec['ru'][k] if k < len(rec['ru']) else {}
            d = dict(a)
            for src in (e, r_):
                for key in ('size_cm', 'century', 'hijri', 'material', 'part', 'school'):
                    if not d.get(key) and src.get(key): d[key] = src[key]
            d.setdefault('name', ''); d.setdefault('obj', ''); d.setdefault('school_az', ''); d.setdefault('school', ''); d.setdefault('material', '')
            d['name_en'] = e.get('name_en', ''); d['name_ru'] = r_.get('name_ru', '')
            d['line'] = a.get('line') or e.get('line') or r_.get('line')
            d['from'] = ''.join(x for x, y in (('a', a), ('e', e), ('r', r_)) if y)
            if not d.get('school') and hs: d['school'] = hs
            caps.append(d)
        pics = [it for it in by_page.get(pg, []) if it.get('how') != 'skip']
        pics.sort(key=lambda it: -(it['px'][0] * it['px'][1]))
        if not pics: nopic.append(pg); continue
        if len(caps) > 1:
            multi.append(pg)
            big = [p for p in pics if p['px'][0] * p['px'][1] > 250000]
            if len(big) != len(caps): continue
            pics = sorted(big, key=lambda it: (it['bbox_pt'][0], it['bbox_pt'][1]))
            for c, p in zip(caps, pics): out.append({'id': p['id'], 'page': pg, **c})
            continue
        out.append({'id': pics[0]['id'], 'page': pg, **caps[0]})
    json.dump(out, open(os.path.join(WORK, f'{slug}_parsed.json'), 'w'), ensure_ascii=False, indent=1)
    print(slug, 'карточек', len(out), '| страниц с 2+ подписями', multi, '| без вырезки', nopic)
    print(' школы', collections.Counter(o['school'] or '?' for o in out).most_common(8))
    print(' без названия', sum(1 for o in out if not o['name']), '| без века', sum(1 for o in out if not o.get('century') and not o.get('year') and not o.get('hijri')), '| без размера', sum(1 for o in out if not o['size_cm']))
    return out


if __name__ == '__main__':
    for s in sys.argv[1:] or ['intangible_2849', 'intangible_2850']:
        build(s)
