import os
REPO = os.environ.get('REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
WORK = os.environ.get('BOOKS_WORK', '.')   # папка с enc_items_full.json (items.jsonl энциклопедии), items_passport.jsonl (Мурадов)
import json, re, unicodedata, collections
rows = json.load(open(os.path.join(WORK, 'enc_items_full.json')))
C = json.load(open(REPO + '/site/model/out/concepts.json'))
def fold(s):
    s = unicodedata.normalize('NFKD', (s or '').lower().replace('ə','e').replace('ı','i').replace('ğ','g').replace('ş','s').replace('ç','c').replace('ö','o').replace('ü','u'))
    return re.sub(r'[^a-z0-9а-я]+', '', ''.join(ch for ch in s if not unicodedata.combining(ch)))
NAME = {}
for c in C:
    for n in [c['headword']] + [x['v'] for x in c.get('names', [])]:
        NAME.setdefault(fold(n), c['id'])
# типы изделий энциклопедии (OCR-варианты → каноническое az)
OBJ = [('dəvə çulu','Dəvə çulu'),('çiyi palaz','Çiyi palaz'),('ciyi palaz','Çiyi palaz'),('dama-dama şəddə','Şəddə'),('dəvəli şəddə','Şəddə'),('şal şəddə','Şəddə'),
       ('duz torbası','Duz torbası'),('yəhər üstü','Yəhərüstü'),('taxt üstü','Taxtüstü'),('dəstərxan','Dəstərxan'),('pərdə','Pərdə'),('ləməqabağı','Ləməqabağı'),
       ('məfrəş','Məfrəş'),('xurcun','Xurcun'),('heybə','Heybə'),('heyba','Heybə'),('çanta','Çanta'),('corab','Corab'),('çul','Çul'),('cul','Çul'),('döşəkçə','Döşəkçə'),
       ('keçə','Keçə'),('buriya','Buriya'),('həsir','Həsir'),('çətən','Çətən'),('catan','Çətən'),('cecim','Cecim'),('sənnə kilim','Kilim'),('kilim','Kilim'),('palaz','Palaz'),
       ('qədirqə','Qədirqə'),('ladı','Ladı'),('sumax','Sumax'),('şəddə','Şəddə'),('vərni','Vərni'),('zili','Zili'),('xalça','Xalça'),('hana','Hana')]
OBJ_RU = {'Kilim':'килим','Palaz':'палас','Çiyi palaz':'палас «чийи»','Cecim':'джеджим','Zili':'зили','Sumax':'сумах','Şəddə':'шадда','Vərni':'верни','Ladı':'лады',
          'Qədirqə':'гадырга','Çul':'чул (попона)','Dəvə çulu':'попона для верблюда','Heybə':'хейбе (перемётная сума)','Xurcun':'хурджин','Məfrəş':'мафраш',
          'Duz torbası':'солонка (дуз торбасы)','Çanta':'сумка','Corab':'носки','Dəstərxan':'дастархан (скатерть)','Pərdə':'занавес','Ləməqabağı':'ламаганабагы',
          'Yəhərüstü':'чепрак (на седло)','Taxtüstü':'покрывало на тахту','Döşəkçə':'тюфячок','Keçə':'войлок','Buriya':'бурья (циновка)','Həsir':'циновка',
          'Çətən':'чатан (камыш)','Xalça':'ковёр','Hana':'ткацкий станок'}
PILE = {'Xalça'}
SCH = [('qarabağ','karabakh','Карабах'),('qarabaş','karabakh','Карабах'),('qazax','gazakh','Газах'),('gəncə','ganja','Гянджа'),('şirvan','shirvan','Ширван'),
       ('bakı','baku','Баку'),('quba','guba','Губа'),('təbriz','tabriz','Тебриз'),('naxçıvan','nakhchivan','Нахчыван'),('lənkəran','talysh','Ленкорань'),
       ('şəki','sheki','Шеки'),('xızı','baku','Хызы (Баку)'),('muğan','mugan','Муган')]
TYPE_OF = {'guba':'guba-shirvan','shirvan':'guba-shirvan','baku':'guba-shirvan','ganja':'ganja-gazakh','gazakh':'ganja-gazakh','karabakh':'karabakh','tabriz':'tabriz'}
PART = [('əvvəl','начало'),('əvgəl','начало'),('əovəl','начало'),('avval','начало'),('aveal','начало'),('əogəl','начало'),('orta','середина'),('sonu','конец'),('sonlar','конец')]
def parse(cap):
    t = re.sub(r'\s+', ' ', cap or '').strip(); low = t.lower(); p = {}
    if re.match(r'^[A-ZƏÖÜÇŞĞİ][A-ZƏÖÜÇŞĞİ ,\-]{2,}\s*[-–—]', t): return {'glossary': True}
    head = re.split(r'\bAz[əa]rbayca', t, maxsplit=1)[0]
    hl = head.lower()
    for k, v in OBJ:
        if re.search(r'(^|[\s.“"])' + re.escape(k) + r'\b', hl): p['obj'] = v; break
    m = re.search(r'[“"«]\s*([^”"»]{2,40}?)\s*[”"»]', head)
    if m and re.fullmatch(r"[A-Za-zƏəÖöÜüÇçŞşĞğİıI' \-]{3,30}", m.group(1).strip()): p['name'] = m.group(1).strip()   # без мусора OCR
    for k, sid, ru in SCH:
        if re.search(r'\b' + k, hl): p['school'] = sid; p['school_ru'] = ru; break
    m = re.search(r'Müəllif:\s*([^.]+(?:\.[A-ZƏÖÜÇŞ][a-zəöüçşığ]+)?)', t)
    if m: p['author'] = m.group(1).strip()
    m = re.search(r'(\d{1,2})\s*-?\s*c[iıu]\w*\s+[əa]sr(\w*)\s*(\S*)', low)
    if m:
        p['century'] = int(m.group(1)); tail = (m.group(3) or '')
        p['century_part'] = next((ru for k, ru in PART if k in tail), '') if m.group(2) else ''
        m2 = re.search(r'(\d0)\s*-?\s*c[iıu]\s+ill', low)
        if m2: p['century_part'] = f'{m2.group(1)}-е годы'
    m = re.search(r'\b(1[5-9]\d\d|20[0-2]\d)\s*(?:-\s*c\w*\s*il|\b)', t)
    if m and ('il' in low or 'Müəllif' in t or re.search(r'\b(19[4-9]\d|20[0-2]\d)\.\s*Material', t)): p['year'] = int(m.group(1))
    m = re.search(r'(\d{4})\s*hicri', low)
    if m: p['hijri'] = int(m.group(1))
    m = re.search(r'Mat\w*\W{0,3}\s*([a-zəöüçşğı ,\-]{2,30})\.', t, re.I)
    if m: p['material_az'] = m.group(1).strip(' ,')
    m = re.search(r'[ÖO0]l[çc]\w*\W{0,3}\s*(\d{1,3}(?:,\d)?)\s*x\s*(\d{1,3}(?:,\d)?)(?:\s*x\s*(\d{1,3}))?', t)
    if m: p['size_cm'] = [float(g.replace(',', '.')) for g in m.groups() if g]
    m = re.search(r'\bD\s?D\s?[KX]\s*(\d{1,5})', t)
    if m: p['inv'] = 'DDK ' + m.group(1)
    if 'xalça muzey' in low or 'xalca muzey' in low: p['museum'] = 'Azərbaycan Xalça Muzeyi'
    if 'şəxsi kolleksiya' in low: p['museum'] = 'Şəxsi kolleksiya'
    return p
out = []; gloss = []
for r in rows:
    p = parse(r.get('caption'))
    if p.get('glossary'): gloss.append({'page': r['page'], 'text': r['caption']}); kind = 'glossary_photo'
    elif p.get('size_cm') or p.get('inv') or (p.get('obj') and (p.get('century') or p.get('year'))): kind = 'item'
    elif p.get('century') and p.get('material_az'): kind = 'item'
    else: kind = 'other'
    obj = p.get('obj')
    cid = None
    for cand in (p.get('name'), obj):
        if cand and fold(cand) in NAME: cid = NAME[fold(cand)]; break
    cid_obj = NAME.get(fold(obj)) if obj else None
    out.append({**{k: r[k] for k in ('id', 'page', 'n', 'file', 'px', 'caption')}, 'kind': kind, **p,
                'concept': NAME.get(fold(p['name'])) if p.get('name') else None, 'concept_obj': cid_obj})
json.dump(out, open(os.path.join(WORK, 'enc_parsed.json'), 'w'), ensure_ascii=False, indent=0)
it = [o for o in out if o['kind'] == 'item']
print('items', len(it), 'other', sum(o['kind'] == 'other' for o in out), 'gloss-photos', len(gloss))
print('obj', collections.Counter(o.get('obj') for o in it).most_common(40))
print('school', collections.Counter(o.get('school') for o in it))
print('inv', sum(1 for o in it if o.get('inv')), 'size', sum(1 for o in it if o.get('size_cm')), 'date', sum(1 for o in it if o.get('century') or o.get('year')))
print('names', [(o['name'], o['concept']) for o in it if o.get('name')])
print('obj concepts', {o['obj']: o['concept_obj'] for o in it if o.get('obj')})
dd = collections.Counter(o.get('inv') for o in it if o.get('inv')); print('dup inv', [k for k, v in dd.items() if v > 1])
