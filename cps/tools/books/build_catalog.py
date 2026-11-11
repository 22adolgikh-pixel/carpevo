"""Каталог книжных ковров для CPS (формат элемента как в rug_catalog_colab.add) + сырьё для Carpet DNA."""
import os
REPO = os.environ.get('REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
WORK = os.environ.get('BOOKS_WORK', '.')   # папка с enc_items_full.json (items.jsonl энциклопедии), items_passport.jsonl (Мурадов)
import json, re, collections, time, sys
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_enc import fold, NAME, OBJ_RU, TYPE_OF
enc = json.load(open(os.path.join(WORK, 'enc_parsed.json')))
mur = [json.loads(l) for l in open(os.path.join(WORK, 'items_passport.jsonl'))]
azcm = json.load(open(REPO + '/cps/tools/colab/data/azcarpetmuseum_items.json'))
ENC = 'azerbaycan_xalchasi_ensiklopediyasi_ii_cild'; MUR = 'azerbaycan_xalchalari_qarabag_qrupu'
TECH = {'Kilim': 'безворсовый: килим', 'Zili': 'безворсовый: зили', 'Sumax': 'безворсовый: сумах', 'Palaz': 'палас', 'Çiyi palaz': 'палас',
        'Şəddə': 'безворсовый: шадда', 'Cecim': 'джеджим', 'Vərni': 'безворсовый: верни', 'Ladı': 'безворсовый: лады', 'Qədirqə': 'безворсовый: гадырга',
        'Keçə': 'войлок', 'Buriya': 'плетение (камыш)', 'Çətən': 'плетение (камыш)', 'Həsir': 'плетение (камыш)', 'Xalça': 'ворсовый'}
BAG = {'Xurcun', 'Heybə', 'Məfrəş', 'Duz torbası', 'Çanta'}
HORSE = {'Çul', 'Dəvə çulu', 'Yəhərüstü', 'Ləməqabağı'}
HOME = {'Corab', 'Dəstərxan', 'Döşəkçə', 'Pərdə', 'Taxtüstü'}
import difflib
KEYS = list(NAME)
def match_concept(*cands):
    for c in cands:
        if not c or fold(c) in ('azerbaycan', ''): continue
        f = fold(c.replace('ii', 'll').replace('iı', 'll'))
        if f in NAME: return NAME[f], 'exact'
        m = difflib.get_close_matches(f, KEYS, 1, 0.86)
        if m and len(f) >= 4: return NAME[m[0]], 'fuzzy:' + m[0]
    return None, None
def num(v): return re.sub(r'\D', '', v or '')
AZ = {}
for x in azcm: AZ.setdefault(num(x.get('inv')), x)
def sz_ok(a, s):
    m = [int(v) for v in re.findall(r'\d+', s or '')][:2]
    if not a or len(m) < 2: return None
    a = sorted(a[:2]); m = sorted(m)
    return max(abs(p - q) / q for p, q in zip(a, m)) < 0.12
items, groups, plates_enc, plates_mur = [], [], [], []
for o in enc:
    if o['kind'] != 'item': continue
    obj = o.get('obj'); sch = o.get('school')
    date = (f"{o['century']} в." + (f", {o['century_part']}" if o.get('century_part') else '')) if o.get('century') else (str(o['year']) if o.get('year') else '')
    if o.get('hijri'): date = f"{date} ({o['hijri']} г. хиджры)".strip()
    title = ' '.join(x for x in [obj or '', f"«{o['name']}»" if o.get('name') else '', o.get('school_ru') or ''] if x) or 'Энциклопедия, т. II, стр. %d' % o['page']
    it = dict(id=o['id'], src='book:' + ENC, img=f"img/{o['id']}.jpg", title=title,
              names=[x for x in (o.get('name'), obj, OBJ_RU.get(obj)) if x], composition=o.get('name') or '',
              date=date, school=TYPE_OF.get(sch, ''), school_why=(f"подпись энциклопедии: «{o['school_ru']}»" if sch else ''),
              technique=TECH.get(obj, ''), fragment=False, text=o['caption'],
              museum=('Национальный музей азербайджанского ковра' if o.get('museum') == 'Azərbaycan Xalça Muzeyi' else (o.get('museum') or '')),
              inv=num(o.get('inv')), place=o.get('school_ru') or '', url='', license='книга: только анализ/обучение, не публиковать',
              size=('x'.join(str(int(v)) if float(v).is_integer() else str(v) for v in o['size_cm']) + ' см') if o.get('size_cm') else '',
              page=o['page'], book=ENC)
    if obj in BAG: it['object'] = 'мешок / сумка'
    elif obj in HORSE: it['object'] = 'конское снаряжение'
    elif obj in HOME: it['object'] = 'бытовое изделие'
    elif obj == 'Hana': it['object'] = 'станок (не ковёр)'
    if o.get('author'): it['author'] = o['author']
    items.append(it)
    tw = AZ.get(num(o.get('inv'))) if o.get('inv') else None
    if tw:
        ok = sz_ok(o.get('size_cm'), tw.get('size'))
        if ok is not False:
            groups.append({'id': 'g_' + o['id'], 'members': [o['id'], f"azcm_{tw['id']}"], 'pairs': [[o['id'], f"azcm_{tw['id']}", 999, 'inv']],
                           'why': f"DDK {num(o['inv'])} = инв. {tw['inv']} музея; размер {'совпал' if ok else 'не указан'}"})
            it['same_group'] = 'g_' + o['id']
    plates_enc.append({**{k: o.get(k) for k in ('id', 'page', 'n', 'caption', 'obj', 'name', 'school', 'century', 'century_part', 'year', 'hijri',
                                                 'material_az', 'size_cm', 'inv', 'museum', 'author', 'concept', 'concept_obj')},
                       'museum_twin': (f"azcm_{tw['id']}" if tw else None), 'museum_twin_size_ok': (sz_ok(o.get('size_cm'), tw.get('size')) if tw else None),
                       'museum_twin_place': ((tw.get('place') or {}).get('az') if tw else None), 'museum_twin_name': ((tw.get('n') or {}).get('az') if tw else None)})
# --- Мурадов 2010: pdf-страница = разворот, печатная ≈ 2*(p-1); с печатной 87 — коллекция «Азер-ильме», с 237 — новые ковры «Азер-ильме»
for r in mur:
    if r.get('kind_guess') != 'item': continue
    p = r.get('passport') or {}; pr = 2 * (r['page'] - 1)
    part = 'introduction' if pr < 87 else ('azerilme_collection' if pr < 237 else 'azerilme_new')
    title_az = re.sub(r'\s+', ' ', p.get('title_az') or '').strip()
    nm = p.get('name') or ''
    m = re.search(r'[“"«]\s*([^”"»]{2,40}?)\s*[”"»]', title_az)
    if m and not nm: nm = m.group(1)
    nm = nm.strip(' .')
    lead = title_az.split('.')[0].strip()
    if re.search(r'\d', nm) or fold(nm) == 'azerbaycan': nm = ''
    nm = {'Çaləbi': 'Çələbi', 'Balyg': 'Balıq'}.get(nm, nm)
    concept, how = match_concept(nm, p.get('name_en'))
    concept_lead = NAME.get(fold(lead)) if lead else None
    cent = p.get('century'); date = f'{cent} в.' if cent else (str(p['years'][0]) if p.get('years') else '')
    it = dict(id=r['id'], src='book:' + MUR, img=f"img/{r['id']}.jpg", title=title_az[:120] or nm or 'Мурадов 2010, разворот %d' % r['page'],
              names=[x for x in (nm, p.get('name_en'), lead, p.get('title_en')) if x], composition=nm, date=date, school='karabakh',
              school_why='книга «Azərbaycan xalçaları. Qarabağ qrupu» (Мурадов 2010)' + ('' if part == 'introduction' else '; коллекция «Азер-ильме» — слабое свидетельство'),
              technique=(p.get('technique') or [''])[0] if isinstance(p.get('technique'), list) else (p.get('technique') or ''),
              fragment=('fraqment' in title_az.lower()), text=title_az, museum=p.get('collection') or '', inv=p.get('inv') or '', place='',
              url='', license='книга: только анализ/обучение, не публиковать',
              size=('x'.join(str(int(v)) for v in p['size_cm'][:2]) + ' см') if p.get('size_cm') else '', page=r['page'], book=MUR)
    if part == 'azerilme_new': it['object'] = 'современная копия (Азер-ильме)'
    items.append(it)
    plates_mur.append({'id': r['id'], 'pdf_page': r['page'], 'printed_page_approx': pr, 'part': part, 'title_az': title_az, 'title_en': p.get('title_en'),
                       'name': nm or None, 'name_en': p.get('name_en'), 'concept': concept, 'concept_how': how, 'concept_lead': concept_lead,
                       **{k: p.get(k) for k in ('century', 'years', 'size_cm', 'field_cm', 'density', 'knot', 'colors_n', 'condition', 'material', 'technique', 'collection', 'item_type')}})
cat = {'built': time.strftime('%Y-%m-%d %H:%M'), 'extra': True, 'about': 'Книги: Энциклопедия азерб. ковра т. II; Мурадов «Qarabağ qrupu» (2010). Фото только для анализа.',
       'items': items, 'groups': groups}
json.dump(cat, open(os.path.join(WORK, 'books_catalog.json'), 'w'), ensure_ascii=False)
json.dump(plates_enc, open(os.path.join(WORK, 'plates_enc.json'), 'w'), ensure_ascii=False, indent=1)
json.dump(plates_mur, open(os.path.join(WORK, 'plates_mur.json'), 'w'), ensure_ascii=False, indent=1)
print('items', len(items), collections.Counter(i['src'] for i in items), 'groups', len(groups))
print('obj', collections.Counter(i.get('object', 'ковёр') for i in items))
print('fuzzy', [(p['name'],p['concept'],p['concept_how']) for p in plates_mur if (p.get('concept_how') or '').startswith('fuzzy')]); print('unmatched', sorted({p['name'] for p in plates_mur if p['name'] and not p['concept']}))
print('mur parts', collections.Counter(p['part'] for p in plates_mur), 'mur concepts', sum(1 for p in plates_mur if p['concept']), [ (p['name'],p['concept']) for p in plates_mur if p['name']][:40])
