"""Источники для Carpet DNA (site/model/sources/<id>/plates.json + glossary_map.json + school_check.json)."""
import os
REPO = os.environ.get('REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
WORK = os.environ.get('BOOKS_WORK', '.')   # папка с enc_items_full.json (items.jsonl энциклопедии), items_passport.jsonl (Мурадов)
import json, os, re, collections, sys
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_enc import fold, NAME, OBJ_RU, TYPE_OF
OUTROOT = os.environ.get('OUT', REPO + '/site/model/sources')
CS = json.load(open(REPO + '/site/model/carpet_schools.json'))['items']
CON = {c['id']: c for c in json.load(open(REPO + '/site/model/out/concepts.json'))}
SCH_RU = {'karabakh': 'Карабах', 'shirvan': 'Ширван', 'gazakh': 'Газах', 'baku': 'Баку', 'guba': 'Губа', 'tabriz': 'Тебриз', 'nakhchivan': 'Нахчыван',
          'talysh': 'Ленкорань (Талыш)', 'sheki': 'Шеки', 'ganja': 'Гянджа', 'mugan': 'Муган'}
GROUP_OF = {'shirvan': 'shirvan', 'baku': 'baku', 'guba': 'guba', 'gazakh': 'gazakh', 'ganja': 'ganja'}
PL_AZ = {'Qarabağ': 'karabakh', 'Şirvan': 'shirvan', 'Qazax': 'gazakh', 'Bakı': 'baku', 'Quba': 'guba', 'Təbriz': 'tabriz', 'Naxçıvan': 'nakhchivan', 'Xızı': 'baku', 'Padar': 'shirvan'}
def date_s(p):
    if p.get('century'):
        s = f"{p['century']} в." + (f", {p['century_part']}" if p.get('century_part') else '')
    elif p.get('year'): s = f"{p['year']} г."
    else: s = None
    if p.get('hijri') and s: s += f" ({p['hijri']} г. хиджры)"
    return s
def check_school(cid, sch, where, page):
    """сравнить школу из книги с привязкой ковра (carpet_schools.json) и с таблицами Керимова (concepts.schools)"""
    if not cid or not sch: return None
    t = TYPE_OF.get(sch); k = CS.get(cid) or {}
    ks = CON.get(cid, {}).get('schools') or {}
    st = 'нет данных в модели'
    if k.get('type'): st = 'совпадает' if k['type'] == t else 'расходится'
    elif ks: st = 'совпадает' if any(x == t or TYPE_OF.get(x) == t or x.startswith(t or '#') for x in ks) else 'расходится'
    return {'concept': cid, 'book_school': SCH_RU.get(sch, sch), 'book_type': t, 'where': where, 'page': page,
            'model_carpet_schools': ({'type': k.get('type'), 'group': k.get('group'), 'confidence': k.get('confidence')} if k else None),
            'model_tables_schools': ks or None, 'status': st}
# ---------------- Энциклопедия т. II ----------------
enc = json.load(open(os.path.join(WORK, 'plates_enc.json')))
src_enc = {'id': 'tagiyeva_enc_2015_v2', 'short': 'Энциклопедия азерб. ковра, т. II', 'author': 'Р. Тагиева (гл. ред.)', 'year': 2015,
           'title': {'az': 'Azərbaycan xalçası ensiklopediyası, II cild', 'ru': 'Энциклопедия азербайджанского ковра, т. II'},
           'type': 'encyclopedia', 'file': 'Drive: scans/intangible/Azerbaycan_Xalchasi_Ensiklopediyasi_II_cild.pdf (472 стр.)', 'page_offset': 0,
           'note': {'ru': 'Т. II — безворсовые ковры (килим, зили, сумах, шадда, верни, джеджим, палас), ковровые изделия (чул, хурджин, хейбе, мафраш, '
                          'дуз торбасы, носки), каймы и орнаменты (схемы), авторские ковры XX–XXI вв., сырьё и красители. Почти все предметы — из '
                          'Азербайджанского музея ковра с инв. номерами DDK. Подпись к фото: тип. Школа. Азербайджан. Век. Материал. Размер. Музей. DDK. '
                          'Текстовый слой PDF испорчен; подписи распознаны заново (tesseract). Год издания — по SOURCES_REVIEW (2 т., 2015), в PDF не сверен.'}}
plates, chk, terms = [], [], collections.OrderedDict()
for p in enc:
    sch = p.get('school'); cid = p.get('concept') or p.get('concept_obj')
    tw = p.get('museum_twin'); note = []
    if tw:
        tsch = next((v for k, v in PL_AZ.items() if k in (p.get('museum_twin_place') or '')), None)
        note.append(f"тот же предмет на сайте музея ({tw}, «{p.get('museum_twin_name')}», {p.get('museum_twin_place')})"
                    + ('' if p.get('museum_twin_size_ok') is not False else '; размер расходится — проверить'))
        if tsch and sch and TYPE_OF.get(tsch) != TYPE_OF.get(sch):
            note.append(f"школа расходится: в энциклопедии {SCH_RU.get(sch)}, у музея {SCH_RU.get(tsch)}")
            chk.append({'concept': cid, 'book_school': SCH_RU.get(sch), 'museum_school': SCH_RU.get(tsch), 'inv': p.get('inv'), 'page': p['page'],
                        'status': 'расходится с сайтом музея'})
    if p.get('author'): note.append('авторский ковёр: ' + p['author'])
    plates.append({'id': 'enc2_' + p['id'].split('_')[-2] + '_' + p['id'].split('_')[-1], 'pdf_page': p['page'], 'printed_page': p['page'],
                   'section': OBJ_RU.get(p.get('obj')) or 'без названия типа в подписи', 'title_raw': p['caption'], 'type_name': p.get('obj'),
                   'group': SCH_RU.get(sch), 'attribution_note': '; '.join(note) or None, 'date': date_s(p), 'collection': p.get('museum'),
                   'inv_no': p.get('inv'), 'dimensions_cm': ' x '.join(str(int(v)) if float(v).is_integer() else str(v) for v in p['size_cm']) if p.get('size_cm') else None,
                   'material_az': p.get('material_az'), 'name': p.get('name'), 'concept': cid, 'concept_type': p.get('concept_obj'),
                   'link_status': 'auto' if cid else None, 'school': {'type': TYPE_OF.get(sch), 'group': GROUP_OF.get(sch)} if sch else None,
                   'cps_id': p['id'], 'museum_twin': tw})
    c = check_school(p.get('concept'), sch, 'подпись', p['page'])
    if c: chk.append(c)
    for t, cc in ((p.get('obj'), p.get('concept_obj')), (p.get('name'), p.get('concept'))):
        if not t or t in ('Xalça', 'Hana') or (p.get('author') and t == p.get('name')): continue
        e = terms.setdefault(t, {'term': t, 'concept': cc, 'definition_ru': OBJ_RU.get(t) if t in OBJ_RU else 'название композиции (по подписям к фото)',
                                 'confidence': 'high' if cc else 'medium', 'new_candidate': not cc, 'pages': [], 'schools': collections.Counter()})
        e['pages'].append(p['page'])
        if sch: e['schools'][SCH_RU.get(sch)] += 1
def fin(terms):
    out = []
    for e in terms.values():
        e = dict(e); e['schools'] = dict(e['schools']); e['n'] = len(e['pages']); e['pages'] = sorted(set(e['pages'])); out.append(e)
    return out
def write(sid, src, plates, terms, chk, about):
    d = os.path.join(OUTROOT, sid); os.makedirs(d, exist_ok=True)
    json.dump({'source': src, 'plates': plates}, open(f'{d}/plates.json', 'w'), ensure_ascii=False, indent=1)
    json.dump({'_about': about, 'terms': fin(terms), 'borders': [], 'type_candidates': [t for t, e in terms.items() if e['new_candidate']]},
              open(f'{d}/glossary_map.json', 'w'), ensure_ascii=False, indent=1)
    json.dump({'_about': 'Сверка школ: книга против модели (carpet_schools.json — привязка ковров; concepts.schools — таблицы Керимова) и против сайта музея. '
                         'Правило 4: расхождение фиксируется, не выбирается.', 'items': chk}, open(f'{d}/school_check.json', 'w'), ensure_ascii=False, indent=1)
    print(sid, 'plates', len(plates), 'terms', len(terms), 'new', sum(e['new_candidate'] for e in terms.values()), 'checks', collections.Counter(c['status'] for c in chk))
write('tagiyeva_enc_2015_v2', src_enc, plates, terms, chk,
      'Типы изделий и названия композиций из подписей к фото энциклопедии, т. II; concept — понятие модели; new_candidate — понятия нет. '
      'Определения статей энциклопедии (словник) добавятся после полного OCR книги.')
# ---------------- Мурадов 2010 ----------------
mur = json.load(open(os.path.join(WORK, 'plates_mur.json')))
src_mur = {'id': 'muradov_2010_qarabag', 'short': 'Мурадов 2010, Qarabağ qrupu', 'author': 'Видади Мурадов', 'year': 2010,
           'title': {'az': 'Azərbaycan xalçaları. Qarabağ qrupu', 'en': 'Azerbaijani carpets. Garabagh group', 'ru': 'Азербайджанские ковры. Карабахская группа'},
           'publisher': 'Bakı: Elm, 2010, 272 с. (при поддержке Фонда Гейдара Алиева)', 'type': 'catalogue',
           'file': 'Drive: scans/intangible/ (PDF 137 стр., каждая страница — разворот)', 'page_offset': None,
           'note': {'ru': 'Серия «Azərbaycan xalçaları bütün məktəbləri ilə». Части книги: история (до с. 86), коллекция галереи «Sehrli ilmələr» фирмы '
                          '«Азер-ильме» (с. 87–236), новые ковры «Азер-ильме» (с. 237+). У каждого ковра — паспорт: размер, поле, плотность, узел, '
                          'число цветов, состояние. Атрибуция — дилерская коллекция, свидетельство слабее музейного (как Мурадов 2008).'}}
plates_m, chk_m, terms_m = [], [], collections.OrderedDict()
SEC = {'introduction': 'история и музейные предметы', 'azerilme_collection': 'коллекция «Азер-ильме»', 'azerilme_new': 'новые ковры «Азер-ильме»'}
for p in mur:
    cid = p.get('concept'); dims = p.get('size_cm')
    plates_m.append({'id': 'mur10_' + p['id'].split('_')[-2] + '_' + p['id'].split('_')[-1], 'pdf_page': p['pdf_page'], 'printed_page': p['printed_page_approx'],
                     'section': SEC[p['part']], 'title_raw': p['title_az'], 'title_en': p.get('title_en'), 'type_name': p.get('item_type'), 'group': 'Карабах',
                     'attribution_note': None if p['part'] == 'introduction' else 'дилерская коллекция / новодел — слабое свидетельство',
                     'date': (f"{p['century']} в." if p.get('century') else (str(p['years'][0]) if p.get('years') else None)), 'collection': p.get('collection'),
                     'inv_no': None, 'dimensions_cm': ' x '.join(str(int(v)) for v in dims[:2]) if dims else None, 'field_cm': p.get('field_cm'),
                     'density': p.get('density'), 'knot': p.get('knot'), 'colors_n': p.get('colors_n'), 'condition': p.get('condition'),
                     'name': p.get('name'), 'concept': cid, 'link_status': ('auto' if (p.get('concept_how') == 'exact' and CON.get(cid, {}).get('type') == 'composition') else ('proposed' if cid else None)),
                     'school': {'type': 'karabakh', 'group': None}, 'cps_id': p['id'], 'printed_page_note': 'печатная страница ≈ 2×(pdf−1), не сверена'})
    c = check_school(cid, 'karabakh', p['part'], p['pdf_page'])
    if c: chk_m.append(c)
    if p.get('name'):
        e = terms_m.setdefault(p['name'], {'term': p['name'], 'concept': cid, 'definition_ru': 'название ковра (по подписи)', 'confidence': 'high' if p.get('concept_how') == 'exact' else 'medium',
                                           'new_candidate': not cid, 'pages': [], 'schools': collections.Counter()})
        e['pages'].append(p['pdf_page']); e['schools']['Карабах'] += 1
write('muradov_2010_qarabag', src_mur, plates_m, terms_m, chk_m,
      'Названия ковров из подписей Мурадова 2010 (Карабахская группа). concept — понятие модели (auto — точное совпадение, иначе proposed).')
