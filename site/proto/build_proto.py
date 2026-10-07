# build_proto.py — прототип «Ковровое ДНК» на модели понятий (вариант B).
# python3 build_proto.py SITE_DATA.json SCHEMES.json OUT.html [метка_версии]
# Берёт site/model/out/* (понятия, связи, очередь), site/model/classify.py (смысловые классы),
# данные сайта (источники, носители) и выгрузку pixel_schemes; всё вшивается в один HTML.
import json, sys, os, csv, re, glob, subprocess, datetime, collections as C

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'model')
sys.path.insert(0, MODEL)
from classify import classify, CLASSES
_mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'classes_manual.json')
MANUAL_CL = json.load(open(_mp))['items'] if os.path.exists(_mp) else {}
_tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'translations_ai.json')
TRANSL = json.load(open(_tp))['items'] if os.path.exists(_tp) else {}
_cp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'carpet_schools.json')
CSCH = json.load(open(_cp))['items'] if os.path.exists(_cp) else {}
_pp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'sources', 'museum_photos', 'catalog_photos.json')
PHOTOS = json.load(open(_pp))['items'] if os.path.exists(_pp) else []
BASIS_RU = {'kerimov': 'по переводу Керимова / легенде таблицы', 'book': 'по тексту книги Керимова', 'ai_language': 'разбор слова (ИИ)',
            'place_name': 'название по месту (ИИ)', 'unknown': 'значение не установлено'}


def knot_aspect(s):
    # пропорция узла (ширина/высота) = узлов на 10 см по длине / узлов на 10 см по ширине; нет данных — None (квадрат)
    k = s.get('knot') or (s.get('grid') or {}).get('knot') or {}
    if s.get('knot_aspect'): return round(float(s['knot_aspect']), 3)
    if k.get('l') and k.get('w'): return round(float(k['l']) / float(k['w']), 3)
    return None


def meaning(c):
    m = {'ru': c['meaning'].get('ru'), 'en': c['meaning'].get('en')}
    if c['meaning'].get('status') == 'checked_ai':
        return m, 'kerimov', None
    t = TRANSL.get(c['id'])
    if not t: return m, 'machine', None
    if t['basis'] == 'unknown':
        return {'ru': None, 'en': None}, 'unknown', (m['ru'] if m['ru'] else None)
    return {'ru': t['ru'], 'en': t['en'] or m['en']}, t['basis'], (m['ru'] if t.get('changed') and m['ru'] else None)

# Иерархия школ по Керимову: тип → группа (отчёт о классификациях 2026-10-05)
SCHOOL_TYPES = [
    ('guba-shirvan', 'Губа-Ширван', 'Quba-Şirvan', [('guba', 'Губа', 'Quba'), ('shirvan', 'Ширван', 'Şirvan'), ('baku', 'Баку (Апшерон)', 'Bakı (Abşeron)')]),
    ('ganja-gazakh', 'Гянджа-Газах', 'Gəncə-Qazax', [('ganja', 'Гянджа', 'Gəncə'), ('gazakh', 'Газах', 'Qazax')]),
    ('karabakh', 'Карабах', 'Qarabağ', []),
    ('tabriz', 'Тебриз', 'Təbriz', []),
]
SCHOOL_NORM = {'guba-shirvan': ('guba-shirvan', None), 'baku': ('guba-shirvan', 'baku'), 'guba': ('guba-shirvan', 'guba'),
               'shirvan': ('guba-shirvan', 'shirvan'), 'ganja-gazakh': ('ganja-gazakh', None), 'ganja': ('ganja-gazakh', 'ganja'),
               'gazakh': ('ganja-gazakh', 'gazakh'), 'karabakh': ('karabakh', None), 'tabriz': ('tabriz', None)}
TYPE_ABBR = {'К.-Ш': 'guba-shirvan', 'К.—Ш': 'guba-shirvan', 'К.-Ш.': 'guba-shirvan', 'Кар.': 'karabakh', 'Г.-К': 'ganja-gazakh',
             'Г.—К': 'ganja-gazakh', 'Г.-К.': 'ganja-gazakh', 'Г.—К.': 'ganja-gazakh', 'Теб.': 'tabriz', 'Тәб.': 'tabriz'}


def main(site_data, schemes_path, out_html, label=None):
    out = os.path.join(MODEL, 'out')
    cs = json.load(open(os.path.join(out, 'concepts.json')))
    rel = json.load(open(os.path.join(out, 'relations.json')))
    tmap = json.load(open(os.path.join(out, 'term_map.json')))
    queue = list(csv.DictReader(open(os.path.join(out, 'review_queue.csv'))))
    D = json.load(open(site_data))
    S = [s for s in json.load(open(schemes_path)) if s.get('id')]
    by = {c['id']: c for c in cs}

    ko = C.defaultdict(list)
    for r in rel:
        if r['type'] == 'kind_of': ko[r['from']].append(by[r['to']]['headword'])

    # схемы: только нужное для рисования
    sch = {}
    for s in S:
        t = s.get('school') or TYPE_ABBR.get((s.get('type') or '').strip())
        sch[s['id']] = {'w': s['w'], 'h': s['h'], 'rows': s['rows'], 'pal': s.get('palette'), 'n': s.get('name'),
                        'tb': s.get('table'), 'fg': s.get('fig'), 'sc': t, 'tr': s.get('translation'),
                        'c': s.get('carpet'), 'nt': s.get('note'), 'ka': knot_aspect(s)}
    desc = {}
    ddir = os.path.join(MODEL, 'descriptions')
    for f in sorted(os.listdir(ddir)) if os.path.isdir(ddir) else []:
        if f.endswith('.json'):
            dj = json.load(open(os.path.join(ddir, f)))
            for cid, d in dj['items'].items():
                d = {**d, 'src': dj['source']['short']}
                if cid in desc and not cid.startswith('school:'):     # второй источник — добавить разделы с его подписью
                    desc[cid].setdefault('more', []).append(d)
                else:
                    desc[cid] = d
    # схема → ковёр/композиция по колонке легенды Керимова «для какого ковра характерен»
    def nz(x):
        x = (x or '').lower().replace('ё', 'е'); x = re.sub(r'[«»"\'|().,]', ' ', x); return re.sub(r'\s+', ' ', x).strip()
    cidx = C.defaultdict(set)
    for c in cs:
        if c['type'] in ('composition', 'object'):
            for n in c['names']:
                if n['kind'] not in ('misreading', 'meaning_rejected', 'meaning'): cidx[nz(n['v'])].add(c['id'])
    for s in S:
        car = nz((s.get('carpet_i18n') or {}).get('ru') or s.get('carpet'))
        cc = []
        for part in [p.strip() for p in re.split(r',| и |;', car) if p.strip()]:
            for x in sorted(cidx.get(part, ())):
                if x not in cc: cc.append(x)
        sch[s['id']]['cc'] = cc
    linked = set()
    concepts = []
    for c in cs:
        cq = 'auto'
        mm, mq, mo = meaning(c)
        if c['id'] in MANUAL_CL:
            cl, cq = MANUAL_CL[c['id']]['classes'], 'ai'
        else:
            cl = classify(c, ko[c['id']]) if c['type'] == 'ornament' else []
        sc = C.Counter()
        for k, v in (c.get('schools') or {}).items():
            if k in SCHOOL_NORM: sc['/'.join(filter(None, SCHOOL_NORM[k]))] += v
        for sid in c.get('schemes') or []:
            linked.add(sid)
            t = sch.get(sid, {}).get('sc')
            if t in SCHOOL_NORM: sc['/'.join(filter(None, SCHOOL_NORM[t]))] += 1
        concepts.append({
            'id': c['id'], 't': c['type'], 'h': c['headword'], 'm': mm, 'mq': mq, 'mo': mo,
            'nm': [[n['v'], n['lang'], n['kind']] + ([n['src']] if n.get('src') else []) for n in c['names']],
            'r': c['roles'], 'sc': dict(sc), 'src': c['sources'],
            'at': [[a['source_id'], a['where'], a['page'], a['quote'], a['note'], a['role']] for a in c['attestations']],
            'rec': c['records'], 'mg': c['merge'], 'cl': cl, 'cq': cq, 'sch': c.get('schemes') or [],
            'ds': desc.get(c['id']),
            'w': [],
            'x': [[r.get('title'), r.get('url')] for r in (c.get('external_references') or []) if isinstance(r, dict)],
        })
    cc_of = C.defaultdict(list)
    for sid, sv in sch.items():
        for x in sv.get('cc') or []: cc_of[x].append(sid)
    for c in concepts:
        k = CSCH.get(c['id'])
        if k:
            c['cs'] = [k.get('type'), k.get('group'), k.get('confidence'), k.get('conflict') or '', k.get('evidence') or []]
            key = '/'.join(filter(None, [k.get('type'), k.get('group') if k.get('group') in ('guba', 'shirvan', 'baku', 'ganja', 'gazakh') else None]))
            if key and key not in c['sc']: c['sc'][key] = 0          # 0 = школа из привязки ковров (carpet_schools.json), не из таблиц
        if not c['sch'] and cc_of.get(c['id']): c['ill'] = cc_of[c['id']][:6]
    ph_of = C.defaultdict(list)
    for i, ph in enumerate(PHOTOS):
        if ph.get('concept'): ph_of[tmap.get(ph['concept'], ph['concept'])].append(i)
    for c in concepts:
        if ph_of.get(c['id']):
            # сначала целые ковры с высокой уверенностью
            c['ph'] = sorted(ph_of[c['id']], key=lambda i: (not PHOTOS[i]['whole'], PHOTOS[i]['conf'] != 'high'))
    carriers = []
    for k in D.get('carriers') or []:
        cid = tmap.get(k.get('term_id'))
        carriers.append([cid, k.get('motif_name'), k.get('date'), k.get('location'), k.get('museum_or_collection'), k.get('source_id'), k.get('source')])
    try:
        commit = subprocess.check_output(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        commit = '?'
    meta = {'label': label or 'прототип', 'built': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), 'commit': commit,
            'unlinked': len([s for s in S if s['id'] not in linked])}
    plates, extra_src = [], []
    sdir = os.path.join(MODEL, 'sources')
    for f in sorted(glob.glob(os.path.join(sdir, '*', 'plates.json'))):
        pj = json.load(open(f)); extra_src.append(pj['source'])
        for pl in pj['plates']:
            plates.append({k: pl.get(k) for k in ('id', 'pdf_page', 'printed_page', 'section', 'title_raw', 'type_name', 'group', 'attribution_note',
                           'date', 'inscribed_date', 'collection', 'inv_no', 'dimensions_cm', 'knots_per_m2', 'is_detail', 'design_ru', 'borders_ru',
                           'motifs_en', 'concept', 'link_status', 'school')} | {'src': pj['source']['id']})
    west = C.defaultdict(list)
    west_open = {'terms': [], 'types': [], 'borders': []}
    for f in sorted(glob.glob(os.path.join(sdir, '*', 'glossary_map.json'))):
        gm = json.load(open(f)); sid = os.path.basename(os.path.dirname(f))
        for t in gm.get('terms', []):
            cid = tmap.get(t.get('concept') or '', t.get('concept'))
            if cid: west[cid].append([t['term'], t.get('definition_ru') or t.get('definition_en'), sid, t.get('confidence')])
            elif t.get('new_candidate'): west_open['terms'].append([t['term'], t.get('definition_ru') or t.get('definition_en'), sid])
        west_open['types'] += [[x, sid] for x in gm.get('type_candidates', [])]
        for b in gm.get('borders', []):
            cid = tmap.get(b.get('concept') or '', b.get('concept'))
            if cid: west[cid].append([b['name_en'], 'кайма, типичная для: ' + (b.get('where') or ''), sid, 'medium'])
            else: west_open['borders'].append([b['name_en'], b.get('where'), sid])
    srcs = {s['id']: s for s in D['sources']}
    for s in extra_src: srcs[s['id']] = s
    for c in concepts: c['w'] = west.get(c['id'], [])
    DATA = {'concepts': concepts, 'rel': rel, 'tmap': tmap, 'queue': queue, 'schemes': sch,
            'sources': srcs, 'carriers': carriers, 'plates': plates,
            'west_open': west_open, 'photos': PHOTOS, 'school_desc': {k[7:]: v for k, v in desc.items() if k.startswith('school:')}, 'classes': CLASSES, 'school_types': SCHOOL_TYPES, 'meta': meta}
    tpl = open(os.path.join(HERE, 'template.html')).read()
    js = json.dumps(DATA, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    open(out_html, 'w').write(tpl.replace('/*__DATA__*/null', js))
    print('ok', out_html, round(os.path.getsize(out_html) / 1e6, 2), 'MB', len(concepts), 'понятий', len(sch), 'схем')


if __name__ == '__main__':
    main(*sys.argv[1:5])
