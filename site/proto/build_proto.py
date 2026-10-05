# build_proto.py — прототип «Ковровое ДНК» на модели понятий (вариант B).
# python3 build_proto.py SITE_DATA.json SCHEMES.json OUT.html [метка_версии]
# Берёт site/model/out/* (понятия, связи, очередь), site/model/classify.py (смысловые классы),
# данные сайта (источники, носители) и выгрузку pixel_schemes; всё вшивается в один HTML.
import json, sys, os, csv, re, glob, subprocess, datetime, collections as C

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'model')
sys.path.insert(0, MODEL)
from classify import classify, CLASSES

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
                        'c': s.get('carpet'), 'nt': s.get('note')}
    desc = {}
    ddir = os.path.join(MODEL, 'descriptions')
    for f in sorted(os.listdir(ddir)) if os.path.isdir(ddir) else []:
        if f.endswith('.json'):
            dj = json.load(open(os.path.join(ddir, f)))
            for cid, d in dj['items'].items():
                desc[cid] = {**d, 'src': dj['source']['short']}
    linked = set()
    concepts = []
    for c in cs:
        cl = classify(c, ko[c['id']]) if c['type'] == 'ornament' else []
        sc = C.Counter()
        for k, v in (c.get('schools') or {}).items():
            if k in SCHOOL_NORM: sc['/'.join(filter(None, SCHOOL_NORM[k]))] += v
        for sid in c.get('schemes') or []:
            linked.add(sid)
            t = sch.get(sid, {}).get('sc')
            if t in SCHOOL_NORM: sc['/'.join(filter(None, SCHOOL_NORM[t]))] += 1
        concepts.append({
            'id': c['id'], 't': c['type'], 'h': c['headword'], 'm': c['meaning'],
            'nm': [[n['v'], n['lang'], n['kind']] + ([n['src']] if n.get('src') else []) for n in c['names']],
            'r': c['roles'], 'sc': dict(sc), 'src': c['sources'],
            'at': [[a['source_id'], a['where'], a['page'], a['quote'], a['note'], a['role']] for a in c['attestations']],
            'rec': c['records'], 'mg': c['merge'], 'cl': cl, 'sch': c.get('schemes') or [],
            'ds': desc.get(c['id']),
            'x': [[r.get('title'), r.get('url')] for r in (c.get('external_references') or []) if isinstance(r, dict)],
        })
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
    srcs = {s['id']: s for s in D['sources']}
    for s in extra_src: srcs[s['id']] = s
    DATA = {'concepts': concepts, 'rel': rel, 'tmap': tmap, 'queue': queue, 'schemes': sch,
            'sources': srcs, 'carriers': carriers, 'plates': plates,
            'classes': CLASSES, 'school_types': SCHOOL_TYPES, 'meta': meta}
    tpl = open(os.path.join(HERE, 'template.html')).read()
    js = json.dumps(DATA, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    open(out_html, 'w').write(tpl.replace('/*__DATA__*/null', js))
    print('ok', out_html, round(os.path.getsize(out_html) / 1e6, 2), 'MB', len(concepts), 'понятий', len(sch), 'схем')


if __name__ == '__main__':
    main(*sys.argv[1:5])
