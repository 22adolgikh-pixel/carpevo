"""Кандидаты в дубли понятий модели (ковры с почти одинаковым названием) → cps/concept_dups.json для карточки «дубли» в /m (v11.56).
Критерий: два понятия-ковра, азербайджанские названия совпадают на ≥86 % без учёта диакритики (ç/c, ı/i, q/g, x/h). Запуск: python3 build_concept_dups.py"""
import os, json, re, unicodedata, difflib, itertools
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
C = [c for c in json.load(open(REPO + '/site/model/out/concepts.json')) if c['type'] == 'composition']
CS = json.load(open(REPO + '/site/model/carpet_schools.json'))['items']
SCH = {'guba': 'Губа', 'shirvan': 'Ширван', 'baku': 'Баку', 'ganja': 'Гянджа', 'gazakh': 'Газах', 'karabakh': 'Карабах', 'tabriz': 'Тебриз', 'guba-shirvan': 'Губа-Ширван', 'ganja-gazakh': 'Гянджа-Газах',
       'nakhchivan': 'Нахчыван', 'mugan': 'Муган', 'talysh': 'Талыш'}
SRC = {'kerimov_v1': 'Керимов, т. I', 'kerimov_v2': 'Керимов, т. II', 'kerimov_v3': 'Керимов, т. III', 'azcarpets_33_2020': 'журнал «Azərbaycan xalçaları» №33 (2020)', 'huseynov_2012': 'Гусейнов 2012'}
def fold(s):
    s = unicodedata.normalize('NFKD', (s or '').lower().replace('ə','e').replace('ı','i').replace('ğ','g').replace('ş','s').replace('ç','c').replace('ö','o').replace('ü','u').replace('q','g').replace('x','h'))
    return re.sub(r'[^a-zа-я]+', '', ''.join(ch for ch in s if not unicodedata.combining(ch)))
def keys(c): return {k for k in ({fold(n['v']) for n in c['names'] if (n.get('lang') or 'az').startswith('az') and n.get('kind') not in ('misreading', 'meaning', 'meaning_rejected')} | {fold(c['headword'])}) if len(k) >= 4}
def side(c):
    s = CS.get(c['id']) or {}
    az = []; ru = []
    for n in c['names']:
        if n.get('kind') in ('misreading', 'meaning', 'meaning_rejected'): continue
        (ru if n.get('lang') == 'ru' else az if (n.get('lang') or 'az') in ('az', 'az-cyrl') else []).append(n['v'])
    q = [{'where': (a.get('where') or '')[:60].split(' (файл')[0], 'quote': (a.get('quote') or '')[:260]} for a in c.get('attestations', []) if a.get('quote')][:2]
    return {'id': c['id'], 'az': c['headword'], 'names': list(dict.fromkeys(az))[:5], 'ru': list(dict.fromkeys(ru))[:3], 'school': SCH.get(s.get('group') or s.get('type') or '', ''),
            'sources': [f"{SRC.get(k, k)} ×{v}" for k, v in (c.get('sources') or {}).items()], 'quotes': q, 'schemes': len(c.get('schemes') or [])}
K = {c['id']: keys(c) for c in C}; out = []
for a, b in itertools.combinations(C, 2):
    r = max((difflib.SequenceMatcher(None, x, y).ratio() for x in K[a['id']] for y in K[b['id']]), default=0)
    if r >= 0.86: out.append({'id': a['id'] + '~' + b['id'], 'score': round(r, 2), 'a': side(a), 'b': side(b)})
out.sort(key=lambda x: -x['score'])
json.dump({'_about': 'Кандидаты в дубли понятий Carpet DNA для карточки «дубли» в /m. Собирает cps/tools/books/build_concept_dups.py.', 'pairs': out}, open(REPO + '/cps/concept_dups.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(out)); print(json.dumps(out[5], ensure_ascii=False, indent=1)[:1500])
