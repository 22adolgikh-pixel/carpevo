"""Указатель упоминаний ковров (композиций) модели в текстах книг (2026-11-13).
Вход: CORPUS — папка с OCR (intangible_28xx_ocr.jsonl, intangible_2885_text.jsonl, sympo_text.jsonl). Выход: site/model/sources/TEXT_MENTIONS_2026-11-13.json.
Поиск — по азербайджанскому названию (и рус./англ. именам понятия), без диакритики, только слова от 6 букв; OCR шумный, поэтому это нижняя оценка."""
import os, sys, json, re, unicodedata, collections
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
CORPUS = os.environ.get('CORPUS', '.')
C = json.load(open(REPO + '/site/model/out/concepts.json'))
def fold(s):
    s = unicodedata.normalize('NFKD', (s or '').lower().replace('ə','e').replace('ı','i').replace('ğ','g').replace('ş','s').replace('ç','c').replace('ö','o').replace('ü','u'))
    return re.sub(r'[^a-z0-9а-я]+', ' ', ''.join(ch for ch in s if not unicodedata.combining(ch)))
FILES = {'muradov_2008': [('intangible_2849_ocr.jsonl', 'm1-'), ('intangible_2850_ocr.jsonl', 'm2-')], 'muradov_2013': [('intangible_2851_ocr.jsonl', '')],
         'irevan_qrupu': [('intangible_2854_ocr.jsonl', '')], 'ahm_kovry': [('intangible_2885_text.jsonl', '')], 'symposium_2007_paris': [('sympo_text.jsonl', '')]}
PAGES = {s: [(pre + str(json.loads(l)['page']), ' ' + fold(json.loads(l)['text']) + ' ') for f, pre in fl for l in open(os.path.join(CORPUS, f), encoding='utf-8')] for s, fl in FILES.items()}
STOP = {'azerbaycan', 'xalca', 'carpet', 'qarabag', 'sirvan', 'tebriz', 'naxcivan', 'gence', 'qazax', 'kilim', 'palaz', 'namazliq'}
DESC = set()
for f in os.listdir(REPO + '/site/model/descriptions'):
    if f.endswith('.json'): DESC |= set(json.load(open(REPO + '/site/model/descriptions/' + f)).get('items', {}))
out = {}
for c in C:
    if c['type'] != 'composition': continue
    keys = {fold(n['v']).strip() for n in c.get('names', []) if n.get('kind') not in ('misreading', 'meaning', 'meaning_rejected')} | {fold(c['headword']).strip()}
    keys = {k for k in keys if len(k.replace(' ', '')) >= 6 and k not in STOP}
    if not keys: continue
    hit = {}
    for s, pp in PAGES.items():
        pg = [p for p, t in pp if any((' ' + k) in t for k in keys)]
        if pg: hit[s] = pg
    if hit: out[c['id']] = {'headword': c['headword'], 'has_description': c['id'] in DESC, 'mentions': hit, 'n_pages': sum(map(len, hit.values()))}
res = {'_about': 'Упоминания ковров модели в текстах пяти книг (страницы файлов; m1-/m2- — части Мурадова 2008). Нижняя оценка: OCR шумный, короткие и общие названия не искались.',
       'summary': {'compositions_mentioned': len(out), 'without_description': sum(1 for v in out.values() if not v['has_description']),
                   'by_source': {s: sum(1 for v in out.values() if s in v['mentions']) for s in FILES}}, 'items': dict(sorted(out.items(), key=lambda kv: -kv[1]['n_pages']))}
json.dump(res, open(REPO + '/site/model/sources/TEXT_MENTIONS_2026-11-13.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res['summary'], ensure_ascii=False)); print([(k, v['n_pages']) for k, v in list(res['items'].items())[:25]])
print('без описания, ≥3 стр.:', [(k, v['n_pages']) for k, v in res['items'].items() if not v['has_description'] and v['n_pages'] >= 3])
