# -*- coding: utf-8 -*-
"""books_vlm_colab.py — модель зрения через OpenRouter читает страницы альбомов (CarpetsCat, 2026-11-12).
Ключ: Colab Secrets → OPENROUTER_API_KEY. TEST=1: 4 страницы в vlm_test_<модель>.jsonl; TEST=0: всё в vlm_plates.jsonl (возобновляемо)."""
import os, json, glob, base64, re, time, requests
from concurrent.futures import ThreadPoolExecutor
import fitz
from google.colab import drive, userdata
drive.mount('/content/drive')

KEY = userdata.get('OPENROUTER_API_KEY')
MODEL = os.environ.get('VLM_MODEL', 'google/gemini-3.7-flash')
TEST = os.environ.get('TEST', '1') == '1'
WORKERS = int(os.environ.get('WORKERS', '8'))
B = '/content/drive/MyDrive/carpet-dna/output/books'
SLUGS = ['intangible_2849', 'intangible_2850', 'intangible_2851', 'intangible_2854', 'azerbaycan_xalchasi', 'intangible_2885']
OUT = f"{B}/vlm_test_{re.sub(r'[^a-z0-9]+', '_', MODEL.lower())}.jsonl" if TEST else f'{B}/vlm_plates.jsonl'
MIN_AREA = 100000

PROMPT = """You are reading one page of a carpet album (Azerbaijani carpets; captions are usually trilingual: Azerbaijani / English / Russian).
The page image is attached. Below are crops that an automatic cutter extracted from this page (id, position as % of page: x0,y0,x1,y1)
and a noisy OCR text of the page (may be garbled - trust the image, use OCR only as a hint).

Crops:
{crops}

OCR text:
{ocr}

For EACH crop return an object. Answer with JSON only: {{"page_kind": "plate|text|scenery|promo|other", "items": [ ... ]}}
Each item: {{"id": "<crop id>", "kind": "rug|prayer_rug|bag|flatweave|fragment|not_textile", "caption": "the caption text that belongs to this crop, or ''",
"name": "carpet name in Azerbaijani if given (e.g. Zeyxur), else ''", "name_en": "", "name_ru": "",
"school": "guba|shirvan|karabakh|gazakh|ganja|baku|tabriz|nakhchivan|irevan|shaki|mugan|lankaran|talish|other|''",
"century": "XIX or XX etc., else ''", "part": "early|mid|late|''", "year": null, "material": "wool|silk|cotton|mixed|''",
"size_cm": null or [h, w], "museum": "", "inv": "", "confidence": 0.0-1.0}}
kind: rug = pile carpet photographed whole; prayer_rug = whole prayer rug; bag = bag/saddlebag/khurjin/mafrash; flatweave = kilim/palaz/zili/sumakh; fragment = close-up or part of a textile;
not_textile = people, landscape, buildings, workshop, tools, wool/yarn, maps, text, portraits, drawings. Do not guess school/century/size when the caption does not give them."""


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_text(slug):
    d = {}
    for fn in ('ocr_pages.jsonl', 'text.jsonl'):
        p = f'{B}/{slug}/{fn}'
        if os.path.exists(p):
            for l in open(p, encoding='utf-8'):
                try:
                    r = json.loads(l); pg = int(r['page'])
                    if len(r.get('text', '')) > len(d.get(pg, '')): d[pg] = r['text']
                except Exception: pass
    return d


def find_pdf(slug):
    key = 'azerbaycan_xalchasi' if slug == 'azerbaycan_xalchasi' else slug
    for p in glob.glob('/content/drive/MyDrive/scans/**/*.pdf', recursive=True):
        if key.lower() in os.path.basename(p).lower(): return p


done = set()
if os.path.exists(OUT) and not TEST:
    for l in open(OUT, encoding='utf-8'):
        try: r = json.loads(l); done.add((r['slug'], r['page']))
        except Exception: pass
if TEST and os.path.exists(OUT): os.remove(OUT)

jobs = []
for slug in SLUGS:
    ip = f'{B}/{slug}/items.jsonl'; pdf = find_pdf(slug)
    if not os.path.exists(ip) or not pdf: print('ПРОПУСК', slug, '(нет items.jsonl или PDF)'); continue
    items = [json.loads(l) for l in open(ip, encoding='utf-8')]
    doc = fitz.open(pdf); txt = load_text(slug); by = {}
    for it in items:
        if it['px'][0] * it['px'][1] >= MIN_AREA: by.setdefault(it['page'], []).append(it)
    for pg, its in sorted(by.items()):
        if (slug, pg) in done: continue
        page = doc[pg - 1]; W, H = page.rect.width, page.rect.height
        jpg = page.get_pixmap(dpi=100 if max(W, H) > 900 else 130).tobytes('jpeg', jpg_quality=70)
        crops = '\n'.join(f"- {it['id']}: {int(100*it['bbox_pt'][0]/W)},{int(100*it['bbox_pt'][1]/H)},{int(100*it['bbox_pt'][2]/W)},{int(100*it['bbox_pt'][3]/H)}" for it in its)
        jobs.append({'slug': slug, 'page': pg, 'ids': [it['id'] for it in its], 'jpg': base64.b64encode(jpg).decode(),
                     'prompt': PROMPT.format(crops=crops, ocr=re.sub(r'\s+', ' ', txt.get(pg, ''))[:2500])})
TOTAL = len(jobs)
log('модель', MODEL, '| страниц к разбору:', TOTAL, '| уже готово:', len(done))
if TEST:  # 4 показательные страницы: по одной-две из альбомов Мурадова и из azerbaycan_xalchasi
    pick = [j for j in jobs if j['slug'] == 'intangible_2849'][10:12] + [j for j in jobs if j['slug'] == 'intangible_2850'][20:21] + [j for j in jobs if j['slug'] == 'azerbaycan_xalchasi'][5:6]
    jobs = pick or jobs[:4]


def call(job):
    body = {'model': MODEL, 'temperature': 0, 'usage': {'include': True}, 'max_tokens': 2500,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': job['prompt']},
                                                       {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + job['jpg']}}]}]}
    err = ''
    for k in range(4):
        try:
            r = requests.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': 'Bearer ' + KEY.strip()}, json=body, timeout=180)
            if r.status_code in (429, 500, 502, 503): err = f'HTTP {r.status_code}'; time.sleep(3 * (k + 1)); continue
            j = r.json()
            if 'choices' not in j: return {'slug': job['slug'], 'page': job['page'], 'error': str(j)[:300]}
            raw = j['choices'][0]['message']['content'] or ''
            m = re.search(r'\{.*\}', raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            u = j.get('usage') or {}
            return {'slug': job['slug'], 'page': job['page'], 'ids': job['ids'], 'model': MODEL, 'data': data, 'cost': u.get('cost'), 'tok': u.get('total_tokens')}
        except Exception as e:
            err = str(e)[:200]; time.sleep(2)
    return {'slug': job['slug'], 'page': job['page'], 'error': err}


n = bad = 0; spent = 0.0
with ThreadPoolExecutor(WORKERS) as ex, open(OUT, 'a', encoding='utf-8') as f:
    for r in ex.map(call, jobs):
        n += 1
        if 'error' in r:
            bad += 1
            if bad <= 3: print('ОШИБКА', r)
            continue
        spent += r.get('cost') or 0
        f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
        if TEST:
            print(f"\n--- {r['slug']} стр.{r['page']}  ${r.get('cost')}  токенов {r.get('tok')}  page_kind={r['data'].get('page_kind')}")
            for it in r['data'].get('items', []):
                print('  ', it.get('id'), '|', it.get('kind'), '|', it.get('name'), '|', it.get('school'), '|', it.get('century'), it.get('part'), '|', it.get('material'), '|', it.get('size_cm'), '| conf', it.get('confidence'))
                if it.get('caption'): print('       подпись:', it['caption'][:160])
        elif n % 25 == 0: log(f'{n}/{len(jobs)}  потрачено ${spent:.2f}')
log(f'ГОТОВО: страниц {n}, ошибок {bad}, потрачено ${spent:.3f}')
if TEST and n - bad:
    print(f'ОЦЕНКА полного прогона этой моделью: ~${spent / (n - bad) * TOTAL:.2f} за {TOTAL} страниц')
