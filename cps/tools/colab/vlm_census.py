# vlm_census.py — перепись каталога: какие именованные элементы есть на каждом ковре (v12.1, без обучения).
#
# Для каждого ковра (не скрытого и не дубликата) модель получает фото с сеткой 10 % и пронумерованные схемы-кандидаты
# (по одной на каждый элемент, который у нас размечен; рамка на схемах убрана) и возвращает, какие элементы видны, где и с какой уверенностью.
# По пробам (30.10): «какие элементы есть» модель называет верно в 93 %, но около четверти ответов «есть» ложные, а рамки грубые (47 %),
# поэтому результат — ПОДСКАЗКИ для человека («ждут проверки»), а не разметка.
#
# Ключ OpenRouter: Colab → Секреты → OPENROUTER_API_KEY. В чат ключ не присылать.
# Запуск:
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env VC_MODEL=anthropic/claude-haiku-5.5      (или qwen/qwen3-vl-32b-instruct, qwen/qwen3-vl-8b-instruct)
#   %env VC_LIMIT=40                              (сначала 40 ковров — пробный прогон и цена; потом убрать)
#   %run /content/carpevo/cps/tools/colab/vlm_census.py
# Параметры: VC_MODEL, VC_LIMIT (0 = все), VC_MAX_USD (стоп по бюджету, по умолчанию 5), VC_THREADS (6), VC_SIDE (1280, длинная сторона фото),
#   VC_PRICE_IN / VC_PRICE_OUT ($ за 1М токенов, если OpenRouter не вернул cost), VC_PRIOR=1 (сначала ковры с разметкой, для сверки).
# Результат: Диск carpet-dna/vlm_census/<модель>.jsonl (по строке на ковёр; перезапуск продолжает с места остановки) и сводка в конце.
import os, io, re, json, glob, time, base64, random, zipfile, subprocess, collections, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw

MODEL = os.environ.get('VC_MODEL', 'anthropic/claude-haiku-5.5'); LIMIT = int(os.environ.get('VC_LIMIT', 0))
MAX_USD = float(os.environ.get('VC_MAX_USD', 5)); THREADS = int(os.environ.get('VC_THREADS', 6)); SIDE = int(os.environ.get('VC_SIDE', 1280))
PIN = float(os.environ.get('VC_PRICE_IN', 0.1)); POUT = float(os.environ.get('VC_PRICE_OUT', 0.5))
W = '/content/vc'; os.makedirs(W, exist_ok=True); DATA = '/content/cpsdata'
if not os.path.isdir(DATA + '/data'):
    subprocess.run(['git', 'clone', '--depth', '1', '-b', 'cps-data', 'https://github.com/22adolgikh-pixel/carpevo.git', DATA], check=True)
D = DATA + '/data'
try:
    from google.colab import userdata; KEY = userdata.get('OPENROUTER_API_KEY')
except Exception:
    KEY = os.environ.get('OPENROUTER_API_KEY', '')
assert KEY, 'нет ключа: Секреты Colab → OPENROUTER_API_KEY'

# ---------- кандидаты: одна схема на элемент (по числу подтверждённых рамок) + рамка схемы убрана ----------
works = {os.path.splitext(f)[0] for f in os.listdir(D + '/work')}
rugs = {}; cnt = collections.Counter(); marked = {}
for f in glob.glob(D + '/rugs/*.json'):
    r = json.load(open(f, encoding='utf-8'))
    if r.get('hidden') or r.get('dup_of'): continue
    rugs[r['id']] = r.get('lib_img') or f"img/{r['id']}.jpg"
    ms = [m for m in (r.get('marks') or []) if m.get('status') == 'confirmed' and m.get('scheme') in works and m.get('box')]
    if ms: marked[r['id']] = [dict(box=m['box'], scheme=m['scheme']) for m in ms]
    for m in ms: cnt[m['scheme']] += 1
def nm(sid):
    w = json.load(open(f'{D}/work/{sid}.json', encoding='utf-8')); mt = w.get('meta') or {}
    return (mt.get('name_az') or mt.get('name') or '').strip()
cands = [s for s, _ in cnt.most_common()]
print('ковров:', len(rugs), 'с разметкой:', len(marked), 'кандидатов-схем:', len(cands))

def strip_frame(a):
    import cv2
    if a.shape[0] < 8 or a.shape[1] < 8: return a
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if (ring == 0).any(): return a
    a = a[1:-1, 1:-1].copy()
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]); col = collections.Counter(ring.tolist()).most_common(1)[0][0]
    n, lab = cv2.connectedComponents((a == col).astype(np.uint8), connectivity=4)
    for l in set(lab[0].tolist() + lab[-1].tolist() + lab[:, 0].tolist() + lab[:, -1].tolist()) - {0}: a[lab == l] = 0
    return a
def scheme_img(sid, px=14):
    w = json.load(open(f'{D}/work/{sid}.json', encoding='utf-8')); rows = w['matrix']['rows']; pal = w.get('palette') or ['#ffffff', '#1a1a1a', '#b8b8b8']
    h = len(rows); ww = max(len(r) for r in rows); idx = np.zeros((h, ww), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in '. ': idx[y, x] = int(ch) if ch.isdigit() else 1
    idx = strip_frame(idx); h, ww = idx.shape; a = np.full((h, ww, 3), 255, np.uint8)
    for k in range(1, int(idx.max()) + 1):
        c = pal[min(k, len(pal) - 1)].lstrip('#'); a[idx == k] = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    s = max(1, min(px, 160 // max(h, ww))); return Image.fromarray(a).resize((ww * s, h * s), Image.NEAREST)
def panels(sids, per=12, cell=150):   # 12 схем на панель, номера сквозные 1..N
    out = []
    for p0 in range(0, len(sids), per):
        chunk = sids[p0:p0 + per]; cols = 4; rows = (len(chunk) + cols - 1) // cols
        P = Image.new('RGB', (cols * cell, rows * (cell + 20)), 'white'); d = ImageDraw.Draw(P)
        for i, sid in enumerate(chunk):
            im = scheme_img(sid); im.thumbnail((cell - 10, cell - 10)); x, y = (i % cols) * cell, (i // cols) * (cell + 20)
            P.paste(im, (x + (cell - im.width) // 2, y + 20 + (cell - im.height) // 2)); d.text((x + 6, y + 4), str(p0 + i + 1), fill='black'); d.rectangle([x, y, x + cell - 1, y + cell + 19], outline='#999')
        out.append(P)
    return out
def b64(im, q=85):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q); return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()
PANELS = [b64(p) for p in panels(cands)]

# ---------- фото из пакета на Диске ----------
OUT = '/content/drive/MyDrive/carpet-dna/vlm_census'
from google.colab import drive; drive.mount('/content/drive'); os.makedirs(OUT, exist_ok=True)
zp = glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/rugs_bundle.zip'); assert zp, 'нет carpet-dna/rug_catalog/rugs_bundle.zip на Диске'
Z = zipfile.ZipFile(zp[0]); names = set(Z.namelist()); zl = threading.Lock()
def photo(rid):
    lib = rugs[rid]
    with zl:
        for cand in (lib, 'img/' + os.path.basename(lib), f'img/{rid}.jpg', f'rug_img/{rid}.jpg'):
            if cand in names: data = Z.read(cand); break
        else: return None
    im = Image.open(io.BytesIO(data)).convert('RGB'); im.thumbnail((SIDE, SIDE)); d = ImageDraw.Draw(im); w, h = im.size
    for k in range(1, 10):
        d.line([(w * k / 10, 0), (w * k / 10, h)], fill=(255, 255, 255), width=1); d.line([(0, h * k / 10), (w, h * k / 10)], fill=(255, 255, 255), width=1)
        d.text((w * k / 10 + 2, 2), str(k * 10), fill=(255, 255, 0)); d.text((2, h * k / 10 + 2), str(k * 10), fill=(255, 255, 0))
    return im

PROMPT = ('Первое изображение — фото азербайджанского ковра (или его фрагмента) с сеткой через 10 %. Следующие изображения — панели пронумерованных чертежей-схем '
          f'элементов узора (1…{len(cands)}; каждая клетка = один узел). Определи, какие из этих элементов ЕСТЬ на ковре (фото может быть повёрнуто, цвета другие, ворс сглаживает форму). '
          'Не называй элемент, если не уверен: лишние ответы хуже пропусков. Для каждого найденного дай номер схемы, уверенность 0..1 и рамки [x,y,w,h] в процентах фото, по одной на каждое место, где он виден (до 6). '
          'Если это не ковёр или узор не разобрать, верни пустой список. Ответь только JSON: {"found":[{"n":<номер>,"conf":<0..1>,"boxes":[[x,y,w,h],...]}]}')
cost = [0.0]; lock = threading.Lock(); ERR = [0]; NOPHOTO = [0]
def ask(im):
    content = [dict(type='text', text=PROMPT), dict(type='image_url', image_url=dict(url=b64(im)))] + [dict(type='image_url', image_url=dict(url=u)) for u in PANELS]
    body = dict(model=MODEL, temperature=0, max_tokens=900, usage=dict(include=True), messages=[dict(role='user', content=content)])
    for t in range(5):
        try:
            rq = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', json.dumps(body).encode(), {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
            out = json.load(urllib.request.urlopen(rq, timeout=180)); u = out.get('usage') or {}
            c = u.get('cost'); c = c if c is not None else (u.get('prompt_tokens', 0) * PIN + u.get('completion_tokens', 0) * POUT) / 1e6
            with lock: cost[0] += c
            s = out['choices'][0]['message']['content']; j = re.search(r'\{.*\}', s, re.S)
            return (json.loads(j.group(0)) if j else {'found': []}), u, c
        except Exception as e:
            msg = str(e)
            if hasattr(e, 'read'):
                try: msg += ' | ' + e.read().decode('utf-8', 'replace')[:400]
                except Exception: pass
            with lock:
                if ERR[0] < 4: print('ОШИБКА ЗАПРОСА:', msg, flush=True)
                ERR[0] += 1
            code = getattr(e, 'code', 0)
            if code in (400, 401, 402, 403, 404): break      # повтор не поможет
            time.sleep(6 * (t + 1))
    return None, {}, 0

fn = f"{OUT}/{MODEL.replace('/', '_').replace(':', '_')}.jsonl"; done = set()
if os.path.exists(fn): done = {json.loads(l)['rug'] for l in open(fn) if l.strip()}
todo = [r for r in sorted(rugs, key=lambda r: (r not in marked if os.environ.get('VC_PRIOR') == '1' else 0, r)) if r not in done]
if LIMIT: todo = todo[:LIMIT]
print(f'модель {MODEL}: уже сделано {len(done)}, в очереди {len(todo)}, бюджет ${MAX_USD}')
fout = open(fn, 'a', encoding='utf-8'); t0 = time.time(); nbad = [0]; nok = [0]
def work(rid):
    if cost[0] >= MAX_USD or (ERR[0] >= 8 and nok[0] == 0): return
    im = photo(rid)
    if im is None: NOPHOTO[0] += 1; return
    res, u, c = ask(im)
    if res is None: nbad[0] += 1; return
    found = [dict(sid=cands[int(f['n']) - 1], conf=f.get('conf'), boxes=f.get('boxes') or []) for f in res.get('found', []) if str(f.get('n', '')).isdigit() and 1 <= int(f['n']) <= len(cands)]
    with lock:
        nok[0] += 1
        fout.write(json.dumps(dict(rug=rid, found=found, tokens=[u.get('prompt_tokens'), u.get('completion_tokens')], usd=round(c, 5)), ensure_ascii=False) + '\n'); fout.flush()
with ThreadPoolExecutor(THREADS) as ex:
    for i, _ in enumerate(ex.map(work, todo)):
        if (i + 1) % 25 == 0: print(f'{i + 1}/{len(todo)}  потрачено ${cost[0]:.3f}  ({time.time() - t0:.0f} с)', flush=True)
rows = [json.loads(l) for l in open(fn) if l.strip()]
n = max(1, len(rows)); tok = [sum((r['tokens'][k] or 0) for r in rows) for k in (0, 1)]
print(f'без фото в пакете: {NOPHOTO[0]}')
print(f'\n=== {MODEL}: ковров обработано {len(rows)}, ошибок запроса {nbad[0]}, потрачено ${cost[0]:.3f} в этой сессии, в среднем ${sum(r["usd"] for r in rows) / n:.5f} на ковёр; токенов на ковёр: вход {tok[0] // n}, выход {tok[1] // n}')
sch = collections.Counter(f['sid'] for r in rows for f in r['found'])
print('элементов на ковёр в среднем:', round(sum(len(r['found']) for r in rows) / n, 1), '; чаще всего:', [(s, c) for s, c in sch.most_common(8)])
hit = tot = 0   # сверка с вашей разметкой там, где она есть
for r in rows:
    if r['rug'] in marked:
        want = {m['scheme'] for m in marked[r['rug']]}; got = {f['sid'] for f in r['found']}; hit += len(want & got); tot += len(want)
print(f'сверка с вашей разметкой: найдено {hit} из {tot} размеченных элементов ({100 * hit / max(1, tot):.0f} %)')
print('Готово:', fn)
