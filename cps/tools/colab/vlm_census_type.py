# vlm_census_type.py — перепись «какие элементы есть на ковре» с ЗАКРЫТЫМ СПИСКОМ ПО ТИПУ ковра (v12.2).
#
# Тип ковра берётся из заголовка/композиции/заметок: название типа = meta.carpet схем (кириллица), сопоставление «скелетов» слов с латиницей;
# кандидаты — схемы этого типа, рамка на схемах убрана. Ковры без типа или у типа без схем пропускаются.
# Проверка 31.10 на 13 коврах: Haiku 5.5 находит 53 % элементов (лишних 18 %), Sonnet 59 % (лишних 35 %); без типа — 5 % и 21 %.
# Результат — подсказки «ждут проверки» (какие элементы вероятно есть), не разметка.
#
# Ключ OpenRouter: Colab → Секреты → OPENROUTER_API_KEY (проверьте, что он не просрочен и на счёте есть $1). В чат ключ не присылать.
# Запуск:
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env VC_MODEL=anthropic/claude-haiku-5.5
#   %run /content/carpevo/cps/tools/colab/vlm_census_type.py
# Параметры: VC_MODEL, VC_LIMIT (0 = все), VC_MAX_USD (3), VC_THREADS (6), VC_SIDE (1280), VC_DRY=1 (только показать список ковров и типов, без запросов).
# Результат: Диск carpet-dna/vlm_census/type_<модель>.jsonl (перезапуск продолжает), сводка в конце.
import os, io, re, json, glob, time, base64, collections, zipfile, subprocess, unicodedata, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw

MODEL = os.environ.get('VC_MODEL', 'anthropic/claude-haiku-5.5'); LIMIT = int(os.environ.get('VC_LIMIT', 0)); DRY = os.environ.get('VC_DRY') == '1'
MAX_USD = float(os.environ.get('VC_MAX_USD', 3)); THREADS = int(os.environ.get('VC_THREADS', 6)); SIDE = int(os.environ.get('VC_SIDE', 1280))
PIN = float(os.environ.get('VC_PRICE_IN', 0.1)); POUT = float(os.environ.get('VC_PRICE_OUT', 0.5))
REPO = '/content/carpevo'; DATA = '/content/cpsdata'
if not os.path.isdir(DATA + '/data'):
    subprocess.run(['git', 'clone', '--depth', '1', '-b', 'cps-data', 'https://github.com/22adolgikh-pixel/carpevo.git', DATA], check=True)
D = DATA + '/data'
KEY = ''
if not DRY:
    try:
        from google.colab import userdata; KEY = userdata.get('OPENROUTER_API_KEY')
    except Exception:
        KEY = os.environ.get('OPENROUTER_API_KEY', '')
    assert KEY, 'нет ключа: Секреты Colab → OPENROUTER_API_KEY'

def fold(s):
    s = (s or '').lower()
    for a, b in (('ə', 'a'), ('ı', 'i'), ('ö', 'o'), ('ü', 'u'), ('ç', 'c'), ('ş', 's'), ('ğ', 'g'), ('й', 'и'), ('ё', 'е')): s = s.replace(a, b)
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return re.sub(r'[^a-zа-я0-9]+', ' ', s).strip()

# ---------- типы и их схемы ----------
# Тип = название ковра в метаданных схем (meta.carpet, кириллица, 193 названия). Ищем его в заголовке/композиции/заметках ковра:
# сравниваем «скелеты» слов (кириллица и латиница → одна латиница, гласные схлопнуты), чтобы «Пиребедиль» = «Pirabadil» = «Pirəbədil».
M = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ж': 'j', 'з': 'z', 'и': 'i', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'c', 'ч': 'c', 'ш': 's', 'щ': 's', 'ы': 'i', 'э': 'e', 'ю': 'u', 'я': 'a', 'ь': '', 'ъ': '', 'ә': 'a', 'һ': 'h', 'ғ': 'g', 'ө': 'o', 'ү': 'u', 'ј': 'j', 'ҝ': 'g', 'ҹ': 'c', ' ': ' '}
def skel(s):
    s = ''.join(M.get(c, c) for c in fold(s)); s = s.replace('q', 'g').replace('h', 'x').replace('w', 'v').replace('y', 'i')
    return re.sub(r'[aeiou]', 'a', re.sub(r'(.)\1+', r'\1', s)).replace(' ', '')
REG = {'карабах', 'ширван', 'куба', 'казах', 'к', 'г', 'хан карабах', 'гянджа', 'генже', 'баку', 'тебриз', 'кубинский', 'карабахский', 'казахский'}   # широкие регионы — не тип
ALIAS = {'hajigayib': 'Гаджигаиб', 'hacigayib': 'Гаджигаиб', 'hajigaib': 'Гаджигаиб', 'hajiqayib': 'Гаджигаиб', 'hacigaib': 'Гаджигаиб', 'гаджигаиыб': 'Гаджигаиб', 'гаджигаиб': 'Гаджигаиб'}
carp = collections.defaultdict(list)
for f in sorted(glob.glob(D + '/work/*.json')):
    w = json.load(open(f, encoding='utf-8')); m = w.get('meta') or {}
    if m.get('carpet') and w.get('matrix') and w['matrix'].get('rows'): carp[m['carpet'].strip()].append(w['id'])
rows = [dict(name=c, schemes=ids, key=skel(c)) for c, ids in sorted(carp.items()) if fold(c) not in REG and len(skel(c)) >= 5 and len(ids) >= 2]
for i, r in enumerate(rows): r['n'] = i + 1
byname = {r['name']: r for r in rows}
def find_type(txt):
    toks = fold(txt).split(); grams = {skel(' '.join(toks[i:i + n])) for n in (1, 2) for i in range(len(toks))}; ft = ' '.join(toks)
    out = {r['n'] for r in rows if r['key'] in grams} | {byname[v]['n'] for a, v in ALIAS.items() if a in ft and v in byname}
    return sorted(out)

rugs = {}; marked = {}
works = {os.path.splitext(f)[0] for f in os.listdir(D + '/work')}
for f in glob.glob(D + '/rugs/*.json'):
    r = json.load(open(f, encoding='utf-8'))
    if r.get('hidden') or r.get('dup_of'): continue
    ty = find_type(' '.join([r.get('title', ''), r.get('composition', ''), r.get('notes', '')]))
    if len(ty) != 1: continue          # нет типа, неоднозначно или у типа нет схем
    rugs[r['id']] = dict(lib=r.get('lib_img') or f"img/{r['id']}.jpg", type=ty[0])
    ms = [m for m in (r.get('marks') or []) if m.get('status') == 'confirmed' and m.get('scheme') in works and m.get('box')]
    if ms: marked[r['id']] = {m['scheme'] for m in ms}
print('ковров с однозначным типом и схемами типа:', len(rugs), '; из них с вашей разметкой:', len(marked), '; типов:', len({v['type'] for v in rugs.values()}))
print('по типам:', collections.Counter(rows[v['type'] - 1]['name'] for v in rugs.values()).most_common(12))
if DRY: raise SystemExit

# ---------- картинки ----------
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
    w = json.load(open(f'{D}/work/{sid}.json', encoding='utf-8')); rws = w['matrix']['rows']; pal = w.get('palette') or ['#ffffff', '#1a1a1a', '#b8b8b8']
    h = len(rws); ww = max(len(r) for r in rws); idx = np.zeros((h, ww), np.uint8)
    for y, row in enumerate(rws):
        for x, ch in enumerate(row):
            if ch not in '. ': idx[y, x] = int(ch) if ch.isdigit() else 1
    idx = strip_frame(idx); h, ww = idx.shape; a = np.full((h, ww, 3), 255, np.uint8)
    for k in range(1, int(idx.max()) + 1):
        c = pal[min(k, len(pal) - 1)].lstrip('#'); a[idx == k] = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    s = max(1, min(px, 160 // max(h, ww))); return Image.fromarray(a).resize((ww * s, h * s), Image.NEAREST)
def panels(sids, per=12, cell=150):
    out = []
    for p0 in range(0, len(sids), per):
        chunk = sids[p0:p0 + per]; cols = 4; rr = (len(chunk) + cols - 1) // cols
        P = Image.new('RGB', (cols * cell, rr * (cell + 20)), 'white'); d = ImageDraw.Draw(P)
        for i, sid in enumerate(chunk):
            im = scheme_img(sid); im.thumbnail((cell - 10, cell - 10)); x, y = (i % cols) * cell, (i // cols) * (cell + 20)
            P.paste(im, (x + (cell - im.width) // 2, y + 20 + (cell - im.height) // 2)); d.text((x + 6, y + 4), str(p0 + i + 1), fill='black'); d.rectangle([x, y, x + cell - 1, y + cell + 19], outline='#999')
        out.append(P)
    return out
def b64(im, q=85):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q); return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()
PANELS = {n: [b64(p) for p in panels(rows[n - 1]['schemes'])] for n in {v['type'] for v in rugs.values()}}

OUT = '/content/drive/MyDrive/carpet-dna/vlm_census'
from google.colab import drive; drive.mount('/content/drive'); os.makedirs(OUT, exist_ok=True)
zp = glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/rugs_bundle.zip'); assert zp, 'нет carpet-dna/rug_catalog/rugs_bundle.zip на Диске'
Z = zipfile.ZipFile(zp[0]); names = set(Z.namelist()); zl = threading.Lock()
def photo(rid):
    lib = rugs[rid]['lib']
    with zl:
        for cand in (lib, 'img/' + os.path.basename(lib), f'img/{rid}.jpg', f'rug_img/{rid}.jpg'):
            if cand in names: data = Z.read(cand); break
        else: return None
    im = Image.open(io.BytesIO(data)).convert('RGB'); im.thumbnail((SIDE, SIDE)); d = ImageDraw.Draw(im); w, h = im.size
    for k in range(1, 10):
        d.line([(w * k / 10, 0), (w * k / 10, h)], fill=(255, 255, 255), width=1); d.line([(0, h * k / 10), (w, h * k / 10)], fill=(255, 255, 255), width=1)
        d.text((w * k / 10 + 2, 2), str(k * 10), fill=(255, 255, 0)); d.text((2, h * k / 10 + 2), str(k * 10), fill=(255, 255, 0))
    return im

cost = [0.0]; lock = threading.Lock(); ERR = [0]; NOPHOTO = [0]; nok = [0]
def ask(im, tn):
    k = len(rows[tn - 1]['schemes'])
    prompt = (f'Первое изображение — фото азербайджанского ковра типа «{rows[tn - 1]["name"]}» (или его фрагмента) с сеткой через 10 %. Следующие изображения — панели пронумерованных чертежей-схем '
              f'элементов узора этого типа (1…{k}; каждая клетка = один узел). Определи, какие из этих элементов ЕСТЬ на ковре (фото может быть повёрнуто, цвета другие, ворс сглаживает форму; элемент обычно повторяется). '
              'Не называй элемент, если не уверен: лишние ответы хуже пропусков. Для каждого найденного дай номер схемы, уверенность 0..1 и рамки [x,y,w,h] в процентах фото, по одной на каждое место (до 6). '
              'Если это не ковёр или узор не разобрать, верни пустой список. Ответь только JSON: {"found":[{"n":<номер>,"conf":<0..1>,"boxes":[[x,y,w,h],...]}]}')
    content = [dict(type='text', text=prompt), dict(type='image_url', image_url=dict(url=b64(im)))] + [dict(type='image_url', image_url=dict(url=u)) for u in PANELS[tn]]
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
                try: msg += ' | ' + e.read().decode('utf-8', 'replace')[:300]
                except Exception: pass
            with lock:
                if ERR[0] < 4: print('ОШИБКА ЗАПРОСА:', msg, flush=True)
                ERR[0] += 1
            if getattr(e, 'code', 0) in (400, 401, 402, 403, 404): break
            time.sleep(6 * (t + 1))
    return None, {}, 0

fn = f"{OUT}/type_{MODEL.replace('/', '_').replace(':', '_')}.jsonl"; done = set()
if os.path.exists(fn): done = {json.loads(l)['rug'] for l in open(fn) if l.strip()}
todo = [r for r in sorted(rugs, key=lambda r: (r not in marked, r)) if r not in done]
if LIMIT: todo = todo[:LIMIT]
print(f'модель {MODEL}: уже сделано {len(done)}, в очереди {len(todo)}, бюджет ${MAX_USD}')
fout = open(fn, 'a', encoding='utf-8'); t0 = time.time()
def work(rid):
    if cost[0] >= MAX_USD or (ERR[0] >= 8 and nok[0] == 0): return
    im = photo(rid)
    if im is None: NOPHOTO[0] += 1; return
    tn = rugs[rid]['type']; sc = rows[tn - 1]['schemes']
    res, u, c = ask(im, tn)
    if res is None: return
    found = [dict(sid=sc[int(f['n']) - 1], conf=f.get('conf'), boxes=f.get('boxes') or []) for f in res.get('found', []) if str(f.get('n', '')).isdigit() and 1 <= int(f['n']) <= len(sc)]
    with lock:
        nok[0] += 1; fout.write(json.dumps(dict(rug=rid, type=rows[tn - 1]['name'], found=found, tokens=[u.get('prompt_tokens'), u.get('completion_tokens')], usd=round(c, 5)), ensure_ascii=False) + '\n'); fout.flush()
with ThreadPoolExecutor(THREADS) as ex:
    for i, _ in enumerate(ex.map(work, todo)):
        if (i + 1) % 20 == 0: print(f'{i + 1}/{len(todo)}  потрачено ${cost[0]:.3f}  ({time.time() - t0:.0f} с)', flush=True)
rws = [json.loads(l) for l in open(fn) if l.strip()]; n = max(1, len(rws))
print(f'без фото в пакете: {NOPHOTO[0]}')
print(f'\n=== {MODEL}: ковров {len(rws)}, потрачено ${cost[0]:.3f} в этой сессии, в среднем ${sum(r["usd"] for r in rws) / n:.5f} на ковёр, элементов на ковёр в среднем {sum(len(r["found"]) for r in rws) / n:.1f}')
hit = tot = fp = 0
for r in rws:
    if r['rug'] in marked:
        want = marked[r['rug']] & set(rows[rugs[r['rug']]['type'] - 1]['schemes']); got = {f['sid'] for f in r['found']}; hit += len(want & got); tot += len(want); fp += len(got - want)
print(f'сверка с вашей разметкой (ковры с рамками): найдено {hit} из {tot} ({100 * hit / max(1, tot):.0f} %), лишних {fp}')
print('Готово:', fn)
