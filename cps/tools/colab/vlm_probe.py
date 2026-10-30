# vlm_probe.py — проба «умеет ли модель зрения-языка узнать элемент по схеме Керимова» (v12.0, без обучения).
#
# Вопрос эксперимента: вырезка рамки с фото ковра + панель из 8 схем (одна верная, 7 чужих, разных по форме) → какая верная?
# Правда — ваши подтверждённые рамки со схемой студии (≈270 рамок, 34 схемы). Случайный ответ = 12,5 %.
# Если модель стабильно > 60 % — «имена по закрытому списку типа» решаются промптом + проверкой человека (план этап 3),
# и дальше строим перепись каталога. Если ≈ случайно — идём в синтетику и обучение своего детектора.
#
# Ключ OpenRouter: в Colab слева значок ключа («Секреты») → имя OPENROUTER_API_KEY → включить доступ. В чат ключ не присылать.
# Запуск (CPU хватает, ≈30–60 мин на модель из-за лимитов бесплатных моделей):
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env VP_MODELS=qwen/qwen2.5-vl-72b-instruct:free
#   %run /content/carpevo/cps/tools/colab/vlm_probe.py
# Параметры: VP_MODELS (через запятую; для сравнения добавьте anthropic/claude-sonnet-4.5), VP_N (сколько рамок, по умолчанию 120),
#   VP_K (схем в панели, 8), VP_SEED, VP_BUNDLE (id пакета на Диске; по умолчанию из rugs_settings.json ветки cps-data).
import os, io, re, json, glob, time, base64, random, zipfile, subprocess, collections, urllib.request
import numpy as np
from PIL import Image, ImageDraw

W = '/content/vp'; os.makedirs(W, exist_ok=True)
DATA = '/content/cpsdata'
if not os.path.isdir(DATA + '/data'):
    subprocess.run(['git', 'clone', '--depth', '1', '-b', 'cps-data', 'https://github.com/22adolgikh-pixel/carpevo.git', DATA], check=True)
D = DATA + '/data'
N = int(os.environ.get('VP_N', 120)); K = int(os.environ.get('VP_K', 8)); SEED = int(os.environ.get('VP_SEED', 1))
MODELS = [m.strip() for m in os.environ.get('VP_MODELS', 'qwen/qwen2.5-vl-72b-instruct:free').split(',') if m.strip()]
EXPORT = os.environ.get('VP_EXPORT', '') == '1' or os.environ.get('VP_FULL', '') == '1'   # 1 = не спрашивать модели, а сложить вопросы в zip (для проверки внутри Claude)
KEY = ''
if not EXPORT:
    try:
        from google.colab import userdata; KEY = userdata.get('OPENROUTER_API_KEY')
    except Exception:
        KEY = os.environ.get('OPENROUTER_API_KEY', '')
    assert KEY, 'нет ключа: Секреты Colab → OPENROUTER_API_KEY'

# ---------- правда: подтверждённые рамки со схемой ----------
works = {os.path.splitext(f)[0] for f in os.listdir(D + '/work')}
marks = []
for f in glob.glob(D + '/rugs/*.json'):
    r = json.load(open(f, encoding='utf-8'))
    for m in r.get('marks') or []:
        if m.get('status') == 'confirmed' and m.get('scheme') in works and m.get('box'):
            marks.append(dict(rug=r['id'], lib=r.get('lib_img') or f"img/{r['id']}.jpg", box=m['box'], scheme=m['scheme'], poly=m.get('poly')))
print('рамок со схемой:', len(marks), 'схем:', len({m['scheme'] for m in marks}))

# ---------- картинки ковров из пакета на Диске ----------
need = {m['rug']: m['lib'] for m in marks}
IMG = W + '/img'; os.makedirs(IMG, exist_ok=True)
def have(rid): return os.path.exists(f'{IMG}/{rid}.jpg')
if not all(have(r) for r in need):
    from google.colab import drive; drive.mount('/content/drive')
    bid = os.environ.get('VP_BUNDLE') or json.load(open(D + '/rugs_settings.json')).get('bundle_id')
    zp = glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/rugs_bundle.zip')
    assert zp, 'нет carpet-dna/rug_catalog/rugs_bundle.zip на Диске (id пакета: %s)' % bid
    with zipfile.ZipFile(zp[0]) as z:
        names = set(z.namelist())
        for rid, lib in need.items():
            for cand in (lib, 'img/' + os.path.basename(lib), f'img/{rid}.jpg', f'rug_img/{rid}.jpg'):
                if cand in names:
                    open(f'{IMG}/{rid}.jpg', 'wb').write(z.read(cand)); break
print('фото ковров:', sum(have(r) for r in need), 'из', len(need))
marks = [m for m in marks if have(m['rug'])]

# ---------- схема → картинка ----------
def strip_frame(a):
    """У части схем (24 из 1084, напр. qapı t40_f03) по периметру нарисована рамка, на ковре её нет:
    убираем сплошное кольцо и однотонную заливку между ним и узором. Проверено 30.10: qapı 3/7 -> 7/7."""
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
    w = json.load(open(f'{D}/work/{sid}.json', encoding='utf-8')); mx = w['matrix']; pal = w.get('palette') or ['#ffffff', '#1a1a1a', '#b8b8b8']
    rows = mx['rows']; h = len(rows); ww = max(len(r) for r in rows); idx = np.zeros((h, ww), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in '. ': continue
            idx[y, x] = int(ch) if ch.isdigit() else 1
    if os.environ.get('VP_KEEP_FRAME', '') != '1': idx = strip_frame(idx)
    h, ww = idx.shape; a = np.full((h, ww, 3), 255, np.uint8)
    for k in range(1, int(idx.max()) + 1):
        c = pal[min(k, len(pal) - 1)].lstrip('#'); a[idx == k] = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    s = max(1, min(px, 160 // max(h, ww)))
    return Image.fromarray(a).resize((ww * s, h * s), Image.NEAREST)

def panel(sids, cell=170):
    cols = 4; rows = (len(sids) + cols - 1) // cols
    P = Image.new('RGB', (cols * cell, rows * (cell + 22)), 'white'); d = ImageDraw.Draw(P)
    for i, sid in enumerate(sids):
        im = scheme_img(sid); im.thumbnail((cell - 10, cell - 10)); x, y = (i % cols) * cell, (i // cols) * (cell + 22)
        P.paste(im, (x + (cell - im.width) // 2, y + 22 + (cell - im.height) // 2)); d.text((x + 6, y + 4), 'ABCDEFGHIJKLMNOP'[i], fill='black')
        d.rectangle([x, y, x + cell - 1, y + cell + 21], outline='#999')
    return P

def crop(m, pad=0.25, side=420):
    im = Image.open(f"{IMG}/{m['rug']}.jpg").convert('RGB'); W_, H_ = im.size; x, y, w, h = m['box']
    x0, y0 = max(0, (x - w * pad) * W_), max(0, (y - h * pad) * H_); x1, y1 = min(W_, (x + w * (1 + pad)) * W_), min(H_, (y + h * (1 + pad)) * H_)
    c = im.crop((int(x0), int(y0), int(x1), int(y1))); c.thumbnail((side, side)); return c

def b64(im):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=88); return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()

# ---------- форма схем: чужие берём непохожие по размеру и плотности, чтобы вопрос был честным, но не издевательским ----------
def sig(sid):
    mx = json.load(open(f'{D}/work/{sid}.json', encoding='utf-8'))['matrix']; rows = mx['rows']
    return (len(rows), max(len(r) for r in rows))
allsch = sorted({m['scheme'] for m in marks})

def ask(model, img_crop, img_panel, k):
    letters = 'ABCDEFGHIJKLMNOP'[:k]
    txt = (f'Первое изображение — вырезка фрагмента с фотографии азербайджанского ковра: один повторяющийся элемент узора. '
           f'Второе — панель из {k} чертежей-схем (каждая клетка = один узел), подписанных буквами {", ".join(letters)}. '
           f'Одна схема изображает тот же элемент, что на вырезке (фото может быть повёрнуто, отражено, цвета и размер другие, '
           f'ворс сглаживает форму). Определи по форме контура и внутреннего рисунка, какая. '
           f'Ответь JSON: {{"answer":"<буква>","second":"<вторая по вероятности>","confidence":<0..1>}}')
    body = dict(model=model, temperature=0, max_tokens=120, messages=[dict(role='user', content=[
        dict(type='text', text=txt), dict(type='image_url', image_url=dict(url=b64(img_crop))), dict(type='image_url', image_url=dict(url=b64(img_panel)))])])
    for t in range(5):
        try:
            rq = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', json.dumps(body).encode(),
                                        {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
            out = json.load(urllib.request.urlopen(rq, timeout=120))
            s = out['choices'][0]['message']['content']; j = re.search(r'\{.*?\}', s, re.S)
            return json.loads(j.group(0)) if j else {'answer': (re.search(r'\b([A-P])\b', s) or [None, ''])[1]}
        except Exception as e:
            time.sleep(8 * (t + 1))
    return None

rng = random.Random(SEED); pick = rng.sample(marks, min(N, len(marks)))
# не больше 6 рамок на схему, чтобы одна схема не занимала всю выборку
cnt = collections.Counter(); sel = []
for m in pick:
    if cnt[m['scheme']] < 6: sel.append(m); cnt[m['scheme']] += 1
print('вопросов:', len(sel), 'схем в выборке:', len(cnt))
OUT = '/content/drive/MyDrive/carpet-dna/vlm_probe' if os.path.isdir('/content/drive/MyDrive') else W
os.makedirs(OUT, exist_ok=True)
if os.environ.get('VP_FULL', '') == '1':   # целые ковры (≤1600 px) + ✓ рамки со схемами — для пробы «найди элемент без рамки»
    zp = f'{OUT}/probe_full.zip'; key = {}
    for m in marks: key.setdefault(m['rug'], []).append(dict(box=m['box'], scheme=m['scheme']))
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, rid in enumerate(sorted(key)):
            im = Image.open(f'{IMG}/{rid}.jpg').convert('RGB'); im.thumbnail((1600, 1600))
            b = io.BytesIO(); im.save(b, 'JPEG', quality=88); z.writestr(f'rug{i:02d}.jpg', b.getvalue()); key[rid] = dict(file=f'rug{i:02d}.jpg', size=im.size, marks=key[rid])
        z.writestr('full_key.json', json.dumps(key, ensure_ascii=False, indent=1))
    print(f'Готово: {zp} ({os.path.getsize(zp) / 1e6:.1f} МБ, ковров {len(key)})')
    MODELS = []; EXPORT = False
if EXPORT:
    zp = f'{OUT}/probe_questions.zip'; key = []
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, m in enumerate(sel):
            others = [s for s in allsch if s != m['scheme']]; r2 = random.Random(SEED * 1000 + i); opts = r2.sample(others, min(K - 1, len(others))) + [m['scheme']]; r2.shuffle(opts)
            b = io.BytesIO(); crop(m).save(b, 'JPEG', quality=88); z.writestr(f'q{i:03d}_photo.jpg', b.getvalue())
            b = io.BytesIO(); panel(opts).save(b, 'PNG'); z.writestr(f'q{i:03d}_schemes.png', b.getvalue())
            key.append(dict(q=i, rug=m['rug'], scheme=m['scheme'], truth='ABCDEFGHIJKLMNOP'[opts.index(m['scheme'])], opts=opts))
        z.writestr('answer_key.json', json.dumps(key, ensure_ascii=False, indent=1))
    print(f'Готово: {zp} ({os.path.getsize(zp) / 1e6:.1f} МБ, вопросов {len(key)}). Скачайте его с Диска и приложите в чат Claude.')
    MODELS = []
for model in MODELS:
    log = []; ok = ok2 = bad = 0; per = collections.defaultdict(lambda: [0, 0]); t0 = time.time()
    for i, m in enumerate(sel):
        others = [s for s in allsch if s != m['scheme']]; r2 = random.Random(SEED * 1000 + i); opts = r2.sample(others, min(K - 1, len(others))) + [m['scheme']]; r2.shuffle(opts)
        res = ask(model, crop(m), panel(opts), len(opts))
        if res is None: bad += 1; continue
        a = str(res.get('answer', '')).strip().upper()[:1]; s2 = str(res.get('second', '')).strip().upper()[:1]
        truth = 'ABCDEFGHIJKLMNOP'[opts.index(m['scheme'])]
        hit, hit2 = a == truth, (a == truth or s2 == truth); ok += hit; ok2 += hit2; per[m['scheme']][0] += hit; per[m['scheme']][1] += 1
        log.append(dict(rug=m['rug'], scheme=m['scheme'], truth=truth, ans=a, second=s2, conf=res.get('confidence'), opts=opts))
        if (i + 1) % 10 == 0: print(f'{model}: {i + 1}/{len(sel)} верно {ok}/{len(log)}  ({time.time() - t0:.0f} c)', flush=True)
    n = max(1, len(log)); print(f'\n=== {model}: верно с 1-й попытки {ok}/{len(log)} = {100 * ok / n:.0f} %, в двух лучших {100 * ok2 / n:.0f} %, ошибок запроса {bad}; случайно было бы {100 / K:.1f} %')
    print('по схемам (верно/всего):', {k: tuple(v) for k, v in sorted(per.items(), key=lambda t: -t[1][1])[:12]})
    json.dump(dict(model=model, k=K, n=len(log), acc=ok / n, acc2=ok2 / n, log=log), open(f"{OUT}/probe_{model.replace('/', '_').replace(':', '_')}.json", 'w'), ensure_ascii=False, indent=1)
print('Готово. Результаты на Диске:', OUT, '— пришлите Claude строки «===».')
