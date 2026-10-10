# rug_catalog_colab.py — единый каталог ковров («паспорта») из всего, что лежит на Диске, и поиск одного и того же ковра в разных источниках.
#
# Источники (папки carpet-dna на Диске, ничего загружать вручную не нужно):
#   • Керимов т. I–III: output/page_images/kerimov_volN/page_NNN.jpg + output/raw_pages_v3/kerimov_volN/page_NNN.json
#     (на странице ищутся фото — плотные прямоугольные области; подписи: «Табл. 217. ПАЛАС…», «Рис. 44» + «carriers_mentioned» → тип ковра, дата, школа);
#   • музейные фото: photos/full/* + site/model/sources/museum_photos/catalog_photos.json из репозитория (Wikimedia, Met, AIC, Cleveland — с подписью музея, школой, техникой);
#   • V&A / Met: photos/museum_open/img + catalog.json;
#   • azerbaijanrugs: photos/azerbaijan_rugs_guide (папки + одноимённые .txt).
# Для каждого ковра — паспорт: название(я), источник, музей/номер, дата, место, техника (ворс/безворс), целый/фрагмент, школа + почему.
# Двойники: признаки DINOv2 (фото, зеркало, поворот на 180°) → кандидаты → проверка точками SIFT + RANSAC (одна и та же вещь, а не «похожий узор»).
# Результат: Drive carpet-dna/rug_catalog/ — catalog.json, groups.json, dup_sheets/*.jpg (листы двойников для проверки глазами),
#            rugs_bundle.zip (каталог + фото ≤1000 px для студии; доступ «по ссылке», id печатается в конце и хранится в bundle_id.txt).
# Время: 40–70 мин на T4 (больше всего — чтение фото с Диска и SIFT). Повторный запуск использует кэш признаков.
import os, re, io, json, glob, csv, time, zipfile, hashlib, collections, subprocess
import numpy as np, cv2, torch
from google.colab import drive, auth
drive.mount('/content/drive')
D = os.environ.get('RC_ROOT', '/content/drive/MyDrive/carpet-dna')
OUT = f'{D}/rug_catalog'; os.makedirs(f'{OUT}/dup_sheets', exist_ok=True)
CACHE = os.environ.get('RC_CACHE', f'{OUT}/_cache'); os.makedirs(CACHE, exist_ok=True)   # кэш на Диске: если Colab отключится, повторный запуск продолжит с места остановки
REPO = os.environ.get('RC_REPO', '/content/carpevo')
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '--depth', '1', '-b', 'cps-v5', 'https://github.com/22adolgikh-pixel/carpevo.git', REPO], check=True)
t0 = time.time()
def log(*a): print(f'[{time.time() - t0:5.0f} c]', *a, flush=True)

# ---------------- словари: школа, техника, фрагмент ----------------
SCHOOL_KEYS = [('karabag', 'karabakh'), ('karabak', 'karabakh'), ('карабах', 'karabakh'), ('shusha', 'karabakh'), ('шуш', 'karabakh'), ('lampa', 'karabakh'),
    ('kazak', 'ganja-gazakh'), ('gazakh', 'ganja-gazakh'), ('казах', 'ganja-gazakh'), ('газах', 'ganja-gazakh'), ('ganja', 'ganja-gazakh'), ('gendje', 'ganja-gazakh'),
    ('genje', 'ganja-gazakh'), ('гянджа', 'ganja-gazakh'), ('borchal', 'ganja-gazakh'), ('bordjal', 'ganja-gazakh'), ('борчал', 'ganja-gazakh'), ('fachral', 'ganja-gazakh'),
    ('fakhral', 'ganja-gazakh'), ('фахрал', 'ganja-gazakh'), ('chelaberd', 'ganja-gazakh'), ('челеберд', 'ganja-gazakh'),
    ('kuba', 'guba-shirvan'), ('quba', 'guba-shirvan'), ('guba', 'guba-shirvan'), ('куба', 'guba-shirvan'), ('губа', 'guba-shirvan'), ('shirvan', 'guba-shirvan'),
    ('ширван', 'guba-shirvan'), ('baku', 'guba-shirvan'), ('баку', 'guba-shirvan'), ('derbend', 'guba-shirvan'), ('daghestan', 'guba-shirvan'), ('дагестан', 'guba-shirvan'),
    ('zakatal', 'guba-shirvan'), ('talish', 'guba-shirvan'), ('талыш', 'guba-shirvan'), ('mughan', 'guba-shirvan'), ('moghan', 'guba-shirvan'), ('муган', 'guba-shirvan'),
    ('khyzy', 'guba-shirvan'), ('хызы', 'guba-shirvan'), ('perepedil', 'guba-shirvan'), ('пиребедил', 'guba-shirvan'), ('seichur', 'guba-shirvan'), ('chichi', 'guba-shirvan'),
    ('чичи', 'guba-shirvan'), ('marasal', 'guba-shirvan'), ('хила', 'guba-shirvan'), ('биджо', 'guba-shirvan'), ('мараза', 'guba-shirvan'), ('сумах', 'guba-shirvan'),
    ('tabriz', 'tabriz'), ('тебриз', 'tabriz'), ('heriz', 'tabriz'), ('serapi', 'tabriz'), ('ardabil', 'tabriz'), ('ardebil', 'tabriz'), ('karaja', 'tabriz'),
    ('sarab', 'tabriz'), ('meshkin', 'tabriz'), ('nakhchivan', 'nakhchivan'), ('нахчыван', 'nakhchivan'), ('нахичеван', 'nakhchivan')]
SCHOOL_RU = {'тебриз': 'tabriz', 'губа-ширван': 'guba-shirvan', 'куба-ширван': 'guba-shirvan', 'карабах': 'karabakh', 'гянджа-газах': 'ganja-gazakh', 'газах': 'ganja-gazakh', 'баку': 'guba-shirvan'}
FLAT = [('kilim', 'безворсовый: килим'), ('килим', 'безворсовый: килим'), ('soumak', 'безворсовый: сумах'), ('sumak', 'безворсовый: сумах'), ('сумах', 'безворсовый: сумах'),
        ('verne', 'безворсовый: верни'), ('verni', 'безворсовый: верни'), ('верни', 'безворсовый: верни'), ('zili', 'безворсовый: зили'), ('зили', 'безворсовый: зили'),
        ('palas', 'палас'), ('палас', 'палас'), ('jajim', 'джеджим'), ('cecim', 'джеджим'), ('джеджим', 'джеджим'), ('shadda', 'безворсовый: шадда'), ('шадда', 'безворсовый: шадда')]
BAGS = ('khorjin', 'mafrash', 'bag', 'чувал', 'хурджун', 'мафраш', 'heybe')

def school_of(*texts):
    t = ' '.join(x or '' for x in texts).lower()
    for k, v in SCHOOL_KEYS:
        if k in t: return v, k
    return '', ''
def tech_of(*texts):
    t = ' '.join(x or '' for x in texts).lower()
    for k, v in FLAT:
        if k in t: return v
    return ''
def frag_of(*texts):
    t = ' '.join(x or '' for x in texts).lower()
    return any(k in t for k in ('fragment', 'фрагмент', 'detail', 'деталь'))

items = []
def add(**kw):
    kw.setdefault('names', []); kw['names'] = [n for n in dict.fromkeys(x.strip() for x in kw['names'] if x and x.strip())]
    items.append(kw)

# ---------------- 1. Керимов (страницы книги) ----------------
def page_photos(img):
    """Фото на странице: крупные области, где почти все пиксели отличаются от бумаги (у текста — лишь штрихи букв)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape
    bg = np.median(np.concatenate([g[:10].ravel(), g[-10:].ravel(), g[:, :10].ravel(), g[:, -10:].ravel()]))
    raw = (np.abs(g.astype(int) - bg) > 28).astype(np.uint8)
    m = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    bx = [list(st[i][:4]) for i in range(1, n) if st[i][2] * st[i][3] > 0.004 * w * h]
    # склейка кусков одного фото (светлые полосы паласа разрывают его на части): друг над другом, почти одна ширина
    gap = 0.03 * max(w, h); changed = True
    while changed:
        changed = False
        for a in range(len(bx)):
            for b in range(a + 1, len(bx)):
                A, B = bx[a], bx[b]
                ox = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0]); oy = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
                vert = ox > 0.9 * max(A[2], B[2]) and oy > -gap      # только по вертикали и при почти одинаковой ширине
                if vert:
                    x0, y0 = min(A[0], B[0]), min(A[1], B[1]); x1, y1 = max(A[0] + A[2], B[0] + B[2]), max(A[1] + A[3], B[1] + B[3])
                    bx[a] = [x0, y0, x1 - x0, y1 - y0]; bx.pop(b); changed = True; break
            if changed: break
    out = []
    for x, y, ww, hh in bx:
        if ww * hh < 0.04 * w * h or min(ww, hh) < 0.12 * min(w, h): continue
        R = raw[y:y + hh, x:x + ww]
        rows = np.where(R.mean(1) > 0.3)[0]; cols = np.where(R.mean(0) > 0.3)[0]     # обрезать приставшие строки текста
        if len(rows) < 0.1 * h or len(cols) < 0.1 * w: continue
        y0, y1, x0, x1 = y + rows[0], y + rows[-1] + 1, x + cols[0], x + cols[-1] + 1
        if raw[y0:y1, x0:x1].mean() < 0.5: continue                                   # текстовый блок
        out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    out.sort(key=lambda b: (round(b[1] / (0.15 * h)), b[0]))
    return out

for vol, roman in (('kerimov_vol1', 'I'), ('kerimov_vol2', 'II'), ('kerimov_vol3', 'III')):
    pdir, jdir = f'{D}/output/page_images/{vol}', f'{D}/output/raw_pages_v3/{vol}'
    if not os.path.isdir(pdir): log('нет папки', pdir); continue
    n0 = len(items)
    for jp in sorted(glob.glob(f'{jdir}/page_*.json')):
        try: J = json.load(open(jp, encoding='utf-8'))
        except Exception: continue
        figs = J.get('figures') or []
        if not figs: continue
        if vol == 'kerimov_vol1' and int(re.sub(r'\D', '', os.path.basename(jp)) or 0) < 318: continue   # т. I до с. 318 — схемы, не фото
        ip = f'{pdir}/{os.path.basename(jp)[:-5]}.jpg'
        if not os.path.exists(ip): continue
        img = cv2.imread(ip)
        if img is None: continue
        boxes = page_photos(img)
        if not boxes: continue
        carriers = {c.get('figure_ref'): c for c in (J.get('carriers_mentioned') or []) if c.get('figure_ref')}
        schools = {m.get('name_ru'): m.get('school') for m in (J.get('motifs_mentioned') or []) if m.get('school')}
        pg = os.path.basename(jp)[5:-5]
        for i, b in enumerate(boxes):
            f = figs[i] if len(figs) == len(boxes) else (figs[0] if len(figs) == 1 else {})
            lab = f.get('figure_label') or ''; cap = f.get('caption') or ''
            c = carriers.get(lab, {}); comp = c.get('motif_name') or f.get('depicts_motif') or ''
            title = (cap if cap and cap != lab else '') or comp
            sc = SCHOOL_RU.get((schools.get(comp) or '').lower(), ''); why = f'Керимов: «{comp}» — {schools.get(comp)}' if sc else ''
            if not sc:
                sc, k = school_of(title, comp)
                why = f'слово «{k}» в подписи Керимова' if sc else ''
            x, y, w, h = b
            add(id=f'{vol}_p{pg}_{i + 1}', src=vol, img_path=ip, crop=[int(x), int(y), int(w), int(h)],
                title=f'Керимов, т. {roman}' + (f', {lab}' if lab else '') + (f'. {title}' if title else '') + f' (стр. скана {int(pg)}, печ. {J.get("page_number_printed") or "?"})',
                names=[comp, title], composition=comp, date=c.get('date') or '', school=sc, school_why=why,
                technique=tech_of(title, comp), fragment=frag_of(title), text=(c.get('quote') or '')[:400], museum='', inv='', place='', url='', license='книга (не публиковать фото)',
                ambiguous=len(figs) != len(boxes) and len(figs) != 1)
    log(vol, 'фото на страницах:', len(items) - n0)

# ---------------- 2. музейные фото (catalog_photos.json) ----------------
def fname(pid): return re.sub(r'[^A-Za-z0-9._-]+', '_', pid)[:80] + '_' + hashlib.md5(pid.encode()).hexdigest()[:6] + '.jpg'
CP = json.load(open(f'{REPO}/site/model/sources/museum_photos/catalog_photos.json', encoding='utf-8'))['items']
n0 = len(items)
for p in CP:
    ip = f'{D}/photos/full/{fname(p["id"])}'
    if not os.path.exists(ip): continue
    sc = p.get('type') or ''; why = f'{p.get("basis") or "подпись источника"} ({p.get("conf") or "?"})' if sc else ''
    if not sc:
        sc, k = school_of(p.get('title'), p.get('place')); why = f'слово «{k}» в подписи' if sc else ''
    add(id=re.sub(r'[^\w]+', '_', p['id']), src=p.get('src') or p['id'].split(':')[0], img_path=ip, title=p.get('title') or '', names=[p.get('title'), p.get('name_seen')],
        composition=p.get('concept') or '', date=p.get('date') or '', school=sc, school_why=why,
        technique={'flatweave': 'безворсовый', 'pile': 'ворсовый'}.get(p.get('kind') or '', '') or tech_of(p.get('title')),
        fragment=p.get('whole') is False, text='', museum={'met': 'Метрополитен-музей', 'aic': 'Институт искусств Чикаго', 'cleveland': 'Кливлендский музей искусств', 'wikimedia': 'Wikimedia Commons'}.get(p.get('src'), p.get('src') or ''),
        inv='', place=p.get('place') or '', url=p.get('url') or '', license=p.get('license') or '', size=p.get('dims') or '')
log('музейные фото (catalog_photos):', len(items) - n0)

# ---------------- 3. V&A / Met (museum_open) ----------------
n0 = len(items); mo = f'{D}/photos/museum_open'
if os.path.exists(f'{mo}/catalog.json'):
    for it in json.load(open(f'{mo}/catalog.json', encoding='utf-8')):
        ip = f'{mo}/img/{it["id"]}.jpg'
        if not os.path.exists(ip): continue
        sc, k = school_of(it.get('title'), it.get('place'), it.get('type'))
        add(id=it['id'], src={'V&A': 'vam', 'Met': 'met'}.get(it.get('museum'), (it.get('museum') or 'museum').lower()), img_path=ip, title=it.get('title') or '', names=[it.get('title')],
            composition='', date=it.get('date') or '', school=sc, school_why=f'слово «{k}» в подписи музея' if sc else '',
            technique=tech_of(it.get('title'), it.get('type')), fragment=frag_of(it.get('title'), it.get('type')), text='', museum=it.get('museum') or '',
            inv=it.get('inv') or '', place=it.get('place') or '', url=it.get('url') or '', license=it.get('license') or '', size='')
log('V&A/Met (museum_open):', len(items) - n0)

# ---------------- 4. azerbaijanrugs ----------------
n0 = len(items); AZ = f'{D}/photos/azerbaijan_rugs_guide'; seen = set()
for dp, dn, fn in sorted(os.walk(AZ)):
    names = set(fn); folder = os.path.relpath(dp, AZ)
    for f in sorted(fn):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in ('.jpg', '.jpeg', '.png', '.webp'): continue
        p = os.path.join(dp, f)
        if os.path.getsize(p) < 12000 or stem in seen: continue
        seen.add(stem)
        txt = ''
        for cnd in (stem + '.txt', stem + '.TXT'):
            if cnd in names: txt = open(os.path.join(dp, cnd), encoding='utf-8', errors='ignore').read().strip(); break
        sc, k = school_of(stem)                                   # сначала имя файла, потом папка, потом описание
        if not sc: sc, k = school_of(folder.split('/')[0])
        if not sc: sc, k = school_of(txt[:300])
        m = re.search(r'(\d{2,3})\s*[x×]\s*(\d{2,3})\s*cm', txt)
        cent = re.search(r'\d{1,2}(?:st|nd|rd|th)[ -]centur(?:y|ies)', txt, re.I)
        add(id='azr_' + re.sub(r'[^\w]+', '_', stem)[:60], src='azerbaijanrugs', img_path=p, title=(txt.split('.')[0][:100] if txt else stem.replace('_', ' ')),
            names=[txt.split('.')[0][:100] if txt else ''], composition='', date=cent.group(0) if cent else '', school=sc,
            school_why=f'слово «{k}» в {"имени файла" if k in stem.lower() else ("папке «" + folder.split("/")[0] + "»" if k in folder.lower() else "описании")}' if sc else '',
            technique=tech_of(stem, folder, txt[:300]), fragment=frag_of(stem, txt[:200]), text=txt[:1500], museum='', inv='', place='', url='', license='azerbaijanrugs.com (только анализ)',
            size=f'{m.group(1)}×{m.group(2)} см' if m else '', folder=folder.split('/')[0])
log('azerbaijanrugs:', len(items) - n0)
for it in items:
    if any(b in (it['title'] + ' ' + it.get('folder', '')).lower() for b in BAGS): it['object'] = 'мешок / сумка'
log('ВСЕГО ковров (фото):', len(items), dict(collections.Counter(it['src'] for it in items)))

# ---------------- картинки ----------------
def load(it, side=1000):
    a = cv2.imread(it['img_path'])
    if a is None: return None
    if it.get('crop'):
        x, y, w, h = it['crop']; a = a[y:y + h, x:x + w]
    s = side / max(a.shape[:2])
    return cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else a
IMGDIR = f'{CACHE}/img'; os.makedirs(IMGDIR, exist_ok=True)
for i, it in enumerate(items):
    p = f'{IMGDIR}/{it["id"]}.jpg'
    if not os.path.exists(p):
        a = load(it)
        if a is None: it['bad'] = True; continue
        cv2.imwrite(p, a, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if i % 300 == 0: log('фото', i, '/', len(items))
items = [it for it in items if not it.get('bad')]

# ---------------- признаки DINOv2 ----------------
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = (lambda x: torch.nn.functional.adaptive_avg_pool2d(x, 6).flatten(1)) if os.environ.get('RC_TEST') else torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(dev).eval()   # RC_TEST — проверка без сети
MEAN = np.array([0.406, 0.456, 0.485], np.float32); STD = np.array([0.225, 0.224, 0.229], np.float32)
FC = f'{CACHE}/dino.npz'; feats = dict(np.load(FC, allow_pickle=True)['d'].item()) if os.path.exists(FC) else {}
todo = [it['id'] for it in items if it['id'] not in feats]
for i in range(0, len(todo), 32):
    ids = todo[i:i + 32]; ims = []
    for k in ids:
        a = cv2.resize(cv2.imread(f'{IMGDIR}/{k}.jpg'), (224, 224), interpolation=cv2.INTER_AREA)
        ims += [a, a[:, ::-1], a[::-1, ::-1]]
    x = torch.from_numpy(np.stack([((m.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for m in ims])).to(dev)
    with torch.no_grad(): f = model(x).float().cpu().numpy()
    f = f.reshape(len(ids), 3, -1).mean(1)
    for k, v in zip(ids, f): feats[k] = v / np.linalg.norm(v)
    if (i // 32) % 20 == 19: np.savez(FC, d=np.array(feats, dtype=object))
np.savez(FC, d=np.array(feats, dtype=object))
ids = [it['id'] for it in items]; X = np.stack([feats[k] for k in ids]); log('признаки готовы')

# ---------------- кандидаты и проверка SIFT ----------------
S = X @ X.T; np.fill_diagonal(S, -1)
K = 12; cand = set()
for i in range(len(ids)):
    for j in np.argsort(-S[i])[:K]:
        if S[i, j] >= 0.72: cand.add((min(i, j), max(i, j)))
log('кандидатов в двойники:', len(cand))
# точки SIFT всех фото — один раз в память (≈10 мин), потом сравнение пар быстрое
sift = cv2.SIFT_create(nfeatures=1500)
need = sorted({i for p in cand for i in p}); KP = {}
for n, i in enumerate(need):
    g = cv2.cvtColor(cv2.imread(f'{IMGDIR}/{ids[i]}.jpg'), cv2.COLOR_BGR2GRAY)
    s_ = 900 / max(g.shape); g = cv2.resize(g, None, fx=s_, fy=s_) if s_ < 1 else g
    k, d = sift.detectAndCompute(g, None)
    if d is not None: d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7)).astype(np.float16)      # RootSIFT
    KP[i] = (np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32), d)
    if n % 500 == 0: log('точки SIFT', n, '/', len(need))
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=40))
def verify(i, j):
    (p1, d1), (p2, d2) = KP[i], KP[j]
    if d1 is None or d2 is None or len(d1) < 30 or len(d2) < 30: return 0
    m = flann.knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.78 * b.distance]
    if len(good) < 15: return 0
    A = p1[[g.queryIdx for g in good]]; B = p2[[g.trainIdx for g in good]]
    H, inl = cv2.findHomography(A, B, cv2.RANSAC, 6.0)
    if inl is None: return 0
    n = int(inl.sum())
    return n if n >= 0.3 * len(good) or n >= 120 else 0     # одна и та же вещь: много совпавших точек и большая их доля
PF = f'{CACHE}/pairs_progress.json'                         # проверенные пары сохраняются — после обрыва продолжаем
prog = json.load(open(PF)) if os.path.exists(PF) else {'checked': [], 'found': []}
checked = {tuple(x) for x in prog['checked']}; found = {(a, b): v for a, b, v in prog['found']}
pos = {k: n for n, k in enumerate(ids)}
todo = [(i, j) for i, j in sorted(cand) if (ids[i], ids[j]) not in checked]
log('пар осталось проверить:', len(todo), '(уже проверено раньше:', len(checked), ')')
for n, (i, j) in enumerate(todo):
    v = verify(i, j); checked.add((ids[i], ids[j]))
    if v >= 40: found[(ids[i], ids[j])] = v
    if n % 1000 == 999 or n == len(todo) - 1:
        json.dump({'checked': [list(x) for x in checked], 'found': [[a, b, v] for (a, b), v in found.items()]}, open(PF, 'w'))
        log('проверено пар', n + 1, '/', len(todo), '| двойников', len(found))
pairs = [(pos[a], pos[b], v, float(S[pos[a], pos[b]])) for (a, b), v in found.items() if a in pos and b in pos]
# группы
par = list(range(len(ids)))
def fd(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
for i, j, v, s in pairs: par[fd(i)] = fd(j)
G = collections.defaultdict(list)
for i in range(len(ids)): G[fd(i)].append(i)
groups = [g for g in G.values() if len(g) > 1]
PRI = {'kerimov_vol1': 0, 'kerimov_vol2': 0, 'kerimov_vol3': 0, 'met': 1, 'vam': 1, 'aic': 1, 'cleveland': 1, 'wikimedia': 2, 'azerbaijanrugs': 3}
out_groups = []
for gi, g in enumerate(sorted(groups, key=len, reverse=True)):
    g.sort(key=lambda i: (PRI.get(items[i]['src'], 2), items[i]['id']))
    gid = f'g{gi + 1:04d}'
    for i in g: items[i]['same_group'] = gid
    votes = collections.Counter(items[i]['school'] for i in g if items[i]['school'])
    out_groups.append({'id': gid, 'members': [ids[i] for i in g], 'pairs': [[ids[a], ids[b], v, round(s, 3)] for a, b, v, s in pairs if a in g and b in g],
                       'school_votes': dict(votes), 'names': list(dict.fromkeys(n for i in g for n in items[i]['names']))[:10]})
    # лист для проверки
    tiles = []
    for i in g[:8]:
        a = cv2.imread(f'{IMGDIR}/{ids[i]}.jpg'); s = 260 / a.shape[0]; a = cv2.resize(a, (max(60, int(a.shape[1] * s)), 260))
        a = cv2.copyMakeBorder(a, 0, 44, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(a, items[i]['src'][:18], (6, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(a, re.sub(r'[^\x20-\x7e]', '?', items[i]['id'])[-26:], (6, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
        tiles.append(a)
    cv2.imwrite(f'{OUT}/dup_sheets/{gid}.jpg', np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
log('групп двойников:', len(out_groups), '| фото в них:', sum(len(g['members']) for g in out_groups))

# ---------------- сохранение и пакет для студии ----------------
for it in items: it.pop('img_path', None); it.pop('crop', None); it['img'] = f'img/{it["id"]}.jpg'
cat = {'built': time.strftime('%Y-%m-%d %H:%M'), 'items': items, 'groups': out_groups}
json.dump(cat, open(f'{OUT}/catalog.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
json.dump(out_groups, open(f'{OUT}/groups.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
BZ = f'{CACHE}/rugs_bundle.zip' if os.environ.get('RC_TEST') else '/content/rugs_bundle.zip'   # пакет собирается локально и загружается на Диск через API
with zipfile.ZipFile(BZ, 'w', zipfile.ZIP_STORED) as z:
    z.writestr('catalog.json', json.dumps(cat, ensure_ascii=False))
    for it in items: z.write(f'{IMGDIR}/{it["id"]}.jpg', it['img'])
log('пакет:', round(os.path.getsize(BZ) / 1e6), 'МБ')
# загрузка на Диск с доступом «по ссылке» (тот же id при повторных запусках — студии не нужно ничего менять)
if os.environ.get('RC_TEST'): raise SystemExit('RC_TEST: без загрузки на Диск')
auth.authenticate_user()
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
svc = build('drive', 'v3')
idf = f'{OUT}/bundle_id.txt'; fid = open(idf).read().strip() if os.path.exists(idf) else ''
media = MediaFileUpload(BZ, mimetype='application/zip', resumable=True, chunksize=50 * 1024 * 1024)
if fid:
    try: svc.files().update(fileId=fid, media_body=media).execute()
    except Exception as e: print('не удалось обновить старый файл, создаю новый:', e); fid = ''
if not fid:
    folder = svc.files().list(q="name='rug_catalog' and mimeType='application/vnd.google-apps.folder' and trashed=false", fields='files(id)').execute()['files']
    meta = {'name': 'rugs_bundle.zip', **({'parents': [folder[0]['id']]} if folder else {})}
    fid = svc.files().create(body=meta, media_body=media, fields='id').execute()['id']
    svc.permissions().create(fileId=fid, body={'type': 'anyone', 'role': 'reader'}).execute()
    open(idf, 'w').write(fid)
print('\nИТОГ')
print('ковров (фото):', len(items), dict(collections.Counter(it['src'] for it in items)))
print('со школой:', sum(1 for it in items if it['school']), '| с названием типа ковра:', sum(1 for it in items if it.get('composition')))
print('групп «один и тот же ковёр»:', len(out_groups), '(листы для проверки: carpet-dna/rug_catalog/dup_sheets/)')
print('id пакета для студии:', fid)
print('Готово. Напишите Claude: «каталог готов».')
