# carpet_model_colab.py — «своя модель для ковров», версия 1 (Colab, T4 GPU, ~1–2 часа).
#
# Идея: берём готовую DINOv2 (ViT-S/14) и доучиваем её последние блоки на коврах двумя задачами сразу:
#   A) «узор ↔ ковёр»: из принятых схем CPS рисуем «искусственный ковёр» (случайные ковровые цвета, клетки-узлы,
#      фон из настоящего фото, поворот, перспектива, размытие, JPEG) и учим модель, что он и чистая схема — одно и то же;
#   B) самообучение на настоящих фото ковров без подписей (два разных кадра одного фото → близко, разные фото → далеко).
# Проверка (честно, на том, что модель не видела):
#   1) узнавание узора: 15% схем отложены; «искусственный ковёр» → ищем среди отложенных схем (top-1 / top-5), до и после доучивания;
#   2) школы: линейный классификатор поверх признаков на фото azerbaijanrugs (групповая проверка), до и после — стала ли модель лучше «видеть ковры»;
#   3) настоящие фото музеев: для каждого фото — 5 самых похожих узоров (контактные листы для проверки глазами; правильных ответов у нас пока нет).
# Результат: Drive carpet-dna/carpet_model/ — report.json, motif_suggestions.csv, sheets/*.jpg, carpet_vits14_v1.pt (веса доученных блоков + головы).
# Параметры: %env CM_STEPS=3000 (шагов обучения), %env CM_TEST=1 (быстрая проверка без GPU на крошечной модели).
import os, sys, json, glob, csv, time, random, subprocess, collections, re
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

TEST = os.environ.get('CM_TEST') == '1'
STEPS = max(40, int(os.environ.get('CM_STEPS', '40' if TEST else '3000')))
BS = 8 if TEST else 48
IMG = 112 if TEST else 224
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if TEST:
    DRIVE = os.environ.get('CM_ROOT', '/tmp/cm'); DATA = os.environ['CPS_DATA']
else:
    from google.colab import drive
    drive.mount('/content/drive'); DRIVE = '/content/drive/MyDrive/carpet-dna'
    DATA = '/content/cpsdata/data'
    if not os.path.isdir(DATA + '/work'):
        print('скачиваю схемы CPS (ветка cps-data)…', flush=True)
        subprocess.run(['git', 'clone', '--depth', '1', '-b', 'cps-data', 'https://github.com/22adolgikh-pixel/carpevo.git', '/content/cpsdata'], check=True)
OUT = f'{DRIVE}/carpet_model'; os.makedirs(f'{OUT}/sheets', exist_ok=True)
print('устройство:', dev, '| шагов:', STEPS, flush=True)
if dev == 'cpu' and not TEST: print('ВНИМАНИЕ: нет GPU (Среда выполнения → Сменить тип → T4 GPU)', flush=True)

# ---------------- схемы CPS ----------------
motifs = []
for f in sorted(glob.glob(f'{DATA}/work/*.json')):
    try: w = json.load(open(f))
    except Exception: continue
    m = w.get('matrix') or {}
    if not (w.get('done') and m.get('rows')) or w.get('kind') in ('photo', 'junk', 'drawing'): continue
    rows = m['rows']; h, wd = len(rows), max(len(r) for r in rows)
    if h < 5 or wd < 5: continue
    A = np.zeros((h, wd), np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            A[y, x] = 0 if ch == '.' else (int(ch) if ch.isdigit() else 1)
    if (A > 0).mean() < 0.03: continue
    meta = w.get('meta') or {}
    motifs.append(dict(id=os.path.basename(f)[:-5], A=A, name=meta.get('name') or '', site=meta.get('site_id') or ''))
random.shuffle(motifs)
nval = max(2, int(len(motifs) * 0.15))
val_m, tr_m = motifs[:nval], motifs[nval:]
print('схем:', len(motifs), '| для обучения:', len(tr_m), '| отложено:', len(val_m), flush=True)

# ---------------- фото ковров (без подписей) ----------------
def photos_in(d):
    return [p for p in glob.glob(f'{d}/**/*', recursive=True) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and os.path.getsize(p) > 12000]
AZ = f'{DRIVE}/photos/azerbaijan_rugs_guide'
real = photos_in(AZ) + photos_in(f'{DRIVE}/photos/full') + photos_in(f'{DRIVE}/photos/museum_open/img')
museum = photos_in(f'{DRIVE}/photos/museum_open/img') + [p for p in photos_in(f'{DRIVE}/photos/full') if '/met_' in p or 'met' in os.path.basename(p)[:4]]
print('настоящих фото:', len(real), '| музейных для показа:', len(museum), flush=True)
_cache = {}
def load(p, side=448):
    if p in _cache: return _cache[p]
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: im = np.full((side, side, 3), 128, np.uint8)
    s = side / max(im.shape[:2]); im = cv2.resize(im, None, fx=min(1, s), fy=min(1, s), interpolation=cv2.INTER_AREA)
    if len(_cache) < 6000: _cache[p] = im
    return im

# ---------------- рисование ----------------
CARPET = [(40, 40, 150), (30, 30, 110), (120, 50, 25), (70, 35, 20), (190, 220, 235), (60, 110, 40), (40, 150, 200), (30, 30, 30), (210, 200, 180), (80, 60, 130)]   # BGR: красный, бордо, синий, т.-синий, слоновая кость, зелёный, охра, чёрный, беж, лиловый
def jitter(c, s=18): return tuple(int(np.clip(v + random.gauss(0, s), 0, 255)) for v in c)

def clean(A):
    """Чистая схема: фон белый, обводка чёрная, тело серое, прочие цвета — оттенки серого; вписана в квадрат."""
    lut = np.array([255, 20, 150, 90, 200, 60, 120, 180, 40, 220], np.uint8)
    g = lut[np.minimum(A, 9)]
    h, w = g.shape; s = max(h, w); pad = np.full((s + 4, s + 4), 255, np.uint8)
    pad[2 + (s - h) // 2: 2 + (s - h) // 2 + h, 2 + (s - w) // 2: 2 + (s - w) // 2 + w] = g
    return cv2.cvtColor(cv2.resize(pad, (IMG, IMG), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

def synth(A):
    """«Искусственный ковёр»: узор в ковровых цветах, клетки-узлы, на фоне куска настоящего ковра, с искажениями."""
    cols = random.sample(CARPET, 4); bg = jitter(cols[0]); lut = [bg] + [jitter(c) for c in cols[1:]] + [jitter(random.choice(CARPET)) for _ in range(7)]
    k = random.randint(4, 9); ky = max(3, int(k * random.uniform(0.75, 1.3)))
    h, w = A.shape
    im = np.zeros((h * ky, w * k, 3), np.uint8)
    for v in np.unique(A):
        m = cv2.resize((A == v).astype(np.uint8), (w * k, h * ky), interpolation=cv2.INTER_NEAREST).astype(bool)
        im[m] = lut[min(int(v), 10)]
    noise = np.random.normal(0, random.uniform(4, 14), im.shape)                                         # неровность шерсти
    knot = np.tile(np.linspace(-8, 8, k)[None, :, None], (h * ky, w, 3))                                  # полоски узлов
    im = np.clip(im + noise + knot * random.uniform(0, 1), 0, 255).astype(np.uint8)
    # фон: кусок настоящего ковра (или однотон)
    S = int(max(im.shape[:2]) * random.uniform(1.1, 1.8))
    if real and random.random() < 0.7:
        ph = load(random.choice(real)); y0 = random.randint(0, max(0, ph.shape[0] - 64)); x0 = random.randint(0, max(0, ph.shape[1] - 64))
        canvas = cv2.resize(ph[y0:y0 + 160, x0:x0 + 160], (S, S))
    else: canvas = np.full((S, S, 3), bg, np.uint8)
    oy, ox = random.randint(0, S - im.shape[0]), random.randint(0, S - im.shape[1])
    msk = (A > 0) if random.random() < 0.5 else np.ones_like(A, bool)                                   # иногда узор «вырезан» по силуэту, иногда с фоном
    msk = cv2.resize(msk.astype(np.uint8), (w * k, h * ky), interpolation=cv2.INTER_NEAREST).astype(bool)
    reg = canvas[oy:oy + im.shape[0], ox:ox + im.shape[1]]; reg[msk] = im[msk]
    # геометрия и съёмка
    M = cv2.getRotationMatrix2D((S / 2, S / 2), random.uniform(-12, 12) + random.choice([0, 0, 0, 90]), random.uniform(0.85, 1.1))
    canvas = cv2.warpAffine(canvas, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    d = S * 0.08; src = np.float32([[0, 0], [S, 0], [S, S], [0, S]]); dst = src + np.float32(np.random.uniform(-d, d, (4, 2)))
    canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst), (S, S), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.7: canvas = cv2.GaussianBlur(canvas, (0, 0), random.uniform(0.3, 1.6))
    canvas = cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 2] *= random.uniform(0.7, 1.2); hsv[..., 1] *= random.uniform(0.7, 1.2)
    canvas = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    _, enc = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 90)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def view(p):
    """Случайный кадр настоящего фото (для самообучения)."""
    im = load(p); h, w = im.shape[:2]; s = int(min(h, w) * random.uniform(0.35, 1.0))
    y, x = random.randint(0, h - s), random.randint(0, w - s); c = cv2.resize(im[y:y + s, x:x + s], (IMG, IMG), interpolation=cv2.INTER_AREA)
    if random.random() < 0.5: c = c[:, ::-1]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 0] = (hsv[..., 0] + random.uniform(-6, 6)) % 180; hsv[..., 1:] *= random.uniform(0.75, 1.2)
    c = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.4: c = cv2.GaussianBlur(c, (0, 0), random.uniform(0.3, 1.5))
    return c

MEAN = np.array([0.406, 0.456, 0.485], np.float32); STD = np.array([0.225, 0.224, 0.229], np.float32)   # BGR
def tens(ims): return torch.from_numpy(np.stack([((i.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for i in ims])).to(dev)

# ---------------- модель ----------------
class Tiny(nn.Module):           # только для CM_TEST
    def __init__(s): super().__init__(); s.c = nn.Sequential(nn.Conv2d(3, 16, 5, 4), nn.ReLU(), nn.Conv2d(16, 32, 3, 2), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 384)); s.blocks = [s.c]
    def forward(s, x): return s.c(x)
back = Tiny().to(dev) if TEST else torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(dev)
for p in back.parameters(): p.requires_grad = False
TUNE = 4
for b in list(back.blocks)[-TUNE:]:
    for p in b.parameters(): p.requires_grad = True
if not TEST:
    for p in back.norm.parameters(): p.requires_grad = True
head = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Linear(512, 256)).to(dev)
trained = False

@torch.no_grad()
def embed(ims, bs=64, proj=False):
    back.eval(); head.eval(); out = []
    for i in range(0, len(ims), bs):
        with torch.autocast(dev, enabled=dev == 'cuda'):
            f = back(tens(ims[i:i + bs])).float()
            if proj and trained: f = head(f).float()      # после обучения узоры сравниваем в пространстве головы (там учили «узор ↔ ковёр»)
        out.append(F.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 384), np.float32)

def nce(a, b, t=0.07):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1); l = a @ b.T / t; y = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(l, y) + F.cross_entropy(l.T, y)) / 2

# ---------------- проверки ----------------
VAL_CLEAN = [clean(m['A']) for m in val_m]
rs = random.getstate(); random.seed(123); VAL_SYN = [synth(m['A']) for m in val_m for _ in range(2)]; random.setstate(rs)
def eval_motif():
    g = embed(VAL_CLEAN, proj=True); q = embed(VAL_SYN, proj=True); S = q @ g.T; truth = np.repeat(np.arange(len(val_m)), 2)
    rank = (S > S[np.arange(len(q)), truth][:, None]).sum(1)
    return dict(top1=float((rank == 0).mean()), top5=float((rank < 5).mean()), gallery=len(val_m), random_top1=1 / len(val_m))

def school_labels():
    p = f'{DRIVE}/photos/azrugs_school/catalog.csv'
    if not os.path.exists(p): return []
    R = [r for r in csv.DictReader(open(p, encoding='utf-8')) if r['label'] and r['conflict'] == '0' and os.path.exists(r['path'])]
    by = collections.defaultdict(list)
    for r in R: by[r['label']].append(r)
    out = []
    for k, v in by.items():
        if len(v) >= 25: random.Random(0).shuffle(v); out += v[:400 if not TEST else 20]
    return out
SCH = school_labels()
def eval_school():
    if len(SCH) < 40: return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X = embed([cv2.resize(load(r['path']), (IMG, IMG), interpolation=cv2.INTER_AREA) for r in SCH]); y = np.array([r['label'] for r in SCH])
    cv = cross_val_predict(LogisticRegression(max_iter=3000, C=10, class_weight='balanced'), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    rec = {k: float((cv[y == k] == k).mean()) for k in sorted(set(y))}
    return dict(accuracy=float((cv == y).mean()), balanced=float(np.mean(list(rec.values()))), recall=rec, n=len(y))

ex = [synth(m['A']) for m in tr_m[:6]] + [clean(m['A']) for m in tr_m[:6]]
cv2.imwrite(f'{OUT}/synth_examples.jpg', np.vstack([np.hstack(ex[:6]), np.hstack(ex[6:])]))   # как выглядят «искусственные ковры» (верх) и схемы (низ)
rep = {'motifs_train': len(tr_m), 'motifs_val': len(val_m), 'real_photos': len(real), 'steps': STEPS}
t0 = time.time()
rep['before'] = {'motif': eval_motif(), 'school': eval_school()}
print('ДО доучивания:', json.dumps(rep['before'], ensure_ascii=False), '| %.0f c' % (time.time() - t0), flush=True)

# ---------------- обучение ----------------
params = [p for p in back.parameters() if p.requires_grad] + list(head.parameters())
opt = torch.optim.AdamW([{'params': [p for p in back.parameters() if p.requires_grad], 'lr': 2e-5}, {'params': head.parameters(), 'lr': 3e-4}], weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 3e-4], total_steps=STEPS, pct_start=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(8)
def make_batch():
    ms = random.sample(tr_m, min(BS, len(tr_m)))
    a = list(pool.map(lambda m: synth(m['A']), ms)); b = [clean(m['A']) for m in ms]
    ps = random.sample(real, min(BS, len(real))) if real else []
    c = list(pool.map(view, ps)); d = list(pool.map(view, ps))
    return a, b, c, d
nxt = pool.submit(make_batch)
for step in range(1, STEPS + 1):
    a, b, c, d = nxt.result(); nxt = pool.submit(make_batch)
    back.train(); head.train()
    with torch.autocast(dev, enabled=dev == 'cuda'):
        z = head(back(tens(a + b + c + d)).float())
        n1, n2 = len(a), len(c)
        loss_m = nce(z[:n1], z[n1:2 * n1])
        loss_r = nce(z[2 * n1:2 * n1 + n2], z[2 * n1 + n2:]) if n2 > 1 else torch.zeros((), device=dev)
        loss = loss_m + 0.5 * loss_r
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step()
    if step % max(1, STEPS // 20) == 0:
        print(f'шаг {step}/{STEPS} | узор↔ковёр {loss_m.item():.3f} | фото {loss_r.item():.3f} | {time.time() - t0:.0f} c', flush=True)
    if step % max(1, STEPS // 4) == 0 and step < STEPS:
        print('  промежуточно, узор:', eval_motif(), flush=True)

trained = True
rep['after'] = {'motif': eval_motif(), 'school': eval_school()}
print('ПОСЛЕ доучивания:', json.dumps(rep['after'], ensure_ascii=False), flush=True)
torch.save({'tuned_blocks': [b.state_dict() for b in list(back.blocks)[-TUNE:]], 'norm': None if TEST else back.norm.state_dict(),
            'head': head.state_dict(), 'img': IMG, 'base': 'dinov2_vits14', 'tune': TUNE}, f'{OUT}/carpet_vits14_v1.pt')

# ---------------- музейные фото → похожие узоры ----------------
gal_ims = [clean(m['A']) for m in motifs]; G = embed(gal_ims, proj=True)
def crops(im):
    h, w = im.shape[:2]; out = [cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)]
    for s in (0.5, 0.33):
        sz = int(min(h, w) * s); st = max(1, sz // 2)
        for y in range(0, h - sz + 1, st):
            for x in range(0, w - sz + 1, st): out.append(cv2.resize(im[y:y + sz, x:x + sz], (IMG, IMG), interpolation=cv2.INTER_AREA))
    return out
show = museum[:200] if museum else real[:100]
rows_out = []
for i, p in enumerate(show):
    E = embed(crops(load(p, 640)), proj=True); s = (E @ G.T).max(0); top = np.argsort(-s)[:5]
    rows_out.append([os.path.relpath(p, DRIVE)] + [f'{motifs[j]["id"]}|{motifs[j]["name"]}|{s[j]:.3f}' for j in top])
    if i < 60:
        ph = load(p, 640); ph = cv2.resize(ph, (int(ph.shape[1] * 300 / ph.shape[0]), 300))
        tiles = [cv2.copyMakeBorder(cv2.resize(gal_ims[j], (150, 150)), 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for j in top]
        for t, j in zip(tiles, top): cv2.putText(t, f'{motifs[j]["id"][-14:]} {s[j]:.2f}', (2, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:] + [np.full((180, 150, 3), 255, np.uint8)])])
        hh = max(ph.shape[0], grid.shape[0])
        sheet = np.hstack([cv2.copyMakeBorder(ph, 0, hh - ph.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
                           cv2.copyMakeBorder(grid, 0, hh - grid.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))])
        cv2.imwrite(f'{OUT}/sheets/{i:03d}.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
csv.writer(open(f'{OUT}/motif_suggestions.csv', 'w', newline='', encoding='utf-8')).writerows([['photo', 'top1', 'top2', 'top3', 'top4', 'top5']] + rows_out)
json.dump(rep, open(f'{OUT}/report.json', 'w'), ensure_ascii=False, indent=1)
b, a = rep['before']['motif'], rep['after']['motif']
print(f"ИТОГ: узнавание узора (отложенные {a['gallery']} схем): top-1 {b['top1']:.2f} → {a['top1']:.2f}, top-5 {b['top5']:.2f} → {a['top5']:.2f} (случайно: {a['random_top1']:.3f})")
if rep['before']['school']: print(f"школы (сбалансированная точность): {rep['before']['school']['balanced']:.3f} → {rep['after']['school']['balanced']:.3f}")
print(f'контактные листы: {OUT}/sheets/ ({min(60, len(show))} шт.) | {time.time() - t0:.0f} c')
print('Готово. Напишите Claude: «модель ковров готова».')
