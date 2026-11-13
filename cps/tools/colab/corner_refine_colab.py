# corner_refine_colab.py — обучить уточнение углов ковра по правкам людей в /m (v11.57).
#
# Что делает: для каждого угла берёт окно фото вокруг машинного угла и учит сеть говорить, куда угол сдвинуть.
# Цель — углы, подтверждённые людьми (cps/tools/corners/labels.json, собирает make_labels.py из резервной копии студии).
# Окна режутся с разным размером и случайным сдвигом от верного угла — так из ~470 ковров получаются десятки тысяч примеров.
# Все четыре угла приводятся к виду «левый верхний» (зеркалом), поэтому сеть одна.
# Сеть — MobileNetV3-small (предобучена на ImageNet), выход — сдвиг (dx, dy) в долях окна.
# Проверка — 20 % ковров, которых сеть не видела (последние по времени правки): ошибка до и после, как в отчёте Claude.
# Итог: carpet-dna/models/corner_refine_v1.pt (TorchScript, ~6 МБ) и corner_refine_v1.json (замеры).
# Время: GPU (T4) — 10–20 мин; чтение пакетов с Диска — ещё 2–5 мин. Ключи не нужны.
import os, io, json, math, random, time, zipfile, glob
import numpy as np, cv2, torch, torch.nn as nn, torchvision
from google.colab import drive
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
REPO = '/content/carpevo'
LAB = json.load(open(f'{REPO}/cps/tools/corners/labels.json', encoding='utf-8'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV, '| ковров с метками', len(LAB))
OUT = '/content/drive/MyDrive/carpet-dna/models'; os.makedirs(OUT, exist_ok=True)

# ---------- фото из пакетов студии ----------
zips = sorted(set(glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/*bundle*.zip') + glob.glob('/content/drive/MyDrive/carpet-dna/**/books_bundle.zip', recursive=True)))
log('пакеты:', [os.path.basename(z) for z in zips])
Z = [zipfile.ZipFile(z) for z in zips]; names = [set(z.namelist()) for z in Z]
LONG = 1024
IM = {}
for x in LAB:
    for z, nm in zip(Z, names):
        if x['img'] in nm:
            a = cv2.imdecode(np.frombuffer(z.read(x['img']), np.uint8), cv2.IMREAD_COLOR)
            if a is None: break
            k = LONG / max(a.shape[:2])
            if k < 1: a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            IM[x['id']] = a; break
LAB = [x for x in LAB if x['id'] in IM]
log('фото найдено', len(IM), '— ковров для обучения', len(LAB))
assert len(LAB) > 100, 'мало фото — проверь, что rugs_bundle.zip и books_bundle.zip лежат в carpet-dna/rug_catalog'

# ---------- окна ----------
W = 128                                   # окно в пикселях на входе сети
FLIP = [(1, 1), (-1, 1), (-1, -1), (1, -1)]   # ЛВ, ПВ, ПН, ЛН → всё к «левому верхнему»
MAG = (255, 0, 255)                       # за краем фото — пурпур (как в vlm_corners)


def window(a, cx, cy, half, k):
    """окно 2*half (в долях длинной стороны) вокруг (cx, cy) (доли кадра) → W×W, зеркалом к виду угла k"""
    h, w = a.shape[:2]; L = max(h, w); hp = half * L
    sx, sy = FLIP[k]
    # аффинное: выходной пиксель (u,v) → точка фото x = cx*w + sx*(u/W*2-1)*hp
    M = np.array([[sx * 2 * hp / W, 0, cx * w - sx * hp], [0, sy * 2 * hp / W, cy * h - sy * hp]], np.float32)
    return cv2.warpAffine(a, M, (W, W), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=MAG)


def offset(a, cx, cy, tx, ty, half, k):
    """сдвиг от (cx,cy) к (tx,ty) в долях полуокна, в зеркальной системе угла k"""
    h, w = a.shape[:2]; L = max(h, w); sx, sy = FLIP[k]
    return np.array([sx * (tx - cx) * w / (half * L), sy * (ty - cy) * h / (half * L)], np.float32)


ids = sorted(LAB, key=lambda x: x['t'])
cut = int(len(ids) * 0.8); TR, TE = ids[:cut], ids[cut:]
log('обучение', len(TR), '| проверка', len(TE), '(последние по времени)')
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def prep(win): return torch.from_numpy(((win[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


class DS(torch.utils.data.Dataset):
    def __init__(s, items, n): s.it, s.n = items, n
    def __len__(s): return s.n
    def __getitem__(s, i):
        x = random.choice(s.it); a = IM[x['id']]; k = random.randrange(4)
        tx, ty = x['quad'][k]; h, w = a.shape[:2]; L = max(h, w)
        half = math.exp(random.uniform(math.log(0.025), math.log(0.14)))
        r = half * random.uniform(0, 0.75); t = random.uniform(0, 2 * math.pi)       # старт — где-то рядом с верным углом
        cx = tx + r * math.cos(t) * L / w; cy = ty + r * math.sin(t) * L / h
        win = window(a, cx, cy, half, k)
        if random.random() < 0.8:                                                  # свет и цвет фото разные
            win = cv2.convertScaleAbs(win, alpha=random.uniform(0.7, 1.3), beta=random.uniform(-25, 25))
        if random.random() < 0.3: win = cv2.GaussianBlur(win, (0, 0), random.uniform(0.5, 1.5))
        return prep(win), torch.from_numpy(offset(a, cx, cy, tx, ty, half, k))


net = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
net.classifier = nn.Sequential(nn.Linear(576, 128), nn.Hardswish(), nn.Linear(128, 2))
net = net.to(DEV)
EP, PER = 14, 6000
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * (PER // 64 + 1))
dl = torch.utils.data.DataLoader(DS(TR, PER), batch_size=64, num_workers=2, drop_last=False)
for ep in range(EP):
    net.train(); tl = 0; n = 0
    for xb, yb in dl:
        xb, yb = xb.to(DEV), yb.to(DEV)
        loss = nn.functional.smooth_l1_loss(net(xb), yb, beta=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(xb); n += len(xb)
    log(f'эпоха {ep + 1}/{EP}: потеря {tl / n:.4f}')

# ---------- уточнение и проверка ----------
net.eval()
HALVES = (0.12, 0.06, 0.03)


@torch.no_grad()
def refine(a, quad):
    h, w = a.shape[:2]; L = max(h, w); q = [list(p) for p in quad]
    for half in HALVES:
        b = torch.stack([prep(window(a, q[k][0], q[k][1], half, k)) for k in range(4)]).to(DEV)
        o = net(b).cpu().numpy()
        for k in range(4):
            sx, sy = FLIP[k]; d = np.clip(o[k], -1, 1)
            q[k][0] += sx * d[0] * half * L / w; q[k][1] += sy * d[1] * half * L / h
    return q


def err(q, t, a):
    h, w = a.shape[:2]; L = max(h, w)                                              # ошибка в долях длинной стороны — как в отчёте
    return float(np.mean([math.hypot((q[k][0] - t[k][0]) * w / L, (q[k][1] - t[k][1]) * h / L) for k in range(4)]))


rep = {}
for start in ('auto', 'vlm'):
    E0, E1 = [], []
    for x in TE:
        if not x.get(start): continue
        a = IM[x['id']]; E0.append(err(x[start], x['quad'], a)); E1.append(err(refine(a, x[start]), x['quad'], a))
    if not E0: continue
    E0, E1 = np.array(E0), np.array(E1)
    rep[start] = {'n': len(E0), 'before_median': round(float(np.median(E0)), 4), 'after_median': round(float(np.median(E1)), 4),
                  'before_p90': round(float(np.percentile(E0, 90)), 4), 'after_p90': round(float(np.percentile(E1, 90)), 4),
                  'better': int((E1 < E0 - 0.002).sum()), 'worse': int((E1 > E0 + 0.002).sum()),
                  'within_1pct_before': round(float((E0 <= 0.01).mean()), 3), 'within_1pct_after': round(float((E1 <= 0.01).mean()), 3)}
    log(start, rep[start])

# ---------- выгрузка ----------
net.cpu().eval()
ts = torch.jit.trace(net, torch.zeros(4, 3, W, W))
p = f'{OUT}/corner_refine_v1.pt'; ts.save(p)
meta = {'ver': 1, 'W': W, 'halves': HALVES, 'mag': MAG, 'long': LONG, 'trained': time.strftime('%Y-%m-%d %H:%M'),
        'n_train': len(TR), 'n_test': len(TE), 'test': rep}
json.dump(meta, open(f'{OUT}/corner_refine_v1.json', 'w'), ensure_ascii=False, indent=1)
print('\nИТОГ:', json.dumps(meta, ensure_ascii=False))
print('файлы:', p, f'({os.path.getsize(p) / 1e6:.1f} МБ) и corner_refine_v1.json — Claude заберёт их с Диска')
