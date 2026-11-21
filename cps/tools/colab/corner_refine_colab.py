# corner_refine_colab.py — обучить уточнение углов ковра по правкам людей в /m (v11.57).
#
# Что делает: для каждого угла берёт окно фото вокруг машинного угла и учит сеть говорить, куда угол сдвинуть.
# Цель — углы, подтверждённые людьми (cps/tools/corners/labels.json, собирает make_labels.py из резервной копии студии).
# Окна режутся с разным размером и случайным сдвигом от верного угла — так из ~470 ковров получаются десятки тысяч примеров.
# Все четыре угла приводятся к виду «левый верхний» (зеркалом), поэтому сеть одна.
# Сеть — MobileNetV3-small (предобучена на ImageNet), выход — сдвиг (dx, dy) в долях окна.
# Проверка — 20 % ковров, которых сеть не видела (последние по времени правки): ошибка до и после, как в отчёте Claude.
# Итог: carpet-dna/models/corner_refine_v{VER}.pt (TorchScript, ~6 МБ) и corner_refine_v{VER}.json (замеры).
# v5 (21.11): разбор 48 худших ковров показал — в 37 из них начальные углы были РАМКОЙ ВСЕГО ФОТО (find_quad не нашёл ковёр), а верные углы дальше окна сети (до 53 %), бахрома — лишь 7.
#   Теперь первая ступень с окном 50 %, обучение включает старт из углов кадра, в отчёте отдельная строка 'frame' (старт — рамка фото).
# v4 (19.11): 1 201 метка, трудные примеры втрое чаще, окно до 24 % (видно бахрому целиком), MobileNetV3-large, строгие замеры (худший угол, «бахрома наружу»).
# Время: GPU (T4) — 25–40 мин; чтение пакетов с Диска — ещё 2–5 мин. Ключи не нужны.
import os, io, json, math, random, time, zipfile, glob
import numpy as np, cv2, torch, torch.nn as nn, torchvision
from google.colab import drive
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
REPO = '/content/carpevo'
# метки: из клона репозитория, а если его нет — файл labels.json, загруженный в Colab (панель «Файлы» слева → загрузить в /content)
_lp = next((p for p in (f'{REPO}/cps/tools/corners/labels.json', '/content/labels.json') if os.path.exists(p)), None)
assert _lp, 'нет labels.json: загрузите его в /content (значок папки слева → загрузить) или клонируйте репозиторий в /content/carpevo'
LAB = json.load(open(_lp, encoding='utf-8'))
VER = 5                                    # v1 — 13.11 (365); v2 — 15.11 (488); v3 — 16.11 (≈830 меток); v4 — 19.11 (1 201 метка, бахрома). Старые веса не перезаписываются
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
        x = random.choice(s.it); a = IM[x['id']]; k = random.randrange(4)   # (в s.it трудные ковры — где человек двигал углы — повторены 3 раза)
        tx, ty = x['quad'][k]; h, w = a.shape[:2]; L = max(h, w)
        if random.random() < 0.25:                                                 # v5: старт — угол рамки фото (find_quad ковёр не нашёл), окно крупное
            half = random.uniform(0.35, 0.55); cx, cy = (0.0 if FLIP[k][0] > 0 else 1.0), (0.0 if FLIP[k][1] > 0 else 1.0)
        else:
            half = math.exp(random.uniform(math.log(0.02), math.log(0.55)))        # v4: окно 24 %; v5: до 55 % — найти ковёр, если старт далеко
            r = half * random.uniform(0, 0.9); t = random.uniform(0, 2 * math.pi)  # старт — где-то рядом с верным углом
            cx = tx + r * math.cos(t) * L / w; cy = ty + r * math.sin(t) * L / h
        win = window(a, cx, cy, half, k)
        if random.random() < 0.8:                                                  # свет и цвет фото разные
            win = cv2.convertScaleAbs(win, alpha=random.uniform(0.7, 1.3), beta=random.uniform(-25, 25))
        if random.random() < 0.3: win = cv2.GaussianBlur(win, (0, 0), random.uniform(0.5, 1.5))
        return prep(win), torch.from_numpy(np.clip(offset(a, cx, cy, tx, ty, half, k), -1, 1))


net = torchvision.models.mobilenet_v3_large(weights='DEFAULT')                 # v4: сеть крупнее (≈ 4 млн весов; на сервере ≈ 0,3 с на ковёр)
net.classifier = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Linear(256, 2))
net = net.to(DEV)
EP, PER = 22, 10000
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * (PER // 64 + 1))
TRW = TR + [x for x in TR if x.get('fix')] * 2                                   # v4: трудные (правленые людьми) — втрое чаще
log('трудных (человек двигал углы):', sum(1 for x in TR if x.get('fix')), 'из', len(TR))
dl = torch.utils.data.DataLoader(DS(TRW, PER), batch_size=64, num_workers=2, drop_last=False)
for ep in range(EP):
    net.train(); tl = 0; n = 0
    for xb, yb in dl:
        xb, yb = xb.to(DEV), yb.to(DEV)
        loss = nn.functional.smooth_l1_loss(net(xb), yb, beta=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(xb); n += len(xb)
    log(f'эпоха {ep + 1}/{EP}: потеря {tl / n:.4f}')

# ---------- уточнение и проверка ----------
net.eval()
HALVES = (0.5, 0.3, 0.16, 0.08, 0.04, 0.02)   # v5: на сервере ≈ 0,45 с на ковёр


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


def maxerr(q, t, a):
    h, w = a.shape[:2]; L = max(h, w)
    return float(max(math.hypot((q[k][0] - t[k][0]) * w / L, (q[k][1] - t[k][1]) * h / L) for k in range(4)))


def fringe_out(q, t, a):
    """1, если хоть один угол вынесен НАРУЖУ вдоль длинной оси ковра больше чем на 1 % длинной стороны фото (типичная ошибка «бахрома = край»)"""
    h, w = a.shape[:2]; L = max(h, w); T = np.array(t) * [w, h]; Q = np.array(q) * [w, h]
    ax = (T[3] - T[0]) if np.linalg.norm(T[3] - T[0]) >= np.linalg.norm(T[1] - T[0]) else (T[1] - T[0]); ax = ax / np.linalg.norm(ax)
    c = T.mean(0)
    for k in range(4):
        d = np.dot(Q[k] - T[k], ax); s = np.sign(np.dot(T[k] - c, ax))
        if d * s > 0.01 * L: return 1
    return 0


rep = {}
for x in TE: x['frame'] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]   # v5: самый трудный старт — рамка всего фото
for start in ('frame', 'auto', 'vlm', 'ref'):            # ref — углы прежней сети (v3): сравнение «было — стало»
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
    # v4: строже — худший из четырёх углов (так смотрит человек) и «бахрома»: угол вынесен наружу по длинной оси ковра больше чем на 1 %
    M0, M1, F0, F1 = [], [], 0, 0
    for x in TE:
        if not x.get(start): continue
        a = IM[x['id']]; q1 = refine(a, x[start])
        M0.append(maxerr(x[start], x['quad'], a)); M1.append(maxerr(q1, x['quad'], a)); F0 += fringe_out(x[start], x['quad'], a); F1 += fringe_out(q1, x['quad'], a)
    M0, M1 = np.array(M0), np.array(M1)
    rep[start].update({'worst_corner_le_1pct_before': round(float((M0 <= 0.01).mean()), 3), 'worst_corner_le_1pct_after': round(float((M1 <= 0.01).mean()), 3),
                       'worst_corner_le_2pct_after': round(float((M1 <= 0.02).mean()), 3), 'fringe_out_before': F0, 'fringe_out_after': F1})
    log(start, rep[start])

# ---------- v3: можно ли принимать углы без человека? ----------
# Уверенность = согласие сети с самой собой: уточняем из 6 слегка сдвинутых стартов (до 1,5 % длинной стороны) и смотрим разброс ответов.
# Если разброс мал — сеть «видит» угол; таблица ниже показывает, какую долю ковров можно принять автоматически и сколько среди них ошибок.
def spread_run(a, quad, n=6, jit=0.015):
    h, w = a.shape[:2]; L = max(h, w); rs = np.random.RandomState(0); Q = []
    for i in range(n):
        q0 = [[p[0] + (rs.uniform(-jit, jit) * L / w if i else 0), p[1] + (rs.uniform(-jit, jit) * L / h if i else 0)] for p in quad]
        Q.append(refine(a, q0))
    Q = np.array(Q); med = np.median(Q, 0)
    d = np.hypot((Q[..., 0] - med[:, 0]) * w / L, (Q[..., 1] - med[:, 1]) * h / L)      # n × 4
    return med.tolist(), float(d.max())


SP = []
for x in TE:
    st = x.get('vlm') or x.get('auto')
    if not st: continue
    a = IM[x['id']]; med, sp = spread_run(a, st); SP.append((sp, maxerr(med, x['quad'], a), err(med, x['quad'], a)))
SP = np.array(SP); conf = []
for thr in (0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 1.0):
    m = SP[:, 0] <= thr
    if m.sum() == 0: continue
    conf.append({'spread_le': thr, 'share': round(float(m.mean()), 3), 'n': int(m.sum()), 'bad_max_gt_2pct': int((SP[m, 1] > 0.02).sum()), 'bad_max_gt_1pct': int((SP[m, 1] > 0.01).sum()),
                 'median_err': round(float(np.median(SP[m, 2])), 4)})
    log('разброс ≤', thr, conf[-1])
rep['confidence'] = conf

# ---------- выгрузка ----------
net.cpu().eval()
ts = torch.jit.trace(net, torch.zeros(4, 3, W, W))   # v4: в студии — «Загрузить» → «Веса сети углов» (оба файла)
p = f'{OUT}/corner_refine_v{VER}.pt'; ts.save(p)
meta = {'ver': VER, 'W': W, 'halves': HALVES, 'mag': MAG, 'long': LONG, 'trained': time.strftime('%Y-%m-%d %H:%M'),
        'n_train': len(TR), 'n_test': len(TE), 'test': rep}
json.dump(meta, open(f'{OUT}/corner_refine_v{VER}.json', 'w'), ensure_ascii=False, indent=1)
print('\nИТОГ:', json.dumps(meta, ensure_ascii=False))
print('файлы:', p, f'({os.path.getsize(p) / 1e6:.1f} МБ) и corner_refine_v{VER}.json — скачать оба и загрузить в студии: «Ковры» → «＋ Загрузить» → «Веса сети углов»')
