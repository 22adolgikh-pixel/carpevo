# element_search_colab.py — обучить поиск элементов узора по ответам людей (v11.58).
#
# Сеть учится превращать вырезку элемента в «отпечаток» (вектор из 128 чисел): у одного и того же элемента ковра отпечатки
# близки, у похожего, но другого — далеки. Учим на группах рамок (cps/tools/elements/labels.json, make_labels.py):
#   одна группа — один элемент; подтверждённые рамки группы — «то же самое», отклонённые машинные — «похоже, но не то»,
#   случайные места того же ковра — «не то». Поворот на 90° и зеркало не важны (у Керимова элемент тот же).
# Проверка — на 20 % ковров, которых сеть не видела:
#   1) проверка предложений: отличает ли сеть верные рамки от отклонённых лучше, чем нынешняя оценка машины;
#   2) поиск: по одному образцу найти на ковре остальные повторы — сколько подтверждённых найдено и сколько лишнего.
# v11.71: полнота — каждая цель один раз; ковры 🧪 всегда в проверке; в конце — сравнение с v1 на тех же коврах тем же кодом.
# v11.65: вырезка ОВАЛОМ (вне овала серый, как у DINOv3: соседи в плотных узорах не мешают) → новая сеть element_embed_v2.
# Итог: carpet-dna/models/element_embed_v2.pt (TorchScript) и element_embed_v2.json (замеры и пороги). v1 не трогается.
# Время: GPU (T4) — 20–35 мин. Ключи не нужны.
import os, json, math, random, time, zipfile, glob
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from google.colab import drive
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
REPO = '/content/carpevo'
LAB = json.load(open(f'{REPO}/cps/tools/elements/labels.json', encoding='utf-8'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV, '| ковров', len(LAB))
if DEV == 'cpu': print('!!! видеокарты нет — будет очень долго. Среда выполнения → Сменить среду → T4 GPU')
OUT = '/content/drive/MyDrive/carpet-dna/models'; os.makedirs(OUT, exist_ok=True)
random.seed(0); np.random.seed(0); torch.manual_seed(0)

zips = sorted(set(glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/*bundle*.zip')))
Z = [zipfile.ZipFile(z) for z in zips]; names = [set(z.namelist()) for z in Z]
LONG = 1400
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
log('фото найдено', len(LAB))
assert len(LAB) >= 20, 'мало фото — проверь пакеты в carpet-dna/rug_catalog'

SHAPE = 'ellipse'                                      # 'rect' — как в v1
S = 96; CTX = 0.15                                     # вход сети 96×96; вокруг рамки 15 % поля
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(a, box, jit=0.0, ctx=CTX, S=S, shape=SHAPE):
    """рамка (доли кадра x,y,w,h) → квадрат S×S с полем вокруг; jit — случайный сдвиг/масштаб рамки"""
    h, w = a.shape[:2]; x, y, bw, bh = box
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * ctx); bx, by = cx, cy      # bx, by — центр самой рамки
    if jit:
        side *= math.exp(random.uniform(-jit, jit)); cx += random.uniform(-jit, jit) * side * 0.5; cy += random.uniform(-jit, jit) * side * 0.5
    M = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
    c = cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    if shape == 'ellipse':                                                  # овал по рамке (при сдвиге — вместе с рамкой), снаружи серое
        m = np.zeros((S, S), np.uint8); ex, ey = (bx - (cx - side / 2)) / side * S, (by - (cy - side / 2)) / side * S
        cv2.ellipse(m, (int(round(ex)), int(round(ey))), (max(2, int(round(bw * w / side * S / 2))), max(2, int(round(bh * h / side * S / 2)))), 0, 0, 360, 255, -1)
        c = np.where(m[:, :, None] > 0, c, np.uint8(127))
    return c


def aug(c):
    c = np.rot90(c, random.randrange(4)); c = c[:, ::-1] if random.random() < 0.5 else c
    c = cv2.convertScaleAbs(np.ascontiguousarray(c), alpha=random.uniform(0.75, 1.25), beta=random.uniform(-20, 20))
    if random.random() < 0.3: c = cv2.GaussianBlur(c, (0, 0), random.uniform(0.4, 1.2))
    return c


def prep(c): return torch.from_numpy(((np.ascontiguousarray(c)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


rugs = sorted(LAB, key=lambda x: x['id']); random.shuffle(rugs)
cut = int(len(rugs) * 0.8); TR, TE = rugs[:cut], rugs[cut:]
TE += [x for x in TR if x.get('eltest')]      # v11.71: ковры «тест сетей» (все повторы отмечены) всегда в проверке
TR = [x for x in TR if not x.get('eltest')]   # v11.63: ковры «тест сетей» не учим — на них мерим
log('ковров: обучение', len(TR), '| проверка', len(TE))


def groups_of(items):
    G = []
    for x in items:
        allb = [m['box'] for g in x['groups'].values() for m in g]
        for gid, g in x['groups'].items():
            pos = [m['box'] for m in g if m['st'] == 'confirmed']; neg = [m['box'] for m in g if m['st'] == 'rejected']
            if pos: G.append({'rug': x['id'], 'pos': pos, 'neg': neg, 'all': allb})
    return G


GTR = groups_of(TR); log('групп для обучения', len(GTR))


def rand_box(x, ref, allb):
    """случайное место того же ковра того же размера, не задевающее известные рамки"""
    for _ in range(20):
        b = [random.uniform(0.02, 0.98 - ref[2]), random.uniform(0.02, 0.98 - ref[3]), ref[2], ref[3]]
        if all(iou(b, o) < 0.1 for o in allb): return b
    return b


class Net(nn.Module):
    def __init__(s):
        super().__init__(); m = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        s.f = m.features; s.h = nn.Sequential(nn.Linear(576, 256), nn.Hardswish(), nn.Linear(256, 128))
    def forward(s, x): return F.normalize(s.h(F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)), dim=1)


net = Net().to(DEV)
B = 48; STEPS = 2600
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS)
TAU = 0.1


def batch():
    A, P, N = [], [], []
    for g in random.sample(GTR, min(B, len(GTR))) if len(GTR) >= B else random.choices(GTR, k=B):
        a = IM[g['rug']]; ab = random.choice(g['pos']); pb = random.choice(g['pos'])           # одна рамка → два разных вида её же
        A.append(prep(aug(crop(a, ab, 0.12)))); P.append(prep(aug(crop(a, pb, 0.12))))
        nb = random.choice(g['neg']) if g['neg'] and random.random() < 0.6 else rand_box(None, ab, g['all'])
        N.append(prep(aug(crop(a, nb, 0.12))))
    return torch.stack(A), torch.stack(P), torch.stack(N)


net.train()
for st in range(STEPS):
    A, P, N = [t.to(DEV) for t in batch()]
    ea, ep, en = net(A), net(P), net(N)
    # InfoNCE: якорь ближе к своей паре, чем к чужим парам и ко всем «не тем» в пачке (свой трудный пример — тоже)
    logits = torch.cat([ea @ ep.T, ea @ en.T], 1) / TAU
    loss = F.cross_entropy(logits, torch.arange(len(A), device=DEV))
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if (st + 1) % 200 == 0: log(f'шаг {st + 1}/{STEPS}: потеря {loss.item():.3f}')

net.eval()


CUR = {'name': 'v2', 'net': net, 'S': S, 'ctx': CTX, 'shape': SHAPE}     # v11.71: сеть и её вырезка — чтобы мерить v1 тем же кодом


@torch.no_grad()
def emb(a, boxes, M=None):
    M = M or CUR; out = []
    for i in range(0, len(boxes), 256):
        out.append(M['net'](torch.stack([prep(crop(a, b, 0, M['ctx'], M['S'], M['shape'])) for b in boxes[i:i + 256]]).to(DEV)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 128), np.float32)


@torch.no_grad()
def emb_rot(a, box, M=None):
    """отпечаток образца во всех 8 положениях (поворот на 90° и зеркало) — у ковра элемент бывает повёрнут"""
    M = M or CUR; c = crop(a, box, 0, M['ctx'], M['S'], M['shape']); V = []
    for f in (0, 1):
        for r in range(4):
            v = np.rot90(c[:, ::-1] if f else c, r); V.append(prep(v))
    return M['net'](torch.stack(V).to(DEV)).float().cpu().numpy()


def sim(q8, E): return (E @ q8.T).max(1)                 # лучшее из 8 положений образца


# ---------- проверка 1: отличить верные предложения машины от отклонённых ----------
from sklearn.metrics import roc_auc_score
ys, ss, sc0 = [], [], []
for x in TE:
    a = IM[x['id']]
    for g in x['groups'].values():
        conf = [m for m in g if m['st'] == 'confirmed']
        cand = [m for m in g if m['k'] != 'hand' and m['st'] in ('confirmed', 'rejected')]
        for m in cand:
            refs = [c for c in conf if c['id'] != m['id']]
            if not refs: continue
            ref = next((c for c in refs if c['k'] == 'hand'), refs[0])
            ys.append(m['st'] == 'confirmed'); ss.append(float(sim(emb_rot(a, ref['box']), emb(a, [m['box']]))[0])); sc0.append(m['sc'] if m['sc'] is not None else 0)
ys, ss, sc0 = np.array(ys), np.array(ss), np.array(sc0)
rep = {'verify': {'n': int(len(ys)), 'right': int(ys.sum()), 'wrong': int((~ys).sum())}}
if len(set(ys)) == 2:
    rep['verify'].update({'auc_net': round(float(roc_auc_score(ys, ss)), 3), 'auc_machine_score': round(float(roc_auc_score(ys, sc0)), 3)})
    # порог «машина сомневается»: ниже него рамка заранее отмечается ✗ — ошибаться в сторону ✗ не больше чем у 5 % верных
    thr_bad = float(np.percentile(ss[ys], 5)); rep['verify'].update({'thr_bad': round(thr_bad, 3),
        'wrong_caught': round(float((ss[~ys] < thr_bad).mean()), 3), 'right_lost': round(float((ss[ys] < thr_bad).mean()), 3)})
log('проверка предложений:', rep['verify'])

# ---------- проверка 2: поиск повторов по одному образцу ----------
def search(a, q, ref, win_area, thr, quad=None, M=None):
    h, w = a.shape[:2]; bw, bh = ref[2], ref[3]; C = []
    for sc in (0.85, 1.0, 1.18):
        for (ww, hh) in {(bw * sc, bh * sc), (bh * sc * h / w, bw * sc * w / h)}:          # и повёрнутый на 90°
            stx, sty = ww / 3, hh / 3; xs = np.arange(0, 1 - ww, stx); ys_ = np.arange(0, 1 - hh, sty)
            if quad:
                qx = [p[0] for p in quad]; qy = [p[1] for p in quad]
                xs = xs[(xs >= min(qx) - ww / 2) & (xs <= max(qx) - ww / 2)]; ys_ = ys_[(ys_ >= min(qy) - hh / 2) & (ys_ <= max(qy) - hh / 2)]
            bs = [[float(x), float(y), ww, hh] for y in ys_ for x in xs]
            if not bs: continue
            s = sim(q, emb(a, bs, M)); C += [(float(v), b) for v, b in zip(s, bs) if v >= thr - 0.15]
    C.sort(key=lambda t: -t[0]); keep = []
    for v, b in C:                                                                          # убрать наложения
        if all(iou(b, k[1]) < 0.3 for k in keep): keep.append((v, b))
    return keep


def run_search(M):
    """по одному образцу каждой группы (≥ 3 подтверждённых) ищем остальные; у каждой находки — какая цель (или None)"""
    res = []
    for x in TE:
        a = IM[x['id']]
        for g in x['groups'].values():
            conf = [m for m in g if m['st'] == 'confirmed']
            if len(conf) < 3: continue
            ref = next((c for c in conf if c['k'] == 'hand'), conf[0]); q = emb_rot(a, ref['box'], M)
            targets = [c['box'] for c in conf if c['id'] != ref['id']]; rej = [m['box'] for m in g if m['st'] == 'rejected']
            found = [(v, b) for v, b in search(a, q, ref['box'], None, 0.0, x.get('quad'), M) if iou(b, ref['box']) < 0.4]   # сам образец не в счёт
            tg = []
            for _, b in found:
                k = int(np.argmax([iou(b, t) for t in targets])) if targets else -1
                tg.append(k if k >= 0 and iou(b, targets[k]) >= 0.4 else None)
            res.append({'rug': x['id'], 'eltest': bool(x.get('eltest')), 'sims': [v for v, _ in found], 'tg': tg, 'n_t': len(targets),
                        'rej': [max([iou(b, t) for t in rej] + [0]) >= 0.4 for _, b in found]})
    return res


def at(res, thr):
    """v11.71: каждая цель засчитывается ОДИН раз (раньше два окна на одной цели давали два попадания → полнота > 100 %)"""
    tp = n = rj = dup = 0; nt = sum(R['n_t'] for R in res); tp_e = n_e = 0
    for R in res:
        seen = set()
        for v, t, j in zip(R['sims'], R['tg'], R['rej']):
            if v < thr: continue
            n += 1; rj += j
            if t is not None and t not in seen: seen.add(t); tp += 1
            elif t is not None: dup += 1
        if R['eltest']: tp_e += len(seen); n_e += sum(1 for v in R['sims'] if v >= thr)
    return {'thr': round(float(thr), 3), 'found_confirmed': f'{tp}/{nt}', 'recall': round(tp / max(1, nt), 3), 'proposed': n,
            'extra': n - tp - dup, 'dup': dup, 'hit_rejected': rj, 'share_confirmed': round(tp / max(1, n), 3),
            'eltest_precision': round(tp_e / n_e, 3) if n_e else None}


def table(res, name):
    rows = [at(res, t) for t in np.arange(0.5, 0.981, 0.0125)]
    for r in rows[::4]: log(f'поиск {name}', r)
    return rows


def same_load(rows, n):
    """порог, при котором находок примерно n (сравнение сетей при одинаковой «нагрузке» на проверку людьми)"""
    return min(rows, key=lambda r: (abs(r['proposed'] - n), -r['thr']))


res2 = run_search(CUR); rows2 = table(res2, 'v2')
rep['search'] = {'groups': len(res2), 'test_rugs': [x['id'] for x in TE], 'by_thr': rows2}
ok = [r for r in rows2 if r['recall'] >= 0.6]
rep['search']['pick'] = max(ok, key=lambda r: (r['share_confirmed'], r['thr'])) if ok else rows2[0]
log('выбран порог поиска v2 (полнота ≥ 0,6, больше всего «в цель»):', rep['search']['pick'])

# ---------- v11.71: та же проверка для v1 (веса с Диска) — на тех же коврах, тем же кодом ----------
V1 = {'pt': '1M_5g-sChZHlGhzNYyiuiL3xzq81ybX-V', 'json': '11_NpAjRhHnZSYBgln7JZxetOebGKWQ0J'}
try:
    p1 = f'{OUT}/element_embed_v1.pt'; j1 = f'{OUT}/element_embed_v1.json'
    if not os.path.exists(p1):
        import gdown; p1 = '/content/element_embed_v1.pt'; j1 = '/content/element_embed_v1.json'
        gdown.download(id=V1['pt'], output=p1, quiet=True); gdown.download(id=V1['json'], output=j1, quiet=True)
    m1 = json.load(open(j1, encoding='utf-8')) if os.path.exists(j1) else {}
    M1 = {'name': 'v1', 'net': torch.jit.load(p1, map_location=DEV).eval(), 'S': int(m1.get('S', 96)), 'ctx': float(m1.get('ctx', 0.15)), 'shape': m1.get('shape', 'rect')}
    rows1 = table(run_search(M1), 'v1')
    r1 = next(r for r in rows1 if r['thr'] >= 0.875)                     # рабочий порог v1 в студии
    r2 = same_load(rows2, r1['proposed'])
    cmp = {'v1_at_0.875': r1, 'v2_same_load': r2, 'v2_pick': rep['search']['pick'],
           'note': 'часть ковров проверки v1 могла видеть при своём обучении — сравнение в пользу v1'}
    rep['compare'] = cmp
    log('СРАВНЕНИЕ при одинаковом числе находок:'); log('  v1 (порог 0,875):', r1); log('  v2:', r2)
    better = r2['recall'] > r1['recall'] and r2['hit_rejected'] <= r1['hit_rejected']
    rep['compare']['v2_better'] = better
    log('  → v2 ' + ('ЛУЧШЕ — можно переключать поиск на v2 (порог %.3f)' % r2['thr'] if better else 'не лучше — поиск остаётся на v1'))
except Exception as e:
    log('сравнение с v1 не вышло:', repr(e)); rep['compare'] = {'error': repr(e)}
log('(«share_confirmed» — доля находок, совпавших с подтверждёнными; «extra» — находки мимо отмеченного: на коврах 🧪 это ошибки, на остальных часть — неотмеченные повторы; «eltest_precision» — точность только на коврах 🧪)')

net.cpu().eval()
ts = torch.jit.trace(net, torch.zeros(8, 3, S, S)); p = f'{OUT}/element_embed_v2.pt'; ts.save(p)
meta = {'ver': 2, 'S': S, 'ctx': CTX, 'shape': SHAPE, 'long': LONG, 'trained': time.strftime('%Y-%m-%d %H:%M'), 'n_rugs_train': len(TR), 'n_rugs_test': len(TE),
        'thr_bad': rep['verify'].get('thr_bad'), 'thr_search': ((rep.get('compare') or {}).get('v2_same_load') if (rep.get('compare') or {}).get('v2_better') else rep['search'].get('pick') or {}).get('thr'), 'test': rep}
json.dump(meta, open(f'{OUT}/element_embed_v2.json', 'w'), ensure_ascii=False, indent=1)
from google.colab import auth; auth.authenticate_user()                                 # доступ «по ссылке» — студия скачает сама
from googleapiclient.discovery import build
svc = build('drive', 'v3'); ids = {}
for fn in ('element_embed_v2.pt', 'element_embed_v2.json'):
    f = svc.files().list(q=f"name='{fn}' and trashed=false", fields='files(id,modifiedTime)', orderBy='modifiedTime desc').execute()['files']
    if f: svc.permissions().create(fileId=f[0]['id'], body={'type': 'anyone', 'role': 'reader'}).execute(); ids[fn] = f[0]['id']
print('\nИТОГ:', json.dumps({k: meta[k] for k in ('trained', 'thr_bad', 'thr_search')} | {'verify': rep['verify'], 'search_pick': rep['search'].get('pick'), 'compare': {k: v for k, v in (rep.get('compare') or {}).items() if k != 'note'}, 'drive': ids}, ensure_ascii=False))
