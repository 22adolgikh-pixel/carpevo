# elements_bench_colab.py — общий замер поиска повторов элемента (v11.63): SAM 3, наша сеть, DINOv3, (по желанию) OWLv2.
#
# Задача для всех одна: дан образец (одна рамка элемента) — найти на ковре все его повторы.
# Метки берутся СВЕЖИЕ из резервной ветки cps-data (обновляется раз в 10 мин) — после вашей разметки ничего ждать не надо.
#
# Два режима счёта:
#  • есть ковры с флажком «🧪 тест сетей элементов» → счёт строгий только на них: там отмечены все повторы всех элементов,
#    поэтому любая находка не на повторе этого элемента — ошибка. Считаются полнота, точность и F1.
#  • таких ковров нет → как раньше: 9 ковров, которых наша сеть не видела; находки на неотмеченных местах — «неизвестно».
#
# SAM 3 (Meta) умеет «найди все такие же, как в этой рамке». Два способа:
#  • целиком: рамка-образец на всём фото;
#  • кусками: фото режется на куски, к каждому справа приклеивается образец, рамка — на нём (мелкие элементы видны крупнее).
# Нужен секрет HF_TOKEN (доступ к facebook/sam3 одобрен). GPU (T4) обязательно. Время: 20–45 мин.
RUN_OWL = False      # True — добавить OWLv2 (ещё ~15 мин)
import os, sys, json, random, time, zipfile, glob, subprocess
import numpy as np, cv2, torch
from PIL import Image
from google.colab import drive, userdata
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV)
if DEV == 'cpu': raise SystemExit('!!! видеокарты нет. Среда выполнения → Сменить среду → T4 GPU, потом запустите блок заново')
HF = None
try: HF = userdata.get('HF_TOKEN')
except Exception: pass
if not HF: raise SystemExit('!!! нет секрета HF_TOKEN (значок ключа слева, «Доступ из блокнота» включён)')
REPO = '/content/carpevo'; MD = '/content/drive/MyDrive/carpet-dna/models'
random.seed(0); np.random.seed(0)

# ---------- свежие метки из cps-data ----------
try:
    subprocess.run('rm -rf /content/cpsdata && git clone -q --depth 1 --filter=blob:none --sparse -b cps-data https://github.com/22adolgikh-pixel/carpevo.git /content/cpsdata'
                   ' && cd /content/cpsdata && git sparse-checkout set data/rugs', shell=True, check=True)
    print(subprocess.run([sys.executable, f'{REPO}/cps/tools/elements/make_labels.py', '/content/cpsdata/data/rugs'], capture_output=True, text=True, check=True).stdout.strip())
except Exception as e:
    log('свежие метки не получены, беру из репозитория:', str(e)[:200])
LAB = json.load(open(f'{REPO}/cps/tools/elements/labels.json', encoding='utf-8'))

zips = sorted(set(glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/*bundle*.zip')))
Z = [zipfile.ZipFile(z) for z in zips]; names = [set(z.namelist()) for z in Z]
LONG = 1400; IM = {}
for x in LAB:
    for z, nm in zip(Z, names):
        if x['img'] in nm:
            a = cv2.imdecode(np.frombuffer(z.read(x['img']), np.uint8), cv2.IMREAD_COLOR)
            if a is None: break
            k = LONG / max(a.shape[:2])
            if k < 1: a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            IM[x['id']] = a; break
miss = [x['id'] for x in LAB if x['id'] not in IM and x.get('eltest')]
if miss: log('!!! нет фото в пакетах для тестовых ковров:', miss, '— их нужно добавить в rugs_bundle (скажите Claude)')
LAB = [x for x in LAB if x['id'] in IM]; log('ковров с фото', len(LAB))
ELT = [x for x in LAB if x.get('eltest')]
STRICT = bool(ELT)
if STRICT: EVAL = {x['id'] for x in ELT}; log('режим: СТРОГИЙ, ковров «тест сетей»', len(ELT))
else:
    rugs = sorted(LAB, key=lambda x: x['id']); random.shuffle(rugs); EVAL = {x['id'] for x in rugs[int(len(rugs) * 0.8):]}
    log('режим: как раньше (ковров «тест сетей» пока нет), ковров проверки', len(EVAL))


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def nms(D, t=0.3):
    D = sorted(D, key=lambda d: -d[0]); keep = []
    for d in D:
        if all(iou(d[1], k[1]) < t for k in keep): keep.append(d)
    return keep


# ---------- задачи: группа с ≥2 подтверждёнными рамками; образец — рука или самая крупная ----------
TASKS = []
for x in LAB:
    if x['id'] not in EVAL: continue
    G = x['groups']
    for gk, g in G.items():
        cf = [m for m in g if m['st'] == 'confirmed']
        if len(cf) < (2 if STRICT else 3): continue
        cf.sort(key=lambda m: (m['k'] != 'hand', -m['box'][2] * m['box'][3])); ref = cf[0]
        TASKS.append({'rug': x['id'], 'ref': ref['box'], 'targets': [m['box'] for m in cf[1:]], 'rej': [m['box'] for m in g if m['st'] == 'rejected'],
                      'other': [m['box'] for k2, g2 in G.items() if k2 != gk for m in g2 if m['st'] == 'confirmed'],
                      'known': [m['box'] for g2 in G.values() for m in g2]})
log('групп-задач', len(TASKS), '| повторов искать', sum(len(t['targets']) for t in TASKS))
if not TASKS: raise SystemExit('нет групп с ≥2 подтверждёнными рамками на тестовых коврах')


def score(t, D, thrs):
    D = [d for d in D if iou(d[1], t['ref']) < 0.3]; out = []
    for thr in thrs:
        used = set(); hit = bad = unk = 0
        for v, b in (d for d in D if d[0] >= thr):
            j = max(range(len(t['targets'])), key=lambda i: iou(b, t['targets'][i]), default=None)
            near = j is not None and iou(b, t['targets'][j]) >= 0.4
            if near and j not in used: used.add(j); hit += 1
            elif near: pass                                                     # второй раз на тот же повтор — не считаем
            elif any(iou(b, r) >= 0.4 for r in t['rej']): bad += 1
            elif STRICT: bad += 1                                               # всё отмечено — любое иное место ошибка
            elif any(iou(b, k) >= 0.4 for k in t['known']): pass
            else: unk += 1
        out.append((hit, len(t['targets']), bad, unk))
    return out


def summary(label, raw, tasks=None):
    tasks = tasks or TASKS; sc = sorted(v for D in raw for v, _ in D)
    if not sc: log(label, 'нет находок'); return {'best': None}
    thrs = sorted(set(np.round(np.percentile(sc, [30, 50, 70, 80, 85, 90, 93, 95, 97, 98, 99, 99.5]), 4)))
    rows = [score(t, D, thrs) for t, D in zip(tasks, raw)]; rep = []
    for i, thr in enumerate(thrs):
        h = sum(r[i][0] for r in rows); n = sum(r[i][1] for r in rows); bd = sum(r[i][2] for r in rows); uk = sum(r[i][3] for r in rows)
        rc, pr = h / max(1, n), h / max(1, h + bd)
        rep.append({'thr': float(thr), 'found': f'{h}/{n}', 'recall': round(rc, 3), 'wrong': bd, 'unmarked': uk, 'precision': round(pr, 3),
                    'f1': round(2 * rc * pr / max(1e-9, rc + pr), 3)})
    best = max(rep, key=(lambda r: r['f1']) if STRICT else (lambda r: r['recall'] - 0.5 * r['wrong'] / max(1, sum(len(t['targets']) for t in tasks)) - 0.002 * r['unmarked']))
    log(label, '→', best); return {'by_thr': rep, 'best': best}


def size_ok(D, area):
    return [d for d in D if 0.3 < d[1][2] * d[1][3] / max(1e-9, area) < 3.5]


def ef_of(a, box):
    h, w = a.shape[:2]; return max(box[2] * w, box[3] * h) / max(h, w)


def cut(a, box, pad=0.0):
    h, w = a.shape[:2]; x, y, bw, bh = box
    return a[int(max(0, (y - bh * pad) * h)):int(min(h, (y + bh * (1 + pad)) * h)), int(max(0, (x - bw * pad) * w)):int(min(w, (x + bw * (1 + pad)) * w))].copy()


REP = {'mode': 'strict' if STRICT else 'old', 'rugs': len(EVAL), 'tasks': len(TASKS)}
RAW = {}

# ---------- сети-отпечатки (наша и DINOv3): окна размера образца, шаг 1/3, 3 масштаба + поворот 90° — как в студии ----------
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
CL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def emb_net(name):
    p = f'{MD}/{name}.pt'
    if not os.path.exists(p): log('нет', p, '— пропуск'); return None
    M = {}
    try: M = json.load(open(p[:-3] + '.json', encoding='utf-8'))
    except Exception: pass
    if name == 'dinov3_elem':          # файл .pt записан на процессоре (в нём зашит cpu для координат) — на GPU собираем ту же сеть из весов Hugging Face
        import torch.nn as nn, torch.nn.functional as F
        from transformers import AutoModel
        class V3(nn.Module):
            def __init__(s, m): super().__init__(); s.m = m; s.nr = int(getattr(m.config, 'num_register_tokens', 4))
            def forward(s, x):
                h = s.m(pixel_values=x).last_hidden_state
                return F.normalize(torch.cat([h[:, 0], h[:, 1 + s.nr:].mean(1)], 1), dim=1)
        net = V3(AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m', token=HF)).to(DEV).eval()
    else: net = torch.jit.load(p, map_location=DEV).eval()
    S, ctx, gray = int(M.get('S', 96)), float(M.get('ctx', 0.15)), bool(M.get('gray'))

    def crop(a, box):
        h, w = a.shape[:2]; x, y, bw, bh = box; cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * ctx)
        Mx = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
        c = cv2.warpAffine(a, Mx, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        if gray: g = CL.apply(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)); c = cv2.merge([g, g, g])
        if name == 'dinov3_elem':      # v11.64: овал, снаружи серое (как в студии)
            m = np.zeros((S, S), np.uint8); cv2.ellipse(m, (S // 2, S // 2), (max(2, int(round(bw * w / side * S / 2))), max(2, int(round(bh * h / side * S / 2)))), 0, 0, 360, 255, -1)
            c = np.where(m[:, :, None] > 0, c, np.uint8(127))
        return c

    @torch.no_grad()
    def run(cs):
        o = []
        for i in range(0, len(cs), 512):
            b = np.stack([((np.ascontiguousarray(c)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for c in cs[i:i + 512]])
            with torch.autocast('cuda', dtype=torch.float16): o.append(net(torch.from_numpy(b).to(DEV)).float().cpu().numpy())
        o = np.concatenate(o); return o / np.linalg.norm(o, axis=1, keepdims=True).clip(1e-9)

    def query(a, box):
        c = crop(a, box); return run([np.ascontiguousarray(np.rot90(c[:, ::-1] if f else c, r)) for f in (0, 1) for r in range(4)])

    def find(a, q, bw, bh):
        h, w = a.shape[:2]; D = []
        for sc in (0.85, 1.0, 1.18):
            for ww, hh in {(bw * sc, bh * sc), (bh * sc * h / w, bw * sc * w / h)}:
                bs = [[float(x), float(y), ww, hh] for y in np.arange(0, 1 - hh, hh / 3) for x in np.arange(0, 1 - ww, ww / 3)]
                if not bs: continue
                s = (run([crop(a, b) for b in bs]) @ q.T).max(1); D += [(float(v), b) for v, b in zip(s, bs)]
        D.sort(key=lambda d: -d[0]); return nms(D[:3000])[:300]
    return query, find


for name, label in (('element_embed_v1', 'наша сеть'), ('dinov3_elem', 'DINOv3')):
    E = emb_net(name)
    if not E: continue
    q_, f_ = E; raw = []
    for t in TASKS:
        a = IM[t['rug']]; raw.append(f_(a, q_(a, t['ref']), t['ref'][2], t['ref'][3]))
    RAW[label] = (E, raw); REP[label] = summary(label, raw)
    if not STRICT and name == 'element_embed_v1': log('  (наша сеть видела часть этих групп? нет — это 9 ковров проверки)')

# ---------- SAM 3 ----------
from transformers import Sam3Processor, Sam3Model
sproc = Sam3Processor.from_pretrained('facebook/sam3', token=HF); sam = Sam3Model.from_pretrained('facebook/sam3', token=HF).to(DEV).eval()
log('SAM 3 загружен')


@torch.inference_mode()
def sam_run(arr, box_xyxy):
    pil = Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]))
    inp = sproc(images=pil, input_boxes=[[list(map(float, box_xyxy))]], input_boxes_labels=[[1]], return_tensors='pt').to(DEV)
    out = sam(**inp); ts = inp['original_sizes'].tolist()
    try: r = sproc.post_process_object_detection(out, threshold=0.01, target_sizes=ts)[0]
    except Exception: r = sproc.post_process_instance_segmentation(out, threshold=0.01, mask_threshold=0.5, target_sizes=ts)[0]
    return list(zip(r['scores'].float().tolist(), r['boxes'].float().tolist()))


def sam_whole(a, ref):
    h, w = a.shape[:2]; x, y, bw, bh = ref
    D = [(s, [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]) for s, (x1, y1, x2, y2) in sam_run(a, [x * w, y * h, (x + bw) * w, (y + bh) * h]) if x2 - x1 > 2 and y2 - y1 > 2]
    return nms(size_ok(D, bw * bh))[:300]


def sam_tiles(a, q, ef, area):
    """куски так, чтобы элемент был ~1/8 стороны; к куску справа приклеен образец q в том же масштабе, рамка — на нём"""
    h, w = a.shape[:2]; L = max(h, w); side = int(min(L, max(320, ef * L * 8))); step = int(side * 0.7)
    qh, qw = q.shape[:2]; k = ef * L / max(qh, qw); q2 = cv2.resize(q, (max(8, int(qw * k)), max(8, int(qh * k))), interpolation=cv2.INTER_AREA)
    D = []
    for y0 in sorted(set(list(range(0, max(1, h - side), step)) + [max(0, h - side)])):
        for x0 in sorted(set(list(range(0, max(1, w - side), step)) + [max(0, w - side)])):
            tile = a[y0:y0 + side, x0:x0 + side]; th, tw = tile.shape[:2]
            cv_ = np.full((max(th, q2.shape[0] + 16), tw + q2.shape[1] + 16, 3), 127, np.uint8); cv_[:th, :tw] = tile
            cv_[8:8 + q2.shape[0], tw + 8:tw + 8 + q2.shape[1]] = q2
            for s, (x1, y1, x2, y2) in sam_run(cv_, [tw + 8, 8, tw + 8 + q2.shape[1], 8 + q2.shape[0]]):
                if x2 > tw + 2 or y2 > th + 2 or x2 - x1 < 3 or y2 - y1 < 3: continue
                D.append((s, [(x0 + x1) / w, (y0 + y1) / h, (x2 - x1) / w, (y2 - y1) / h]))
    return nms(size_ok(D, area))[:300]


raw_w, raw_t = [], []
for i, t in enumerate(TASKS):
    a = IM[t['rug']]
    try: raw_w.append(sam_whole(a, t['ref']))
    except Exception as e:
        if i == 0: raise SystemExit(f'SAM 3 упал на первом же вызове — пришлите Claude: {e!r}')
        raw_w.append([])
    raw_t.append(sam_tiles(a, cut(a, t['ref']), ef_of(a, t['ref']), t['ref'][2] * t['ref'][3]))
    if (i + 1) % 10 == 0: log(f'SAM 3: {i + 1}/{len(TASKS)}')
REP['SAM 3 целиком'] = summary('SAM 3 целиком', raw_w); REP['SAM 3 кусками'] = summary('SAM 3 кусками', raw_t)

# ---------- OWLv2 (по желанию) ----------
if RUN_OWL:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    oproc = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble'); owl = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble').to(DEV).eval()

    @torch.no_grad()
    def owl_find(a, q, ef, area):
        h, w = a.shape[:2]; L = max(h, w); side = int(min(L, max(240, ef * L * 6))); step = int(side * 0.66); D = []
        qi = Image.fromarray(np.ascontiguousarray(q[:, :, ::-1]))
        for y0 in sorted(set(list(range(0, max(1, h - side), step)) + [max(0, h - side)])):
            for x0 in sorted(set(list(range(0, max(1, w - side), step)) + [max(0, w - side)])):
                tile = Image.fromarray(np.ascontiguousarray(a[y0:y0 + side, x0:x0 + side, ::-1])); tw, th = tile.size; S = max(tw, th)
                out = owl.image_guided_detection(**oproc(images=tile, query_images=qi, return_tensors='pt').to(DEV))
                r = oproc.post_process_image_guided_detection(outputs=out, threshold=0.0, nms_threshold=1.0, target_sizes=torch.tensor([[S, S]]))[0]
                o = r['scores'].argsort(descending=True)[:60]
                for sc, (x1, y1, x2, y2) in zip(r['scores'][o].tolist(), r['boxes'][o].tolist()):
                    if x2 - x1 < 4 or y2 - y1 < 4 or x2 > tw + 2 or y2 > th + 2: continue
                    D.append((sc, [(x0 + x1) / w, (y0 + y1) / h, (x2 - x1) / w, (y2 - y1) / h]))
        return nms(size_ok(D, area))[:300]
    raw_o = [owl_find(IM[t['rug']], cut(IM[t['rug']], t['ref'], 0.15), ef_of(IM[t['rug']], t['ref']), t['ref'][2] * t['ref'][3]) for t in TASKS]
    REP['OWLv2'] = summary('OWLv2', raw_o)

# ---------- между коврами: образец с ковра A, ищем на ковре B, где элемент с тем же именем (из старой разметки) ----------
U = {}
for x in LAB:
    for g in x['groups'].values():
        cf = [m for m in g if m['st'] == 'confirmed']; n = next((m['motif'] or m['scheme'] for m in cf if m['motif'] or m['scheme']), '')
        if n and cf: cf.sort(key=lambda m: (m['k'] != 'hand', -m['box'][2] * m['box'][3])); U.setdefault(n, []).append((x['id'], cf[0]['box'], [m['box'] for m in cf]))
pairs = [(p, q) for us in U.values() for p in us for q in us if p[0] != q[0]]; random.shuffle(pairs); pairs = pairs[:40]


def rank(D, tb):
    return next((i for i, (_, b) in enumerate(sorted(D, key=lambda d: -d[0])) if any(iou(b, t) >= 0.4 for t in tb)), None)


def xs_sum(r):
    return {'pairs': len(r), 'top1': round(sum(v == 0 for v in r) / max(1, len(r)), 2), 'top5': round(sum(v is not None and v < 5 for v in r) / max(1, len(r)), 2),
            'never': sum(v is None for v in r)}


XS = {}
for (ra, qa, _), (rb, _, tb) in pairs:
    A, B = IM[ra], IM[rb]; tw_ = float(np.median([b[2] for b in tb])); th_ = float(np.median([b[3] for b in tb])); ef = max(ef_of(B, b) for b in tb)
    for label in RAW:
        (q_, f_), _ = RAW[label]; XS.setdefault(label, []).append(rank(f_(B, q_(A, qa), tw_, th_), tb))
    XS.setdefault('SAM 3 кусками', []).append(rank(sam_tiles(B, cut(A, qa), ef, tw_ * th_), tb))
REP['между коврами'] = {k: xs_sum(v) for k, v in XS.items()}
log('между коврами (на каком месте первое верное):', REP['между коврами'])

print('\nИТОГ:', json.dumps({k: (v['best'] if isinstance(v, dict) and 'best' in v else v) for k, v in REP.items()}, ensure_ascii=False))
json.dump(REP, open(f'{MD}/elements_bench.json', 'w'), ensure_ascii=False, indent=1)
print('полный отчёт: carpet-dna/models/elements_bench.json')
