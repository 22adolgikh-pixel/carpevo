# owl_probe_colab.py — проба: найдёт ли OWLv2 (Google) повторы элемента по образцу лучше нашей сети (v11.62).
#
# OWLv2 умеет «поиск по картинке»: даёшь вырезку-образец — находит похожие места на другом изображении.
# Лицензия Apache-2.0, файлы открыты (заявки не нужны). Ничего не обучается — только замер на ваших ответах.
# Для каждой группы, где подтверждено ≥3 рамок: образец — рамка руки или самая крупная; цели — остальные подтверждённые;
# «не то» — отклонённые. Ковёр режется на перекрывающиеся куски, чтобы элемент был крупным (OWLv2 плохо видит мелочь).
# Счёт честный: каждая цель засчитывается один раз; находка на отклонённом месте — ошибка; находка на неотмеченном месте —
# «неизвестно» (может быть неотмеченным повтором).
# Сравнение: наша сеть element_embed_v1 тем же способом, что в студии; на 9 коврах, которых она не видела при обучении,
# и (для справки) на всех. Плюс между коврами: образец с одного ковра, поиск на другом, где есть элемент с тем же именем.
# Время: GPU (T4) — 10–20 мин. Ключи не нужны.
import os, json, math, random, time, zipfile, glob
import numpy as np, cv2, torch
from google.colab import drive
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
REPO = '/content/carpevo'
LAB = json.load(open(f'{REPO}/cps/tools/elements/labels.json', encoding='utf-8'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV)
if DEV == 'cpu': print('!!! видеокарты нет — OWLv2 на процессоре очень медленный. Среда выполнения → Сменить среду → T4 GPU, потом перезапуск')
random.seed(0); np.random.seed(0)

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
LAB = [x for x in LAB if x['id'] in IM]; log('ковров с фото', len(LAB))
rugs = sorted(LAB, key=lambda x: x['id']); random.shuffle(rugs); TEST = {x['id'] for x in rugs[int(len(rugs) * 0.8):]}   # как при обучении нашей сети


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def nms(D, t=0.3):
    D = sorted(D, key=lambda d: -d[0]); keep = []
    for d in D:
        if all(iou(d[1], k[1]) < t for k in keep): keep.append(d)
    return keep


# ---------- задачи ----------
TASKS = []
for x in LAB:
    for g in x['groups'].values():
        cf = [m for m in g if m['st'] == 'confirmed']
        if len(cf) < 3: continue
        cf.sort(key=lambda m: (m['k'] != 'hand', -m['box'][2] * m['box'][3])); ref = cf[0]
        TASKS.append({'rug': x['id'], 'ref': ref['box'], 'targets': [m['box'] for m in cf[1:]], 'rej': [m['box'] for m in g if m['st'] == 'rejected'],
                      'known': [m['box'] for gg in x['groups'].values() for m in gg], 'quad': x.get('quad'), 'test': x['id'] in TEST})
log('групп для проверки', len(TASKS), '| из них на коврах проверки', sum(t['test'] for t in TASKS))


def score(t, D):
    """D — [(оценка, рамка)] после nms; → строки по порогам"""
    D = [d for d in D if iou(d[1], t['ref']) < 0.3]
    out = []
    for thr in THRS:
        S = [d for d in D if d[0] >= thr]; used = set(); hit = rej = unk = 0
        for v, b in S:
            j = max(range(len(t['targets'])), key=lambda i: iou(b, t['targets'][i]), default=None)
            if j is not None and j not in used and iou(b, t['targets'][j]) >= 0.4: used.add(j); hit += 1
            elif any(iou(b, r) >= 0.4 for r in t['rej']): rej += 1
            elif any(iou(b, k) >= 0.4 for k in t['known']): pass
            else: unk += 1
        out.append((hit, len(t['targets']), rej, unk))
    return out


def summary(rows, label):
    rep = []
    for i, thr in enumerate(THRS):
        h = sum(r[i][0] for r in rows); n = sum(r[i][1] for r in rows); rj = sum(r[i][2] for r in rows); uk = sum(r[i][3] for r in rows)
        rep.append({'thr': round(float(thr), 3), 'found': f'{h}/{n}', 'recall': round(h / max(1, n), 3), 'on_rejected': rj, 'unmarked': uk,
                    'precision_known': round(h / max(1, h + rj), 3)})
    best = max(rep, key=lambda r: r['recall'] - 0.5 * r['on_rejected'] / max(1, sum(rr[0][1] for rr in rows)) - 0.002 * r['unmarked'])
    log(label, 'лучший порог:', best); return {'by_thr': rep, 'best': best}


# ---------- OWLv2 ----------
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
OWL = 'google/owlv2-base-patch16-ensemble'
proc = Owlv2Processor.from_pretrained(OWL); owl = Owlv2ForObjectDetection.from_pretrained(OWL).to(DEV).eval()
log('OWLv2 загружен')


def qcrop(a, box, pad=0.15):
    h, w = a.shape[:2]; x, y, bw, bh = box
    x0, y0 = int(max(0, (x - bw * pad) * w)), int(max(0, (y - bh * pad) * h)); x1, y1 = int(min(w, (x + bw * (1 + pad)) * w)), int(min(h, (y + bh * (1 + pad)) * h))
    return Image.fromarray(a[y0:y1, x0:x1, ::-1].copy())


@torch.no_grad()
def owl_find(a, qimg, elem_frac):
    """ищет образец qimg на фото a кусками: кусок такой, чтобы элемент был ~1/6 его стороны"""
    h, w = a.shape[:2]; L = max(h, w); side = int(min(L, max(240, elem_frac * L * 6))); step = int(side * 0.66); D = []
    for y0 in list(range(0, max(1, h - side), step)) + [max(0, h - side)]:
        for x0 in list(range(0, max(1, w - side), step)) + [max(0, w - side)]:
            tile = Image.fromarray(a[y0:y0 + side, x0:x0 + side, ::-1].copy()); tw, th = tile.size
            inp = proc(images=tile, query_images=qimg, return_tensors='pt').to(DEV)
            out = owl.image_guided_detection(**inp)
            # OWLv2 дополняет кусок до квадрата — рамки в долях квадрата max(tw,th)
            S = max(tw, th); res = proc.post_process_image_guided_detection(outputs=out, threshold=0.0, nms_threshold=1.0, target_sizes=torch.tensor([[S, S]]))[0]
            o = res['scores'].argsort(descending=True)[:60]                          # с каждого куска — 60 лучших
            for sc, bx in zip(res['scores'][o].tolist(), res['boxes'][o].tolist()):
                x1, y1, x2, y2 = bx
                if x2 - x1 < 4 or y2 - y1 < 4 or x2 > tw + 2 or y2 > th + 2: continue
                D.append((sc, [(x0 + x1) / w, (y0 + y1) / h, (x2 - x1) / w, (y2 - y1) / h]))
    D = [d for d in D if 0.3 < (d[1][2] * d[1][3]) / max(1e-9, elem_frac ** 2 * (L * L) / (w * h)) < 3.5]   # размер — как у образца
    return nms(D)[:300]


THRS = None
OWL_RAW = []
for i, t in enumerate(TASKS):
    a = IM[t['rug']]; h, w = a.shape[:2]; ef = max(t['ref'][2] * w, t['ref'][3] * h) / max(h, w)
    OWL_RAW.append(owl_find(a, qcrop(a, t['ref']), ef))
    if (i + 1) % 10 == 0: log(f'OWLv2: {i + 1}/{len(TASKS)}')
allsc = sorted(v for D in OWL_RAW for v, _ in D)
THRS = sorted(set(np.round(np.percentile(allsc, [50, 70, 80, 85, 90, 93, 95, 97, 98, 99]), 3))) if allsc else [0.5]
rep = {'owl_all': summary([score(t, D) for t, D in zip(TASKS, OWL_RAW)], 'OWLv2, все ковры'),
       'owl_test': summary([score(t, D) for t, D in zip(TASKS, OWL_RAW) if t['test']], 'OWLv2, 9 ковров проверки')}

# ---------- наша сеть (element_embed_v1, как в студии) ----------
P = '/content/drive/MyDrive/carpet-dna/models/element_embed_v1.pt'
if os.path.exists(P):
    net = torch.jit.load(P, map_location=DEV).eval()
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def crop(a, box, S=96, ctx=0.15):
        h, w = a.shape[:2]; x, y, bw, bh = box; cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * ctx)
        M = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
        return cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    @torch.no_grad()
    def run(cs):
        o = []
        for i in range(0, len(cs), 512):
            b = np.stack([((np.ascontiguousarray(c)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for c in cs[i:i + 512]])
            o.append(net(torch.from_numpy(b).to(DEV)).cpu().numpy())
        return np.concatenate(o)

    def net_find(a, ref):
        h, w = a.shape[:2]; c = crop(a, ref); q = run([np.rot90(c[:, ::-1] if f else c, r) for f in (0, 1) for r in range(4)]); D = []
        for sc in (0.85, 1.0, 1.18):
            for ww, hh in {(ref[2] * sc, ref[3] * sc), (ref[3] * sc * h / w, ref[2] * sc * w / h)}:
                bs = [[float(x), float(y), ww, hh] for y in np.arange(0, 1 - hh, hh / 3) for x in np.arange(0, 1 - ww, ww / 3)]
                if not bs: continue
                s = (run([crop(a, b) for b in bs]) @ q.T).max(1); D += [(float(v), b) for v, b in zip(s, bs) if v >= 0.5]
        return nms(D)[:300]

    NET_RAW = [net_find(IM[t['rug']], t['ref']) for t in TASKS]
    THRS_OWL = THRS; THRS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.95]
    rep['net_test'] = summary([score(t, D) for t, D in zip(TASKS, NET_RAW) if t['test']], 'наша сеть, 9 ковров проверки')
    rep['net_all'] = summary([score(t, D) for t, D in zip(TASKS, NET_RAW)], 'наша сеть, все ковры (часть она видела при обучении)')
    THRS = THRS_OWL
else:
    log('нет element_embed_v1.pt на Диске — сравнение с нашей сетью пропущено')

# ---------- между коврами (OWLv2): образец с ковра A, поиск на ковре B с тем же именем элемента ----------
U = {}
for x in LAB:
    for g in x['groups'].values():
        cf = [m for m in g if m['st'] == 'confirmed']; n = next((m['motif'] or m['scheme'] for m in cf if m['motif'] or m['scheme']), '')
        if n and cf: cf.sort(key=lambda m: (m['k'] != 'hand', -m['box'][2] * m['box'][3])); U.setdefault(n, []).append((x['id'], cf[0]['box'], [m['box'] for m in cf]))
pairs = [(a, b) for n, us in U.items() for a in us for b in us if a[0] != b[0]][:60]
xs = []
for (ra, qa, _), (rb, _, tb) in pairs:
    A, B = IM[ra], IM[rb]; h, w = B.shape[:2]; ef = max(max(bx[2] * w, bx[3] * h) for bx in tb) / max(h, w)
    D = owl_find(B, qcrop(A, qa), ef)
    rank = next((i for i, (_, b) in enumerate(sorted(D, key=lambda d: -d[0])) if any(iou(b, t) >= 0.4 for t in tb)), None)
    xs.append(rank)
rep['cross_owl'] = {'pairs': len(xs), 'first_hit_in_top1': round(sum(r == 0 for r in xs) / max(1, len(xs)), 2),
                    'in_top5': round(sum(r is not None and r < 5 for r in xs) / max(1, len(xs)), 2), 'never': sum(r is None for r in xs)}
log('между коврами (OWLv2):', rep['cross_owl'])

print('\nИТОГ:', json.dumps({k: (v['best'] if 'best' in v else v) for k, v in rep.items()}, ensure_ascii=False))
json.dump(rep, open('/content/drive/MyDrive/carpet-dna/models/owl_probe.json', 'w'), ensure_ascii=False, indent=1)
print('полный отчёт: carpet-dna/models/owl_probe.json')
