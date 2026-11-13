# dino_elements_colab.py — готовая сеть DINOv2 (без обучения) для сравнения элементов (v11.60).
#
# Скачивает DINOv2 ViT-S/14 (Meta, лицензия Apache-2.0), проверяет её на наших ответах и выгружает для студии.
# Отпечаток вырезки = [CLS-токен, среднее по участкам], нормированный. Сеть не учится — только замер.
# Замеры (те же ковры для проверки, что и в element_search_colab.py, — цифры можно сравнить):
#   1) проверка предложений: отличает ли верные рамки от отклонённых (AUC; у оценки машины — 0,64);
#   2) между коврами: у элемента с именем — сколько из 5 ближайших на других коврах носят то же имя.
# Размер входа подбирается из 112 / 168 / 224 (больше — точнее, но медленнее на сервере).
# Итог: carpet-dna/models/dinov2_s14.pt (TorchScript) и dinov2_s14.json; доступ «по ссылке» открывается сам.
# Время: GPU (T4) — 10–15 мин. Ключи не нужны.
import os, json, math, random, time, zipfile, glob
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from google.colab import drive
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
REPO = '/content/carpevo'
LAB = json.load(open(f'{REPO}/cps/tools/elements/labels.json', encoding='utf-8'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV)
OUT = '/content/drive/MyDrive/carpet-dna/models'; os.makedirs(OUT, exist_ok=True)
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
rugs = sorted(LAB, key=lambda x: x['id']); random.shuffle(rugs); TE = rugs[int(len(rugs) * 0.8):]   # как в element_search_colab.py

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEV).eval()


class Emb(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        o = s.m.forward_features(x)
        v = torch.cat([o['x_norm_clstoken'], o['x_norm_patchtokens'].mean(1)], 1)
        return F.normalize(v, dim=1)


net = Emb(dino).eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
CTX = 0.15


def crop(a, box, S):
    h, w = a.shape[:2]; x, y, bw, bh = box
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * CTX)
    M = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
    return cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def prep(c): return torch.from_numpy(((np.ascontiguousarray(c)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


@torch.no_grad()
def run(cs):
    out = []
    for i in range(0, len(cs), 64): out.append(net(torch.stack([prep(c) for c in cs[i:i + 64]]).to(DEV)).cpu().numpy())
    return np.concatenate(out)


def rot8(c): return [np.rot90(c[:, ::-1] if f else c, r) for f in (0, 1) for r in range(4)]


from sklearn.metrics import roc_auc_score


def evaluate(S, items):
    ys, ss, sc0 = [], [], []
    for x in items:
        a = IM[x['id']]
        for g in x['groups'].values():
            conf = [m for m in g if m['st'] == 'confirmed']
            for m in [m for m in g if m['k'] != 'hand' and m['st'] in ('confirmed', 'rejected')]:
                refs = [c for c in conf if c['id'] != m['id']]
                if not refs: continue
                ref = next((c for c in refs if c['k'] == 'hand'), refs[0])
                q = run(rot8(crop(a, ref['box'], S))); e = run([crop(a, m['box'], S)])
                ys.append(m['st'] == 'confirmed'); ss.append(float((e @ q.T).max())); sc0.append(m['sc'] or 0)
    ys, ss, sc0 = np.array(ys), np.array(ss), np.array(sc0)
    r = {'n': int(len(ys)), 'auc': round(float(roc_auc_score(ys, ss)), 3), 'auc_machine_score': round(float(roc_auc_score(ys, sc0)), 3)}
    thr = float(np.percentile(ss[ys], 5)); r.update(thr_bad=round(thr, 3), wrong_caught=round(float((ss[~ys] < thr).mean()), 3))
    return r


def cross(S):
    U = []
    for x in LAB:
        a = IM[x['id']]
        for g in x['groups'].values():
            cf = [m for m in g if m['st'] == 'confirmed']
            if not cf: continue
            n = next((m['motif'] or m['scheme'] for m in cf if m['motif'] or m['scheme']), '')
            cf.sort(key=lambda m: (m['k'] != 'hand', -m['box'][2] * m['box'][3]))
            c = crop(a, cf[0]['box'], S); U.append((x['id'], n, run(rot8(c))))
    A8 = np.stack([u[2] for u in U]); A0 = A8[:, 0]
    Sm = np.einsum('irk,jk->ijr', A8, A0).max(2)
    hits, tot = 0, 0; named = [i for i, u in enumerate(U) if u[1] and sum(1 for v in U if v[1] == u[1] and v[0] != u[0]) > 0]
    for i in named:
        o = [j for j in np.argsort(-Sm[i]) if U[j][0] != U[i][0]][:5]
        hits += sum(U[j][1] == U[i][1] for j in o); tot += 1
    return {'units': len(U), 'named_with_partner': tot, 'top5_same_name': round(hits / max(1, tot), 2),
            'any_in_top5': round(sum(any(U[j][1] == U[i][1] for j in [j for j in np.argsort(-Sm[i]) if U[j][0] != U[i][0]][:5]) for i in named) / max(1, tot), 2)}


rep = {}
for S in (112, 168, 224):
    t = time.time(); rep[S] = {'verify_test': evaluate(S, TE), 'verify_all': evaluate(S, LAB), 'cross': cross(S)}
    log('вход', S, rep[S], f'{time.time() - t:.0f} c')
best = max(rep, key=lambda S: rep[S]['verify_all']['auc'] - (0.01 if S == 224 else 0))
log('выбран вход', best)
net.cpu().eval()
ts = torch.jit.trace(net, torch.zeros(8, 3, best, best), check_trace=False); p = f'{OUT}/dinov2_s14.pt'; ts.save(p)
x = torch.randn(2, 3, best, best); dif = float((ts(x) - net(x)).abs().max()); log('сверка выгрузки', dif)
meta = {'ver': 1, 'kind': 'dinov2_vits14', 'S': best, 'ctx': CTX, 'long': LONG, 'trained': time.strftime('%Y-%m-%d %H:%M'),
        'thr_bad': rep[best]['verify_all']['thr_bad'], 'thr_search': None, 'test': {str(k): v for k, v in rep.items()}}
json.dump(meta, open(f'{OUT}/dinov2_s14.json', 'w'), ensure_ascii=False, indent=1)
from google.colab import auth; auth.authenticate_user()
from googleapiclient.discovery import build
svc = build('drive', 'v3'); ids = {}
for fn in ('dinov2_s14.pt', 'dinov2_s14.json'):
    f = svc.files().list(q=f"name='{fn}' and trashed=false", fields='files(id,modifiedTime)', orderBy='modifiedTime desc').execute()['files']
    if f: svc.permissions().create(fileId=f[0]['id'], body={'type': 'anyone', 'role': 'reader'}).execute(); ids[fn] = f[0]['id']
print('\nИТОГ:', json.dumps({'S': best, 'thr_bad': meta['thr_bad'], 'by_size': meta['test'], 'drive': ids, 'size_mb': round(os.path.getsize(p) / 1e6, 1)}, ensure_ascii=False))
