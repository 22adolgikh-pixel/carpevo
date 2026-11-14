# dino3_elements_colab.py — DINOv3 против DINOv2 на наших элементах (v11.62). Без обучения — только замер и выгрузка.
#
# Нужен доступ к DINOv3 на Hugging Face (лицензия Meta принята) и токен «только чтение» в Секретах Colab: HF_TOKEN.
# Сравниваются на одних и тех же рамках:
#   DINOv2 ViT-S/14 (как сейчас в студии), DINOv3 ViT-S/16 и ViT-B/16; вход 112 и 224;
#   цветная вырезка и чёрно-белая с выровненным контрастом (чтобы сеть смотрела на форму, а не на цвет).
# Замеры (сети не обучаются, поэтому честно мерить можно на всех коврах):
#   auc     — отличает ли верные рамки от отклонённых (1,0 — идеально, 0,5 — наугад; у оценки машины ~0,70);
#   top5    — у элемента с именем: доля «того же имени» среди 5 ближайших на других коврах (0..1);
#   any5    — у скольких элементов тот же элемент есть хотя бы среди 5 ближайших.
# Лучший вариант выгружается для студии: carpet-dna/models/dinov3_elem.pt + .json, доступ «по ссылке».
# Время: GPU (T4) — 15–25 мин.
import os, json, random, time, zipfile, glob
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from google.colab import drive, userdata
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV)
if DEV == 'cpu': print('!!! видеокарты нет — будет долго. Среда выполнения → Сменить среду → T4 GPU')
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'))
import transformers; log('transformers', transformers.__version__)
from transformers import AutoModel

REPO = '/content/carpevo'; OUT = '/content/drive/MyDrive/carpet-dna/models'
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
LAB = [x for x in LAB if x['id'] in IM]; log('ковров с фото', len(LAB))
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
CTX = 0.15
CL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def crop(a, box, S, gray):
    h, w = a.shape[:2]; x, y, bw, bh = box
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * CTX)
    M = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
    c = cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    if gray: g = CL.apply(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)); c = cv2.merge([g, g, g])
    return c


def prep(c): return torch.from_numpy(((np.ascontiguousarray(c)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


class V2(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        o = s.m.forward_features(x); return F.normalize(torch.cat([o['x_norm_clstoken'], o['x_norm_patchtokens'].mean(1)], 1), dim=1)


class V3(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m; s.nr = int(getattr(m.config, 'num_register_tokens', 4))
    def forward(s, x):
        h = s.m(pixel_values=x).last_hidden_state                 # [CLS, 4 регистра, участки]
        return F.normalize(torch.cat([h[:, 0], h[:, 1 + s.nr:].mean(1)], 1), dim=1)


NETS = {}
NETS['dinov2_s14'] = (V2(torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')).to(DEV).eval(), 14)
for nm, hf in (('dinov3_s16', 'facebook/dinov3-vits16-pretrain-lvd1689m'), ('dinov3_b16', 'facebook/dinov3-vitb16-pretrain-lvd1689m')):
    try: NETS[nm] = (V3(AutoModel.from_pretrained(hf)).to(DEV).eval(), 16); log('загружена', nm)
    except Exception as e: log('!!! не загрузилась', nm, str(e)[:300])

from sklearn.metrics import roc_auc_score


@torch.no_grad()
def run(net, cs):
    o = []
    for i in range(0, len(cs), 64): o.append(net(torch.stack([prep(c) for c in cs[i:i + 64]]).to(DEV)).float().cpu().numpy())
    return np.concatenate(o)


def rot8(c): return [np.rot90(c[:, ::-1] if f else c, r) for f in (0, 1) for r in range(4)]


def measure(net, S, gray):
    ys, ss, sc0 = [], [], []
    for x in LAB:
        a = IM[x['id']]
        for g in x['groups'].values():
            conf = [m for m in g if m['st'] == 'confirmed']
            cand = [m for m in g if m['k'] != 'hand' and m['st'] in ('confirmed', 'rejected')]
            if not cand: continue
            for m in cand:
                refs = [c for c in conf if c['id'] != m['id']]
                if not refs: continue
                ref = next((c for c in refs if c['k'] == 'hand'), refs[0])
                q = run(net, rot8(crop(a, ref['box'], S, gray))); e = run(net, [crop(a, m['box'], S, gray)])
                ys.append(m['st'] == 'confirmed'); ss.append(float((e @ q.T).max())); sc0.append(m['sc'] or 0)
    ys, ss = np.array(ys), np.array(ss)
    U = []
    for x in LAB:
        a = IM[x['id']]
        for g in x['groups'].values():
            cf = [m for m in g if m['st'] == 'confirmed']
            if not cf: continue
            n = next((m['motif'] or m['scheme'] for m in cf if m['motif'] or m['scheme']), '')
            cf.sort(key=lambda m: (m['k'] != 'hand', -m['box'][2] * m['box'][3]))
            U.append((x['id'], n, run(net, rot8(crop(a, cf[0]['box'], S, gray)))))
    A8 = np.stack([u[2] for u in U]); Sm = np.einsum('irk,jk->ijr', A8, A8[:, 0]).max(2)
    named = [i for i, u in enumerate(U) if u[1] and any(v[1] == u[1] and v[0] != u[0] for v in U)]
    t5 = a5 = 0
    for i in named:
        o = [j for j in np.argsort(-Sm[i]) if U[j][0] != U[i][0]][:5]; k = sum(U[j][1] == U[i][1] for j in o)
        t5 += k / 5; a5 += k > 0
    return {'auc': round(float(roc_auc_score(ys, ss)), 3), 'auc_machine': round(float(roc_auc_score(ys, sc0)), 3), 'n': int(len(ys)),
            'top5': round(t5 / max(1, len(named)), 3), 'any5': round(a5 / max(1, len(named)), 3), 'named': len(named),
            'thr_pre': round(float(np.percentile(ss[ys], 5)), 3), 'catch_at_thr_pre': round(float((ss[~ys] < np.percentile(ss[ys], 5)).mean()), 3)}


R = {}
for nm, (net, p) in NETS.items():
    for S in ((112, 224) if nm != 'dinov2_s14' else (112,)):
        S = S // p * p
        for gray in (False, True):
            t = time.time(); key = f'{nm}@{S}{"-ч/б" if gray else ""}'
            R[key] = measure(net, S, gray); log(key, R[key], f'{time.time() - t:.0f} c')

best = max(R, key=lambda k: R[k]['auc'] + R[k]['any5'])
log('лучший:', best, R[best])
nm = best.split('@')[0]; S = int(best.split('@')[1].split('-')[0]); gray = best.endswith('ч/б')
net = NETS[nm][0].cpu().eval()
ts = torch.jit.trace(net, torch.zeros(8, 3, S, S), check_trace=False, strict=False)
x = torch.randn(2, 3, S, S); dif = float((ts(x) - net(x)).abs().max().detach()); log('сверка выгрузки', dif)
assert dif < 1e-3, 'выгрузка разошлась с сетью — пришли лог Claude'
fn = 'dinov3_elem'; p = f'{OUT}/{fn}.pt'; ts.save(p)
meta = {'ver': 1, 'kind': nm, 'S': S, 'gray': gray, 'ctx': CTX, 'long': LONG, 'trained': time.strftime('%Y-%m-%d %H:%M'), 'thr_search': None,
        'measured': R}
json.dump(meta, open(f'{OUT}/{fn}.json', 'w'), ensure_ascii=False, indent=1)
from google.colab import auth; auth.authenticate_user()
from googleapiclient.discovery import build
svc = build('drive', 'v3'); ids = {}
for f in (fn + '.pt', fn + '.json'):
    r = svc.files().list(q=f"name='{f}' and trashed=false", fields='files(id)', orderBy='modifiedTime desc', spaces='drive').execute().get('files', [])
    if r: svc.permissions().create(fileId=r[0]['id'], body={'type': 'anyone', 'role': 'reader'}).execute(); ids[f] = r[0]['id']
print('\nИТОГ:', json.dumps({'best': best, 'all': R, 'drive': ids, 'size_mb': round(os.path.getsize(p) / 1e6, 1)}, ensure_ascii=False))
