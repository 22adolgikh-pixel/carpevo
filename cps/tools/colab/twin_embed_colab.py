# twin_embed_colab.py — «отпечатки» всех фото ковров для поиска двойников (v11.86, 17.11).
#
# Зачем: сравнение фото по общим точкам (сервер) точное, но медленное — все 4 400 × 4 400 пар не проверить.
# Отпечаток (DINOv2) почти не зависит от обрезки, перспективы, цвета съёмки и поворота: у каждого ковра
# берём 12 ближайших по отпечатку, и только их сервер проверяет точно.
# Что делает: отпечатки фото целиком и центра (70 %), в 4 поворотах; сходство пары — лучшее из вариантов.
# Проверка: на парах, решённых человеком (tools/twins/photos.json), — на каком месте у ковра стоит его настоящий двойник
# и какое сходство у «тот же» / «разные». Порог берётся так, чтобы 95 % настоящих двойников были выше него.
# Итог на Диске: carpet-dna/models/twin_nn_v1.json → загрузить в студии («Ковры» → строка «двойники» → «⬆ отпечатки»).
# Время: T4 — 5–10 мин (большая часть — чтение фото из пакетов).
#
# Запуск:
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %run /content/carpevo/cps/tools/colab/twin_embed_colab.py
import os, json, glob, zipfile, time
import numpy as np, cv2, torch
from google.colab import drive
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
REPO = '/content/carpevo'; OUT = '/content/drive/MyDrive/carpet-dna/models'; os.makedirs(OUT, exist_ok=True)
J = json.load(open(f'{REPO}/cps/tools/twins/photos.json', encoding='utf-8')); PH, PAIRS = J['photos'], J['pairs']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV, '| фото', len(PH), '| пар с ответом', len(PAIRS))
zips = sorted(set(glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/*bundle*.zip') + glob.glob('/content/drive/MyDrive/carpet-dna/**/books_bundle.zip', recursive=True)))
log('пакеты:', [os.path.basename(z) for z in zips]); Z = [zipfile.ZipFile(z) for z in zips]; names = [set(z.namelist()) for z in Z]
MODEL = os.environ.get('TW_MODEL', 'dinov2_vitb14')
net = torch.hub.load('facebookresearch/dinov2', MODEL).to(DEV).eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32); S = 224


def raw(x):
    for z, nm in zip(Z, names):
        if x['img'] in nm:
            return cv2.imdecode(np.frombuffer(z.read(x['img']), np.uint8), cv2.IMREAD_COLOR)
    return None


def tens(a):
    h, w = a.shape[:2]; k = S / max(h, w); a = cv2.resize(a, (max(14, int(w * k)), max(14, int(h * k))), interpolation=cv2.INTER_AREA)
    c = np.full((S, S, 3), 127, np.uint8); y, x0 = (S - a.shape[0]) // 2, (S - a.shape[1]) // 2; c[y:y + a.shape[0], x0:x0 + a.shape[1]] = a
    return torch.from_numpy(((c[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


def views(a):
    """8 видов: фото целиком и центр 70 %, каждый в 4 поворотах (ковёр мог быть снят боком или вверх ногами)"""
    h, w = a.shape[:2]; c = a[int(h * .15):int(h * .85), int(w * .15):int(w * .85)]
    return [tens(np.ascontiguousarray(np.rot90(v, k))) for v in (a, c) for k in range(4)]


E_, ids, buf, bid = [], [], [], []
@torch.no_grad()
def flush():
    if not buf: return
    o = net.forward_features(torch.stack(buf).to(DEV)); f = torch.cat([o['x_norm_clstoken'], o['x_norm_patchtokens'].mean(1)], 1)
    f = torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float32)
    E_.append(f.reshape(-1, 8, f.shape[1])); buf.clear()
miss = 0
for i, x in enumerate(PH):
    a = raw(x)
    if a is None: miss += 1; continue
    buf.extend(views(a)); ids.append(x['id'])
    if len(buf) >= 256: flush()
    if i % 500 == 0: log('отпечатки', i, '/', len(PH))
flush(); X = np.concatenate(E_); pos = {i: k for k, i in enumerate(ids)}; log('фото с отпечатком', len(ids), '| нет в пакетах', miss)

# сходство пары: вид 0 (целиком) и вид 4 (центр) ковра a против всех 8 видов ковра b — лучшее
Xt = torch.from_numpy(X).to(DEV); Q = Xt[:, [0, 4]]                       # N×2×D
N = len(ids); K = 12; NN = {}; SIM = np.zeros((N, N), np.float16) if N <= 6000 else None
with torch.no_grad():
    for s in range(0, N, 256):
        q = Q[s:s + 256]                                                       # b×2×D
        sim = torch.einsum('qvd,nwd->qnvw', q, Xt).amax((2, 3))               # b×N
        sim = torch.maximum(sim, torch.einsum('qvd,nwd->qnvw', Xt[s:s + 256], Q).amax((2, 3)))
        for j in range(sim.shape[0]): sim[j, s + j] = -1
        if SIM is not None: SIM[s:s + 256] = sim.cpu().numpy().astype(np.float16)
        v, ix = sim.topk(K, 1)
        for j in range(sim.shape[0]):
            NN[ids[s + j]] = [[ids[int(t)], round(float(u), 4)] for u, t in zip(v[j].tolist(), ix[j].tolist())]
log('соседи посчитаны')

# проверка на ответах людей
same = [(a, b) for a, b, l in PAIRS if l == 'same' and a in pos and b in pos]; diff = [(a, b) for a, b, l in PAIRS if l != 'same' and a in pos and b in pos]
rank = []; ss, sd = [], []
for a, b in same:
    r = SIM[pos[a]].astype(np.float32); rank.append(int((r > r[pos[b]]).sum()) + 1); ss.append(float(r[pos[b]]))
for a, b in diff: sd.append(float(SIM[pos[a], pos[b]]))
rank = np.array(rank) if rank else np.array([0]); thr = float(np.percentile(ss, 5)) if ss else 0.9
rep = {'same_pairs': len(same), 'diff_pairs': len(diff), 'rank_median': float(np.median(rank)), 'top1': float((rank <= 1).mean()), 'top5': float((rank <= 5).mean()),
       'top12': float((rank <= 12).mean()), 'sim_same_p5': round(thr, 4), 'sim_same_median': round(float(np.median(ss)), 4) if ss else None,
       'sim_diff_median': round(float(np.median(sd)), 4) if sd else None, 'diff_over_thr': int(sum(1 for x in sd if x >= thr))}
log('проверка:', rep)
res = {'ver': 1, 'made': time.strftime('%Y-%m-%d %H:%M'), 'model': MODEL, 'thr': round(thr, 4), 'check': rep, 'nn': NN}
json.dump(res, open(f'{OUT}/twin_nn_v1.json', 'w'), ensure_ascii=False)
print('\nИТОГ:', json.dumps(rep, ensure_ascii=False))
print(f'файл: {OUT}/twin_nn_v1.json — загрузите его в студии (Ковры → «двойники» → «⬆ отпечатки») или скажите Claude, он заберёт с Диска')
