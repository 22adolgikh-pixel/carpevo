# frame_classifier_colab.py — «целый ли ковёр на фото?» за человека (16.11).
#
# Зачем: без ответа «целый?» ковру не ставят углы, а без углов нет выпрямленной рабочей картинки. Не отвечено ≈3 200 фото.
# Что делает: признаки DINOv2 (ViT-S/14) по каждому фото → простая модель по ответам людей (≈1 200: целый / обрезан / деталь / не ковёр).
# Проверка — 5 раз на отложенных пятых частях: при какой уверенности модель почти не ошибается.
# Итог на Диске: carpet-dna/models/frame_pred_v1.json — для каждого неотвеченного фото вероятность «целый» + таблица точности.
# По таблице Claude решит, какие ответы принять автоматически, а какие оставить человеку. Время: T4 — 5–10 мин.
import os, json, glob, zipfile, time
import numpy as np, cv2, torch
from google.colab import drive
drive.mount('/content/drive')
T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
REPO = '/content/carpevo'; OUT = '/content/drive/MyDrive/carpet-dna/models'; os.makedirs(OUT, exist_ok=True)
FR = json.load(open(f'{REPO}/cps/tools/corners/frames.json', encoding='utf-8'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; log('устройство', DEV, '| фото', len(FR), '| с ответом', sum(1 for x in FR if x['frame']))
zips = sorted(set(glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/*bundle*.zip') + glob.glob('/content/drive/MyDrive/carpet-dna/**/books_bundle.zip', recursive=True)))
log('пакеты:', [os.path.basename(z) for z in zips]); Z = [zipfile.ZipFile(z) for z in zips]; names = [set(z.namelist()) for z in Z]
net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEV).eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def read(x):
    for z, nm in zip(Z, names):
        if x['img'] in nm:
            a = cv2.imdecode(np.frombuffer(z.read(x['img']), np.uint8), cv2.IMREAD_COLOR)
            if a is None: return None
            h, w = a.shape[:2]; k = 224 / max(h, w); a = cv2.resize(a, (max(14, int(w * k)), max(14, int(h * k))), interpolation=cv2.INTER_AREA)
            c = np.full((224, 224, 3), 127, np.uint8); y, x0 = (224 - a.shape[0]) // 2, (224 - a.shape[1]) // 2; c[y:y + a.shape[0], x0:x0 + a.shape[1]] = a   # всё фото целиком, с полями
            return torch.from_numpy(((c[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    return None


E_, ids, buf, bid = [], [], [], []
@torch.no_grad()
def flush():
    if not buf: return
    o = net.forward_features(torch.stack(buf).to(DEV)); f = torch.cat([o['x_norm_clstoken'], o['x_norm_patchtokens'].mean(1)], 1)
    E_.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy()); ids.extend(bid); buf.clear(); bid.clear()
for i, x in enumerate(FR):
    t = read(x)
    if t is None: continue
    buf.append(t); bid.append(x['id'])
    if len(buf) == 64: flush()
    if i % 500 == 0: log('признаки', i, '/', len(FR))
flush(); X = np.concatenate(E_); pos = {i: k for k, i in enumerate(ids)}; log('фото найдено', len(ids))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
lab = [x for x in FR if x['frame'] and x['id'] in pos and x['frame'] != 'multi']
Xl = X[[pos[x['id']] for x in lab]]; y = np.array([x['frame'] == 'whole' for x in lab]); log('с ответом и фото', len(lab), '| целых', int(y.sum()))
P = np.zeros(len(lab))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xl, y):
    P[te] = LogisticRegression(C=3, max_iter=2000, class_weight='balanced').fit(Xl[tr], y[tr]).predict_proba(Xl[te])[:, 1]
tab = []
for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
    m = P >= t; n = P <= 1 - t
    tab.append({'p': t, 'whole_share': round(float(m.mean()), 3), 'whole_wrong': int((~y[m]).sum()), 'whole_n': int(m.sum()),
                'notwhole_share': round(float(n.mean()), 3), 'notwhole_wrong': int(y[n].sum()), 'notwhole_n': int(n.sum())})
    log(tab[-1])
log('точность при 0,5:', round(float(((P >= 0.5) == y).mean()), 3))
clf = LogisticRegression(C=3, max_iter=2000, class_weight='balanced').fit(Xl, y)
un = [x for x in FR if not x['frame'] and x['id'] in pos]
pu = clf.predict_proba(X[[pos[x['id']] for x in un]])[:, 1]
res = {'ver': 1, 'trained': time.strftime('%Y-%m-%d %H:%M'), 'n_labeled': len(lab), 'cv': tab, 'pred': {x['id']: round(float(p), 3) for x, p in zip(un, pu)},
       'cv_errors': [{'id': x['id'], 'frame': x['frame'], 'p': round(float(p), 3)} for x, p in zip(lab, P) if (p >= 0.9) != (x['frame'] == 'whole') and (p >= 0.9 or p <= 0.1)]}
json.dump(res, open(f'{OUT}/frame_pred_v1.json', 'w'), ensure_ascii=False)
print('\nИТОГ:', json.dumps({'n_labeled': len(lab), 'cv': tab, 'unanswered': len(un), 'p_whole>=0.9': int((pu >= 0.9).sum()), 'p_whole<=0.1': int((pu <= 0.1).sum())}, ensure_ascii=False))
print('файл:', f'{OUT}/frame_pred_v1.json — Claude заберёт его с Диска')
