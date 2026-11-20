# corners_worst_colab.py — v12.00: обзорный лист худших углов (человек против сети) — чтобы понять, где сеть путает бахрому с краем.
#
# Видеокарта не нужна, 3–6 минут. Берёт cps/tools/corners/labels.json (углы, подтверждённые людьми, и углы сети ref на тот момент),
# фото — из пакетов студии на Диске (carpet-dna/rug_catalog/*bundle*.zip), как corner_refine_colab.py.
# Итог в carpet-dna/models/: corners_worst_v1.jpg (36 худших: целиком + крупно худший угол; зелёный — человек, красный — сеть)
# и corners_worst_v1.json (по каждому ковру: худший угол, ошибка, «бахрома наружу», источник) — Claude читает их с Диска.
import os, json, math, zipfile, glob
import numpy as np, cv2
from google.colab import drive
drive.mount('/content/drive')
REPO = '/content/carpevo'
LAB = json.load(open(f'{REPO}/cps/tools/corners/labels.json', encoding='utf-8'))
LAB = [x for x in LAB if x.get('ref')]
OUT = '/content/drive/MyDrive/carpet-dna/models'; os.makedirs(OUT, exist_ok=True)
zips = sorted(set(glob.glob('/content/drive/MyDrive/carpet-dna/rug_catalog/*bundle*.zip') + glob.glob('/content/drive/MyDrive/carpet-dna/**/books_bundle.zip', recursive=True)))
Z = [zipfile.ZipFile(z) for z in zips]; names = [set(z.namelist()) for z in Z]
print('пакеты', [os.path.basename(z) for z in zips], '| ковров с углами сети', len(LAB))


def img(x):
    for z, nm in zip(Z, names):
        if x['img'] in nm:
            a = cv2.imdecode(np.frombuffer(z.read(x['img']), np.uint8), cv2.IMREAD_COLOR)
            if a is not None:
                k = 1024 / max(a.shape[:2])
                return cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else a
    return None


def stats(q, t, h, w):
    L = max(h, w); e = [math.hypot((q[k][0] - t[k][0]) * w / L, (q[k][1] - t[k][1]) * h / L) for k in range(4)]
    T = np.array(t) * [w, h]; Q = np.array(q) * [w, h]
    ax = (T[3] - T[0]) if np.linalg.norm(T[3] - T[0]) >= np.linalg.norm(T[1] - T[0]) else (T[1] - T[0]); ax = ax / np.linalg.norm(ax); c = T.mean(0)
    along = [float(np.dot(Q[k] - T[k], ax) * np.sign(np.dot(T[k] - c, ax)) / L) for k in range(4)]   # + наружу по длинной оси (бахрома)
    k = int(np.argmax(e))
    return {'worst_k': k, 'worst': round(e[k], 4), 'mean': round(float(np.mean(e)), 4), 'along_out': [round(v, 4) for v in along],
            'fringe_out': int(max(along) > 0.01), 'inward': int(min(along) < -0.01)}


rows = []
for x in LAB:
    a = img(x)
    if a is None: continue
    h, w = a.shape[:2]; s = stats(x['ref'], x['quad'], h, w); s.update(id=x['id'], src=x.get('src'), fix=x.get('fix')); rows.append((s, x, a))
rows.sort(key=lambda r: -r[0]['worst'])
W = [r[0]['worst'] for r in rows]
summ = {'n': len(rows), 'worst_le_1pct': round(float(np.mean([v <= 0.01 for v in W])), 3), 'worst_le_2pct': round(float(np.mean([v <= 0.02 for v in W])), 3),
        'fringe_out': sum(r[0]['fringe_out'] for r in rows), 'inward': sum(r[0]['inward'] for r in rows),
        'by_src': {}}
for r in rows:
    d = summ['by_src'].setdefault(r[0]['src'] or '?', {'n': 0, 'bad2': 0}); d['n'] += 1; d['bad2'] += r[0]['worst'] > 0.02
print(json.dumps({k: v for k, v in summ.items() if k != 'by_src'}, ensure_ascii=False)); print(summ['by_src'])

TW, TH = 360, 240                                                       # плитка: слева ковёр целиком, справа худший угол крупно
tiles = []
for s, x, a in rows[:36]:
    h, w = a.shape[:2]; im = a.copy()
    P = lambda q: (np.array(q) * [w, h]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(im, [P(x['quad'])], True, (60, 220, 60), max(2, w // 300)); cv2.polylines(im, [P(x['ref'])], True, (40, 40, 240), max(2, w // 300))
    k = TH / h; full = cv2.resize(im, (max(1, int(w * k)), TH)); full = full[:, :TW // 2] if full.shape[1] > TW // 2 else cv2.copyMakeBorder(full, 0, 0, 0, TW // 2 - full.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cx, cy = int(x['quad'][s['worst_k']][0] * w), int(x['quad'][s['worst_k']][1] * h); r = int(0.12 * max(h, w))
    zz = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_CONSTANT, value=(255, 0, 255))[cy:cy + 2 * r, cx:cx + 2 * r]
    zz = cv2.resize(zz, (TW // 2, TH))
    t = np.hstack([full, zz])
    cv2.putText(t, f"{s['worst']*100:.1f}% {'FRINGE' if s['fringe_out'] else ('IN' if s['inward'] else '')} {x['id'][:22]}", (4, TH - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)
while len(tiles) % 4: tiles.append(np.zeros((TH, TW, 3), np.uint8))
sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
cv2.imwrite(f'{OUT}/corners_worst_v1.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump({'summary': summ, 'rows': [r[0] for r in rows]}, open(f'{OUT}/corners_worst_v1.json', 'w'), ensure_ascii=False)
print('ИТОГ: записано', f'{OUT}/corners_worst_v1.jpg', 'и corners_worst_v1.json')
