# dino3_shape_colab.py — мешают ли соседи? (v11.64) Тот же замер, что в dino3_elements_colab.py, но меняем ТОЛЬКО вырезку:
#   запас вокруг рамки (0 / 5 / 15 %)  ×  форма (прямоугольник / овал, всё вне овала закрашено серым).
# Сеть одна — DINOv3 S/16 (лучшая прошлого замера), цвет, вход 112 и 224. Время на T4: 8–12 мин. Нужен секрет HF_TOKEN.
# Читать так: auc (выше лучше), any5 (выше лучше). Если у овала/малого запаса метрики заметно выше — соседи мешают.
import os, re, json, time
src = open('/content/carpevo/cps/tools/colab/dino3_elements_colab.py', encoding='utf-8').read()
head = src[:src.index('\nR = {}\n')]              # всё до цикла замеров: загрузка данных, сетей, crop, measure
exec(compile(head, 'dino3_elements_colab(head)', 'exec'))
for k in [k for k in NETS if k != 'dinov3_s16']: NETS.pop(k)     # остальные сети тут не нужны
SHAPE = 'rect'


def crop(a, box, S, gray):
    h, w = a.shape[:2]; x, y, bw, bh = box
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * CTX)
    M = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
    c = cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    if SHAPE == 'ellipse':
        m = np.zeros((S, S), np.uint8); ax, ay = int(round(bw * w / side * S / 2)), int(round(bh * h / side * S / 2))
        cv2.ellipse(m, (S // 2, S // 2), (max(2, ax), max(2, ay)), 0, 0, 360, 255, -1)
        c = np.where(m[:, :, None] > 0, c, np.uint8(127))
    return c


net, p = NETS['dinov3_s16']; R2 = {}
for S in (112, 224):
    S = S // p * p
    for shape in ('rect', 'ellipse'):
        for ctx in (0.0, 0.05, 0.15):
            SHAPE, CTX = shape, ctx; t = time.time(); key = f'S{S} {"овал" if shape == "ellipse" else "прямоугольник"} запас {int(ctx * 100)}%'
            R2[key] = measure(net, S, False); log(key, {k: R2[key][k] for k in ('auc', 'top5', 'any5', 'catch_at_thr_pre')}, f'{time.time() - t:.0f} c')
base = R2['S112 прямоугольник запас 15%']
best = max(R2, key=lambda k: R2[k]['auc'] + R2[k]['any5'])
print('\nИТОГ: базовая (как в студии):', {k: base[k] for k in ('auc', 'top5', 'any5')}, '| лучшая:', best, {k: R2[best][k] for k in ('auc', 'top5', 'any5')})
json.dump(R2, open(f'{OUT}/dino3_shape.json', 'w'), ensure_ascii=False, indent=1)
