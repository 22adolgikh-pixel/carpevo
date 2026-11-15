"""v11.82: подгонка рамки элемента к самому похожему месту рядом.

Машина предлагает рамки повторов элемента, но часто промахивается на полрамки: элемент виден сбоку, в рамку влез кусок.
Здесь рамка сдвигается (и чуть меняет размер) к месту, где картинка лучше всего совпадает с образцом —
подтверждёнными рамками той же группы на этом же ковре. Образец берётся во всех 8 положениях (повороты на 90° и отражения):
элементы на ковре зеркалятся и поворачиваются.

Сравнение — нормированная корреляция (cv2.TM_CCOEFF_NORMED) по яркости и двум цветовым каналам Lab, по уменьшенной картинке.
Всё в долях фото: (x, y, w, h).
"""
import cv2
import numpy as np

SCALES = (0.8, 0.9, 1.0, 1.12, 1.25)
TW = 40                       # сторона образца при сравнении (px): быстро и устойчиво к шуму


def _prep(a):
    """Lab, размытие — меньше шума вязки и JPEG"""
    lab = cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab


def _cut(img, b):
    H, W = img.shape[:2]
    x0, y0 = int(round(b[0] * W)), int(round(b[1] * H)); x1, y1 = int(round((b[0] + b[2]) * W)), int(round((b[1] + b[3]) * H))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    return img[y0:y1, x0:x1]


def _orients(t):
    out = []
    for k in range(4):
        r = np.rot90(t, k)
        out += [r, r[:, ::-1]]
    return out


def refine(a, refs, box, search=0.75, scales=SCALES, exclude=None):
    """a — фото (BGR), refs — рамки-образцы (доли), box — проверяемая рамка (доли).
    → {"box": лучшая рамка, "s": её совпадение (−1..1), "s0": совпадение на исходном месте, "moved": сдвиг в долях размера рамки}
    или None, если сравнить не с чем. exclude — не брать место, перекрытое с этой рамкой больше чем на 50 % (для «рядом, но промах»)."""
    if not refs: return None
    H, W = a.shape[:2]
    bw, bh = box[2] * W, box[3] * H
    if bw < 6 or bh < 6: return None
    k = TW / max(bw, bh)                                          # всё уменьшаем так, чтобы рамка была ≈TW px
    sm = cv2.resize(a, (max(8, int(W * k)), max(8, int(H * k))), interpolation=cv2.INTER_AREA)
    L = _prep(sm); Hs, Ws = L.shape[:2]
    # область поиска
    sx0 = max(0, int((box[0] - search * box[2]) * Ws)); sy0 = max(0, int((box[1] - search * box[3]) * Hs))
    sx1 = min(Ws, int((box[0] + (1 + search) * box[2]) * Ws) + 1); sy1 = min(Hs, int((box[1] + (1 + search) * box[3]) * Hs) + 1)
    R = L[sy0:sy1, sx0:sx1]
    if R.shape[0] < 6 or R.shape[1] < 6: return None
    tw, th = box[2] * Ws, box[3] * Hs
    temps = []
    for rb in refs[:3]:
        c = _cut(L, rb)
        if c is None: continue
        for o in _orients(c):
            temps.append(o)
    if not temps: return None
    best = (-2.0, None); s0 = -2.0
    ox, oy = box[0] * Ws - sx0, box[1] * Hs - sy0                 # исходное место в области поиска
    for t in temps:
        ar = t.shape[1] / t.shape[0]
        for sc in scales:
            # размер образца — по площади рамки, пропорции — свои (повёрнутый образец меняет пропорции)
            area = tw * th * sc * sc; w_ = int(round(np.sqrt(area * ar))); h_ = int(round(np.sqrt(area / ar)))
            if w_ < 5 or h_ < 5 or w_ > R.shape[1] or h_ > R.shape[0]: continue
            T = cv2.resize(t, (w_, h_), interpolation=cv2.INTER_AREA)
            if float(T.std()) < 1.0: continue
            M = np.mean([cv2.matchTemplate(np.ascontiguousarray(R[..., ch]), np.ascontiguousarray(T[..., ch]), cv2.TM_CCOEFF_NORMED) for ch in range(3)], axis=0)
            M = np.nan_to_num(M, nan=-1.0)
            if exclude is not None:
                ex = exclude
                ys, xs = np.mgrid[0:M.shape[0], 0:M.shape[1]]
                cx0 = (xs + sx0) / Ws; cy0 = (ys + sy0) / Hs; cw = w_ / Ws; ch_ = h_ / Hs
                ix = np.clip(np.minimum(cx0 + cw, ex[0] + ex[2]) - np.maximum(cx0, ex[0]), 0, None)
                iy = np.clip(np.minimum(cy0 + ch_, ex[1] + ex[3]) - np.maximum(cy0, ex[1]), 0, None)
                inter = ix * iy; iou = inter / (cw * ch_ + ex[2] * ex[3] - inter + 1e-9)
                M = np.where(iou > 0.5, -1.0, M)
            j = int(np.argmax(M)); yy, xx = divmod(j, M.shape[1]); v = float(M.flat[j])
            if v > best[0]:
                best = (v, ((xx + sx0) / Ws, (yy + sy0) / Hs, w_ / Ws, h_ / Hs))
            if abs(sc - 1.0) < 1e-6 and abs(ar - tw / th) / (tw / th) < 0.35:
                xi, yi = int(round(ox + (tw - w_) / 2)), int(round(oy + (th - h_) / 2))
                if 0 <= yi < M.shape[0] and 0 <= xi < M.shape[1] and (exclude is None):
                    s0 = max(s0, float(M[yi, xi]))
    if best[1] is None: return None
    b = best[1]
    b = (max(0.0, b[0]), max(0.0, b[1]), min(b[2], 1 - max(0.0, b[0])), min(b[3], 1 - max(0.0, b[1])))
    mv = float(np.hypot((b[0] + b[2] / 2) - (box[0] + box[2] / 2), (b[1] + b[3] / 2) - (box[1] + box[3] / 2)) / max(box[2], box[3]))
    return {"box": [round(float(v), 5) for v in b], "s": round(best[0], 3), "s0": round(s0, 3) if s0 > -2 else None, "moved": round(mv, 3)}


def iou(p, s):
    ix = max(0, min(p[0] + p[2], s[0] + s[2]) - max(p[0], s[0])); iy = max(0, min(p[1] + p[3], s[1] + s[3]) - max(p[1], s[1])); i = ix * iy
    return i / (p[2] * p[3] + s[2] * s[3] - i + 1e-9)
