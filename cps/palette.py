"""v11.98: палитра ковра полоской — для TouchDesigner и подобных программ.

Полоска W×H пикселей (по умолчанию 1024×1): цвета ковра по убыванию площади, слева — самый крупный (обычно фон поля),
справа — мелкие детали. Ширина цвета = его доля в ковре: цвет 30 % → ≈ 307 px из 1024.

Как считается: ковёр выпрямляется по углам (если углы есть), края кадра за ковром не считаются; фото уменьшается до ~320 px,
слегка сглаживается (ворс и нити не должны давать отдельных «цветов»), цвета группируются k-средними в пространстве Lab
и близкие (ΔE < 12) сливаются. Доли меньше 0,3 % отбрасываются как шум.
"""
import io, json, math
import cv2, numpy as np

VER = 1
BORDER = (40, 40, 40)                       # чем rug_rect.rectify заливает место за краем кадра


def body(img, quad=None):
    """ковёр без фона: выпрямленный по углам или (без углов) центр кадра 80 %"""
    if quad:
        import rug_rect
        a = rug_rect.rectify(img, quad, side=480)
        m = ~np.all(a == BORDER, axis=2)
        return a, m
    h, w = img.shape[:2]
    a = img[int(h * .1):int(h * .9), int(w * .1):int(w * .9)]
    return a, np.ones(a.shape[:2], bool)


def colors(img, quad=None, k=12, merge=12.0, min_share=0.003):
    """→ [{hex, rgb, lab, share}] по убыванию доли"""
    a, m = body(img, quad)
    s = 320 / max(a.shape[:2])
    if s < 1:
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    a = cv2.medianBlur(a, 3)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)[m.reshape(-1)]
    if len(lab) < 50: return []
    k = max(2, min(int(k), 24, len(lab) // 20))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    cv2.setRNGSeed(7)
    _, lbl, cen = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lbl.reshape(-1), minlength=k).astype(float)
    cl = [[cen[i].astype(float), cnt[i]] for i in range(k) if cnt[i] > 0]
    # ΔE в «настоящем» Lab (в OpenCV L = L*·2,55, a,b сдвинуты на 128)
    real = lambda c: np.array([c[0] / 2.55, c[1] - 128, c[2] - 128])
    while True:
        best = None
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                d = float(np.linalg.norm(real(cl[i][0]) - real(cl[j][0])))
                if d < merge and (best is None or d < best[0]): best = (d, i, j)
        if not best: break
        _, i, j = best; n = cl[i][1] + cl[j][1]
        cl[i] = [(cl[i][0] * cl[i][1] + cl[j][0] * cl[j][1]) / n, n]; cl.pop(j)
    tot = sum(c[1] for c in cl)
    cl = [c for c in cl if c[1] / tot >= min_share]
    tot = sum(c[1] for c in cl)
    out = []
    for c, n in sorted(cl, key=lambda x: -x[1]):
        bgr = cv2.cvtColor(np.uint8([[np.clip(c, 0, 255)]]), cv2.COLOR_LAB2BGR)[0, 0]
        rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
        out.append({"hex": "#%02x%02x%02x" % tuple(rgb), "rgb": rgb, "lab": [round(x, 1) for x in real(c)], "share": round(n / tot, 4)})
    return out


def widths(shares, W):
    """доли → целые пиксели, сумма ровно W (наибольший остаток); у каждого цвета ≥ 1 px"""
    raw = [s * W for s in shares]
    px = [max(1, int(math.floor(x))) for x in raw]
    d = W - sum(px)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - math.floor(raw[i])))
    i = 0
    while d > 0: px[order[i % len(px)]] += 1; d -= 1; i += 1
    order = sorted(range(len(px)), key=lambda i: -px[i])
    i = 0
    while d < 0:
        j = order[i % len(px)]
        if px[j] > 1: px[j] -= 1; d += 1
        i += 1
    return px


def strip(cols, W=1024, H=1, blend=0):
    """полоска H×W (RGB). blend > 0 — мягкие переходы («растяжки») шириной до blend px на стыках"""
    W = max(len(cols), int(W)); H = max(1, int(H))
    px = widths([c["share"] for c in cols], W)
    row = np.zeros((W, 3), np.float32); x = 0; edges = []
    for c, n in zip(cols, px):
        row[x:x + n] = c["rgb"]; x += n; edges.append(x)
    if blend and len(cols) > 1:
        src = row.copy(); x0 = 0
        for i, e in enumerate(edges[:-1]):
            hw = int(min(blend, px[i] // 2, px[i + 1] // 2))
            if hw < 1: continue
            a, b = np.array(cols[i]["rgb"], np.float32), np.array(cols[i + 1]["rgb"], np.float32)
            for t in range(-hw, hw):
                f = (t + hw + .5) / (2 * hw); src[e + t] = a * (1 - f) + b * f
        row = src
    return np.repeat(row[None], H, axis=0).astype(np.uint8), px


def png(cols, W=1024, H=1, blend=0):
    a, _ = strip(cols, W, H, blend)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    return buf.tobytes()
