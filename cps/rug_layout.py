# rug_layout.py — автоматическая разметка зон ковра на фото: край ковра, полосы кайм, центральное поле (v10.21).
#
# 1) Край ковра: фон фото (серый/белый стол) отличается от ковра — берём цвет по краям кадра, всё сильно отличное — ковёр,
#    крупнейшая связная область → прямоугольник (бахрома по коротким сторонам режется: её цвет близок к фону, а полосы узкие).
# 2) Каймы: от каждой стороны ковра внутрь считаем «профиль» — средний цвет линии на расстоянии d (только средние 60% стороны,
#    чтобы не мешали углы). Граница полосы — резкая смена этого профиля. Каймы на коврах обычно одинаковой ширины со всех сторон,
#    поэтому силу смены усредняем по 4 сторонам (в долях короткой стороны) и берём пики, которые есть хотя бы на 3 сторонах.
# 3) Поле — прямоугольник внутри самой глубокой уверенной границы; самая широкая полоса — главная кайма, узкие — малые/бордюрные.
# Результат — предложение: человек двигает границы и подтверждает.
import cv2, numpy as np


def _carpet_rect(a):
    h, w = a.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 2), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = max(4, int(min(h, w) * 0.02))
    edge = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3), lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    bg = np.median(edge, 0); spread = np.percentile(np.linalg.norm(edge - bg, axis=1), 80)
    d = np.linalg.norm(lab - bg, axis=2)
    fg = (d > max(18, spread * 2.5)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2: return (0, 0, w, h), 0.0
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); M = lb == i
    rows = np.where(M.mean(1) > 0.5)[0]; cols = np.where(M.mean(0) > 0.5)[0]   # строки/столбцы, где ковёр — больше половины (бахрома отпадает)
    if len(rows) < 10 or len(cols) < 10: return (0, 0, w, h), 0.0
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    conf = float(M[y0:y1, x0:x1].mean())
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), conf


def _profiles(lab, rect, depth):
    x, y, w, h = rect; out = []
    for side in ("l", "r", "t", "b"):
        if side in "lr":
            lo, hi = y + int(h * 0.35), y + int(h * 0.65)      # узкая середина стороны: широкие каймы соседних сторон сюда не попадают
            band = lab[lo:hi, x:x + depth] if side == "l" else lab[lo:hi, x + w - depth:x + w][:, ::-1]
            p = band.mean(0)                                  # depth × 3
        else:
            lo, hi = x + int(w * 0.42), x + int(w * 0.58)
            band = lab[y:y + depth, lo:hi] if side == "t" else lab[y + h - depth:y + h, lo:hi][::-1]
            p = band.mean(1)
        out.append(p)
    return out


def detect(img):
    """→ {carpet:[x,y,w,h] в долях, bounds:[{ix, iy, strength}] — вложенные границы (отступы от края ковра в долях его ширины/высоты),
           rings:[zone…] (на одну больше, чем границ: последнее — поле), conf}"""
    H0, W0 = img.shape[:2]; s = 700 / max(H0, W0)
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    h, w = a.shape[:2]
    rect, conf = _carpet_rect(a)
    x, y, cw, ch = rect
    lab = cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 1.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    short = min(cw, ch); depth = max(10, int(short * 0.42))
    P = _profiles(lab, rect, depth)
    L = min(len(p) for p in P)
    # сила смены цвета на каждой стороне: разница среднего цвета «до» и «после» точки (окно k); пики каждой стороны отдельно,
    # потом ищем расстояния, где пик есть хотя бы на 3 сторонах из 4 (каймы одной ширины со всех сторон)
    k = max(2, int(short * 0.012)); tol = max(3, int(short * 0.012))
    side_peaks = []
    for p in P:
        p = p[:L]; g = np.zeros(L)
        for d in range(k, L - k): g[d] = np.linalg.norm(p[d:d + k].mean(0) - p[d - k:d].mean(0))
        base = np.median(g[k:L - k]) + 1e-6; pk = []
        for d in range(k + 1, L - k):
            if g[d] >= g[max(0, d - k):d + k + 1].max() and g[d] > 2.0 * base: pk.append((d, g[d] / base))
        side_peaks.append(pk)
    cands = sorted({d for pk in side_peaks for d, _ in pk})
    clusters = []
    for d in cands:
        hits = []
        for pk in side_peaks:
            near = [v for e, v in pk if abs(e - d) <= tol]
            if near: hits.append(max(near))
        if len(hits) >= 3: clusters.append((d, float(np.mean(hits)), len(hits)))
    kept = []
    for d, v, n in sorted(clusters, key=lambda t: (-t[2], -t[1])):
        if d > short * 0.008 and all(abs(d - e) > max(tol * 2, short * 0.02) for e, _ in kept): kept.append((d, v))
    kept = sorted(kept[:7])
    bounds = [{"ix": d / cw, "iy": d / ch, "strength": round(v, 2)} for d, v in kept]
    # зоны колец: самое широкое кольцо — главная кайма, узкие (<35% главной) — бордюрные полоски, остальные — малые каймы
    edges = [0] + [d for d, _ in kept]
    widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    rings = []
    if widths:
        wmax = max(widths)
        for wd in widths: rings.append("border" if wd == wmax else ("guard" if wd < 0.35 * wmax else "minor"))
    rings.append("field")
    return {"carpet": [x / w, y / h, cw / w, ch / h], "bounds": bounds, "rings": rings, "conf": round(conf, 2)}
