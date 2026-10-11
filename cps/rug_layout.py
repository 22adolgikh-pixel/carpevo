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


def _ring(lab, rect, d, nseg=32, per=10):
    """Средний цвет каждого из nseg участков «кольца» на глубине d от края ковра (кольцо = прямоугольник, отступивший на d)."""
    x, y, w, h = rect
    x0, y0, x1, y1 = x + d, y + d, x + w - 1 - d, y + h - 1 - d
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    P = 2 * ((x1 - x0) + (y1 - y0)); t = (np.arange(nseg * per) + 0.5) / (nseg * per) * P
    px = np.empty_like(t); py = np.empty_like(t); a, b = x1 - x0, y1 - y0
    for i, v in enumerate(t):
        if v < a: px[i], py[i] = x0 + v, y0
        elif v < a + b: px[i], py[i] = x1, y0 + (v - a)
        elif v < 2 * a + b: px[i], py[i] = x1 - (v - a - b), y1
        else: px[i], py[i] = x0, y1 - (v - 2 * a - b)
    s = lab[py.astype(int), px.astype(int)]                 # nseg*per × 3
    return s.reshape(nseg, per, 3).mean(1)


def detect(img):
    """→ {carpet:[x,y,w,h] в долях, bounds:[{ix, iy, strength}] — вложенные границы (отступы от края ковра в долях его ширины/высоты),
           rings:[zone…] (на одну больше, чем границ: последнее — поле), conf}

    Кайма — полоса, которая идёт вокруг всего ковра. Поэтому границу ищем не по одному месту, а по всему «кольцу»:
    кольцо на глубине d делим на 32 участка и смотрим, где цвет меняется одновременно на большинстве участков (медиана по участкам).
    Медальоны и узоры поля меняют цвет лишь местами — медиана их не пропускает. Дальше 30% короткой стороны каймы не ищем."""
    H0, W0 = img.shape[:2]; s = 700 / max(H0, W0)
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    h, w = a.shape[:2]
    rect, conf = _carpet_rect(a)
    x, y, cw, ch = rect
    lab = cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 1.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    short = min(cw, ch); dmax = max(10, int(short * 0.30))
    rings = []
    for d in range(dmax):
        r = _ring(lab, rect, d)
        if r is None: break
        rings.append(r)
    if len(rings) < 8: return {"carpet": [x / w, y / h, cw / w, ch / h], "bounds": [], "rings": ["field"], "conf": round(conf, 2)}
    Rg = np.stack(rings)                                       # depth × seg × 3
    L = len(Rg); k = max(2, int(short * 0.012))
    g = np.zeros(L)
    for d in range(k, L - k):
        ch_seg = np.linalg.norm(Rg[d:d + k].mean(0) - Rg[d - k:d].mean(0), axis=1)   # смена цвета на каждом участке кольца
        g[d] = np.percentile(ch_seg, 40)                     # «почти везде по кругу» (40-й перцентиль — терпим износ/фрагменты)
    base = np.median(g[k:L - k]) + 1e-6
    pk = [(d, g[d] / base) for d in range(k + 1, L - k) if g[d] >= g[max(0, d - k):d + k + 1].max() and g[d] > 2.2 * base and g[d] > 4.0]
    kept = []
    for d, v in sorted(pk, key=lambda t: -t[1]):
        if d > short * 0.008 and all(abs(d - e) > max(2 * k, short * 0.02) for e, _ in kept): kept.append((d, v))
    kept = sorted(kept[:6])
    bounds = [{"ix": d / cw, "iy": d / ch, "strength": round(float(v), 2)} for d, v in kept]
    edges = [0] + [d for d, _ in kept]
    widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    zr = []
    if widths:
        wmax = max(widths)
        for wd in widths: zr.append("border" if wd == wmax else ("guard" if wd < 0.35 * wmax else "minor"))
    zr.append("field")
    return {"carpet": [x / w, y / h, cw / w, ch / h], "bounds": bounds, "rings": zr, "conf": round(conf, 2)}
