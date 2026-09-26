# autogrid.py — автоматическая сетка и пиксели для рисунков на миллиметровке (Carpet Pattern Studio)
# Вход: серый кроп и рамка рисунка (x, y, w, h) в координатах кропа.
# Шаги: 1) шаг сетки по гармоникам профиля «тонких линий» (тушь рисунка исключена); 2) угол поворота перебором;
#       3) точные положения линий (под-пиксель, робастная прямая); 4) яркость каждой клетки;
#       5) тоны: фон / контур (тёмный) / заливка (средний серый) — пороги откалиброваны
#          по уже размеченным вручную рисункам (табл. 9–18, 63); 6) чистка шума.
import numpy as np, cv2

# ---------------------------------------------------------------- профили линий
def _line_response(g):
    g = g.astype(np.float32)
    bl = cv2.GaussianBlur(g, (0, 0), 2.2)
    return np.clip(bl - g, 0, None)          # тонкие тёмные линии > 0


def _profiles(hp, keep=None):
    """Средний отклик линий по столбцам/строкам только по «бумаге» (тушь рисунка исключена маской keep)."""
    if keep is None:
        return np.median(hp, axis=0), np.median(hp, axis=1)
    k = keep.astype(np.float32)
    sx = k.sum(0); sy = k.sum(1)
    px = (hp * k).sum(0) / np.maximum(sx, 1); py = (hp * k).sum(1) / np.maximum(sy, 1)
    # столбцы/строки почти целиком под тушью — берём медиану соседей
    mx, my = np.median(px), np.median(py)
    px[sx < 0.15 * k.shape[0]] = mx; py[sy < 0.15 * k.shape[1]] = my
    return px, py


def _paper_mask(reg):
    g = cv2.GaussianBlur(reg.astype(np.float32), (0, 0), 1.2)
    paper = float(np.percentile(g, 75))
    ink = g < paper * 0.62
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return ~ink


def _harm_period(pr, pmin=3.0, pmax=40.0, K=4):
    """Шаг по гармоникам: сильные k/p и слабые (k-½)/p. Кратные (2p, 5p) и половинные (p/2) шаги проигрывают."""
    z = pr - pr.mean(); n = len(z); i = np.arange(n)
    pmax = min(pmax, n / 3.0)
    ps = np.arange(pmin, max(pmax, pmin + 0.1), 0.02)
    def mag(f):                                  # f: (len(ps), K) циклов на пиксель
        ph = np.exp(-2j * np.pi * f[..., None] * i)
        return np.abs(ph @ z) / n
    k = np.arange(1, K + 1)[None, :]
    fpos = k / ps[:, None]; fneg = (k - 0.5) / ps[:, None]
    valid = fpos < 0.5
    vn = fneg < 0.5
    sp = (mag(np.minimum(fpos, 0.5)) * valid).sum(1) / np.maximum(1, valid.sum(1))
    sn = (mag(np.minimum(fneg, 0.5)) * vn).sum(1) / np.maximum(1, vn.sum(1))
    score = sp - sn
    return ps, score


def _pick_periods(prx, pry, pmin=3.0, pmax=30.0):
    """Общий шаг для обеих осей (клетки почти всегда квадратные); разный — только если это заметно лучше."""
    pmax = min(pmax, len(prx) / 5.0, len(pry) / 5.0)
    psx, sx = _harm_period(prx, pmin, pmax); psy, sy = _harm_period(pry, pmin, pmax)
    n = min(len(sx), len(sy)); ps = psx[:n]
    nx = sx[:n] / (sx[:n].max() + 1e-9); ny = sy[:n] / (sy[:n].max() + 1e-9)
    j = int(np.argmax(nx + ny)); psq = float(ps[j]); ssq = nx[j] + ny[j]
    jx, jy = int(np.argmax(nx)), int(np.argmax(ny))
    px, py = float(ps[jx]), float(ps[jy])
    ratio = max(px, py) / min(px, py)
    harmonic = abs(ratio - round(ratio)) < 0.08
    if not harmonic and 2.0 - ssq > 0.25 and ratio > 1.15:     # явно прямоугольные клетки (как в табл. 13)
        return px, py
    return psq, psq


def _comb(pr, p):
    i = np.arange(len(pr)); w = 2 * np.pi / p
    z = pr - pr.mean()
    re = (z * np.cos(w * i)).sum(); im = -(z * np.sin(w * i)).sum()
    return np.hypot(re, im) / len(pr), ((-np.arctan2(im, re) / (2 * np.pi)) * p) % p


def _refine_period(pr, p):
    best = (-1, p, 0)
    for q in np.arange(p * 0.93, p * 1.07, 0.004):
        s, ph = _comb(pr, q)
        if s > best[0]: best = (s, q, ph)
    return best[1], best[2], best[0]


def _fit_lines(pr, p, ph):
    """Уточняет каждую линию по локальному максимуму профиля и подгоняет прямую pos = a + p*k (робастно)."""
    n = len(pr); ks, pos = [], []
    k0 = int(np.floor(-ph / p)); k1 = int(np.ceil((n - ph) / p))
    for k in range(k0, k1 + 1):
        c = ph + k * p
        lo, hi = int(round(c - p * 0.3)), int(round(c + p * 0.3))
        if lo < 1 or hi > n - 2: continue
        seg = pr[lo:hi + 1]; j = int(np.argmax(seg)) + lo
        if 0 < j < n - 1:
            a, b, cc = pr[j - 1], pr[j], pr[j + 1]; d = a - 2 * b + cc
            off = 0.5 * (a - cc) / d if d else 0
            ks.append(k); pos.append(j + float(np.clip(off, -0.5, 0.5)))
    ks, pos = np.array(ks, float), np.array(pos, float)
    if len(ks) < 3: return p, ph
    wts = np.ones_like(ks)
    for _ in range(6):                          # IRLS (Хьюбер)
        A = np.stack([np.ones_like(ks), ks], 1) * wts[:, None]
        sol, *_ = np.linalg.lstsq(A, pos * wts, rcond=None)
        r = pos - (sol[0] + sol[1] * ks)
        s = max(0.15, 1.4826 * np.median(np.abs(r)))
        wts = np.where(np.abs(r) <= 1.5 * s, 1.0, 1.5 * s / np.abs(r))
    a, pp = sol
    return float(pp), float(a % pp)


def _rotate(img, ang, center):
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), M


def detect_grid(g, frame, square_hint=None):
    """Возвращает dict: angle, px, py (шаг), ox, oy (фаза в повёрнутой рамке), cols, rows, corners (4 угла блока клеток в кропе)."""
    x, y, w, h = [int(round(v)) for v in frame]
    ins = 2
    reg = g[y + ins:y + h - ins, x + ins:x + w - ins]
    hp = _line_response(reg)
    keep = _paper_mask(reg).astype(np.uint8)
    cx, cy = reg.shape[1] / 2, reg.shape[0] / 2
    prx, pry = _profiles(hp, keep)
    p0x, p0y = _pick_periods(prx, pry)
    # угол: максимум суммарной «гребёнки» по обеим осям
    best = (-1, 0)
    for ang in np.arange(-3, 3.001, 0.2):
        r, _ = _rotate(hp, ang, (cx, cy)); kr, _ = _rotate(keep, ang, (cx, cy)); a, b = _profiles(r, kr)
        s = _comb(a, p0x)[0] + _comb(b, p0y)[0]
        if s > best[0]: best = (s, ang)
    for ang in np.arange(best[1] - 0.2, best[1] + 0.2001, 0.04):
        r, _ = _rotate(hp, ang, (cx, cy)); kr, _ = _rotate(keep, ang, (cx, cy)); a, b = _profiles(r, kr)
        s = _comb(a, p0x)[0] + _comb(b, p0y)[0]
        if s > best[0]: best = (s, ang)
    ang = float(best[1])
    r, M = _rotate(hp, ang, (cx, cy)); kr, _ = _rotate(keep, ang, (cx, cy)); a, b = _profiles(r, kr)
    px, phx, sx = _refine_period(a, p0x); py, phy, sy = _refine_period(b, p0y)
    if square_hint or (square_hint is None and abs(px - py) / ((px + py) / 2) < 0.02):
        pm = (px + py) / 2; px = py = pm
        phx = _comb(a, pm)[1]; phy = _comb(b, pm)[1]
    px, phx = _fit_lines(a, px, phx); py, phy = _fit_lines(b, py, phy)
    # блок целых клеток внутри рамки (в повёрнутых координатах области)
    W, H = r.shape[1], r.shape[0]
    cols = int(np.floor((W - phx) / px + 0.02)); rows = int(np.floor((H - phy) / py + 0.02))
    Minv = cv2.invertAffineTransform(M)
    to_crop = lambda P: (P @ Minv[:, :2].T) + Minv[:, 2] + [x + ins, y + ins]
    U, V = np.meshgrid(phx + px * np.arange(cols + 1), phy + py * np.arange(rows + 1))
    nodes = to_crop(np.stack([U, V], -1).reshape(-1, 2)).reshape(rows + 1, cols + 1, 2)
    centers = (nodes[:-1, :-1] + nodes[1:, :-1] + nodes[:-1, 1:] + nodes[1:, 1:]) / 4
    # лучший «четырёхугольник» (перспектива) для CPS: гомография идеальной сетки → найденные узлы
    ideal = np.stack(np.meshgrid(np.arange(cols + 1), np.arange(rows + 1)), -1).reshape(-1, 2).astype(np.float32)
    Hm, _ = cv2.findHomography(ideal, nodes.reshape(-1, 2).astype(np.float32), 0)
    cq = cv2.perspectiveTransform(np.array([[[0, 0], [cols, 0], [cols, rows], [0, rows]]], np.float32), Hm)[0]
    fit = cv2.perspectiveTransform(ideal[None], Hm)[0] - nodes.reshape(-1, 2)
    return {"angle": ang, "px": px, "py": py, "ox": phx, "oy": phy, "cols": cols, "rows": rows,
            "corners": cq.tolist(), "centers": centers, "nodes": nodes,
            "quad_resid": float(np.percentile(np.hypot(*fit.T), 95) / min(px, py)), "strength": float(sx + sy)}


def cell_centers(G):
    """Центры клеток блока (rows×cols×2) в координатах кропа."""
    if G.get("centers") is not None: return G["centers"]
    c = np.array(G["corners"]); cols, rows = G["cols"], G["rows"]
    u = (np.arange(cols) + 0.5) / cols; v = (np.arange(rows) + 0.5) / rows
    U, Vv = np.meshgrid(u, v)
    P = (c[0] * ((1 - U) * (1 - Vv))[..., None] + c[1] * (U * (1 - Vv))[..., None]
         + c[2] * (U * Vv)[..., None] + c[3] * ((1 - U) * Vv)[..., None])
    return P


def cell_means(g, C, pitch, inner=0.5):
    gf = cv2.GaussianBlur(g.astype(np.float32), (0, 0), max(0.3, pitch * inner / 3.5))
    r = pitch * inner / 2; acc = np.zeros(C.shape[:2], np.float32)
    for dy in (-r, 0, r):
        for dx in (-r, 0, r):
            acc += cv2.remap(gf, (C[..., 0] + dx).astype(np.float32), (C[..., 1] + dy).astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return acc / 9


# ---------------------------------------------------------------- тоны
DARK_T = 0.40        # ниже — контур (калибровка по ручной разметке: тёмные p99 ≈ 0.35)
MID_CAND = 0.88      # кандидаты в заливку; окончательно решают области (см. clean)
MID_STRIP = 0.78     # узкая полоса вдоль контура остаётся заливкой, только если заметно серее
MID_FILL = 0.845     # внутренняя часть области заливки (клетки не у контура)
BLEED = 0.045        # поправка порога на каждого тёмного соседа (ореол от размытия)

def classify(v):
    """v — яркости клеток. Возвращает (M, info). 0 фон, 1 тёмный контур, 2 средний серый (кандидаты)."""
    flat = v.ravel()
    hist, edges = np.histogram(flat, bins=64, range=(0, 256))
    hs = np.convolve(hist, [1, 2, 3, 2, 1], 'same')
    top = int(np.argmax(hs[16:])) + 16                  # фон — самый массовый светлый пик
    bg = float(np.median(flat[np.abs(flat - (edges[top] + 2)) < 14]))
    darkish = flat[flat < bg * 0.62]
    dk = float(np.percentile(darkish, 25)) if len(darkish) >= 3 else bg * 0.45
    span = max(bg - dk, 25.0)
    t = (v - dk) / span
    M = np.zeros(v.shape, np.uint8)
    M[t < DARK_T] = 1
    M[(t >= DARK_T) & (t < MID_CAND)] = 2
    return M, {"bg": bg, "dark": dk, "t": t}


SOLID_MIN = 8        # v6: минимальная площадь сплошного серого пятна без контура
SOLID_T = 0.80       # v6: …и насколько оно должно быть серее фона
SOLID_STD = 0.15     # v6: пятно ровное (тени и грязь бумаги — пёстрые)
SOLID_EDGE = 0.02    # v6: тени и грязь по краям кадра прилегают к краю матрицы, настоящая заливка — нет
SOLID_CORE = 0.25    # v6: доля «толстой» части (после эрозии) — отсекает ореолы вдоль линий
REGION_T = 0.86      # замкнутая контуром область — заливка, если её типичная яркость ниже

def clean(M, t):
    """Заливка — это области внутри контура.
    1) Каждая замкнутая тёмным контуром область решается целиком: по медиане яркости (с поправкой на ореол у линий).
    2) Во внешней (незамкнутой) области серое остаётся, только если это явная заливка, прижатая к контуру.
    Серый ореол вдоль линий, пятна и тени бумаги — в фон."""
    M = M.copy()
    dark = (M == 1).astype(np.uint8)
    H, W = M.shape
    nd4 = cv2.filter2D(dark.astype(np.float32), -1, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32),
                       borderType=cv2.BORDER_CONSTANT)
    tc = t + BLEED * nd4                                    # яркость «как без соседних линий»
    near = cv2.dilate(dark, np.ones((3, 3), np.uint8)) > 0
    k4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((1 - dark).astype(np.uint8), connectivity=4)
    outside = np.zeros_like(dark, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]; comp = lab == i
        if x == 0 or y == 0 or x + w == W or y + h == H:
            outside |= comp; continue
        inner = comp & ~near
        tm = float(np.median(tc[inner])) if inner.sum() >= 3 else float(np.median(tc[comp]))
        M[comp] = 2 if tm < REGION_T else 0
    # внешняя область: только явные серые пятна, прижатые к контуру
    cand = outside & (tc < MID_CAND - 0.03)
    M[outside] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=4)
    for i in range(1, n):
        x, y, w, h, area = stats[i]; comp = lab == i
        ring = (cv2.dilate(comp.astype(np.uint8), k4) > 0) & ~comp
        edge_cells = int(comp[0].sum() + comp[-1].sum() + comp[:, 0].sum() + comp[:, -1].sum())
        encl = float(dark[ring].sum()) / max(1, int(ring.sum()) + edge_cells)
        inner = comp & ~near
        tm = float(np.median(tc[inner])) if inner.sum() >= 2 else float(np.median(tc[comp]))
        if area >= 3 and encl >= 0.45 and tm < MID_FILL: M[comp] = 2
        # v6: сплошная серая фигура без тёмного контура (напр. табл. 4) — толстое (не ореол линии) и заметно серое пятно
        elif area >= SOLID_MIN and tm < SOLID_T and edge_cells <= SOLID_EDGE * area and float(np.std(tc[comp])) < SOLID_STD:
            core = cv2.erode(comp.astype(np.uint8), k4, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            if core.sum() >= max(2, SOLID_CORE * area) and float(near[comp].mean()) < 0.8: M[comp] = 2
    return M


def trim(M, margin=1):
    ys, xs = np.nonzero(M)
    if not len(xs): return M, (0, 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
    x1 = min(M.shape[1] - 1, x1 + margin); y1 = min(M.shape[0] - 1, y1 + margin)
    return M[y0:y1 + 1, x0:x1 + 1], (int(x0), int(y0))


def auto_figure(g, frame, dark_only=False):
    """dark_only=True (по умолчанию): только тёмный контур, без попытки угадать серую заливку —
    заливка чаще даёт «мусор», чем пользу, и её проще дорисовать вручную."""
    G = detect_grid(g, frame)
    C = cell_centers(G)
    v = cell_means(g, C, min(G["px"], G["py"]))
    M, info = classify(v)
    if dark_only: M[M == 2] = 0
    else: M = clean(M, info["t"])
    info["confidence"], info["flags"] = confidence(G, M, info["t"])
    return G, C, M, info


def confidence(G, M, t):
    """Грубая оценка: доля «полутоновых» клеток среди закрашенных (сдвиг сетки / размытие) + проверки сетки."""
    flags = []
    ink = t < 0.85
    amb = ((t > 0.25) & (t < 0.6)).sum() / max(1, ink.sum())
    if amb > 0.12: flags.append("много полутонов — проверьте сетку")
    if abs(G["px"] - G["py"]) / min(G["px"], G["py"]) > 0.05: flags.append("клетки не квадратные")
    if abs(G["angle"]) >= 2.5: flags.append("большой наклон")
    if G["quad_resid"] > 0.25: flags.append("сетка неровная")
    if (M > 0).sum() < 3: flags.append("рисунок не найден")
    if (M == 2).any(): flags.append("серую заливку стоит проверить")
    conf = float(np.clip(1 - 2.5 * amb, 0, 1))
    return round(conf, 2), flags


def quad_size(q):
    """Как в интерфейсе (index.html quadSize): размер выпрямленного кадра."""
    d = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    return (max(1, round(max(d(q[0], q[1]), d(q[3], q[2])))), max(1, round(max(d(q[0], q[3]), d(q[1], q[2])))))


def to_work(G, M, info, margin=1):
    """Поля work-файла CPS: четырёхугольник = углы блока клеток, сетка ровно по нему, матрица с полями margin."""
    q = [[round(float(x), 2), round(float(y), 2)] for x, y in G["corners"]]
    W, H = quad_size(q); cols, rows = G["cols"], G["rows"]
    Mt, (ox, oy) = trim(M, margin)
    rows_s = ["".join("." if v == 0 else str(int(v)) for v in r) for r in Mt]
    has_mid = bool((Mt == 2).any())
    span = max(info["bg"] - info["dark"], 25.0)
    return {
        "quad": q,
        "grid": {"pw": round(W / cols, 4), "ph": round(H / rows, 4), "ox": 0, "oy": 0, "cols": cols, "rows": rows,
                 "square": abs(G["px"] - G["py"]) / min(G["px"], G["py"]) < 0.03},
        "classify": {"tones": 1, "inner": 55, "thr": [int(round(info["dark"] + DARK_T * span))],
                     "centers": [int(round(info["bg"])), int(round(info["dark"]))]},
        "matrix": {"w": int(Mt.shape[1]), "h": int(Mt.shape[0]), "origin": [int(ox), int(oy)], "rows": rows_s},
        "palette": ["#ffffff", "#1a1a1a", "#b8b8b8"] if has_mid else ["#ffffff", "#1a1a1a"],
        "palette_names": ["фон", "обводка", "тело 1"] if has_mid else ["фон", "обводка"],
        "palette_touched": False,
        "auto": {"version": 1, "confidence": info["confidence"], "flags": info["flags"],
                 "pitch": [round(G["px"], 3), round(G["py"], 3)], "angle": round(G["angle"], 2),
                 "reviewed": False},
    }
