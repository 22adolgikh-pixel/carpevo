# autocolor.py — автооцифровка ЦВЕТНЫХ схем на миллиметровке (Carpet Pattern Studio, режим «⚡ цвет»)
#
# Чем отличается от autogrid.py (ЧБ):
#  • шаг сетки ищется не по тонким линиям бумаги (в цветной печати они еле видны), а по границам
#    цветных пятен: в узоре на сетке любая смена цвета проходит по линии сетки, поэтому положения
#    границ строго периодичны. Период ищется тестом Рэлея (когерентность фаз exp(2πi·x/p)) —
#    кратные и дробные шаги проигрывают сами (замер: табл. 164 — R=0.82 на верном шаге против 0.44 на половинном);
#  • каждая клетка получает цвет (медиана Lab в центре клетки), цвета кластеризуются в палитру ≤ 10
#    и получают название по словарю красителей (az/ru) — человек подтверждает в студии;
#  • фон = самый частый цвет на краю рисунка → индекс 0 (в матрице «.»).
# Формат результата — тот же work-файл, что и у ЧБ (quad/grid/matrix/palette/palette_names/auto), так что
# весь остальной интерфейс (правка пикселей, экспорт PNG/SVG, сайт) работает без изменений.
import numpy as np, cv2
import autogrid

PHOTO_R = 0.30         # когерентность границ ниже — сетки нет (фото ковра, а не схема)
GOOD_R = 0.60          # выше — шаг надёжный, им можно подсказывать узким полосам того же листа

# ---------------------------------------------------------------- словарь красителей
# опорные цвета — как они выглядят в печати книги (выцветшие), не «идеальные» красители
def _load_dyes():
    """Библиотека красителей — файл dyes.json рядом (общая для студии и автооцифровки)."""
    import json, os
    try:
        d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dyes.json"), encoding="utf-8"))["dyes"]
        return [(x["az"], x["ru"], x["hex"], x["id"]) for x in d]
    except Exception:
        return [("qırmızı", "красный", "#b8402f", "qirmizi"), ("qara", "чёрный", "#1e1c1f", "qara"), ("ağ", "белый", "#f3efe4", "ag")]


DYES = _load_dyes()


def _hex2lab(h):
    rgb = np.array([[[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)]]], np.uint8)   # BGR
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _lab8_to_ref(lab8):
    """OpenCV 8-бит Lab → L 0..100, a/b −128..127 (для честного ΔE)."""
    lab8 = np.asarray(lab8, np.float32)
    return np.stack([lab8[..., 0] * 100 / 255, lab8[..., 1] - 128, lab8[..., 2] - 128], -1)


_DYE_LAB = _lab8_to_ref(np.array([_hex2lab(h) for _, _, h, _ in DYES]))


def dye_name(lab8):
    d = np.linalg.norm(_DYE_LAB - _lab8_to_ref(lab8), axis=1)
    i = int(np.argmin(d))
    return DYES[i][0], DYES[i][1], float(d[i]), DYES[i][3]


def _lab2hex(lab8):
    bgr = cv2.cvtColor(np.clip(np.array([[lab8]]), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
    return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


# ---------------------------------------------------------------- признак «цветной рисунок»
def is_colorful(bgr, frame=None):
    """Доля заметно окрашенных пикселей: ЧБ-схемы (серая бумага, чёрная тушь) дают почти 0."""
    if frame is not None:
        x, y, w, h = [int(round(v)) for v in frame]
        bgr = bgr[max(0, y):y + h, max(0, x):x + w]
    lab = _lab8_to_ref(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return float((chroma > 12).mean()) > 0.08          # замер: ЧБ-листы 0.00, цветные 0.26–0.84


# ---------------------------------------------------------------- шаг сетки по границам цвета
def _edge_positions(lab, axis):
    """Под-пиксельные положения границ цвета вдоль оси (axis=1 — по x, границы вертикальные)."""
    d = np.linalg.norm(np.diff(lab, axis=axis), axis=2)          # граница между пикселями i и i+1 → позиция i+1
    thr = max(12.0, float(np.percentile(d, 97)) * 0.35)
    if axis == 1:
        a, b, c = d[:, :-2], d[:, 1:-1], d[:, 2:]
    else:
        a, b, c = d[:-2], d[1:-1], d[2:]
    m = (b > thr) & (b >= a) & (b > c)
    den = a - 2 * b + c
    off = np.where(den != 0, 0.5 * (a - c) / np.where(den != 0, den, 1), 0)
    idx = np.nonzero(m)
    base = (idx[1] if axis == 1 else idx[0]).astype(np.float32) + 1 + 1   # +1 (срез) +1 (diff)
    w = b[m]
    return base + np.clip(off[m], -0.5, 0.5), w


def _rayleigh(pos, w, ps):
    """R(p) = |Σ w·exp(2πi·x/p)| / Σ w  для набора p; ещё и фаза."""
    if len(pos) == 0: return np.zeros(len(ps)), np.zeros(len(ps))
    ph = np.exp(2j * np.pi * pos[None, :] / ps[:, None]) @ w / max(w.sum(), 1e-9)
    return np.abs(ph), np.angle(ph)


def _period(pos, w, pmin, pmax):
    if len(pos) > 20000:                                       # быстрее и не хуже
        sel = np.random.default_rng(0).choice(len(pos), 20000, replace=False); pos, w = pos[sel], w[sel]
    ps = np.arange(pmin, pmax, 0.02)
    R, _ = _rayleigh(pos, w, ps)
    j = int(np.argmax(R)); p = float(ps[j])
    # защита от «половинного» шага: вдвое больший шаг почти так же когерентен → берём его
    R2, _ = _rayleigh(pos, w, np.array([2 * p]))
    if 2 * p < pmax and R2[0] > 0.92 * R[j]: p = 2 * p
    ps = np.arange(p * 0.985, p * 1.015, 0.002)
    R, A = _rayleigh(pos, w, ps); j = int(np.argmax(R))
    p = float(ps[j]); phase = float((A[j] / (2 * np.pi) * p) % p)
    return p, phase, float(R[j])


def _rot(img, ang, center):
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE), M


def detect_grid_color(bgr, frame, pmin=4.5, pmax=40.0, pitch_hint=None):
    """Сетка цветного рисунка. Возвращает тот же dict, что autogrid.detect_grid (+ R — когерентность).
    pitch_hint=(px, py) — шаг, уверенно найденный на других рисунках того же листа: у узких полос (каймы)
    границ вдоль короткой стороны мало, и без подсказки шаг по ней ненадёжен."""
    x, y, w, h = [int(round(v)) for v in frame]
    ins = 3
    reg = bgr[y + ins:y + h - ins, x + ins:x + w - ins]
    lab = cv2.cvtColor(cv2.GaussianBlur(reg, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    cx, cy = reg.shape[1] / 2, reg.shape[0] / 2
    pmax = min(pmax, reg.shape[1] / 4, reg.shape[0] / 4)

    def est(ang, px=None, py=None):
        r = lab if ang == 0 else _rot(lab, ang, (cx, cy))[0]
        m = 6                                                   # края после поворота — мусор
        r = r[m:-m, m:-m]
        xs, wx = _edge_positions(r, 1); ys, wy = _edge_positions(r, 0)
        xs, ys = xs + m, ys + m
        if px is None:
            if pitch_hint:
                hx, hy = pitch_hint
                return _period(xs, wx, hx * 0.96, hx * 1.04), _period(ys, wy, hy * 0.96, hy * 1.04)
            return _period(xs, wx, pmin, pmax), _period(ys, wy, pmin, pmax)
        rx = _rayleigh(xs, wx, np.array([px]))[0][0]; ry = _rayleigh(ys, wy, np.array([py]))[0][0]
        return rx + ry

    (px, _, _), (py, _, _) = est(0)
    if abs(px - py) / min(px, py) < 0.04: px = py = (px + py) / 2          # клетки почти всегда квадратные
    best = (-1, 0.0)
    for ang in np.arange(-1.5, 1.501, 0.25):
        s = est(ang, px, py)
        if s > best[0]: best = (s, ang)
    for ang in np.arange(best[1] - 0.2, best[1] + 0.2001, 0.05):
        s = est(ang, px, py)
        if s > best[0]: best = (s, ang)
    ang = float(best[1])
    (px, phx, Rx), (py, phy, Ry) = est(ang)
    if abs(px - py) / min(px, py) < 0.04:
        pm = (px + py) / 2
        r = _rot(lab, ang, (cx, cy))[0][6:-6, 6:-6]
        xs, wx = _edge_positions(r, 1); ys, wy = _edge_positions(r, 0)
        Rx, Ax = _rayleigh(xs + 6, wx, np.array([pm])); Ry, Ay = _rayleigh(ys + 6, wy, np.array([pm]))
        px = py = pm; phx = float((Ax[0] / (2 * np.pi) * pm) % pm); phy = float((Ay[0] / (2 * np.pi) * pm) % pm)
        Rx, Ry = float(Rx[0]), float(Ry[0])
    # первая линия: крайнюю неполную клетку берём, только если в рамку попадает ≥ 75% её
    # (иначе в неё попадает бумага за краем рисунка → лишний «цвет» по краю)
    W, H = reg.shape[1], reg.shape[0]
    ox = phx - px if phx > 0.75 * px else phx
    oy = phy - py if phy > 0.75 * py else phy
    cols = int(np.floor((W - ox) / px + 0.25)); rows = int(np.floor((H - oy) / py + 0.25))
    _, M = _rot(lab[..., 0], ang, (cx, cy))
    Minv = cv2.invertAffineTransform(M)
    to_crop = lambda P: (P @ Minv[:, :2].T) + Minv[:, 2] + [x + ins, y + ins]
    U, V = np.meshgrid(ox + px * np.arange(cols + 1), oy + py * np.arange(rows + 1))
    nodes = to_crop(np.stack([U, V], -1).reshape(-1, 2)).reshape(rows + 1, cols + 1, 2)
    centers = (nodes[:-1, :-1] + nodes[1:, :-1] + nodes[:-1, 1:] + nodes[1:, 1:]) / 4
    cq = [nodes[0, 0], nodes[0, -1], nodes[-1, -1], nodes[-1, 0]]
    return {"angle": ang, "px": float(px), "py": float(py), "ox": float(ox), "oy": float(oy), "cols": cols, "rows": rows,
            "corners": [list(map(float, c)) for c in cq], "centers": centers, "nodes": nodes,
            "quad_resid": 0.0, "strength": float(Rx + Ry), "R": [round(float(Rx), 3), round(float(Ry), 3)]}


# ---------------------------------------------------------------- цвет клеток и палитра
def cell_colors(bgr, C, pitch, inner=0.45):
    """Цвет каждой клетки: медиана Lab по 3×3 точкам в центральной части клетки (линии сетки не попадают)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), max(0.4, pitch * inner / 4)), cv2.COLOR_BGR2LAB).astype(np.float32)
    r = pitch * inner / 2; samples = []
    for dy in (-r, 0, r):
        for dx in (-r, 0, r):
            mx = (C[..., 0] + dx).astype(np.float32); my = (C[..., 1] + dy).astype(np.float32)
            samples.append(np.stack([cv2.remap(lab[..., k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                                     for k in range(3)], -1))
    return np.median(np.stack(samples, 0), 0)                   # rows×cols×3 (Lab 8-бит OpenCV)


MERGE_DE = 10.0        # цвета ближе этого (ΔE76) — один цвет
MIN_SHARE = 0.002      # цвет, которого меньше 0.2% клеток (и < 4 клеток) — шум, растворяется в ближайшем
MAX_COLORS = 12
CHROMA_W = 1.3         # вес a/b относительно L при кластеризации


def palette_kmeans(V, ncolors=None):
    """V — N×3 Lab (OpenCV). k-means с запасом → слияние близких → удаление редких.
    ncolors — сколько цветов оставить (задаёт человек в студии, если авто разделило один цвет на два)."""
    # оттенок важнее светлоты: тёмно-зелёный и тёмно-синий в выцветшей печати почти одной яркости
    Wt = np.array([1.0, CHROMA_W, CHROMA_W], np.float32)
    X = _lab8_to_ref(V).astype(np.float32) * Wt
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    # 1) фон отдельно: он часто занимает больше половины клеток и «съедает» центры кластеров
    _, l0, c0 = cv2.kmeans(X, min(4, max(2, len(X) // 3)), None, crit, 3, cv2.KMEANS_PP_CENTERS)
    l0 = l0.ravel(); big = int(np.bincount(l0).argmax()); cen, cnt = [], []
    if (l0 == big).mean() > 0.35:
        bgc = X[l0 == big].mean(0); near = np.linalg.norm(X - bgc, axis=1) < MERGE_DE
        cen.append(X[near].mean(0)); cnt.append(int(near.sum())); rest = X[~near]
    else:
        rest = X
    # 2) остальные цвета
    K = min(16, max(2, len(rest) // 3))
    if len(rest) >= K:
        _, lab, cc = cv2.kmeans(rest, K, None, crit, 4, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel(); cen += list(cc); cnt += [int((lab == k).sum()) for k in range(K)]

    def merge_once(limit):
        best = None
        for i in range(len(cen)):
            for j in range(i + 1, len(cen)):
                d = float(np.linalg.norm(cen[i] - cen[j]))
                if d < limit and (best is None or d < best[0]): best = (d, i, j)
        if not best: return False
        _, i, j = best; n = cnt[i] + cnt[j]
        cen[i] = (cen[i] * cnt[i] + cen[j] * cnt[j]) / max(n, 1); cnt[i] = n
        del cen[j]; del cnt[j]
        return True

    while merge_once(MERGE_DE): pass
    while len(cen) > min(MAX_COLORS, ncolors or MAX_COLORS) and merge_once(1e9): pass
    C = np.array(cen)
    for _ in range(2):                                          # переназначение и чистка редких
        d = np.linalg.norm(X[:, None, :] - C[None], axis=2); a = d.argmin(1)
        keep = [k for k in range(len(C)) if (a == k).sum() >= max(4, MIN_SHARE * len(X))]
        C = np.array([X[a == k].mean(0) for k in keep])
    d = np.linalg.norm(X[:, None, :] - C[None], axis=2); a = d.argmin(1)
    s = np.sort(d, 1)
    ambiguous = (s[:, 1] - s[:, 0]) < 4.0 if C.shape[0] > 1 else np.zeros(len(X), bool)
    return a, C / Wt, ambiguous


SCATTER_DE = 17.0      # «рассыпанный» цвет (не образует пятен) сливается с соседним, если ближе этого
COMPACT_MIN = 0.45     # доля клеток цвета, у которых ≥2 из 4 соседей того же цвета; ниже — это шум печати


def _compactness(M, k):
    m = (M == k).astype(np.uint8)
    if m.sum() == 0: return 1.0
    nb = cv2.filter2D(m, -1, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8), borderType=cv2.BORDER_CONSTANT)
    return float(((nb >= 2) & (m > 0)).sum() / m.sum())


def merge_scattered(M, V, C):
    """Два близких оттенка, один из которых рассыпан по пятнам другого (полосы печати, выцветание) —
    это один цвет. Настоящие отдельные цвета образуют сплошные фигуры и не сливаются."""
    X = _lab8_to_ref(V)
    while len(C) > 2:
        best = None
        for k in range(len(C)):
            cp = _compactness(M, k)
            if cp >= COMPACT_MIN: continue
            d = np.linalg.norm(C - C[k], axis=1); d[k] = 1e9
            # сосед по месту: с каким цветом этот чаще всего граничит
            m = (M == k).astype(np.uint8)
            ring = (cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0) & (m == 0)
            if not ring.any(): continue
            touch = np.bincount(M[ring], minlength=len(C)).astype(float); touch[k] = 0
            j = int(np.argmax(touch - 1e3 * (d > SCATTER_DE)))
            if d[j] <= SCATTER_DE and (best is None or cp < best[0]): best = (cp, k, j)
        if not best: break
        _, k, j = best
        M = np.where(M == k, j, M)
        C[j] = X[M == j].mean(0)
        keep = [i for i in range(len(C)) if i != k]
        remap = np.full(len(C), -1); remap[keep] = np.arange(len(keep))
        M = remap[M]; C = C[keep]
    return M, C


def despeckle(M, V, C):
    """Одиночная клетка, не похожая ни на одного из 8 соседей, — чаще всего пятно печати/JPEG.
    Перекрашиваем её в цвет большинства соседей, только если измеренный цвет к нему близок (ΔE < 25)."""
    H, W = M.shape; out = M.copy(); X = _lab8_to_ref(V); n = 0
    for j in range(H):
        for i in range(W):
            nb = [M[jj, ii] for jj in range(max(0, j - 1), min(H, j + 2)) for ii in range(max(0, i - 1), min(W, i + 2))
                  if (jj, ii) != (j, i)]
            if len(nb) < 5 or M[j, i] in nb: continue
            vals, cnts = np.unique(nb, return_counts=True); k = int(vals[np.argmax(cnts)])
            if cnts.max() >= 0.6 * len(nb) and np.linalg.norm(X[j, i] - C[k]) < 25: out[j, i] = k; n += 1
    return out, n


BG_GROW_DE = 14.0


def grow_background(M, V, Cen, de=None):
    """Фон на сканах — градиент (тень, неравномерная заливка), и k-means режет его на 2–3 близких цвета.
    Клетка, достижимая от края по цепочке клеток с цветом, близким к фону (ΔE < de), тоже фон.
    Пользователи вручную вырезают такой фон (напр. синее поле вокруг медальона)."""
    de = BG_GROW_DE if de is None else de
    if de <= 0: return M
    H, W = M.shape; X = _lab8_to_ref(V).reshape(H, W, 3)
    close = np.linalg.norm(X - Cen[0][None, None, :], axis=2) < de
    close |= (M == 0)
    seen = np.zeros((H, W), bool); st = []
    for j in range(H):
        for i in range(W):
            if (j in (0, H - 1) or i in (0, W - 1)) and close[j, i] and not seen[j, i]: seen[j, i] = True; st.append((j, i))
    while st:
        j, i = st.pop()
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jj, ii = j + dj, i + di
            if 0 <= jj < H and 0 <= ii < W and close[jj, ii] and not seen[jj, ii]: seen[jj, ii] = True; st.append((jj, ii))
    out = M.copy(); out[seen] = 0
    return out


DYE_MAXSHIFT = 38.0     # якорь красителя, найденный на скане, не может уйти от библиотечного цвета дальше (иначе это другой цвет)


def dye_anchors(X, dye_ids, iters=6):
    """Палитра по красителям, выбранным человеком. Для каждого красителя ищем его цвет НА СКАНЕ: старт — библиотечный цвет,
    дальше k-means с якорями (клетки → ближайший якорь, якорь → среднее своих клеток, но не дальше DYE_MAXSHIFT от библиотеки).
    Красители без клеток выпадают. → (метки по X, якоря Lab-ref, список использованных dye id)"""
    lib = {d[3]: d for d in DYES}
    ids = [i for i in dye_ids if i in lib]
    if not ids: return None
    L = np.array([_DYE_LAB[[d[3] for d in DYES].index(i)] for i in ids], np.float32)
    Wt = np.array([1.0, CHROMA_W, CHROMA_W], np.float32)
    A = L.copy()
    for _ in range(iters):
        a = np.linalg.norm((X[:, None, :] - A[None]) * Wt, axis=2).argmin(1)
        for k in range(len(ids)):
            m = a == k
            if m.sum() >= max(3, MIN_SHARE * len(X)):
                c = X[m].mean(0)
                A[k] = c if np.linalg.norm(c - L[k]) <= DYE_MAXSHIFT else L[k] + (c - L[k]) * DYE_MAXSHIFT / np.linalg.norm(c - L[k])
    a = np.linalg.norm((X[:, None, :] - A[None]) * Wt, axis=2).argmin(1)
    keep = [k for k in range(len(ids)) if (a == k).sum() >= max(3, MIN_SHARE * len(X))]
    if not keep: return None
    remap = np.full(len(ids), -1); remap[keep] = np.arange(len(keep))
    return remap[a], A[keep], [ids[k] for k in keep]


def forced_grid(frame, cols, rows=None):
    """Сетка задана человеком (фото ковра: печатной сетки нет). Рамка → cols×rows клеток; rows по умолчанию — квадратные клетки."""
    x, y, w, h = [float(v) for v in frame]
    cols = max(2, int(cols)); rows = max(2, int(rows)) if rows else max(2, int(round(cols * h / w)))
    px, py = w / cols, h / rows
    u = x + px * (np.arange(cols) + 0.5); v = y + py * (np.arange(rows) + 0.5)
    U, V = np.meshgrid(u, v)
    centers = np.stack([U, V], -1)
    cq = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return {"angle": 0.0, "px": px, "py": py, "ox": 0.0, "oy": 0.0, "cols": cols, "rows": rows, "corners": cq,
            "centers": centers, "quad_resid": 0.0, "strength": 1.0, "R": [1.0, 1.0], "forced": True}


def auto_figure_color(bgr, frame, pitch_hint=None, ncolors=None, dyes=None, canon=False, force_cols=None, force_rows=None):
    G = forced_grid(frame, force_cols, force_rows) if force_cols else detect_grid_color(bgr, frame, pitch_hint=pitch_hint)
    if not force_cols and pitch_hint and abs(pitch_hint[0] - pitch_hint[1]) / min(pitch_hint) > 0.08:
        # клетки прямоугольные: вертикальные полосы кайм в книге часто напечатаны повёрнутыми на 90°
        G2 = detect_grid_color(bgr, frame, pitch_hint=(pitch_hint[1], pitch_hint[0]))
        if sum(G2["R"]) > sum(G["R"]): G = G2; G["rotated_cells"] = True
    C = autogrid.cell_centers(G)
    V = cell_colors(bgr, C, min(G["px"], G["py"]))
    rows, cols = V.shape[:2]
    guided = None
    if dyes:
        guided = dye_anchors(_lab8_to_ref(V).reshape(-1, 3), dyes)
    if guided is not None:
        a, Cen, used = guided
        M, Cen = a.reshape(rows, cols), np.array(Cen, np.float32)
        if canon:
            Cen = np.array([_DYE_LAB[[d[3] for d in DYES].index(i)] for i in used], np.float32)
    else:
        a, Cen, _ = palette_kmeans(V.reshape(-1, 3), ncolors)
        M, Cen = merge_scattered(a.reshape(rows, cols), V, np.array(Cen, np.float32)) if not ncolors else (a.reshape(rows, cols), np.array(Cen, np.float32))
    X = _lab8_to_ref(V).reshape(-1, 3)
    d = np.sort(np.linalg.norm(X[:, None, :] - Cen[None], axis=2), 1)
    amb = (d[:, 1] - d[:, 0]) < 4.0 if len(Cen) > 1 else np.zeros(len(X), bool)
    # фон = самый частый цвет на краю
    ring = np.concatenate([M[0], M[-1], M[:, 0], M[:, -1]])
    bg = int(np.bincount(ring, minlength=len(Cen)).argmax())
    order = [bg] + sorted([k for k in range(len(Cen)) if k != bg], key=lambda k: -(M == k).sum())
    remap = np.zeros(len(Cen), int); remap[order] = np.arange(len(order))
    M = remap[M]; Cen = Cen[order]
    M, fixed = despeckle(M, V, Cen)
    if max(M.shape) / max(1, min(M.shape)) < 1.9: M = grow_background(M, V, Cen)   # каймы (узкие) — фон не вырезаем: поле в них часть узора
    lab8 = np.stack([Cen[:, 0] * 255 / 100, Cen[:, 1] + 128, Cen[:, 2] + 128], -1)
    if guided is not None:
        lib = {d[3]: d for d in DYES}
        names = [(lib[i][0], lib[i][1], float(np.linalg.norm(_DYE_LAB[[d[3] for d in DYES].index(i)] - Cen[k])), i) for k, i in enumerate([used[o] for o in order])]
    else:
        names = [dye_name(c) for c in lab8]
    seen, uaz, uru = {}, [], []
    for az, ru, _, _ in names:                                  # «zoğalı», «zoğalı 2» — два оттенка одного красителя
        seen[az] = seen.get(az, 0) + 1
        uaz.append(az if seen[az] == 1 else f"{az} {seen[az]}"); uru.append(ru if seen[az] == 1 else f"{ru} {seen[az]}")
    info = {"palette_hex": [_lab2hex(c) for c in lab8],
            "names_az": uaz, "names_ru": uru, "dye_ids": [n[3] for n in names],
            "dye_dist": [round(n[2], 1) for n in names],
            "counts": [int((M == k).sum()) for k in range(len(Cen))], "despeckled": int(fixed)}
    flags = []
    R = min(G["R"])
    if R < 0.35: flags.append("сетка найдена неуверенно — возможно, это фото ковра, а не схема")
    elif R < 0.55: flags.append("проверьте сетку (границы цвета не везде совпали с ней)")
    amb_share = float(amb.mean())
    if amb_share > 0.05: flags.append(f"{amb_share:.0%} клеток между двумя цветами — проверьте похожие цвета")
    far = [info["names_az"][k] for k in range(len(Cen)) if info["dye_dist"][k] > 22]
    if far and guided is None: flags.append("названия цветов приблизительные: " + ", ".join(far))
    if guided is not None:
        miss = [i for i in dyes if i not in used]
        flags.append("палитра по выбранным красителям" + (f"; не найдены на скане: {', '.join(miss)}" if miss else ""))
    else:
        flags.append("цвета и их названия — подтвердить")
    info["confidence"] = round(float(np.clip((R - 0.3) / 0.5, 0, 1) * (1 - min(amb_share * 3, 0.5))), 2)
    if G.get("forced"):
        info["confidence"] = round(min(0.5, info["confidence"]), 2)       # по фото уверенность всегда низкая: сетку задал человек, а не бумага
        flags.insert(0, f"из фото: сетка задана вручную ({G['cols']}×{G['rows']}) — клетки усреднены, проверьте глазами")
    info["flags"] = flags
    return G, C, M, info


def strip_frame(M, info, depth=3, thr=0.6, max_loss=0.35):
    """Цвета, живущие почти только в рамке сетки (тень/край скана, соседний рисунок), — в фон; цвет без клеток убираем из палитры.
    Сверено на 19 цветных рисунках, которые пользователь обрезал вручную: точная рамка 7 → 10 из 19."""
    H, W = M.shape; yy, xx = np.mgrid[:H, :W]
    d = np.minimum(np.minimum(yy, H - 1 - yy), np.minimum(xx, W - 1 - xx))
    tot = int((M > 0).sum()); drop = []
    for k in range(1, int(M.max()) + 1):
        n = int((M == k).sum())
        if n and int(((M == k) & (d < depth)).sum()) / n > thr: drop.append(k)
    if not drop or sum(int((M == k).sum()) for k in drop) > max_loss * max(tot, 1): return M, info
    M = M.copy()
    for k in drop: M[M == k] = 0
    keep = [k for k in range(len(info["palette_hex"])) if k not in drop]
    remap = np.zeros(len(info["palette_hex"]), int); remap[keep] = np.arange(len(keep)); M = remap[M]
    info = dict(info)
    for key in ("palette_hex", "names_az", "names_ru", "dye_ids", "dye_dist"): info[key] = [info[key][k] for k in keep]
    info["counts"] = [int((M == k).sum()) for k in range(len(keep))]
    info["flags"] = list(info["flags"]) + [f"убраны краевые цвета (тень/край скана): {len(drop)}"]
    return M, info


def to_work_color(G, M, info, margin=0):
    q = [[round(float(x), 2), round(float(y), 2)] for x, y in G["corners"]]
    W, H = autogrid.quad_size(q); cols, rows = G["cols"], G["rows"]
    M, info = strip_frame(M, info)
    Mt, (ox, oy) = autogrid.trim(M, margin)
    rows_s = ["".join("." if v == 0 else np.base_repr(int(v), 36).lower() for v in r) for r in Mt]
    return {
        "quad": q,
        "grid": {"pw": round(W / cols, 4), "ph": round(H / rows, 4), "ox": 0, "oy": 0, "cols": cols, "rows": rows,
                 "square": abs(G["px"] - G["py"]) / min(G["px"], G["py"]) < 0.03},
        "classify": {"tones": len(info["palette_hex"]) - 1, "inner": 45, "mode": "color"},
        "matrix": {"w": int(Mt.shape[1]), "h": int(Mt.shape[0]), "origin": [int(ox), int(oy)], "rows": rows_s},
        "palette": info["palette_hex"],
        "palette_names": info["names_az"],
        "palette_names_ru": info["names_ru"],
        "palette_dyes": info["dye_ids"],
        "palette_touched": False,
        "color_mode": True,
        "auto": {"version": 1, "mode": "color", "confidence": info["confidence"], "flags": info["flags"],
                 "pitch": [round(G["px"], 3), round(G["py"], 3)], "angle": round(G["angle"], 2), "R": G["R"],
                 "counts": info["counts"], "dye_dist": info["dye_dist"], "reviewed": False},
    }
