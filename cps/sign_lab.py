# sign_lab.py — v11.19: «расшифровка» ковра без базы. Каймы ковра — фризы: один и тот же отрезок узора (раппорт),
# повторённый вдоль полосы. Машина сама находит полосы кайм, шаг повтора и «чистый» раппорт (медиана всех повторов —
# грязь фото, блики и ошибки ткача в отдельных повторах уходят). Название не нужно: раппорт — это новый «знак»,
# его можно назвать потом, сравнить с другими коврами и собрать из него кайму любой длины (генератор).
#
# Шаги:
#  1) край ковра и внешняя граница поля — rug_layout.detect;
#  2) каждая сторона каймы разворачивается в вертикальную полосу (внешний край слева);
#  3) полоса режется на узкие колонки; для каждой — шаг повтора по автокорреляции цвета вдоль полосы;
#     соседние колонки с одинаковым шагом — одна кайма (бордюр);
#  4) повторы подгоняются по одному (рука ткача: шаг «плывёт») и усредняются медианой → раппорт;
#  5) одинаковые каймы на 4 сторонах объединяются (по расстоянию от края), места повторов переводятся в доли фото.
import numpy as np, cv2
import rug_layout


def _period(sig, lo, hi):
    s = sig - sig.mean(0)
    hi = min(hi, len(s) - 4)
    if hi <= lo + 2: return 0, 0.0
    ac = np.array([((s[:-L] * s[L:]).sum() / (np.sqrt((s[:-L] ** 2).sum() * (s[L:] ** 2).sum()) + 1e-6)) for L in range(lo, hi)])
    g = float(ac.max())
    for i in range(1, len(ac) - 1):            # первый заметный пик, а не 2T / 3T
        if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] >= 0.85 * g: return lo + i, float(ac[i])
    return lo + int(ac.argmax()), g


def _strip(a, rect, bw, bh, side):
    x, y, w, h = rect
    if side == "l": return a[y + bh:y + h - bh, x:x + bw]
    if side == "r": return cv2.flip(a[y + bh:y + h - bh, x + w - bw:x + w], 1)
    if side == "t": return cv2.rotate(a[y:y + bh, x + bw:x + w - bw], cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.rotate(a[y + h - bh:y + h, x + bw:x + w - bw], cv2.ROTATE_90_CLOCKWISE)


def _to_img(side, rect, bw, bh, u0, u1, v0, v1):
    """прямоугольник в координатах полосы (u — поперёк от внешнего края, v — вдоль) → (x0, y0, x1, y1) на фото"""
    x, y, w, h = rect
    if side == "l": return x + u0, y + bh + v0, x + u1, y + bh + v1
    if side == "r": return x + w - u1, y + bh + v0, x + w - u0, y + bh + v1
    L = w - 2 * bw
    if side == "t": return x + bw + (L - v1), y + u0, x + bw + (L - v0), y + u1      # поворот против часовой: v идёт справа налево
    return x + bw + v0, y + h - u1, x + bw + v1, y + h - u0


def _bands(s, k=28):
    lab = cv2.cvtColor(cv2.GaussianBlur(s, (0, 0), 1), cv2.COLOR_BGR2LAB).astype(np.float32)
    W = s.shape[1]; cw = max(2, W // k); cols = []
    for c0 in range(0, W - cw + 1, cw):
        T, q = _period(lab[:, c0:c0 + cw].mean(1), max(6, s.shape[0] // 60), s.shape[0] // 3)
        cols.append((c0, cw, T, q))
    grp = []
    for c in cols:
        if c[3] < 0.3 or not c[2]: grp.append(None); continue
        last = grp[-1] if grp else None
        if last and abs(last[-1][2] - c[2]) <= max(2, 0.06 * c[2]): last.append(c)
        else: grp.append([c])
    out = []
    for g in grp:
        if not g or len(g) < 2: continue
        out.append({"u0": g[0][0], "u1": g[-1][0] + g[-1][1], "T": int(np.median([c[2] for c in g])), "q": float(np.mean([c[3] for c in g]))})
    # одна кайма, разрезанная полоской со слабым сигналом, — склеить (тот же шаг, зазор меньше 2 колонок)
    m = []
    for b in out:
        if m and abs(m[-1]["T"] - b["T"]) <= max(2, 0.06 * b["T"]) and b["u0"] - m[-1]["u1"] <= 2 * cw:
            m[-1]["u1"] = b["u1"]; m[-1]["q"] = (m[-1]["q"] + b["q"]) / 2
        else: m.append(dict(b))
    return m


def _rapport(s, u0, u1, T):
    """повторы, подогнанные по одному, и их медиана → (раппорт BGR, начала повторов вдоль полосы)"""
    H = s.shape[0]
    if T < 4 or H < 2 * T: return None, []
    starts = list(range(0, H - T + 1, T)); ref = s[starts[len(starts) // 2]:starts[len(starts) // 2] + T, u0:u1].astype(np.float32)
    al, pos = [], []
    for i in starts:
        best = None
        for d in range(-T // 6, T // 6 + 1):
            j = i + d
            if j < 0 or j + T > H: continue
            c = s[j:j + T, u0:u1].astype(np.float32); v = -float(np.abs(c - ref).mean())
            if best is None or v > best[0]: best = (v, c, j)
        if best: al.append(best[1]); pos.append(best[2])
    med = np.median(np.stack(al), 0).astype(np.uint8)
    # начало раппорта — там, где ряд «самый пустой» (стык мотивов), чтобы раппорт был цельным мотивом
    e = np.abs(np.diff(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY).astype(np.float32), axis=1)).mean(1)
    sh = int(np.argmin(np.convolve(e, np.ones(3) / 3, "same")))
    med = np.roll(med, -sh, axis=0)
    return med, [p + sh for p in pos if p + sh + T <= H]


def friezes(img, work=1400):
    """→ {"carpet": [x,y,w,h] доли, "bands": [{"inset": [u0,u1] доли короткой стороны, "T": шаг (доля длины), "q": чёткость,
          "sides": {...}, "tile": BGR раппорт (вдоль полосы — сверху вниз, внешний край слева), "boxes": [[x,y,w,h] доли, ...]}]}"""
    H0, W0 = img.shape[:2]; k = min(1.0, work / max(H0, W0))
    a = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    H, W = a.shape[:2]
    lay = rug_layout.detect(a)
    cx, cy, cw, ch = lay["carpet"]; rect = (int(cx * W), int(cy * H), int(cw * W), int(ch * H))
    b = lay["bounds"][-1] if lay.get("bounds") else {"ix": 0.15, "iy": 0.15}
    bw, bh = max(8, int(b["ix"] * rect[2])), max(8, int(b["iy"] * rect[3]))
    found = []
    for side in "lrtb":
        s = _strip(a, rect, bw, bh, side)
        if s.shape[0] < 60 or s.shape[1] < 8: continue
        for bd in _bands(s):
            if bd["q"] < 0.45: continue
            tile, pos = _rapport(s, bd["u0"], bd["u1"], bd["T"])
            if tile is None or len(pos) < 3: continue
            if float(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).std()) < 14: continue      # однотонная полоса, край фото, бахрома — не узор
            boxes = []
            for p in pos:
                x0, y0, x1, y1 = _to_img(side, rect, bw, bh, bd["u0"], bd["u1"], p, p + bd["T"])
                boxes.append([round(x0 / W, 5), round(y0 / H, 5), round((x1 - x0) / W, 5), round((y1 - y0) / H, 5)])
            found.append({"side": side, "u0": bd["u0"], "u1": bd["u1"], "T": bd["T"], "q": round(bd["q"], 3), "tile": tile, "boxes": boxes,
                          "depth": (bw if side in "lr" else bh)})
    # одна и та же кайма на разных сторонах: близкое расстояние от края (в долях ширины каймы)
    groups = []
    for f in sorted(found, key=lambda f: -f["q"]):
        mid = (f["u0"] + f["u1"]) / 2 / f["depth"]
        g = next((g for g in groups if abs(g["mid"] - mid) < 0.08 and abs(g["items"][0]["T"] - f["T"]) <= max(2, 0.12 * f["T"])), None)
        if g is None: groups.append({"mid": mid, "items": [f]})
        else: g["items"].append(f)
    bands = []
    for g in sorted(groups, key=lambda g: g["mid"]):
        best = g["items"][0]
        bands.append({"inset": round(g["mid"], 3), "q": best["q"], "T": best["T"], "tile": best["tile"],
                      "sides": {f["side"]: {"T": f["T"], "q": f["q"], "n": len(f["boxes"])} for f in g["items"]},
                      "boxes": [bx for f in g["items"] for bx in f["boxes"]],
                      "box_sides": [f["side"] for f in g["items"] for _ in f["boxes"]]})   # l — как раппорт; r — зеркально; t — повёрнут на 90°, b — на 270°
    return {"carpet": [round(v, 4) for v in lay["carpet"]], "bands": bands}
