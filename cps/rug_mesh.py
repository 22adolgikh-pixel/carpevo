# rug_mesh.py — выпрямление ковра сеткой поверх перспективы (проба, v11.71).
#
# rug_rect.rectify растягивает по 4 углам, и края между углами остаются такими, какими были на фото (волна, прогиб).
# Здесь второй шаг:
# 1) rectify_padded: то же преобразование, но с запасом по краям — чтобы видеть, где край ковра за линией углов.
# 2) edge_curves: по каждой из 4 сторон ищем линию края — по смене цвета «снаружи / внутри» на полосках вдоль стороны,
#    путь выбираем динамическим программированием (край непрерывен), затем сглаживаем; у углов отклонение сводим к нулю.
# 3) coons_warp: кривые 4 сторон → прямые стороны артборда (лоскут Кунса: точки внутри сдвигаются плавно).
# Результат — прямоугольный ковёр с ровными границами; сами кривые возвращаются, чтобы разметку можно было перевести обратно.
import cv2, numpy as np
from rug_rect import rect_size


def rectify_padded(img, quad, side=1200, pad=0.07):
    """→ canvas (BGR), valid (маска «внутри кадра»), geo: {W,H,px,py} — ковёр W×H лежит в canvas со смещением (px,py)"""
    h, w = img.shape[:2]
    if max(h, w) > 2 * side:
        k = 2 * side / max(h, w); img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA); h, w = img.shape[:2]
    src = np.array([[x * w, y * h] for x, y in quad], np.float32)
    W, H = rect_size(src, side, (w, h))
    px, py = int(round(W * pad)), int(round(H * pad))
    dst = np.array([[px, py], [px + W, py], [px + W, py + H], [px, py + H]], np.float32)
    T = cv2.getPerspectiveTransform(src, dst)
    size = (W + 2 * px, H + 2 * py)
    canvas = cv2.warpPerspective(img, T, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return canvas, valid, {"W": W, "H": H, "px": px, "py": py}


def _smooth(v, sigma):
    n = len(v); k = int(sigma * 3) * 2 + 1
    x = np.arange(k) - k // 2; g = np.exp(-0.5 * (x / sigma) ** 2); g /= g.sum()
    p = np.pad(v, (k // 2, k // 2), mode="edge")
    return np.convolve(p, g, mode="valid")[:n]


def _track_side(lab, valid, p0, nrm, tan, length, nst, R, band, strip):
    """Идём вдоль стороны от p0 по направлению tan на длину length, на nst станциях ищем смещение края вдоль нормали nrm (внутрь ковра).
    Возвращает (offsets в пикселях, сила)"""
    H, W = lab.shape[:2]
    js = np.arange(-R, R + 1)
    S = np.zeros((nst, len(js)), np.float32)
    ts = np.linspace(0.03, 0.97, nst)
    for i, t in enumerate(ts):
        c = p0 + tan * (t * length)
        ks = np.arange(-strip, strip + 1, 3)
        # профиль по нормали: средний цвет на расстоянии d от линии (вдоль полоски)
        ds = np.arange(-R - band, R + band + 1)
        pts = c[None, None, :] + ks[:, None, None] * tan[None, None, :] + ds[None, :, None] * nrm[None, None, :]
        xs = np.clip(np.rint(pts[..., 0]).astype(int), 0, W - 1); ys = np.clip(np.rint(pts[..., 1]).astype(int), 0, H - 1)
        col = lab[ys, xs].mean(0)                    # len(ds) × 3
        ok = valid[ys, xs].mean(0) > 0.7             # len(ds)
        cs = np.concatenate([np.zeros((1, 3)), np.cumsum(col, 0)])
        for jj, j in enumerate(js):
            a = j + R + band                          # индекс ds для расстояния j
            if a - band < 0 or a + band >= len(ds): continue
            out_ = (cs[a] - cs[a - band]) / band; in_ = (cs[a + band] - cs[a]) / band
            if not (ok[a - band:a + band].all()): S[i, jj] = 0; continue   # внутри кадра ли обе стороны
            S[i, jj] = np.linalg.norm(in_ - out_)
    return js, S, ts


def _viterbi(S, js, lam=0.9, mu=0.015, maxstep=3):
    n, m = S.shape
    sc = S / (np.percentile(S, 95) + 1e-6)
    cost = -sc + mu * (js[None, :] / max(1, abs(js).max())) ** 2 * 10
    D = np.full((n, m), 1e9, np.float32); B = np.zeros((n, m), np.int32)
    D[0] = cost[0]
    for i in range(1, n):
        for j in range(m):
            lo, hi = max(0, j - maxstep), min(m, j + maxstep + 1)
            seg = D[i - 1, lo:hi] + lam * np.abs(np.arange(lo, hi) - j) / maxstep * 0.5
            k = int(np.argmin(seg)); D[i, j] = seg[k] + cost[i, j]; B[i, j] = lo + k
    j = int(np.argmin(D[-1])); path = [j]
    for i in range(n - 1, 0, -1): j = B[i, j]; path.append(j)
    path = np.array(path[::-1]); return js[path].astype(np.float64), sc[np.arange(n), path]


def edge_curves(canvas, valid, geo, nst=64, search=0.05, taper=0.06):
    """→ {top,bot,left,right: массив смещений края внутрь (px) на nst станциях t∈[0,1]}, strength"""
    W, H, px, py = geo["W"], geo["H"], geo["px"], geo["py"]
    lab = cv2.cvtColor(cv2.GaussianBlur(canvas, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    R = int(round(search * min(W, H))); band = max(4, int(0.008 * min(W, H)))
    sides = {
        "top":   (np.array([px, py], float),          np.array([0, 1.0]),  np.array([1.0, 0]), W),
        "bot":   (np.array([px, py + H], float),      np.array([0, -1.0]), np.array([1.0, 0]), W),
        "left":  (np.array([px, py], float),          np.array([1.0, 0]),  np.array([0, 1.0]), H),
        "right": (np.array([px + W, py], float),      np.array([-1.0, 0]), np.array([0, 1.0]), H),
    }
    out, strength = {}, {}
    tt = np.linspace(0, 1, nst)
    for k, (p0, nrm, tan, L) in sides.items():
        js, S, ts = _track_side(lab, valid, p0, nrm, tan, L, nst, R, band, strip=max(6, int(L / nst * 0.7)))
        off, sc = _viterbi(S, js)
        off = _smooth(np.median(np.stack([np.roll(off, s) for s in (-2, -1, 0, 1, 2)]), 0), 1.6)
        full = np.interp(tt, ts, off)                       # края t<0.03 и t>0.97 — продолжаем ближайшим значением
        w = np.clip(np.minimum(tt, 1 - tt) / taper, 0, 1); w = w * w * (3 - 2 * w)
        out[k] = full * w; strength[k] = float(np.mean(sc))
    return out, strength


def coons_warp(canvas, geo, curves):
    """4 кривые → ровный прямоугольник W×H. curves — смещения внутрь (px) вдоль каждой стороны."""
    W, H, px, py = geo["W"], geo["H"], geo["px"], geo["py"]
    nst = len(curves["top"]); tt = np.linspace(0, 1, nst)
    u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
    fu = lambda k: np.interp(u, tt, curves[k]); fv = lambda k: np.interp(v, tt, curves[k])
    Tx, Ty = px + u * W, py + fu("top")
    Bx, By = px + u * W, py + H - fu("bot")
    Lx, Ly = px + fv("left"), py + v * H
    Rx, Ry = px + W - fv("right"), py + v * H
    U, V = u[None, :], v[:, None]
    def blend(T, B, L, R, c00, c10, c01, c11, TT, BB, LL, RR):
        return ((1 - V) * TT[None, :] + V * BB[None, :] + (1 - U) * LL[:, None] + U * RR[:, None]
                - ((1 - U) * (1 - V) * c00 + U * (1 - V) * c10 + (1 - U) * V * c01 + U * V * c11))
    X = blend(None, None, None, None, px, px + W, px, px + W, Tx, Bx, Lx, Rx)
    Y = blend(None, None, None, None, py, py, py + H, py + H, Ty, By, Ly, Ry)
    return cv2.remap(canvas, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def mesh_rectify(img, quad, side=1200):
    """→ (картинка, сведения). Сведения: кривые краёв (px), их размах (% короткой стороны), сила краёв."""
    canvas, valid, geo = rectify_padded(img, quad, side)
    curves, strength = edge_curves(canvas, valid, geo)
    out = coons_warp(canvas, geo, curves)
    sh = min(geo["W"], geo["H"])
    rng = {k: float((v.max() - v.min()) / sh * 100) for k, v in curves.items()}
    return out, {"geo": geo, "curves": {k: v.tolist() for k, v in curves.items()}, "range_pct": rng, "strength": strength}


# ---------- второй шаг: линия поля (внутренняя рамка) ----------
def field_curves(rect, nst=64, search=0.035, taper=0.08):
    """rect — ковёр после mesh_rectify (края уже ровные). Линию поля берём из rug_layout.detect(full=True) (самая глубокая граница),
    затем по каждой стороне уточняем её кривую так же, как край. → {top,bot,left,right: расстояние линии от края (px) на nst станциях}, или None"""
    import rug_layout
    H, W = rect.shape[:2]
    det = rug_layout.detect(rect, full=True)
    if not det["bounds"]: return None
    b = max(det["bounds"], key=lambda x: x["ix"])
    d0 = {"left": b["ix"] * W, "right": b["ix"] * W, "top": b["iy"] * H, "bot": b["iy"] * H}
    lab = cv2.cvtColor(cv2.GaussianBlur(rect, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = np.ones((H, W), bool)
    R = max(6, int(round(search * min(W, H)))); band = max(4, int(0.008 * min(W, H)))
    sides = {"top": (np.array([0, 0.]), np.array([0, 1.]), np.array([1., 0]), W), "bot": (np.array([0, H - 1.]), np.array([0, -1.]), np.array([1., 0]), W),
             "left": (np.array([0, 0.]), np.array([1., 0]), np.array([0, 1.]), H), "right": (np.array([W - 1., 0]), np.array([-1., 0]), np.array([0, 1.]), H)}
    tt = np.linspace(0, 1, nst); out = {}
    for k, (p0, nrm, tan, L) in sides.items():
        js, S, ts = _track_side(lab, valid, p0 + nrm * d0[k], nrm, tan, L, nst, R, band, strip=max(6, int(L / nst * 0.7)))
        off, _ = _viterbi(S, js, mu=0.05)
        off = _smooth(np.median(np.stack([np.roll(off, s) for s in (-2, -1, 0, 1, 2)]), 0), 1.6)
        out[k] = d0[k] + np.interp(tt, ts, off)
    # у углов линия поля идёт к своему среднему значению (углы рамки считаем прямоугольными)
    for k, v in out.items():
        m = float(np.median(v)); w = np.clip(np.minimum(tt, 1 - tt) / taper, 0, 1); w = w * w * (3 - 2 * w)
        out[k] = m + (v - m) * w
    return out


def field_warp(rect, fc):
    """внутренняя рамка → прямоугольник: у каждой стороны линия поля становится прямой на своём среднем расстоянии от края;
    смещение плавно гаснет к краю ковра и (противоположными сторонами) в глубине поля"""
    H, W = rect.shape[:2]; nst = len(fc["top"]); tt = np.linspace(0, 1, nst)
    u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
    med = {k: float(np.median(x)) for k, x in fc.items()}
    et = np.interp(u, tt, fc["top"] - med["top"]); eb = np.interp(u, tt, fc["bot"] - med["bot"])
    el = np.interp(v, tt, fc["left"] - med["left"]); er = np.interp(v, tt, fc["right"] - med["right"])
    y = np.arange(H, dtype=np.float64); x = np.arange(W, dtype=np.float64)
    def prof(c, d0, d1, L):                      # 0 у края, 1 у линии, затем линейно до 0 у противоположной линии
        s = np.clip(c / max(d0, 1), 0, 1); s = s * s * (3 - 2 * s)
        tail = np.clip((L - d1 - c) / max(L - d0 - d1, 1), 0, 1)
        return np.where(c < d0, s, tail)
    at = prof(y, med["top"], med["bot"], H); ab = prof(H - 1 - y, med["bot"], med["top"], H)
    al = prof(x, med["left"], med["right"], W); ar = prof(W - 1 - x, med["right"], med["left"], W)
    Y = y[:, None] + at[:, None] * et[None, :] - ab[:, None] * eb[None, :]
    X = x[None, :] + al[None, :] * el[:, None] - ar[None, :] * er[:, None]
    return cv2.remap(rect, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), med


def mesh_rectify2(img, quad, side=1200):
    """край + линия поля. → (картинка, сведения; field — расстояния линии поля от краёв, px)"""
    out, info = mesh_rectify(img, quad, side)
    fc = field_curves(out)
    if fc is None: info["field"] = None; return out, info
    out2, med = field_warp(out, fc)
    sh = min(out.shape[:2])
    info["field"] = {"median_px": med, "range_pct": {k: float((v.max() - v.min()) / sh * 100) for k, v in fc.items()}}
    return out2, info
