"""v12.54: форма элемента без нейросети — симметрия, округлость, диагональность, число цветовых областей (обычная геометрия на CPU).

Проверено 28.11.2026 глазами на 331 подтверждённом элементе (по 9 крайним в каждой категории): «круглые», «симметричные»,
«ромбы», «острые» отбираются хорошо; «прямые углы», «простые / сложные» — хуже (путаются с размытыми мелкими вырезками).
Пороги — 20-й и 80-й процентили на тех же 331 элементах.
"""
import cv2
import numpy as np

S = 96
TAGS = [("round", "круглые"), ("sharp", "острые"), ("rhomb", "ромбы"), ("square", "прямые углы"),
        ("sym", "симметричные"), ("asym", "несимметричные"), ("simple", "простые"), ("complex", "сложные")]


def _crop(a, box, size):
    H, W = a.shape[:2]; x, y, w, h = box
    cx, cy = (x + w / 2) * W, (y + h / 2) * H; side = max(w * W, h * H)
    M = np.array([[side / size, 0, cx - side / 2], [0, side / size, cy - side / 2]], np.float32)
    return cv2.warpAffine(a, M, (size, size), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), side


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def feats(a, box):
    """→ dict признаков или None (рамка слишком большая — «пол-ковра» — или слишком мелкая на фото)"""
    if box[2] * box[3] > 0.06: return None
    c, side = _crop(a, box, S)
    if side < 28: return None
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
    X = lab / (lab.reshape(-1, 3).std(0) + 1e-6)
    sym = lambda f: float(np.mean([_ncc(X[:, :, k], f(X[:, :, k])) for k in range(3)]))
    s_lr, s_ud, s_r90 = sym(lambda z: z[:, ::-1]), sym(lambda z: z[::-1]), sym(lambda z: np.rot90(z))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, lb, _ = cv2.kmeans(lab.reshape(-1, 3), 3, None, crit, 3, cv2.KMEANS_PP_CENTERS); lb = lb.reshape(S, S)
    border = np.concatenate([lb[0], lb[-1], lb[:, 0], lb[:, -1]]); bg = np.bincount(border, minlength=3).argmax()
    fg = cv2.morphologyEx((lb != bg).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(fg); circ = 0.0
    if n > 1:
        best = max(range(1, n), key=lambda i: st[i, 4] * (1.0 if cc[S // 2, S // 2] == i else 0.5))
        mk = cv2.morphologyEx((cc == best).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); ct = max(cs, key=cv2.contourArea)
        P = cv2.arcLength(ct, True); circ = float(4 * np.pi * cv2.contourArea(ct) / (P * P + 1e-9))
    g = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0); gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy); ang = (np.degrees(np.arctan2(gy, gx)) % 180)[mag > np.percentile(mag, 70)]
    d = np.minimum(np.abs(ang - 45), np.abs(ang - 135)); o = np.minimum(np.minimum(ang, 180 - ang), np.abs(ang - 90))
    diag = float((d < 15).mean() - (o < 15).mean()) if len(ang) else 0.0
    c2, _ = _crop(a, box, int(max(24, min(96, side))))                          # области — без увеличения мелкого
    l2 = cv2.cvtColor(c2, cv2.COLOR_BGR2LAB).astype(np.float32); s2 = c2.shape[0]
    _, lb2, _ = cv2.kmeans(l2.reshape(-1, 3), 4, None, crit, 3, cv2.KMEANS_PP_CENTERS); lb2 = lb2.reshape(s2, s2); reg = 0
    for k in range(4):
        _, _, st2, _ = cv2.connectedComponentsWithStats((lb2 == k).astype(np.uint8), connectivity=4)
        reg += int((st2[1:, 4] >= max(3, 0.004 * s2 * s2)).sum())
    return {"sym": round(max(s_lr, s_ud), 3), "sym4": round(min(s_lr, s_ud, s_r90), 3), "circ": round(circ, 3), "diag": round(diag, 3), "reg": reg}


def tags(f):
    if not f: return []
    t = []
    if f["circ"] >= 0.54: t.append("round")
    if f["circ"] <= 0.18: t.append("sharp")
    if f["diag"] >= -0.01: t.append("rhomb")
    if f["diag"] <= -0.23: t.append("square")
    if f["sym4"] >= 0.24: t.append("sym")
    if f["sym"] <= 0.15: t.append("asym")
    if f["reg"] <= 18: t.append("simple")
    if f["reg"] >= 32: t.append("complex")
    return t
