# frame_edge.py — v12.00: признаки «что у края фото» для модели «целый ли ковёр».
#
# Считаются по самой картинке (уменьшенной до 256 px по длинной стороне) для всех ковров, независимо от ответов людей,
# и кладутся в r.auto_edge = {"v": VER, "f": [...]}. Так в обучении нет утечки (уточнённые углы считаются только у
# «целых», поэтому их использовать нельзя).
#
# Идея: у целого ковра вокруг — фон (ровный, похожий на цвет углов кадра, без узора), у обрезанного узор доходит
# до самого края. Для каждой стороны: разброс яркости в полосе у края, «энергия» перепадов у края против энергии внутри,
# цветовое расстояние полосы у края от полосы внутри, доля пикселей кольца, похожих на цвет углов кадра.
# Стороны сводятся в min / mean / max — так признаки не зависят от поворота.
import numpy as np

VER = 1
NAMES = ["e_std_min", "e_std_mean", "e_std_max",
         "e_grad_min", "e_grad_mean", "e_grad_max",
         "e_cdist_min", "e_cdist_mean", "e_cdist_max",
         "e_bg_min", "e_bg_mean", "e_bg_max",
         "e_corner_std", "e_center_vs_corner"]
EMPTY = [-1.0] * len(NAMES)


def _lab(a):
    """sRGB 0..255 → грубый Lab (достаточно для расстояний)"""
    x = a.astype(np.float32) / 255.0
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = x @ M.T / np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def features(path):
    """список len(NAMES) или None, если картинку не прочитать"""
    try:
        from PIL import Image
        im = Image.open(path); im.draft("RGB", (512, 512)); im = im.convert("RGB")
        im.thumbnail((256, 256))
        a = np.asarray(im)
    except Exception:
        return None
    return features_arr(a)


def features_arr(a):
    h, w = a.shape[:2]
    if h < 32 or w < 32: return None
    L = _lab(a); g = L[..., 0]
    gy, gx = np.gradient(g); G = np.hypot(gx, gy)
    b = max(2, int(round(0.03 * min(h, w)))); i0, i1 = int(0.08 * min(h, w)), int(0.14 * min(h, w))
    c = max(3, int(0.06 * min(h, w)))
    corners = np.concatenate([L[:c, :c].reshape(-1, 3), L[:c, -c:].reshape(-1, 3), L[-c:, :c].reshape(-1, 3), L[-c:, -c:].reshape(-1, 3)])
    bg = np.median(corners, 0)
    sides = {  # (полоса у края, полоса внутри)
        "t": (np.s_[:b, :], np.s_[i0:i1, i1:w - i1]), "b": (np.s_[h - b:, :], np.s_[h - i1:h - i0, i1:w - i1]),
        "l": (np.s_[:, :b], np.s_[i1:h - i1, i0:i1]), "r": (np.s_[:, w - b:], np.s_[i1:h - i1, w - i1:w - i0])}
    S, GR, CD, BG = [], [], [], []
    for out, inn in sides.values():
        S.append(float(g[out].std()))
        GR.append(float(G[out].mean() / (G[inn].mean() + 1.0)))
        CD.append(float(np.linalg.norm(L[out].reshape(-1, 3).mean(0) - L[inn].reshape(-1, 3).mean(0))))
        BG.append(float((np.linalg.norm(L[out].reshape(-1, 3) - bg, axis=1) < 12).mean()))
    ctr = L[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3).mean(0)
    agg = lambda v: [min(v), sum(v) / 4, max(v)]
    return [round(x, 4) for x in agg(S) + agg(GR) + agg(CD) + agg(BG) + [float(np.linalg.norm(corners - bg, axis=1).mean()), float(np.linalg.norm(ctr - bg))]]


def get(r):
    e = (r.get("auto_edge") or {})
    return e["f"] if e.get("v") == VER and isinstance(e.get("f"), list) and len(e["f"]) == len(NAMES) else EMPTY
