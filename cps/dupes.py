"""Поиск повторов узоров между рисунками (напр. ЧБ-схема и цветная схема того же узора у Керимова).
Сравнивается силуэт «не фон» после нормализации размера (48×48), во всех поворотах/отражениях (D4).
Работает по матрицам data/work; сканы не нужны."""
import glob, json, os
import numpy as np

S = 48


def _mask(w):
    m = w.get("matrix")
    if not m or not m.get("rows"):
        return None
    h, ww = int(m["h"]), int(m["w"])
    K = np.zeros((h, ww), np.uint8)
    for j, row in enumerate(m["rows"][:h]):
        for i, ch in enumerate(row[:ww]):
            if ch not in ".0":
                K[j, i] = 1
    return K


def _sig(K):
    h, w = K.shape
    ys = (np.arange(S) * h / S).astype(int); xs = (np.arange(S) * w / S).astype(int)
    return K[np.ix_(ys, xs)].astype(np.float32)


def load(work_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(work_dir, "*.json"))):
        try:
            w = json.load(open(f))
        except Exception:
            continue
        K = _mask(w)
        if K is None or K.sum() < 30:
            continue
        out.append({"id": w["id"], "K": K, "color": bool(w.get("color_mode")),
                    "table": w.get("table"), "name": (w.get("meta") or {}).get("name_az") or (w.get("meta") or {}).get("name") or ""})
    return out


def find(items, min_score=0.93, asp_tol=0.12):
    """Пары похожих рисунков. score = доля совпавших клеток силуэта (после нормализации), лучший из 8 преобразований."""
    n = len(items)
    asp = np.array([i["K"].shape[1] / i["K"].shape[0] for i in items])
    sig = [_sig(i["K"]) for i in items]
    # преобразования: для одинаковой формы — отражения и 180°, для «повёрнутой» — транспонирования
    def tr(a, t):
        return [a, a[:, ::-1], a[::-1], a[::-1, ::-1], a.T, a.T[:, ::-1], a.T[::-1], a.T[::-1, ::-1]][t]
    A = np.stack([s.ravel() for s in sig])
    res = []
    for t in range(8):
        B = np.stack([tr(s, t).ravel() for s in sig])
        inter = A @ B.T
        union_pairs = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
        agree = 1 - (A.sum(1)[:, None] + B.sum(1)[None, :] - 2 * inter) / (S * S)
        iou = inter / np.maximum(union_pairs, 1)
        for a in range(n):
            for b in range(a + 1, n):
                aa = asp[b] if t < 4 else 1 / asp[b]
                if abs(np.log(asp[a] / aa)) > asp_tol:
                    continue
                if iou[a, b] >= min_score - 0.05 and agree[a, b] >= min_score:
                    res.append((a, b, t, float(agree[a, b]), float(iou[a, b])))
    best = {}
    for a, b, t, ag, iou in res:
        k = (a, b)
        if k not in best or iou > best[k][1]:
            best[k] = (ag, iou, t)
    out = []
    for (a, b), (ag, iou, t) in best.items():
        ia, ib = items[a], items[b]
        out.append({"a": ia["id"], "b": ib["id"], "iou": round(iou, 3), "agree": round(ag, 3), "transform": t,
                    "cross": ia["color"] != ib["color"], "a_color": ia["color"], "b_color": ib["color"],
                    "same_dims": ia["K"].shape == ib["K"].shape or ia["K"].shape == ib["K"].shape[::-1],
                    "name_a": ia["name"], "name_b": ib["name"]})
    out.sort(key=lambda r: -r["iou"])
    return out
