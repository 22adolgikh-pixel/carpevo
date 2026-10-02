# risk.py — оценка «насколько рисунок надо проверять» и поиск подозрительных клеток по симметрии.
# Ковровые мотивы почти всегда симметричны (зеркало, поворот на 180°, диагональ). Если рисунок в целом симметричен,
# клетки, нарушающие симметрию, — вероятные ошибки оцифровки (или реальная асимметрия — решает человек).
import numpy as np

SYM_MIN = 0.90        # доля совпадающих закрашенных клеток, начиная с которой рисунок считаем симметричным


def _trim(K):
    ys, xs = np.nonzero(K)
    if not len(xs): return K, (0, 0)
    return K[ys.min():ys.max() + 1, xs.min():xs.max() + 1], (int(xs.min()), int(ys.min()))


def _transforms(K):
    h, w = K.shape
    t = {"mirror_v": K[:, ::-1], "mirror_h": K[::-1, :], "rot180": K[::-1, ::-1]}
    if h == w:
        t["diag"] = K.T; t["antidiag"] = K[::-1, ::-1].T; t["rot90"] = np.rot90(K)
    return t


def symmetry(K, min_sym=SYM_MIN):
    """K — матрица меток (0 = фон). Возвращает {kind, score, suspects:[(x,y)...] в координатах K} или None."""
    Kt, (ox, oy) = _trim(np.asarray(K))
    if (Kt > 0).sum() < 12: return None
    best = None
    for name, T in _transforms(Kt).items():
        m = (Kt > 0) | (T > 0)
        sc = float(((Kt == T) & m).sum() / max(1, m.sum()))
        if best is None or sc > best[1]: best = (name, sc, T)
    if best is None or best[1] < min_sym: return {"kind": None, "score": round(best[1], 3) if best else 0.0, "suspects": []}
    name, sc, T = best
    ys, xs = np.nonzero(Kt != T)
    return {"kind": name, "score": round(sc, 3), "suspects": [(int(x) + ox, int(y) + oy) for x, y in zip(xs, ys)]}


def risk_score(auto, sym):
    """0..1, больше — рискованнее. auto — work['auto']; sym — результат symmetry()."""
    conf = float((auto or {}).get("confidence") or 0.0)
    if any("не найден" in f for f in ((auto or {}).get("flags") or [])): return 1.5      # пустой результат — никогда не «надёжный»
    r = 1.0 - conf
    flags = (auto or {}).get("flags") or []
    r += 0.08 * min(len([f for f in flags if "подтвердить" not in f]), 4)
    if sym and sym.get("kind"):
        r += min(len(sym["suspects"]) / 40.0, 0.4)
    return round(float(min(r, 1.5)), 3)
