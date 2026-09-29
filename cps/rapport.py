# rapport.py — раппорт (повторяющийся фрагмент) каймы: Carpet Pattern Studio
#
# Кайма — это один и тот же фрагмент, повторённый вдоль полосы. Храним сам фрагмент и шаг повтора,
# а не всю полосу: из такой библиотеки потом собирается кайма любой длины (генератор кайм, «ковростроение»).
#
# Что считается:
#   • ось (вдоль длинной стороны полосы) и шаг T — наименьший сдвиг, при котором полоса совпадает сама с собой;
#     концы полосы (углы, обрез) не учитываются;
#   • «чистый» фрагмент — голосование большинством по всем полным повторам (заодно исправляет случайные
#     ошибки оцифровки; клетки, где оцифровка расходится с большинством, возвращаются списком — их стоит глянуть);
#   • начало фрагмента выбирается там, где ряд самый «пустой» (стык двух мотивов), чтобы фрагмент был цельным мотивом;
#   • симметрии: зеркальная поперёк полосы и «скользящая» (через полшага мотив повторяется зеркально).
import numpy as np

MAX_MISMATCH = 0.12        # больше — повтора нет (или полоса слишком короткая / оцифровка грязная)
MIN_AGREE = 0.8           # повторы должны совпадать и по цветам (голосование) не хуже этого
TIE = 0.02                 # из почти равных шагов берём наименьший (иначе найдётся 2T, 3T…)


def matrix_to_array(matrix):
    w, h, rows = int(matrix["w"]), int(matrix["h"]), matrix["rows"]
    K = np.zeros((h, w), np.int16)
    for j, row in enumerate(rows[:h]):
        for i, ch in enumerate(row[:w]): K[j, i] = 0 if ch == "." else int(ch, 36)
    return K


def _rows_to_str(U):
    return ["".join("." if v == 0 else np.base_repr(int(v), 36).lower() for v in r) for r in U]


def _find_along_rows(K, min_rep=1.3, min_overlap=12):
    """Повтор вдоль оси строк (вертикальная полоса). K: h×w.
    Шаг ищется по «силуэту» (рисунок / фон) — он устойчив к путанице близких тонов при оцифровке;
    в книге часто показано всего полтора-два повтора, поэтому достаточно перекрытия min_overlap рядов."""
    h = K.shape[0]
    trim = int(h * 0.05) if h >= 60 else 0
    Z = K[trim:h - trim] if trim else K
    n = Z.shape[0]
    # силуэт: либо весь рисунок, либо один цвет (у ЧБ-схем серая заливка распознаётся шумнее контура) —
    # берём тот, на котором повтор виден чище
    return min((r for r in (_scan(Z, Z != 0, n, trim, min_rep, min_overlap),
                            *[_scan(Z, Z == v, n, trim, min_rep, min_overlap) for v in np.unique(Z) if v != 0
                              and (Z == v).mean() > 0.04])
                if r), key=lambda r: r["mismatch"], default=None)


def _scan(Z, B, n, trim, min_rep, min_overlap):
    # ряды, одинаковые по всей длине полосы (рамочные линии по краям), ничего не говорят о шаге
    const_cols = (B == B[:1]).all(0)
    Bc = B[:, ~const_cols] if (~const_cols).any() else B
    best = []
    for T in range(3, int(n / min_rep) + 1):
        if n - T < min_overlap: break
        a, b = Bc[:-T], Bc[T:]
        ink = a | b
        if ink.sum() < 10: continue
        mm = float(((a != b) & ink).sum() / ink.sum())
        best.append((mm, T))
    if not best: return None
    mmin = min(m for m, _ in best)
    for m, T in sorted(best, key=lambda x: x[1]):
        if m <= mmin + TIE: return {"period": T, "mismatch": m, "trim": trim, "Z": Z}
    return None


def _unit(Z, T):
    """Голосование по повторам → чистый фрагмент; выбор начала фрагмента по самому «пустому» ряду."""
    n = Z.shape[0]
    reps = n // T
    stack = np.stack([Z[k * T:(k + 1) * T] for k in range(reps)], 0)            # reps×T×w
    vals = np.zeros(stack.shape[1:], np.int16); agree = np.zeros(stack.shape[1:], float)
    for j in range(T):
        for i in range(stack.shape[2]):
            v, c = np.unique(stack[:, j, i], return_counts=True)
            vals[j, i] = v[np.argmax(c)]; agree[j, i] = c.max() / reps
    fill = (vals != 0).sum(1)                                  # сколько «рисунка» в каждом ряду
    o = int(np.argmin(fill + 1e-3 * np.arange(T)))             # начало — самый пустой ряд
    unit = np.roll(vals, -o, axis=0)
    # клетки, расходящиеся с большинством (кандидаты в ошибки оцифровки) — в координатах полосы Z
    wrong = []
    for k in range(reps):
        d = np.argwhere(stack[k] != vals)
        wrong += [[int(k * T + j), int(i)] for j, i in d]
    return unit, o, float(agree.mean()), wrong


def find_rapport(K):
    """K — матрица рисунка (h×w, 0 = фон). Возвращает dict или None, если повтора нет."""
    h, w = K.shape
    cands = []
    if h >= 1.5 * w or h >= w:
        r = _find_along_rows(K)
        if r: cands.append(("v", r, K))
    if w >= 1.5 * h or w > h:
        r = _find_along_rows(K.T)
        if r: cands.append(("h", r, K.T))
    if not cands: return None
    axis, r, KK = min(cands, key=lambda c: c[1]["mismatch"])
    if r["mismatch"] > MAX_MISMATCH: return {"found": False, "axis": axis, "best_mismatch": round(r["mismatch"], 3)}
    T, Z = r["period"], r["Z"]
    unit, o, agree, wrong = _unit(Z, T)
    if Z.shape[0] // T >= 2 and agree < MIN_AGREE:          # силуэт повторяется, а цвета — нет: скорее совпадение
        return {"found": False, "axis": axis, "best_mismatch": round(r["mismatch"], 3), "agreement": round(agree, 3)}
    ink = unit != 0
    mirror = float((unit == unit[:, ::-1])[ink | ink[:, ::-1]].mean()) if ink.any() else 0.0      # поперёк полосы
    glide = None
    if T % 2 == 0:
        a, b = unit[:T // 2], unit[T // 2:][::-1]              # вторая половина = первая, отражённая вдоль полосы
        m = (a != 0) | (b != 0)
        glide = float((a == b)[m].mean()) if m.any() else None
    U = unit if axis == "v" else unit.T                        # фрагмент в ориентации исходного рисунка
    off = r["trim"] + o                                        # где начинается первый полный фрагмент (в клетках вдоль оси)
    return {
        "found": True, "axis": axis, "period": int(T), "offset": int(off),
        "match": round(1 - r["mismatch"], 3), "repeats": round(Z.shape[0] / T, 1), "agreement": round(agree, 3),
        "mirror_across": round(mirror, 3), "is_mirror": mirror >= 0.95,
        "glide": None if glide is None else round(glide, 3), "is_glide": bool(glide is not None and glide >= 0.95),
        "check": bool(agree < 0.9 or r["mismatch"] > 0.06 or Z.shape[0] / T < 2),   # показать человеку
        "unit": {"w": int(U.shape[1]), "h": int(U.shape[0]), "rows": _rows_to_str(U)},
        # клетки, где оцифровка расходится с большинством повторов: [ряд, столбец] в координатах рисунка
        "suspect_cells": [([j + r["trim"], i] if axis == "v" else [i, j + r["trim"]]) for j, i in wrong][:300],
    }


def for_work(work):
    """Посчитать раппорт для work-файла студии (если есть матрица и рисунок похож на полосу)."""
    m = work.get("matrix")
    if not m: return None
    K = matrix_to_array(m)
    h, w = K.shape
    is_strip = max(h, w) >= 1.9 * min(h, w)
    sect = ((work.get("meta") or {}).get("section") or "").lower()
    if not is_strip and not any(s in sect for s in ("кайм", "бордюр", "haşiy", "border")): return None
    res = find_rapport(K)
    if res is not None: res["computed_at_matrix"] = [w, h]
    return res
