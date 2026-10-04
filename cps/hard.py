# hard.py — «трудные места» (v10.13): куски TILE×TILE клеток, где пороги и модель клеток расходятся.
# Человек проверяет только эти куски; остальное (где два независимых метода согласны) считается верным.
# Замер на 574 принятых ч/б рисунках (cps-data, 2026-10-04): куски с ≥1 расхождением — медиана 7 на рисунок,
# в них 98% всех ошибок автомата; после проверки только этих кусков остаток >2 ошибочных клеток — у 7% рисунков.
# Проверенные куски сразу идут в обучение модели (learn.load_examples), даже если рисунок ещё не принят.
import hashlib, json
import numpy as np

TILE = 10
MIN_N = 1          # кусок показывается, если в нём хотя бы столько расходящихся клеток (≥3 — вдвое хуже по остатку ошибок)
TOO_MANY = 25      # «слишком много»: больше стольких кусков И они покрывают больше TOO_MANY_SHARE рисунка —
TOO_MANY_SHARE = 0.7   # тогда кусками дольше, чем открыть целиком (в очереди 2026-10-04: 123 из 402 рисунков; остальные — ~19 кусков)


WEAK = 2.0         # сила линий сетки (autogrid.detect_grid, strength) ниже — подозрение «это фото ковра, а не схема»:
                   # в очереди 2026-10-04 таких 48 из 277 ЧБ, среди принятых 1 из 748 (из 12 просмотренных: 8 фото, 4 настоящие схемы)


def weak(w):
    """Рисунок со слабой сеткой, который человек ещё не подтвердил как схему."""
    a = w.get("auto") or {}
    s = a.get("grid_strength")
    return s is not None and s < WEAK and not a.get("scheme_ok")


def ensure_strength(w, g):
    """Досчитать auto.grid_strength для старых рисунков. → True, если записали."""
    a = w.get("auto")
    if not isinstance(a, dict) or a.get("grid_strength") is not None: return False
    import autogrid
    fr = w.get("frame") or [0, 0, g.shape[1], g.shape[0]]
    try: a["grid_strength"] = round(float(autogrid.detect_grid(g, fr)["strength"]), 3)
    except Exception: return False
    return True


def too_many(w):
    hd = w.get("hard") or {}; rem = remaining(w) or []
    return len(rem) > TOO_MANY and len(hd.get("tiles", [])) > TOO_MANY_SHARE * max(1, hd.get("ntiles", 1))


def sig(w):
    """Подпись геометрии: кэш кусков годен, пока сетка и рамка матрицы не менялись (правка пикселей его не сбивает)."""
    m = w.get("matrix") or {}
    s = json.dumps([w.get("quad"), w.get("grid"), m.get("w"), m.get("h"), m.get("origin")], sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def eligible(w):
    a = w.get("auto") or {}
    return bool(w.get("matrix") and w.get("quad") and w.get("grid") and a
                and not w.get("color_mode") and w.get("kind") != "photo" and not a.get("photo"))


def compute(w, g, P):
    """g — серый кроп, P — модель клеток. → {sig, tiles:[{x,y,w,h,n,cells}], ntiles} в координатах матрицы."""
    import autogrid, learn
    C, pitch = learn.final_centers(w)
    v = autogrid.cell_means(g, C, pitch)
    M0, info = autogrid.classify(v)
    Mt = autogrid.clean(M0, info["t"])
    Mm = learn.classify_cells(P, learn.cell_features(g, C, pitch))
    D = Mt != Mm
    h, wd = D.shape
    tiles, ntot = [], 0
    for y in range(0, h, TILE):
        for x in range(0, wd, TILE):
            d = D[y:y + TILE, x:x + TILE]; ntot += 1
            n = int(d.sum())
            if n >= MIN_N:
                ys, xs = np.nonzero(d)
                tiles.append({"x": x, "y": y, "w": int(d.shape[1]), "h": int(d.shape[0]), "n": n,
                              "cells": [[int(a) + x, int(b) + y] for a, b in zip(xs, ys)]})
    tiles.sort(key=lambda t: -t["n"])
    return {"sig": sig(w), "tiles": tiles, "ntiles": ntot}


def ensure(w, g, P):
    """Кэш в w['hard'] (пересчёт, если геометрия изменилась — тогда и отметки «проверено» сбрасываются). → True, если пересчитали."""
    hd = w.get("hard")
    if hd and hd.get("sig") == sig(w) and "tiles" in hd: return False
    r = compute(w, g, P); r["verified"] = []
    w["hard"] = r
    return True


def remaining(w):
    hd = w.get("hard") or {}
    if hd.get("sig") != sig(w): return None
    done = {(v[0], v[1]) for v in hd.get("verified", [])}
    return [t for t in hd.get("tiles", []) if (t["x"], t["y"]) not in done]


def verified_mask(w):
    """Маска проверенных человеком клеток (h×w матрицы) — для обучения на частично проверенных рисунках. None, если нет."""
    hd = w.get("hard") or {}; m = w.get("matrix") or {}
    if not hd.get("verified") or hd.get("sig") != sig(w): return None
    M = np.zeros((m["h"], m["w"]), bool)
    for x, y in hd["verified"]:
        M[y:y + TILE, x:x + TILE] = True
    return M


def apply_cells(w, cells):
    """cells: [[x, y, v]] в координатах матрицы, v ∈ {0,1,2}. → число изменённых клеток."""
    m = w["matrix"]; rows = [list(r) for r in m["rows"]]; n = 0
    for x, y, v in cells:
        x, y, v = int(x), int(y), int(v)
        if not (0 <= y < len(rows) and 0 <= x < len(rows[y]) and v in (0, 1, 2)): continue
        ch = "." if v == 0 else str(v)
        if rows[y][x] != ch: rows[y][x] = ch; n += 1
    m["rows"] = ["".join(r) for r in rows]
    if n and "2" in "".join(m["rows"]) and len(w.get("palette") or []) < 3:     # появилось серое в рисунке без серого — дополнить палитру
        w["palette"] = list(w.get("palette") or ["#ffffff", "#1a1a1a"])[:2] + ["#b8b8b8"]
        w["palette_names"] = list(w.get("palette_names") or ["фон", "обводка"])[:2] + ["тело 1"]
    return n
