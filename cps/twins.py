"""v11.86: двойники быстрее.

Кандидаты в «тот же ковёр» берутся из нескольких источников (не только у ковров с подтверждёнными углами):
  catalog — пары из групп каталога, ответа нет;
  size    — одинаковый размер (±2 см по обеим сторонам) у ковров из разных источников;
  inv     — один музей и один инвентарный номер;
  nn      — ближайшие соседи по «отпечатку» всего фото (DINOv2, считается в Colab: models/twin_nn_v1.json);
  calib   — пары, уже решённые человеком: по ним подбирается порог авторешения.
Каждая пара проверяется совмещением фото (rug_match.align): число общих точек n и перекрытие cov
(насколько ковёр одного фото ложится на ковёр другого; по углам ковров, если они есть, иначе по кадрам).

Авторешение «тот же» — только если на ответах людей нашёлся порог (n, cov) почти без ошибок (≤ 1 %)
и под ним не меньше MIN_SAME подтверждённых пар. Каждая десятая такая пара не ставится сама,
а идёт человеку контрольной карточкой: так видна настоящая точность.
Цепочки: A = B и B = C → A = C; A ≠ X и A = B → B ≠ X (группы не больше MAX_GROUP ковров).
Здесь только расчёты без файлов и сервера: их проверяют тесты."""
import hashlib, re

VER = 1
SIZE_TOL = 2              # см
MIN_SAME = 25             # столько подтверждённых «тот же» должно оказаться над порогом
BAD_SHARE = 0.01          # допустимая доля ответов «разные» над порогом (от числа «тот же» над ним)
CTL_EVERY = 10            # каждая десятая пара над порогом — контрольная
MAX_GROUP = 8             # больше — цепочка, скорее всего, держится на ошибке: не распространяем
GRID_N = (40, 60, 80, 100, 150, 200, 300, 500, 800)
GRID_COV = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
NN_TOP = 5                # первые 5 соседей по отпечатку проверяются всегда
NN_K = 12                 # дальше — до 12, если сходство не ниже порога из Colab (95 % настоящих двойников выше него)


def key(a, b):
    return "~".join(sorted((a, b)))


def size_of(r):
    t = r.get("size") or (r.get("cat") or {}).get("size") or ""
    m = re.search(r"(\d{2,3}(?:[.,]\d)?)\s*[x×х*]\s*(\d{2,3}(?:[.,]\d)?)", str(t))
    if not m: return None
    a, b = (float(m.group(i).replace(",", ".")) for i in (1, 2))
    if a < 20 or b < 20: return None
    return tuple(sorted((a, b)))


def inv_of(r):
    inv = re.sub(r"\D", "", r.get("inv") or "")
    mus = re.sub(r"\W+", "", (r.get("museum") or "").lower())[:24]
    return (mus, inv) if inv and mus and not re.search(r"wikimedia|commons", mus) else None


def src_of(r):
    return (r.get("cat") or {}).get("src") or ("upload" if not r.get("src_id") else str(r.get("src_id")).split(":")[0])


def same_page(ra, rb):
    """части одного разворота или одной страницы книги — разные ковры, даже если фото «совпадают» (общие поля страницы)"""
    fa, fb = ra.get("split_from") or "", rb.get("split_from") or ""
    if fa and fa == fb: return True
    if ra["id"] in (fb,) or rb["id"] in (fa,): return True
    pa, pb = re.sub(r"_\d+$", "", ra["id"]), re.sub(r"_\d+$", "", rb["id"])
    return pa == pb and re.search(r"_p\d+$", pa) is not None


def decided(ra, b):
    v = (ra.get("same_ok") or {}).get(b)
    if v is None: return None
    return "same" if v.get("ok") else ("similar" if v.get("similar") else "diff")


def human(v):
    """решение поставил человек (а не машина и не цепочка)"""
    by = str((v or {}).get("by") or "")
    return bool(by) and not by.startswith("машина") and not by.startswith("по цепочке")


def candidates(rugs, nn=None, alive=lambda r: True, nn_thr=1.0):
    """rugs: {id: ковёр}; nn: {id: [[id2, сходство], …]} — соседи по отпечатку (по убыванию). → {ключ пары: set(источники)}"""
    L = {i: r for i, r in rugs.items() if alive(r)}
    out = {}

    def add(a, b, why):
        if a == b or a not in L or b not in L or same_page(L[a], L[b]): return
        if decided(L[a], b) is not None and why != "calib": return
        out.setdefault(key(a, b), set()).add(why)

    for i, r in L.items():
        for x in r.get("same") or []:
            o = x.get("id")
            if o and decided(r, o) is None: add(i, o, "catalog")
        for o, v in (r.get("same_ok") or {}).items():
            if v is not None and human(v) and o in L and i < o: add(i, o, "calib")
    S = sorted((s, i) for i, s in ((i, size_of(r)) for i, r in L.items()) if s)
    for k, (s1, a) in enumerate(S):                                      # отсортировано по короткой стороне: смотрим только соседей
        for s2, b in S[k + 1:]:
            if s2[0] - s1[0] > SIZE_TOL: break
            if abs(s2[1] - s1[1]) <= SIZE_TOL and src_of(L[a]) != src_of(L[b]): add(a, b, "size")
    by_inv = {}
    for i, r in L.items():
        v = inv_of(r)
        if v: by_inv.setdefault(v, []).append(i)
    for ids in by_inv.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)): add(ids[x], ids[y], "inv")
    for i, lst in (nn or {}).items():
        for j, (o, s) in enumerate((lst or [])[:NN_K]):
            if j < NN_TOP or s >= nn_thr: add(i, o, "nn")
    return out


def calibrate(rows):
    """rows: [(n, cov, ответ человека)] → порог {n, cov, same, bad, total}, где «тот же» больше всего, а «разные» над порогом — не больше 1 %
    подтверждённых (люди тоже ошибаются: ≈1 % случайных промахов по контрольным карточкам; меньше 100 «тот же» — ни одного); None — нельзя"""
    best = None
    for tn in GRID_N:
        for tc in GRID_COV:
            up = [a for n, c, a in rows if n >= tn and c >= tc]
            s = sum(1 for a in up if a == "same"); bad = len(up) - s
            if bad <= int(s * BAD_SHARE) and s >= MIN_SAME and (best is None or s > best["same"] or (s == best["same"] and (tn, tc) < (best["n"], best["cov"]))):
                best = {"n": tn, "cov": tc, "same": s, "bad": bad}
    if best: best["total"] = len(rows); best["same_all"] = sum(1 for *_, a in rows if a == "same")
    return best


def over(v, thr):
    return bool(thr) and v.get("n", 0) >= thr["n"] and v.get("cov", 0) >= thr["cov"]


def is_ctl(k):
    return int(hashlib.md5(k.encode()).hexdigest()[:6], 16) % CTL_EVERY == 0


def closure(rugs):
    """→ (новые «тот же»: [(a, b, группа)], новые «разные»: [(a, b, почему)], противоречия: [(a, b)], большие группы: [[ids]])"""
    par = {}

    def f(x):
        par.setdefault(x, x)
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    diff = []
    for i, r in rugs.items():
        for o, v in (r.get("same_ok") or {}).items():
            if v is None or o not in rugs: continue
            if v.get("ok"): par[f(i)] = f(o)
            elif i < o: diff.append((i, o))
    G = {}
    for x in list(par): G.setdefault(f(x), []).append(x)
    big = [sorted(g) for g in G.values() if len(g) > MAX_GROUP]
    new_same, new_diff, conflict = [], [], []
    for g in G.values():
        if len(g) > MAX_GROUP: continue
        g = sorted(g)
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                a, b = g[x], g[y]
                d = decided(rugs[a], b)
                if d is None: new_same.append((a, b, g))
                elif d != "same": conflict.append((a, b))
    seen = set()
    for a, b in diff:
        ga = G.get(f(a), [a]) if a in par else [a]; gb = G.get(f(b), [b]) if b in par else [b]
        if f(a) == f(b) or len(ga) > MAX_GROUP or len(gb) > MAX_GROUP: continue
        for x in ga:
            for y in gb:
                k = key(x, y)
                if k in seen or decided(rugs[x], y) is not None: continue
                seen.add(k); new_diff.append((x, y, (a, b)))
    return new_same, new_diff, conflict, big


def suspects(rows_k, thr=None, k=20):
    """ответы людей «разные» с самым сильным совпадением фото — их стоит перепроверить. rows_k: [(ключ, n, cov, ответ)]"""
    L = [(n * (0.5 + c), key_, n, c) for key_, n, c, a in rows_k if a != "same" and n >= 40]
    return [{"k": key_, "n": n, "cov": c} for _, key_, n, c in sorted(L, reverse=True)[:k]]
