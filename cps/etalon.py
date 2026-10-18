# etalon.py — v11.24: «эталоны» и детектор по образцам.
#
# Элемент = все ПОДТВЕРЖДЁННЫЕ (✓) рамки с одним названием (узор словаря, или название текстом, или схема).
# Детектор элемента учится на этих рамках: признаки окон — те же, что у поиска «где ещё встречается» (DINOv2 по сетке
# квадратов, motif_search.vec / _feat). Положительные примеры — ✓ рамки элемента; отрицательные — рамки, которые человек
# отклонил (✗), ✓ рамки других элементов и случайные окна тех же ковров вне элемента. Линейный классификатор
# (логистическая регрессия) — точнее, чем «среднее образцов»: он учится, ЧЕМ элемент отличается от соседних узоров.
# Чем больше ✓ и ✗, тем точнее. Находки становятся машинными рамками (пунктир «?», note «эталон: …») — их проверяет человек.
import os, json, time, threading, math, random
import numpy as np
import motif_search as MS

state = {"running": False, "done": 0, "total": 0, "stage": "", "found": 0, "error": "", "last": "", "finished": ""}
NOTE = "эталон: "
MIN_POS = 3


_alias = {}      # v11.29: «t:dəvə» (название текстом) = «m:deve» (узор словаря с тем же названием) — один элемент


def key_of(m):
    """ключ элемента: узор словаря → «m:<id>»; название текстом → «t:<текст>» (или узор словаря с таким названием); только схема → «s:<id>»"""
    if m.get("motif"): return "m:" + m["motif"]
    if (m.get("motif_text") or "").strip():
        k = "t:" + m["motif_text"].strip().lower()
        return _alias.get(k, k)
    if m.get("scheme"): return "s:" + m["scheme"]
    return ""


def elements(A, names=None):
    """A — {rid: ковёр}. → {key: {name, pos: [(rid, box)], neg: [(rid, box)], rugs, types, schools, pending}}"""
    E = {}
    for mid, nm in (names or {}).items():
        for v in {mid, nm or ""}:
            if v.strip(): _alias["t:" + v.strip().lower()] = "m:" + mid
    for rid, r in A.items():
        if r.get("dup_of"): continue
        for m in r.get("marks") or []:
            k = key_of(m)
            if not k: continue
            e = E.setdefault(k, {"key": k, "name": "", "motif": m.get("motif", ""), "motif_text": m.get("motif_text", ""), "scheme": m.get("scheme", ""),
                                 "pos": [], "neg": [], "rugs": set(), "types": {}, "schools": {}, "pending": 0})
            if not e["name"]: e["name"] = (names or {}).get(m.get("motif")) or m.get("motif_text") or m.get("motif") or m.get("scheme")
            if not e["scheme"] and m.get("scheme"): e["scheme"] = m["scheme"]
            st = m.get("status")
            if st == "confirmed":
                e["pos"].append((rid, m["box"])); e["rugs"].add(rid); e.setdefault("pos_m", []).append((rid, m.get("id", ""), m.get("angle") or 0))
                c = r.get("composition") or ""; s = r.get("school") or ""
                if c: e["types"][c] = e["types"].get(c, 0) + 1
                if s: e["schools"][s] = e["schools"].get(s, 0) + 1
            elif st == "rejected": e["neg"].append((rid, m["box"]))
            elif m.get("machine") and str(m.get("note") or "").startswith(NOTE): e["pending"] += 1
    return E


def shapes(e, A):
    """v11.29: образцы одного названия часто — РАЗНЫЕ фигуры (длинная планка и крюк по бокам под одним «qeyçi»), или одна
    фигура в разных положениях (лёжа / стоя). Окно детектора одно на модель и не поворачивается — смесь форм даёт
    «среднее окно», которое режет одно и захватывает лишнее у другого. Делим образцы по форме рамки на фото
    (пропорции в пикселях и площадь): близкие — одна форма. → [[индексы e["pos"]], …], крупные группы первыми"""
    P = e["pos"]; n = len(P)
    if n < 2: return [list(range(n))]
    f = []
    for rid, b in P:
        r = A.get(rid) or {}; W, H = r.get("w") or 1, r.get("h") or 1
        f.append((math.log(max(1e-4, b[2] * W) / max(1e-4, b[3] * H)), math.log(max(1e-6, b[2] * b[3]))))
    par = list(range(n))
    def fd(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if ((f[i][0] - f[j][0]) / 0.35) ** 2 + ((f[i][1] - f[j][1]) / 0.9) ** 2 < 1: par[fd(i)] = fd(j)
    g = {}
    for i in range(n): g.setdefault(fd(i), []).append(i)
    return sorted(g.values(), key=len, reverse=True)


def sub(e, idx, i, total):
    """элемент только из образцов одной формы (ключ тот же — находки остаются этим элементом)"""
    s = dict(e); s["pos"] = [e["pos"][k] for k in idx]; s["rugs"] = {p[0] for p in s["pos"]}
    s["shape"] = i if total > 1 else None
    return s


def _sig(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _fit(X, y, l2=1e-2, it=400, lr=0.5):
    """логистическая регрессия (веса классов уравнены)"""
    n, d = X.shape; w = np.zeros(d, np.float32); b = 0.0
    cw = np.where(y > 0, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum())).astype(np.float32)
    for _ in range(it):
        p = _sig(X @ w + b); g = (p - y) * cw
        w -= lr * (X.T @ g + l2 * w); b -= lr * float(g.sum())
    return w, b


def _rand_boxes(rng, size, avoid, n):
    out = []; w, h = size
    for _ in range(n * 6):
        b = [rng.uniform(0, max(0.0, 1 - w)), rng.uniform(0, max(0.0, 1 - h)), w, h]
        if all(MS._iou(b, a) < 0.1 for a in avoid): out.append(b)
        if len(out) >= n: break
    return out


_CR = {}


def carpet_of(rid, A, load_img):
    """прямоугольник ковра на фото (доли): разметка зон человека, иначе — автоматически (rug_layout), с кэшем"""
    L = (A.get(rid) or {}).get("layout") or {}
    if L.get("carpet"): return L["carpet"]
    if rid in _CR: return _CR[rid]
    c = [0, 0, 1, 1]
    if load_img:
        try:
            import rug_layout, cv2
            a = load_img(rid)
            if a is not None:
                k = min(1.0, 700.0 / max(a.shape[:2])); a = cv2.resize(a, None, fx=k, fy=k) if k < 1 else a
                d = rug_layout.detect(a)
                if d.get("conf", 0) >= 0.6 and d["carpet"][2] * d["carpet"][3] > 0.2: c = d["carpet"]
        except Exception: pass
    _CR[rid] = c
    return c


# варианты настройки: какой лучше — решает самопроверка на ваших же ✓ (оставить один ковёр, найти на нём, сравнить рамки)
CFGS = {
    "A": dict(parts=False, scale="photo", scales=(0.75, 1.0, 1.3), refine=None),
    "B": dict(parts=False, scale="carpet", scales=(0.8, 1.0, 1.25), refine=None),
    "C": dict(parts=False, scale="carpet", scales=(0.8, 1.0, 1.25), refine="tight"),
    "D": dict(parts=True, scale="carpet", scales=(0.8, 1.0, 1.25), refine="tight"),
    "E": dict(parts=False, scale="photo", scales=(0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5), refine="wide"),
}
DEFAULT_CFG = "C"


def train(e, E, A, load_img=None, parts=True):
    """→ модель {w, b, size (доли фото), thr, npos, nneg} или None (мало примеров / нет индекса)"""
    rng = random.Random(1)
    pos = e["pos"][:120]
    if len(pos) < MIN_POS: return None
    Xp, sizes = [], []
    for rid, bx in pos:
        G = MS.ensure(rid)
        if G is None: continue
        v, _ = MS.vec(G, bx); Xp.append(v); cr = carpet_of(rid, A, load_img)
        sizes.append((bx[2], bx[3], bx[2] / max(0.05, cr[2]), bx[3] / max(0.05, cr[3])))
    if len(Xp) < MIN_POS: return None
    Xn = []
    for rid, bx in e["neg"][:200]:                                   # отклонённые человеком
        G = MS.grid(rid)
        if G is not None: Xn.append(MS.vec(G, bx)[0])
    posby = {}
    for rid, bx in pos: posby.setdefault(rid, []).append(bx)
    # ✓ рамки других элементов — отрицательные, кроме тех, что лежат на тех же местах (тот же узор под другим названием)
    others = [p for k, o in E.items() if k != e["key"] for p in o["pos"] if not any(MS._iou(p[1], b) > 0.3 for b in posby.get(p[0], []))]
    rng.shuffle(others)
    for rid, bx in others[:150]:                                     # другие элементы
        G = MS.grid(rid)
        if G is not None: Xn.append(MS.vec(G, bx)[0])
    size = (float(np.median([s[0] for s in sizes])), float(np.median([s[1] for s in sizes])))
    csize = (float(np.median([s[2] for s in sizes])), float(np.median([s[3] for s in sizes])))      # доля ковра
    by = {}
    for rid, bx in pos: by.setdefault(rid, []).append(bx)
    for rid, bxs in by.items():                                      # фон тех же ковров вне элемента
        G = MS.grid(rid)
        if G is None: continue
        for b in _rand_boxes(rng, size, bxs + [m["box"] for m in (A.get(rid) or {}).get("marks") or [] if key_of(m) == e["key"]], 8):
            Xn.append(MS.vec(G, b)[0])
    # куски элемента и окно «с лишним» — тоже отрицательные: детектор учится выбирать окно ЦЕЛОГО элемента, а не его середину
    for rid, bx in (pos[:60] if parts else []):
        G = MS.grid(rid)
        if G is None: continue
        x, y, w, h = bx
        for b in ([x + w * .25, y + h * .25, w * .5, h * .5], [x, y, w, h * .5], [x, y + h * .5, w, h * .5], [x, y, w * .5, h], [x + w * .5, y, w * .5, h],
                  [x - w * .45, y - h * .45, w * 1.9, h * 1.9]):
            b = [max(0.0, b[0]), max(0.0, b[1]), min(b[2], 1 - max(0.0, b[0])), min(b[3], 1 - max(0.0, b[1]))]
            if b[2] > 0.01 and b[3] > 0.01: Xn.append(MS.vec(G, b)[0])
    if len(Xn) < 5: return None
    X = np.stack(Xp + Xn).astype(np.float32); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))].astype(np.float32)
    w, b = _fit(X, y)
    pp = _sig(np.stack(Xp) @ w + b); pn = _sig(np.stack(Xn) @ w + b)
    thr = float(max(0.6, min(0.9, np.percentile(pp, 20)), np.percentile(pn, 98) if len(pn) else 0))   # не ниже почти всех отрицательных
    return {"w": w, "b": b, "size": size, "csize": csize, "thr": min(thr, 0.97), "npos": len(Xp), "nneg": len(Xn),
            "fit_pos": float(np.mean(pp > thr)), "fit_neg": float(np.mean(pn > thr)) if len(pn) else 0.0}


def detect(model, rid, per_rug=12, carpet=None, scales=(0.8, 1.0, 1.25)):
    """окна ковра с вероятностью ≥ порога: [(p, box)] без перекрытий (признаков нет — посчитать)"""
    G = MS.grid(rid)
    if G is None:
        try: G = MS.ensure(rid)
        except Exception: G = None
    if G is None: return []
    gh, gw, d = G.shape; S = MS._integral(G); gmean = S[-1, -1] / (gh * gw)
    cand = []
    bw, bh = (model["csize"][0] * carpet[2], model["csize"][1] * carpet[3]) if carpet and model.get("csize") else model["size"]
    for k in scales:
        sw = max(1, min(gw, int(round(bw * gw * k)))); sh = max(1, min(gh, int(round(bh * gh * k))))
        P = _sig(MS._feat(S, sh, sw, gmean) @ model["w"] + model["b"])
        ys, xs = np.where(P >= model["thr"])
        for yy, xx in zip(ys, xs): cand.append((float(P[yy, xx]), [xx / gw, yy / gh, sw / gw, sh / gh]))
    cand.sort(key=lambda c: -c[0]); out = []
    for p, b in cand:
        if all(MS._iou(b, o[1]) < 0.25 for o in out): out.append((p, b))
        if len(out) >= per_rug: break
    return out


TF8 = [(a, f) for f in (0, 1) for a in (0, 90, 180, 270)]


def templates(e, load_img, n=3):
    """образцы элемента в пикселях: самые крупные ✓ рамки (с их фото)"""
    out = []
    for rid, bx in e["pos"]:
        a = load_img(rid)
        if a is None: continue
        H, W = a.shape[:2]; x0, y0, x1, y1 = int(bx[0] * W), int(bx[1] * H), int((bx[0] + bx[2]) * W), int((bx[1] + bx[3]) * H)
        if x1 - x0 < 8 or y1 - y0 < 8: continue
        out.append(((x1 - x0) * (y1 - y0), a[y0:y1, x0:x1].copy()))
    out.sort(key=lambda t: -t[0])
    return [t for _, t in out[:n]]


REF = {"tight": dict(reg=(1.0, 2.0), ctr=(0.4, 1.4), sc=(0.8, 0.9, 1.0, 1.12, 1.25)),
       "wide": dict(reg=(1.8, 2.8), ctr=(0.6, 1.6), sc=(0.7, 0.9, 1.15, 1.45, 1.85, 2.35, 3.0))}


def refine(img, tpls, box, mode="tight"):
    """уточнить рамку находки по пикселям: окно детектора грубое (сетка признаков ≈ 14 px на 518 px фото) — часто режет
    элемент. Ищем образец (8 положений, 5 масштабов, пропорции — как у образца) рядом с находкой по контурам.
    → (box, angle, flip, score) или None"""
    import cv2, rug_match as RM
    H, W = img.shape[:2]; x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    side = max(w, h); k = min(1.0, 72.0 / max(1.0, side))                   # элемент ≈ 72 px — хватает и быстро
    P_ = REF[mode]; a_, b_ = P_["reg"]
    rx0, ry0 = max(0, int(x - a_ * w)), max(0, int(y - a_ * h)); rx1, ry1 = min(W, int(x + b_ * w)), min(H, int(y + b_ * h))
    reg = img[ry0:ry1, rx0:rx1]
    if reg.size == 0: return None
    reg = cv2.resize(reg, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    _, ER = RM._maps(reg); best = None
    for t in tpls:
        _, ET0 = RM._maps(t)
        for ang, fl in TF8:
            et = RM._tf(ET0, ang, fl); th, tw = et.shape[:2]
            area = w * h * k * k
            for sc in P_["sc"]:
                f = math.sqrt(area * sc * sc / max(1, th * tw)); nw, nh = max(6, int(tw * f)), max(6, int(th * f))
                if nw >= ER.shape[1] or nh >= ER.shape[0]: continue
                r = cv2.matchTemplate(ER, cv2.resize(et, (nw, nh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
                # центр найденного должен оставаться около окна детектора (окно нашло элемент, уточняем — не ищем заново)
                c0, c1 = P_["ctr"]
                cx0, cx1 = (x - c0 * w - rx0) * k - nw / 2, (x + c1 * w - rx0) * k - nw / 2
                cy0, cy1 = (y - c0 * h - ry0) * k - nh / 2, (y + c1 * h - ry0) * k - nh / 2
                msk = np.full(r.shape, -1.0, np.float32)
                ya, yb, xa, xb = max(0, int(cy0)), min(r.shape[0], int(cy1) + 1), max(0, int(cx0)), min(r.shape[1], int(cx1) + 1)
                if ya >= yb or xa >= xb: continue
                msk[ya:yb, xa:xb] = r[ya:yb, xa:xb]
                S1, S2 = cv2.integral(ER), cv2.integral(ER * ER)          # ровный фон даёт ложную «1.0» — окна без контуров не берём
                n_ = nw * nh; ws = S1[nh:, nw:] - S1[:-nh, nw:] - S1[nh:, :-nw] + S1[:-nh, :-nw]; wq = S2[nh:, nw:] - S2[:-nh, nw:] - S2[nh:, :-nw] + S2[:-nh, :-nw]
                sd = np.sqrt(np.maximum(wq / n_ - (ws / n_) ** 2, 0))[:r.shape[0], :r.shape[1]]
                msk[sd < 0.35 * float(ER.std() + 1e-6)] = -1
                _, v, _, p = cv2.minMaxLoc(msk)
                if best is None or v > best[0]: best = (v, p, nw, nh, ang, fl)
    if not best or best[0] < 0.3: return None
    v, (px, py), nw, nh, ang, fl = best
    bx = [(rx0 + px / k) / W, (ry0 + py / k) / H, nw / k / W, nh / k / H]
    return [round(min(max(c, 0.0), 1.0), 5) for c in bx], ang, fl, round(float(v), 3)


def find(cfg, model, tpls, rid, A, load_img):
    """находки на одном ковре по настройке cfg → [(p, box, angle, flip)]"""
    C = CFGS[cfg]
    cr = carpet_of(rid, A, load_img) if C["scale"] == "carpet" else None
    hits = detect(model, rid, carpet=cr, scales=C["scales"])
    hits = [(p, b, 0, 0) for p, b in hits]
    if C["refine"] and tpls and hits and load_img:
        img = load_img(rid)
        if img is not None:
            out = []
            for p, b, _, _ in hits:
                r = refine(img, tpls, b, C["refine"])
                if r: out.append((p, r[0], r[1], r[2]))
                elif C["refine"] == "wide": out.append((p, b, 0, 0))
            keep = []
            for h in out:                                       # уточнённые рамки могли съехаться на один элемент
                if all(MS._iou(h[1], k_[1]) < 0.3 for k_ in keep): keep.append(h)
            hits = keep
    return hits


def evaluate(e, E, A, load_img, cfgs=None):
    """самопроверка: по очереди прячем ✓ одного ковра, учимся на остальных, ищем на нём и сравниваем рамки с вашими.
    → {cfg: {recall (доля ваших ✓, найденных с перекрытием ≥ 50 %), iou (среднее лучшее перекрытие), hits (находок на ковёр)}}"""
    rugs = sorted(e["rugs"])
    if len(rugs) < 2: return {}
    res = {}
    for cfg in (cfgs or CFGS):
        C = CFGS[cfg]; rec = []; ious = []; nh = []
        for r in rugs[:8]:
            e2 = {**e, "pos": [p for p in e["pos"] if p[0] != r], "rugs": set(rugs) - {r}}
            if len(e2["pos"]) < 2: continue
            m = train(e2, E, A, load_img, parts=C["parts"])
            if not m: continue
            tp = templates(e2, load_img) if (C["refine"] and load_img) else []
            hits = find(cfg, m, tp, r, A, load_img); nh.append(len(hits))
            for _, gt in [p for p in e["pos"] if p[0] == r]:
                best = max([MS._iou(gt, h[1]) for h in hits] or [0.0])
                ious.append(best); rec.append(best >= 0.5)
        if rec: res[cfg] = {"recall": round(float(np.mean(rec)), 3), "iou": round(float(np.mean(ious)), 3), "hits": round(float(np.mean(nh)), 1), "n": len(rec)}
    return res


def pick(ev):
    if not ev: return DEFAULT_CFG
    return max(ev, key=lambda c: ev[c]["recall"] + 0.5 * ev[c]["iou"] - 0.01 * max(0.0, ev[c]["hits"] - 6))


def targets(e, A):
    """где искать: ковры тех же типов, что у образцов; если у образцов нет типа — той же школы"""
    types = set(e["types"]); schools = set(e["schools"])
    out = []
    for rid, r in A.items():
        if r.get("dup_of") or r.get("split_into") or r.get("hidden") or rid in e["rugs"]: continue
        if (types and r.get("composition") in types) or (not types and schools and r.get("school") in schools): out.append(rid)
    return out


def run(keys, load_all, add_marks, heavy, names=None, on_done=None, limit_rugs=400, load_img=None, save_eval=None):
    """keys — какие элементы (None — все, у кого ≥ MIN_POS ✓); add_marks(rid, [mark]) → сколько добавлено"""
    if state["running"]: return False
    state.update(running=True, done=0, total=0, found=0, error="", stage="обучение")

    def go():
        try:
            A = load_all(); E = elements(A, names)
            todo, skipped = [], []
            for k in (keys or E):
                if k not in E or len(E[k]["pos"]) < MIN_POS: continue
                cl = shapes(E[k], A); big = [c for c in cl if len(c) >= MIN_POS]
                if not big: skipped.append(E[k]["name"]); continue
                todo += [sub(E[k], c, i, len(big)) for i, c in enumerate(big)]
            state["total"] = len(todo); rep = [f"{n}: образцы разной формы, каждой меньше {MIN_POS} ✓ — не искал" for n in skipped]
            for e in todo:
                sh = f" · форма {e['shape'] + 1}" if e.get("shape") is not None else ""
                state["stage"] = f"«{e['name']}{sh}»: самопроверка на {len(e['rugs'])} коврах с ✓"
                ev = heavy(lambda: evaluate(e, E, A, load_img), "эталоны: самопроверка")
                cfg = pick(ev); C = CFGS[cfg]
                if save_eval: save_eval(e["key"] + (f"#{e['shape'] + 1}" if sh else ""), {"name": e["name"] + sh, "eval": ev, "cfg": cfg, "npos": len(e["pos"]), "rugs": len(e["rugs"]), "at": int(time.time())})
                state["stage"] = f"«{e['name']}{sh}»: учусь на {len(e['pos'])} образцах"
                model = heavy(lambda: train(e, E, A, load_img, parts=C["parts"]), "эталоны: обучение")
                if not model: state["done"] += 1; continue
                tg = targets(e, A)[:limit_rugs]; n = 0
                tpls = heavy(lambda: templates(e, load_img), "эталоны: образцы") if (load_img and C["refine"]) else []
                state["stage"] = f"«{e['name']}{sh}»: ищу на {len(tg)} коврах"
                for rid in tg:
                    hits = heavy(lambda: find(cfg, model, tpls, rid, A, load_img), "эталоны: поиск")
                    if not hits: continue
                    g = ""; ms = []
                    for j, (p, b, ang, fl) in enumerate(hits):
                        mid = f"m{int(time.time()):x}e{abs(hash((e['key'], rid))) % 10**6}_{j}"; g = g or mid
                        ms.append({"id": mid, "box": [round(v, 5) for v in b], "zone": "other", "motif": e["motif"], "motif_text": e["motif_text"] if not e["motif"] else "",
                                   "scheme": e["scheme"], "group": g, "certainty": "sure" if p >= 0.9 else "maybe", "status": "proposed",
                                   "note": NOTE + e["name"] + sh, "machine": 1, "score": round(p, 3), "angle": ang, "flip": fl, "by": "машина", "refined": 3,
                                   "box0": [round(v, 5) for v in b]})
                    n += add_marks(rid, ms)
                state["found"] += n; state["done"] += 1
                ev_ = ev.get(cfg) or {}
                rep.append(f"{e['name']}{sh}: {n}" + (f" (самопроверка: рамки совпали в {round(100 * ev_['recall'])} %)" if ev_ else ""))
            state["last"] = "; ".join(rep) or "нечего искать — нужно ≥ 3 проверенных образца элемента"
            state["finished"] = time.strftime("%Y-%m-%d %H:%M")
            if on_done: on_done(f"найдено {state['found']} мест ({state['last'][:300]})")
        except Exception as ex:
            state["error"] = str(ex)[:300]
        finally:
            state.update(running=False, stage="")
    threading.Thread(target=go, daemon=True).start()
    return True
