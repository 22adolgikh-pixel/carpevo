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


def key_of(m):
    """ключ элемента: узор словаря → «m:<id>»; название текстом → «t:<текст>»; только схема → «s:<id>»"""
    if m.get("motif"): return "m:" + m["motif"]
    if (m.get("motif_text") or "").strip(): return "t:" + m["motif_text"].strip().lower()
    if m.get("scheme"): return "s:" + m["scheme"]
    return ""


def elements(A, names=None):
    """A — {rid: ковёр}. → {key: {name, pos: [(rid, box)], neg: [(rid, box)], rugs, types, schools, pending}}"""
    E = {}
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
                e["pos"].append((rid, m["box"])); e["rugs"].add(rid)
                c = r.get("composition") or ""; s = r.get("school") or ""
                if c: e["types"][c] = e["types"].get(c, 0) + 1
                if s: e["schools"][s] = e["schools"].get(s, 0) + 1
            elif st == "rejected": e["neg"].append((rid, m["box"]))
            elif m.get("machine") and str(m.get("note") or "").startswith(NOTE): e["pending"] += 1
    return E


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


def train(e, E, A):
    """→ модель {w, b, size (доли фото), thr, npos, nneg} или None (мало примеров / нет индекса)"""
    rng = random.Random(1)
    pos = e["pos"][:120]
    if len(pos) < MIN_POS: return None
    Xp, sizes = [], []
    for rid, bx in pos:
        G = MS.ensure(rid)
        if G is None: continue
        v, _ = MS.vec(G, bx); Xp.append(v); sizes.append((bx[2], bx[3]))
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
    by = {}
    for rid, bx in pos: by.setdefault(rid, []).append(bx)
    for rid, bxs in by.items():                                      # фон тех же ковров вне элемента
        G = MS.grid(rid)
        if G is None: continue
        for b in _rand_boxes(rng, size, bxs + [m["box"] for m in (A.get(rid) or {}).get("marks") or [] if key_of(m) == e["key"]], 8):
            Xn.append(MS.vec(G, b)[0])
    if len(Xn) < 5: return None
    X = np.stack(Xp + Xn).astype(np.float32); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))].astype(np.float32)
    w, b = _fit(X, y)
    pp = _sig(np.stack(Xp) @ w + b); pn = _sig(np.stack(Xn) @ w + b)
    thr = float(max(0.6, min(0.9, np.percentile(pp, 20)), np.percentile(pn, 98) if len(pn) else 0))   # не ниже почти всех отрицательных
    return {"w": w, "b": b, "size": size, "thr": min(thr, 0.97), "npos": len(Xp), "nneg": len(Xn),
            "fit_pos": float(np.mean(pp > thr)), "fit_neg": float(np.mean(pn > thr)) if len(pn) else 0.0}


def detect(model, rid, per_rug=12):
    """окна ковра с вероятностью ≥ порога: [(p, box)] без перекрытий (признаков нет — посчитать)"""
    G = MS.grid(rid)
    if G is None:
        try: G = MS.ensure(rid)
        except Exception: G = None
    if G is None: return []
    gh, gw, d = G.shape; S = MS._integral(G); gmean = S[-1, -1] / (gh * gw)
    cand = []
    for k in (0.75, 1.0, 1.3):
        sw = max(1, min(gw, int(round(model["size"][0] * gw * k)))); sh = max(1, min(gh, int(round(model["size"][1] * gh * k))))
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


def refine(img, tpls, box):
    """уточнить рамку находки по пикселям: окно детектора грубое (сетка признаков ≈ 14 px на 518 px фото) — часто режет
    элемент. Ищем образец (8 положений, 5 масштабов, пропорции — как у образца) рядом с находкой по контурам.
    → (box, angle, flip, score) или None"""
    import cv2, rug_match as RM
    H, W = img.shape[:2]; x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    side = max(w, h); k = min(1.0, 72.0 / max(1.0, side))                   # элемент ≈ 72 px — хватает и быстро
    rx0, ry0 = max(0, int(x - 1.4 * w)), max(0, int(y - 1.4 * h)); rx1, ry1 = min(W, int(x + 2.4 * w)), min(H, int(y + 2.4 * h))
    reg = img[ry0:ry1, rx0:rx1]
    if reg.size == 0: return None
    reg = cv2.resize(reg, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    _, ER = RM._maps(reg); best = None
    for t in tpls:
        _, ET0 = RM._maps(t)
        for ang, fl in TF8:
            et = RM._tf(ET0, ang, fl); th, tw = et.shape[:2]
            area = w * h * k * k
            for sc in (0.9, 1.1, 1.35, 1.65, 2.0):            # окно детектора обычно МЕНЬШЕ элемента (режет его) — ищем крупнее
                f = math.sqrt(area * sc * sc / max(1, th * tw)); nw, nh = max(6, int(tw * f)), max(6, int(th * f))
                if nw >= ER.shape[1] or nh >= ER.shape[0]: continue
                r = cv2.matchTemplate(ER, cv2.resize(et, (nw, nh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
                # центр найденного должен оставаться около окна детектора (окно нашло элемент, уточняем — не ищем заново)
                cx0, cx1 = (x - 0.25 * w - rx0) * k - nw / 2, (x + 1.25 * w - rx0) * k - nw / 2
                cy0, cy1 = (y - 0.25 * h - ry0) * k - nh / 2, (y + 1.25 * h - ry0) * k - nh / 2
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
    if not best or best[0] < 0.25: return None
    v, (px, py), nw, nh, ang, fl = best
    bx = [(rx0 + px / k) / W, (ry0 + py / k) / H, nw / k / W, nh / k / H]
    return [round(min(max(c, 0.0), 1.0), 5) for c in bx], ang, fl, round(float(v), 3)


def targets(e, A):
    """где искать: ковры тех же типов, что у образцов; если у образцов нет типа — той же школы"""
    types = set(e["types"]); schools = set(e["schools"])
    out = []
    for rid, r in A.items():
        if r.get("dup_of") or r.get("split_into") or r.get("hidden") or rid in e["rugs"]: continue
        if (types and r.get("composition") in types) or (not types and schools and r.get("school") in schools): out.append(rid)
    return out


def run(keys, load_all, add_marks, heavy, names=None, on_done=None, limit_rugs=400, load_img=None):
    """keys — какие элементы (None — все, у кого ≥ MIN_POS ✓); add_marks(rid, [mark]) → сколько добавлено"""
    if state["running"]: return False
    state.update(running=True, done=0, total=0, found=0, error="", stage="обучение")

    def go():
        try:
            A = load_all(); E = elements(A, names)
            todo = [E[k] for k in (keys or E) if k in E and len(E[k]["pos"]) >= MIN_POS]
            state["total"] = len(todo); rep = []
            for e in todo:
                state["stage"] = f"«{e['name']}»: учусь на {len(e['pos'])} образцах"
                model = heavy(lambda: train(e, E, A), "эталоны: обучение")
                if not model: state["done"] += 1; continue
                tg = targets(e, A)[:limit_rugs]; n = 0
                tpls = heavy(lambda: templates(e, load_img), "эталоны: образцы") if load_img else []
                state["stage"] = f"«{e['name']}»: ищу на {len(tg)} коврах"
                for rid in tg:
                    hits = heavy(lambda: detect(model, rid), "эталоны: поиск")
                    if not hits: continue
                    if tpls:                                     # рамка — по пикселям, с положением (поворот / зеркало)
                        img = load_img(rid)
                        if img is not None:
                            rf = [heavy(lambda: refine(img, tpls, b), "эталоны: уточнение") for _, b in hits]
                            hits = [(p, r[0], r[1], r[2]) if r else (p, b, 0, 0) for (p, b), r in zip(hits, rf)]
                    hits = [h if len(h) == 4 else (h[0], h[1], 0, 0) for h in hits]
                    g = ""; ms = []
                    for j, (p, b, ang, fl) in enumerate(hits):
                        mid = f"m{int(time.time()):x}e{abs(hash((e['key'], rid))) % 10**6}_{j}"; g = g or mid
                        ms.append({"id": mid, "box": [round(v, 5) for v in b], "zone": "other", "motif": e["motif"], "motif_text": e["motif_text"] if not e["motif"] else "",
                                   "scheme": e["scheme"], "group": g, "certainty": "sure" if p >= 0.9 else "maybe", "status": "proposed",
                                   "note": NOTE + e["name"], "machine": 1, "score": round(p, 3), "angle": ang, "flip": fl, "by": "машина", "refined": 1 if tpls else 0})
                    n += add_marks(rid, ms)
                state["found"] += n; state["done"] += 1
                rep.append(f"{e['name']}: {n}")
            state["last"] = "; ".join(rep) or "нечего искать — нужно ≥ 3 проверенных образца элемента"
            state["finished"] = time.strftime("%Y-%m-%d %H:%M")
            if on_done: on_done(f"найдено {state['found']} мест ({state['last'][:300]})")
        except Exception as ex:
            state["error"] = str(ex)[:300]
        finally:
            state.update(running=False, stage="")
    threading.Thread(target=go, daemon=True).start()
    return True
