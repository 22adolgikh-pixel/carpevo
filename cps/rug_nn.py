# rug_nn.py — тип ковра (композиция) по соседям ПО КАРТИНКЕ (v12.05).
#
# Тип указан только у 16 % ковров. Подсказка: ближайшие по отпечатку всего ковра (выпрямленного, если есть углы)
# ковры с известным типом, плюс рисунки Gans-Ruedin (211 целых ковров, подпись — название типа: «Lori-Pambak Kazak»…;
# рисунки от руки, в пиксели не переводятся, но по форме узора сравнимы с фото).
# Отпечаток — сеть element_embed «match» на всём ковре (длинная сторона по вертикали, 2 положения: как есть и 180°).
# Замер: каждый ковёр с типом ищется без себя и своих двойников; доля, где верный тип первый / среди трёх.
import numpy as np, cv2


def vec(EE, a):
    """BGR → вектор ковра (нормирован)"""
    if a is None or not a.size: return None
    if a.shape[1] > a.shape[0]: a = np.rot90(a).copy()
    M = EE.meta("match"); S = int(M.get("S", 112))
    c = cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA)
    v = EE._run([EE._prep(c), EE._prep(np.ascontiguousarray(c[::-1, ::-1]))], "match").mean(0)
    n = np.linalg.norm(v); return v / n if n else None


def group_of(r):
    """ковёр и его двойники — чтобы не искать ковёр по самому себе"""
    g = {r["id"]} | {x.get("id") for x in r.get("same") or [] if x.get("id")}
    if r.get("same_group"): g.add("g:" + str(r["same_group"]))
    return g


def build(EE, rugs, get_img, gans=(), log=print):
    """rugs — живые ковры; get_img(r) → BGR (выпрямленный или фото); gans — [(id, BGR, тип)] → банк"""
    ids, X, comp, grp = [], [], [], []
    for r in rugs:
        try: v = vec(EE, get_img(r))
        except Exception: v = None
        if v is None: continue
        ids.append(r["id"]); X.append(v); comp.append(r.get("composition") or ""); grp.append(group_of(r))
    G, GX, GC = [], [], []
    for gid, a, c in gans:
        v = vec(EE, a)
        if v is not None and c: G.append(gid); GX.append(v); GC.append(c)
    log(f"rug_nn: ковров {len(ids)}, с типом {sum(1 for c in comp if c)}, рисунков Gans-Ruedin с типом {len(G)}")
    return {"ids": ids, "X": np.stack(X) if X else np.zeros((0, 1)), "comp": comp, "grp": grp,
            "G": G, "GX": np.stack(GX) if GX else np.zeros((0, 1)), "GC": GC}


def _vote(sims, labs, k):
    o = [j for j in np.argsort(-sims) if labs[j]][:k]; sc = {}                  # только ковры с типом (свой и двойники уже без метки)
    for j in o:
        if labs[j]: sc[labs[j]] = sc.get(labs[j], 0) + float(sims[j])
    tot = sum(sc.values()) or 1
    return sorted(((c, v / tot, float(max(sims[j] for j in o if labs[j] == c))) for c, v in sc.items()), key=lambda t: -t[1])


def suggest(B, i=None, q=None, k=10, use_gans=True, skip=None):
    """→ [(тип, доля голосов, лучшая близость, откуда)] для ковра i банка (без него и его двойников) или вектора q"""
    if q is None: q = B["X"][i]
    skip = skip if skip is not None else (B["grp"][i] if i is not None else set())
    labs = [c if not (B["grp"][j] & skip) and B["ids"][j] not in skip else "" for j, c in enumerate(B["comp"])]
    s = B["X"] @ q
    out = {c: (v, b, "rugs") for c, v, b in _vote(s, labs, k)}
    if use_gans and len(B["G"]):
        gs = B["GX"] @ q; j = int(np.argmax(gs)); c = B["GC"][j]
        if c not in out or out[c][0] < 0.5: out[c] = (max(out.get(c, (0,))[0], 0.5), float(gs[j]), "gans:" + B["G"][j])
    return sorted(((c, *v) for c, v in out.items()), key=lambda t: -t[1])


def evaluate(B, k=10):
    idx = [i for i, c in enumerate(B["comp"]) if c]
    from collections import Counter
    prior = Counter(B["comp"][i] for i in idx)
    res = {"rugs": [], "both": [], "gans": []}
    conf_hit = []
    for i in idx:
        t = B["comp"][i]
        r1 = suggest(B, i, k=k, use_gans=False); rb = suggest(B, i, k=k, use_gans=True)
        res["rugs"].append(next((n for n, x in enumerate(r1) if x[0] == t), None))
        res["both"].append(next((n for n, x in enumerate(rb) if x[0] == t), None))
        if len(B["G"]):
            gs = B["GX"] @ B["X"][i]; o = np.argsort(-gs); seen = []
            for j in o:
                if B["GC"][j] not in seen: seen.append(B["GC"][j])
                if len(seen) >= 3: break
            res["gans"].append(next((n for n, c in enumerate(seen) if c == t), None))
        if r1: conf_hit.append((r1[0][1], r1[0][0] == t))
    def top(v, n): return round(sum(1 for p in v if p is not None and p < n) / max(1, len(v)), 3)
    out = {"n": len(idx), "types": len(prior), "baseline_most_common": round(prior.most_common(1)[0][1] / max(1, len(idx)), 3) if prior else 0,
           "gans_with_type": len(B["G"]), "gans_types": len(set(B["GC"]))}
    for m, v in res.items():
        if v: out[m] = {"top1": top(v, 1), "top3": top(v, 3)}
    for thr in (0.5, 0.6, 0.7, 0.8):                                            # если показывать только уверенные: сколько и как точно
        sel = [h for c, h in conf_hit if c >= thr]
        out[f"share>={thr}"] = {"n": len(sel), "acc": round(sum(sel) / max(1, len(sel)), 3)}
    return out
