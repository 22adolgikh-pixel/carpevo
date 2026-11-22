# elem_name.py — подсказка названия элемента ПО КАРТИНКЕ среди всех элементов (v12.04).
#
# Было (v11.52): карточка «имена» в /m показывает закрытый список схем Керимова для типа ковра — одинаковый для всех рамок
# этого типа, картинку не смотрит; 30 % названных рамок получили одну и ту же схему.
# Здесь: банк образцов — (1) подтверждённые названные рамки на всех коврах, (2) все схемы студии (рисунки Керимова,
# Gans-Ruedin…). Отпечаток — сеть element_embed «match» (DINOv3 на овальной вырезке). Запрос — образец группы во всех
# 8 положениях; по каждому названию берётся лучшая близость; итог — лучшие названия, без фильтра по школе и типу.
# Замер (eval): каждый названный элемент ищется по банку без своего ковра; доля, где верное название в первых 1/3/6,
# по способам: другие ковры, схемы, вместе, форма (shape_hint), прежний закрытый список.
import os, json, time
import numpy as np, cv2


def label_of(m, smeta):
    """название рамки для сравнения: узор словаря, иначе узор схемы, иначе текст названия"""
    mo = m.get("motif") or (smeta.get(m.get("scheme") or "") or {}).get("motif") or ""
    if mo: return mo
    t = (m.get("motif_text") or (smeta.get(m.get("scheme") or "") or {}).get("name") or "").strip().lower()
    return ("n:" + t) if t else ""


def scheme_label(sid, smeta):
    d = smeta.get(sid) or {}
    return d.get("motif") or (("n:" + d["name"].strip().lower()) if d.get("name") else "")


def units(rugs, smeta):
    """группы подтверждённых рамок: [{rid, g, box, lab, comp}] — образец: нарисованная рукой, иначе самая крупная"""
    out = []
    for r in rugs:
        G = {}
        for m in r.get("marks") or []:
            if m.get("status") == "confirmed": G.setdefault(m.get("group") or m["id"], []).append(m)
        for g, ms in G.items():
            ms.sort(key=lambda m: (bool(m.get("by") == "machine" or m.get("auto")), -m["box"][2] * m["box"][3]))
            lm = next((m for m in ms if label_of(m, smeta)), None); lab = label_of(lm, smeta) if lm else ""
            out.append({"rid": r["id"], "g": g, "box": ms[0]["box"], "lab": lab, "comp": r.get("composition") or "", "n": len(ms),
                        "sc": (lm or {}).get("scheme") or "", "none": g in (r.get("name_none") or [])})
    return out


def _scheme_img(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if a is None: return None
    if a.ndim == 3 and a.shape[2] == 4:                                         # прозрачное — на серый фон
        al = a[:, :, 3:4].astype(np.float32) / 255; a = (a[:, :, :3] * al + 127 * (1 - al)).astype(np.uint8)
    elif a.ndim == 2: a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return a


def build(EE, rugs, smeta, d_out, img_path, read_small, log=print):
    """→ банк {U: единицы с векторами (8 положений), S: схемы с векторами}"""
    U = units(rugs, smeta); byr = {}
    for u in U: byr.setdefault(u["rid"], []).append(u)
    keep, V8 = [], []
    for rid, us in byr.items():
        p = img_path(rid); a = read_small(p, 1400) if p else None
        if a is None: continue
        a = EE.small(a)
        for u in us:
            V8.append(EE.emb_rot(a, u["box"])); keep.append(u)
    S, SV = [], []
    for sid in sorted(smeta):
        lab = scheme_label(sid, smeta)
        if not lab: continue
        f = next((os.path.join(d_out, sid + e) for e in ("_x12.png", ".png") if os.path.exists(os.path.join(d_out, sid + e))), None)
        a = _scheme_img(f) if f else None
        if a is None: continue
        SV.append(EE.emb(a, [[0.0, 0.0, 1.0, 1.0]])[0]); S.append({"id": sid, "lab": lab, "img": "/out/" + os.path.basename(f)})
    log(f"elem_name: единиц {len(keep)}, схем {len(S)}")
    return {"U": keep, "V8": np.stack(V8) if V8 else np.zeros((0, 8, 1), np.float32),
            "S": S, "SV": np.stack(SV) if SV else np.zeros((0, 1), np.float32), "t": time.time(), "v": EE.version("match")}


def _rank(sims, labs):
    best = {}
    for s, l in zip(sims, labs):
        if l and (l not in best or s > best[l]): best[l] = float(s)
    return sorted(best.items(), key=lambda kv: -kv[1])


def suggest(B, i=None, q8=None, exclude_rid=None, k=6, w_s=1.0):
    """лучшие названия для единицы i банка (или вектора q8 [8×D]): [(название, близость, откуда)]; свой ковёр не берётся"""
    if q8 is None: q8 = B["V8"][i]
    out = {}
    U, V8 = B["U"], B["V8"]
    if len(U):
        A0 = V8[:, 0, :]
        s = (A0 @ q8.T).max(1)
        for j, u in enumerate(U):
            if not u["lab"] or u["rid"] == exclude_rid: continue
            if u["lab"] not in out or s[j] > out[u["lab"]][0]: out[u["lab"]] = (float(s[j]), "rug", j)
    if len(B["S"]):
        s = (B["SV"] @ q8.T).max(1) * w_s
        for j, x in enumerate(B["S"]):
            if x["lab"] not in out or s[j] > out[x["lab"]][0]: out[x["lab"]] = (float(s[j]), "scheme", j)
    return sorted(((l, v[0], v[1], v[2]) for l, v in out.items()), key=lambda t: -t[1])[:k]


def evaluate(B, closed=None, shape=None):
    """замер на названных единицах: верное название среди первых 1/3/6. closed(u) → список названий закрытого списка;
    shape(u) → список названий по форме. Свой ковёр из банка исключается."""
    U, V8 = B["U"], B["V8"]; idx = [i for i, u in enumerate(U) if u["lab"]]
    A0 = V8[:, 0, :] if len(U) else None
    res = {m: [] for m in ("rugs", "schemes", "both", "shape", "closed")}
    cover = {"in_other_rugs": 0, "in_schemes": 0}
    slabs = {x["lab"] for x in B["S"]}
    for i in idx:
        u = U[i]; q8 = V8[i]
        other = {U[j]["lab"] for j in range(len(U)) if U[j]["rid"] != u["rid"] and U[j]["lab"]}
        cover["in_other_rugs"] += u["lab"] in other; cover["in_schemes"] += u["lab"] in slabs
        s = (A0 @ q8.T).max(1)
        r1 = _rank([s[j] for j in range(len(U)) if U[j]["rid"] != u["rid"]], [U[j]["lab"] for j in range(len(U)) if U[j]["rid"] != u["rid"]])
        r2 = _rank((B["SV"] @ q8.T).max(1), [x["lab"] for x in B["S"]]) if len(B["S"]) else []
        rb = sorted({**dict(r2), **{l: v for l, v in r1 if v > dict(r2).get(l, -9)}}.items(), key=lambda kv: -kv[1])
        def pos(r): return next((n for n, (l, _) in enumerate(r) if l == u["lab"]), None)
        res["rugs"].append(pos(r1)); res["schemes"].append(pos(r2)); res["both"].append(pos(rb))
        if shape:
            sh = shape(u); res["shape"].append(next((n for n, l in enumerate(sh) if l == u["lab"]), None) if sh is not None else None)
        if closed:
            cl = closed(u); res["closed"].append(None if cl is None else (next((n for n, l in enumerate(cl) if l == u["lab"]), None), len(cl)))
    def top(v, k): return round(sum(1 for p in v if p is not None and p < k) / max(1, len(v)), 3)
    out = {"n": len(idx), "units": len(U), "n_schemes": len(B["S"]), "rugs_named": len({U[i]["rid"] for i in idx}),
           "labels": len({U[i]["lab"] for i in idx}), "coverage": {k: round(v / max(1, len(idx)), 3) for k, v in cover.items()}}
    for m in ("rugs", "schemes", "both", "shape"):
        v = res[m]
        if v: out[m] = {"top1": top(v, 1), "top3": top(v, 3), "top6": top(v, 6)}
    cl = [c for c in res["closed"] if c is not None]
    if cl: out["closed"] = {"has_list": round(len(cl) / max(1, len(idx)), 3), "in_list": round(sum(1 for p, n in cl if p is not None) / max(1, len(idx)), 3),
                            "median_list": int(np.median([n for _, n in cl]))}
    from collections import Counter
    out["top_labels"] = Counter(U[i]["lab"] for i in idx).most_common(10)
    return out


def options(B, sug, name_of, k=6):
    """подсказки → варианты для карточки: [{id, name, src, sim, rid+box (пример с ковра) или img (схема)}]
    id: «mo:<узор словаря>» или «tx:<название>»; у схемы добавляется «sc» — её номер."""
    out = []
    for lab, sim, src, j in sug[:k]:
        o = {"id": ("tx:" + lab[2:]) if lab.startswith("n:") else ("mo:" + lab), "name": name_of(lab), "src": src, "sim": round(sim, 3)}
        if src == "rug":
            u = B["U"][j]; o.update(rid=u["rid"], box=u["box"])
            if u.get("sc"): o["sc"] = u["sc"]
        else:
            x = B["S"][j]; o.update(img=x["img"], sc=x["id"])
        out.append(o)
    return out
