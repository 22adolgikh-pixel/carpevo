"""Метки для обучения поиска элементов (v11.58).

    python3 cps/tools/elements/make_labels.py <папка data/rugs>   → cps/tools/elements/labels.json

Каждый ковёр — группы рамок: одна группа = один элемент узора, повторённый на ковре. Подтверждённые людьми рамки группы —
«тот же элемент», отклонённые машинные предложения — «похоже, но не то» (трудные примеры). Для каждой рамки — откуда она
(машина: repeat / etalon / kerimov / motif / rapport, или рука) и её оценка машиной — чтобы сравнить с новой сетью.
"""
import json, os, sys


def kind(m):
    mc = m.get("machine")
    if not mc: return "hand"
    if isinstance(mc, str): return mc
    n = (m.get("note") or "")
    return "etalon" if n.startswith("эталон") else "rapport" if n.startswith("раппорт") else "xfer" if n.startswith("перенес") else "border_sign" if n.startswith("знак") else "repeat"


def main():
    d = sys.argv[1]; out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"): continue
        r = json.load(open(os.path.join(d, f), encoding="utf-8"))
        if r.get("dup_of") or r.get("hidden") or r.get("split_into"): continue
        G = {}
        for m in r.get("marks") or []:
            b = m.get("box")
            if not (isinstance(b, list) and len(b) == 4) or m.get("status") not in ("confirmed", "rejected", "proposed"): continue
            G.setdefault(m.get("group") or m["id"], []).append({"id": m["id"], "box": [round(float(v), 5) for v in b], "st": m["status"], "k": kind(m),
                                                               "sc": m.get("score"), "ang": m.get("angle", 0), "fl": m.get("flip", 0), "scheme": m.get("scheme") or "", "motif": m.get("motif") or ""})
        if not any(x["st"] in ("confirmed", "rejected") for g in G.values() for x in g): continue
        out.append({"id": r["id"], "img": r.get("lib_img") or ("img/" + r["id"] + ".jpg"), "w": r.get("w"), "h": r.get("h"),
                    "quad": (r.get("layout") or {}).get("quad"), "groups": G})
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    n = lambda s: sum(1 for x in out for g in x["groups"].values() for m in g if m["st"] == s)
    print(len(out), "ковров;", "подтверждено", n("confirmed"), "отклонено", n("rejected"), "ждут", n("proposed"), "→", p)


if __name__ == "__main__": main()
