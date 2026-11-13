"""Метки для обучения уточнения углов (v11.57).

    python3 cps/tools/corners/make_labels.py <папка data/rugs> [<micro.jsonl>]   → cps/tools/corners/labels.json

Цель — углы, которые подтвердил человек: layout.quad при quad_status confirmed (правка или «да» в /m, рука в студии)
и accepted (проверка пачкой). Для сравнения рядом кладутся машинные углы: auto.quad (find_quad) и auto.vlm_quad (модель по картинке).
Из micro.jsonl — время первой правки, чтобы отложенная выборка была «последними» коврами, а не случайными.
"""
import json, os, sys

def main():
    d = sys.argv[1]; mj = sys.argv[2] if len(sys.argv) > 2 else None
    first = {}
    if mj and os.path.exists(mj):
        for l in open(mj, encoding="utf-8"):
            try: r = json.loads(l)
            except ValueError: continue
            if r.get("type") == "quad": first.setdefault(r["rid"], r["t"])
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"): continue
        r = json.load(open(os.path.join(d, f), encoding="utf-8")); L = r.get("layout") or {}; a = r.get("auto") or {}
        if not L.get("quad") or L.get("quad_status") not in ("confirmed", "accepted") or r.get("frame") not in ("whole", "", None): continue
        if r.get("dup_of") or r.get("hidden") or r.get("split_into"): continue
        out.append({"id": r["id"], "img": r.get("lib_img") or ("img/" + r["id"] + ".jpg"), "w": r.get("w"), "h": r.get("h"),
                    "quad": L["quad"], "how": L["quad_status"] + "/" + (r.get("tri") or ""), "src": ((r.get("cat") or {}).get("src") or "").split("_")[0],
                    "auto": a.get("quad"), "vlm": (a.get("vlm_quad") or {}).get("quad"), "t": first.get(r["id"], 0)})
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(len(out), "ковров с подтверждёнными углами →", p)

if __name__ == "__main__": main()
