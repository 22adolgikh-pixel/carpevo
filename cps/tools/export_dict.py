# export_dict.py — словарь узоров и композиций из вики-части сайта для студии (экран «Ковры»).
# Запуск из корня репозитория: python3 cps/tools/export_dict.py → cps/models/dictionary.json
import json, os, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
C = json.load(open(f"{ROOT}/site/model/out/concepts.json", encoding="utf-8"))
T = json.load(open(f"{ROOT}/site/model/out/term_map.json", encoding="utf-8"))
alias = collections.defaultdict(list)
for k, v in T.items():
    if k != v: alias[v].append(k)
out = []
for c in C:
    ns = c.get("names") or []
    pick = lambda pred: next((n["v"] for n in ns if pred(n)), "")
    ru = pick(lambda n: n.get("lang") == "ru" and n.get("kind") != "kerimov_translation")
    tr = pick(lambda n: n.get("kind") == "kerimov_translation") or ((c.get("meaning") or {}).get("ru") or "")
    roles = c.get("roles") or {}
    out.append({"id": c["id"], "type": c["type"], "az": c.get("headword") or c["id"], "ru": ru, "tr": tr[:120],
                "role": max(roles, key=roles.get) if roles else "", "schools": sorted((c.get("schools") or {}), key=lambda s: -c["schools"][s]),
                "names": sorted({n["v"] for n in ns if n.get("kind") != "kerimov_translation"} | set(alias.get(c["id"], [])))})
out.sort(key=lambda x: (x["type"] != "ornament", x["az"]))
json.dump({"_about": "из site/model/out/concepts.json (вики-часть), пересобрать: python3 cps/tools/export_dict.py", "items": out},
          open(f"{ROOT}/cps/models/dictionary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(out), collections.Counter(x["type"] for x in out))
