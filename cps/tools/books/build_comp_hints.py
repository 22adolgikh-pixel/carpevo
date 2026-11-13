"""Подсказки для карточки «тип» в /m (v11.57): что говорят книги о типе ковра.

    python3 cps/tools/books/build_comp_hints.py      → cps/comp_hints.json

Берёт описания Carpet DNA (site/model/descriptions/*.json, черновики ИИ по книгам) и для каждого типа ковра (composition)
оставляет короткий текст «как узнать»: сначала раздел о поле, потом «Что это», потом каймы. Керимов — первым.
Описания — черновики ИИ, людьми не проверены; на карточке это сказано.
"""
import glob, json, os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
DICT = json.load(open(os.path.join(ROOT, "cps", "models", "dictionary.json"), encoding="utf-8"))["items"]
COMPS = {d["id"]: d for d in DICT if d["type"] == "composition"}
ORDER = ("Как устроено поле", "Особенности", "Узор", "Узоры", "Что это", "Каймы", "Каймы и цвет")
MAXC = 520


def short(t, n):
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= n: return t
    cut = t[:n]; k = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:k + 1] if k > n * 0.5 else cut.rstrip() + "…")


def main():
    cand = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "site", "model", "descriptions", "*.json"))):
        j = json.load(open(f, encoding="utf-8")); src = (j.get("source") or {}).get("short", "") or os.path.basename(f)
        rank = 0 if "Керимов" in src else 1
        for cid, v in (j.get("items") or {}).items():
            if cid not in COMPS: continue
            secs = {s["title"]: s for s in v.get("sections") or [] if s.get("text")}
            parts, pages = [], []
            for t in ORDER:
                if t in secs and sum(len(p) for p in parts) < MAXC:
                    parts.append(short(secs[t]["text"], MAXC - sum(len(p) for p in parts)))
                    pages += secs[t].get("pages") or []
            if parts: cand.setdefault(cid, []).append((rank, -len(" ".join(parts)), {"text": " ".join(parts), "src": src, "pages": sorted(set(pages), key=str)[:4]}))
    out = {cid: [c[2] for c in sorted(v, key=lambda c: c[:2])][:2] for cid, v in cand.items()}
    p = os.path.join(ROOT, "cps", "comp_hints.json")
    json.dump({"_about": "подсказки к карточке «тип» в /m; черновики ИИ по книгам, не проверены людьми; build_comp_hints.py", "items": out},
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"типов ковра {len(COMPS)}, с описанием {len(out)} → {p}")


if __name__ == "__main__":
    main()
