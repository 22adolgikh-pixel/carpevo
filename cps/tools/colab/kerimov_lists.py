# kerimov_lists.py — v11.7: подписи рисунков Керимова из «Списка иллюстраций» в конце тома.
#
# У многих фото на страницах книги подписи нет (подпись к «Рис. 30» Керимов вынес в конец тома): тогда у карточки ковра
# нет ни типа, ни музея. В конце тома есть «СПИСОК ЧЕРНО-БЕЛЫХ ИЛЛЮСТРАЦИЙ» («1. Плетень…», «4. Ковер «Ширван»… Инв. № 682»)
# и «СПИСОК ЦВЕТНЫХ ТАБЛИЦ». Разбираем их из raw_pages_v3 (full_text страниц-указателей) → {("рис"|"табл", номер): подпись}.
# Списки на английском (LIST OF …) пропускаются — нумерация та же, русская подпись полнее.
import re

SEC = [(re.compile(r"СПИСОК\s+Ч[ЕЁ]РНО[\s\-–—]*БЕЛЫХ\s+ИЛЛЮСТРАЦИЙ", re.I), "рис"),
       (re.compile(r"СПИСОК\s+ЦВЕТНЫХ\s+(?:ТАБЛИЦ|ИЛЛЮСТРАЦИЙ|ВКЛЕЕК)", re.I), "табл"),
       (re.compile(r"СПИСОК\s+ИЛЛЮСТРАЦИЙ", re.I), "рис"),
       (re.compile(r"\bLIST\s+OF\b|ОГЛАВЛЕНИЕ|СОДЕРЖАНИЕ|RESUME|SUMMARY", re.I), None)]
ITEM = re.compile(r"^\s*(\d{1,3})\s*[.)]\s+(\S.*)$")


def parse_pages(pages):
    """pages — [(номер скана, json страницы)] по порядку. → {(вид, n): подпись}. Раздел продолжается на следующих страницах,
    пока не встретится другой заголовок; строки без номера — продолжение предыдущей подписи."""
    out = {}; kind = None; last = None
    for _, J in pages:
        txt = (J.get("full_text") or "").replace("\\n", "\n")
        if not txt: continue
        heads = " ".join(J.get("headings") or [])
        # заголовок раздела может быть и в headings, и в тексте; страницы-указатели помечены page_kind=index
        found_any = False
        for line in txt.split("\n"):
            s = line.strip()
            if not s: continue
            hit = None
            for rx, k in SEC:
                if rx.search(s): hit = (k,); break
            if hit:
                kind = hit[0]; last = None; found_any = True; continue
            if not kind:
                continue
            m = ITEM.match(s)
            if m:
                n = int(m.group(1)); key = (kind, n)
                if key in out and last != key and n == 1 and kind == "рис":   # второй «1.» — уже другой список
                    kind = None; continue
                out[key] = m.group(2).strip(); last = key
            elif last and not re.match(r"^\d+$", s) and len(s) > 2:
                out[last] = (out[last] + " " + s).strip()
        if not found_any and kind and (J.get("page_kind") not in ("index", "mixed", None)):
            kind = None; last = None
    return out


def label_key(label):
    """«Рис. 30» → ("рис", 30); «Табл. 210» / «Таблица 5» → ("табл", n); иначе None"""
    s = (label or "").lower().strip()
    m = re.match(r"(рис|табл|таб|fig|pl)\w*\.?\s*(\d{1,3})", s)
    if not m: return None
    return ("рис" if m.group(1) in ("рис", "fig") else "табл", int(m.group(2)))


def caption_info(cap):
    """подпись → {museum, inv, date, name} (name — первое «название» в кавычках)"""
    c = cap or ""
    inv = re.search(r"Инв\.?\s*№\s*([A-Za-zА-Яа-я]{0,4}[\s\-]?\d[\w\-/.]*)", c)
    mus = re.search(r"(?:Хранится|хранится|находится|Находится|хранился)\s+(?:в|во)\s+(.+?)(?:\s+под\s+названием|\.\s*Инв|\.|,\s*а\s|$)", c)
    date = re.search(r"\b((?:[XVI]{1,5}(?:\s*[—–-]\s*[XVI]{1,5})?)\s*(?:век|века|веке|вв?\.)|(?:1[5-9]\d\d)\s*г)", c)
    nm = re.search(r"[«\"]([^»\"]{2,40})[»\"]", c)
    return {"museum": mus.group(1).strip() if mus else "", "inv": inv.group(1).strip().rstrip(".,") if inv else "",
            "date": date.group(1).strip() if date else "", "name": nm.group(1).strip() if nm else ""}


if __name__ == "__main__":     # самопроверка на фрагменте т. II, с. 227
    J = {"page_kind": "index", "headings": ["СПИСОК ЧЕРНО-БЕЛЫХ ИЛЛЮСТРАЦИЙ"], "full_text":
         "СПИСОК ЧЕРНО-БЕЛЫХ ИЛЛЮСТРАЦИЙ\n\n1. Плетень, изготовленная способом простого продевания.\n\n4. Ковер «Ширван», сотканный в XII — XIII веках, "
         "хранится в стамбульском музее Тюрк ве Ислам эсерлери музеси под названием «Сельджуг». Инв. № 682.\n\n13. Трон Шаха Исмаила Хатаи, сделанный "
         "тебризскими мастерами. XVI век. Хранится в стамбульском музее Топгапы сарайы.\n\n16. Кусочек ковра «Карабах», сотканного в XIV веке. Хранится в Нью-Йорке в музее Метрополитен.\n\n17. Ковер «Борьба"}
    J2 = {"page_kind": "index", "headings": [], "full_text": "дракона с фениксом», Гянджа — Казахского типа. XV век.\n18. Ковер «Казах». Инв. № 1036.\n"
          "СПИСОК ЦВЕТНЫХ ТАБЛИЦ\n1. Ковер «Пиребедиль». XIX век. Хранится в Государственном музее азербайджанского ковра. Инв. № 3456.\nLIST OF ILLUSTRATIONS\n1. Wattle"}
    L = parse_pages([(117, J), (118, J2)])
    assert L[("рис", 4)].startswith("Ковер «Ширван»"), L
    assert L[("рис", 17)].endswith("XV век."), L[("рис", 17)]
    assert L[("табл", 1)].startswith("Ковер «Пиребедиль»") and ("рис", 1) in L and L[("рис", 1)].startswith("Плетень"), L
    assert label_key("Рис. 30") == ("рис", 30) and label_key("Табл. 210") == ("табл", 210)
    i = caption_info(L[("рис", 4)]); assert i["inv"] == "682" and "стамбульском" in i["museum"] and i["name"] == "Ширван", i
    i = caption_info(L[("табл", 1)]); assert i["inv"] == "3456" and "ковра" in i["museum"], i
    print("ok", len(L))
