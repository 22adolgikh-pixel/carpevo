# az_names.py — азербайджанские названия элементов для Carpet Pattern Studio
#  • az_draft(): черновик латиницы (Azərbaycan əlifbası) из кириллицы книги
#    (кириллица Керимова — русская передача азербайджанских слов, поэтому это эвристика:
#     результат помечается как «черновик» и проверяется глазами);
#  • az_key(): «скелет» слова для сравнения латиницы, кириллицы и русской передачи между собой
#    (q/ğ/g → g, ə → e, x/h → h …) — по нему строятся связи со страницами сайта «Ковровое ДНК».
import re, os, json

_CYR = re.compile(r"[Ѐ-ӿ]")
_VOW = set("аоуыеиәёюүөэяі")
_BACK = set("аоуы")
_FRONT = set("еиәёюүөэяі")
_MAP = {"а": "a", "б": "b", "в": "v", "д": "d", "е": "e", "ә": "ə", "ж": "j", "з": "z", "и": "i", "і": "i",
        "й": "y", "ј": "y", "к": "k", "ҝ": "g", "ғ": "ğ", "л": "l", "м": "m", "н": "n", "о": "o", "ө": "ö",
        "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ү": "ü", "ф": "f", "х": "x", "һ": "h", "ц": "s",
        "ч": "ç", "ҹ": "c", "ш": "ş", "щ": "şç", "ъ": "", "ь": "", "ы": "ı", "э": "e"}
# слова, которые эвристика передаёт неверно (дополняйте по мере работы; ключ — строчная кириллица)
_WORDS = {"гошагуш": "qoşaquş", "гоша": "qoşa", "гуш": "quş", "агадж": "ağac", "бута": "buta",
          "гёль": "göl", "гель": "göl", "гюль": "gül", "гуль": "gül", "чичек": "çiçək", "чичак": "çiçək",
          "дашлы": "daşlı", "даг": "dağ", "дагдан": "dağdan", "ат": "at", "ит": "it", "адам": "adam",
          "гармоши": "qarmoşi", "сачаглы": "saçaqlı", "сачаг": "saçaq", "гармагуш": "qarmaquş",
          "илан": "ilan", "буйнуз": "buynuz", "гыз": "qız", "оглан": "oğlan", "гарагуш": "qaraquş",
          "гарга": "qarğa", "гарлы": "qarlı", "гапы": "qapı", "дарак": "daraq", "гармаг": "qarmaq",
          "гамыш": "qamış", "гарыш": "qarış", "гушлу": "quşlu", "гуйруг": "quyruq", "аяг": "ayaq"}


def has_cyr(s): return bool(_CYR.search(s or ""))


def _word(w):
    lw = w.lower()
    if lw in _WORDS:
        r = _WORDS[lw]
        if not w[:1].isupper(): return r
        return ("İ" + r[1:]) if r[0] == "i" else (r[0].upper() + r[1:])
    out, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]; lo = ch.lower(); up = ch != lo
        prev = w[i - 1].lower() if i else ""
        nxt = w[i + 1].lower() if i + 1 < n else ""
        pcons = bool(prev) and prev.isalpha() and prev not in _VOW and prev not in "ьъ"
        if lo == "д" and nxt == "ж": r = "c"; i += 1
        elif lo == "г":
            if prev == "г": r = "q"                                            # тогга → toqqa
            elif nxt in _FRONT or nxt == "ь" or prev == "н": r = "g"             # гёз → göz, ченг → çeng             # гёз → göz, ченг → çeng
            elif not nxt and prev in _VOW and sum(c in _VOW for c in w.lower()) == 1: r = "ğ"   # даг → dağ, быг → bığ
            elif not nxt or nxt in "гк": r = "q"                               # балыг → balıq, тогга → toqqa
            elif not prev: r = "q"                                             # гуш → quş
            elif prev in _VOW or nxt in _VOW: r = "ğ"                          # агадж → ağac, сырга → sırğa
            else: r = "q"
        elif lo == "к": r = "q" if (nxt in _BACK or (nxt == "к" and i + 2 < n and w[i + 2].lower() in _BACK)) else "k"  # Карабах → Qarabağ, саккал → saqqal
        elif lo == "ё": r = "ö" if pcons else "yo"
        elif lo == "ю": r = "ü" if pcons else "yu"
        elif lo == "я": r = "ə" if pcons else "ya"                               # биляк → bilək, Гянджа → Gəncə
        elif lo == "е": r = "ə" if (prev and prev.isalpha() and i == n - 1) else "e"   # Саде → Sadə
        else: r = _MAP.get(lo, ch)
        if up and r: r = "İ" + r[1:] if r[0] == "i" else r[0].upper() + r[1:]
        out.append(r); i += 1
    return "".join(out)


def az_draft(s):
    """Черновик азербайджанской латиницы из кириллической записи. Латиница возвращается как есть."""
    s = (s or "").strip()
    if not s or s == "?" or not has_cyr(s): return s if s != "?" else ""
    return re.sub(r"[Ѐ-ӿ]+", lambda m: _word(m.group(0)), s)


_FOLD = str.maketrans({"ə": "e", "ä": "e", "ğ": "g", "q": "g", "x": "h", "ı": "i", "ö": "o", "ü": "u",
                       "ç": "c", "ş": "s", "w": "v", "İ": "i", "I": "i"})


def az_key(s):
    """Сравнительный ключ: одинаковый для «Ağac», «ağac», «Агадж», «агач»."""
    s = s or ""
    if has_cyr(s): s = az_draft(s)
    s = s.translate(_FOLD).lower().translate(_FOLD)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"(.)\1+", r"\1", s)


def ru_key(s):
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", s).strip()


# ---------------- подбор страниц сайта ----------------
_site_cache = {"mtime": None, "rows": []}


def _site_rows(path):
    try: mt = os.path.getmtime(path)
    except OSError: return []
    if _site_cache["mtime"] != mt:
        rows = []
        for s in json.load(open(path, encoding="utf-8")):
            base = re.sub(r"^(elem|motif)_", "", s.get("id", "")).replace("_", " ")
            az_vars = [v.strip() for v in re.split(r"[,/;]", s.get("az") or "") if v.strip()]
            akeys = {k for k in [az_key(base)] + [az_key(v) for v in az_vars] if k}
            rkeys = {k for k in [ru_key(v) for v in re.split(r"[,/;]", s.get("ru") or "")] if k}
            rows.append((s, akeys, rkeys))
        _site_cache.update(mtime=mt, rows=rows)
    return _site_cache["rows"]


def site_suggest(path, names=(), translation="", limit=8):
    """names — название(я) элемента (латиница или кириллица), translation — русский перевод."""
    qk = [k for k in (az_key(n) for n in names if n) if k]
    qr = ru_key(translation)
    qr_words = set(qr.split())
    hits = []
    for s, akeys, rkeys in _site_rows(path):
        sc = 0
        for k in qk:
            for a in akeys:
                if a == k: sc = max(sc, 10)
                elif k in a.split() or a in k.split(): sc = max(sc, 6)
                elif len(k) > 3 and (a.startswith(k) or k.startswith(a)) and len(a) > 3: sc = max(sc, 4)
        if qr:
            for r in rkeys:
                if r == qr: sc = max(sc, 8) if sc < 8 else sc + 1
                elif qr_words & set(r.split()) and len(r) > 2: sc = max(sc, 3)
        if sc:
            hits.append((sc + (0.5 if s.get("kind") == "element" else 0), s))
    hits.sort(key=lambda h: -h[0])
    return [dict(h[1], score=round(h[0], 1)) for h in hits[:limit]]
