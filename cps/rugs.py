# rugs.py — экран «Ковры» (v10.20): фото ковров, разметка элементов (рамки), связь с узорами словаря и схемами студии.
#
# Ковёр: data/rugs/<id>.json, фото data/rug_img/<id>.jpg (≤1600 px по длинной стороне), миниатюры data/rug_thumbs/ (не в бэкап).
# Метка (элемент на ковре): рамка в долях фото [x, y, w, h] (0..1), зона (поле / кайма / угол / медальон …), узор из словаря
# (models/dictionary.json — выгрузка вики-части сайта), схема студии (id рисунка), группа повторов, уверенность, статус проверки,
# кто отметил и кто подтвердил. Повторы одного элемента — метки с одинаковой group.
import os, re, json, time, threading, io, urllib.request, hashlib
import cv2, numpy as np
from fastapi import Request, UploadFile, File, Body, Query
from fastapi.responses import FileResponse, JSONResponse, Response

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = {"field": "поле", "medallion": "медальон", "spandrel": "угол поля", "border": "главная кайма",
         "minor": "малая кайма", "guard": "бордюрная полоска", "end": "торцевая полоса", "other": "другое"}
STATUSES = ("proposed", "confirmed", "rejected")
SCHOOLS = {"guba-shirvan": "Губа-Ширван", "ganja-gazakh": "Гянджа-Газах", "karabakh": "Карабах", "tabriz": "Тебриз", "nakhchivan": "Нахчыван", "": "—"}
MAXSIDE = 1600
_lock = threading.Lock()
imp = {"running": False, "done": 0, "total": 0, "errors": 0, "last": ""}


CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғҹһөүҝ", ["a","b","v","g","d","e","e","j","z","i","y","k","l","m","n","o","p","r","s","t","u","f","h","ts","c","s","s","","i","","e","yu","ya","e","g","c","h","o","u","g"]))


def fold(s):
    """Для поиска: «goch» найдёт «qoça», «Гоча» и «гоча» — азербайджанские буквы и кириллица сводятся к простой латинице."""
    s = (s or "").lower()
    s = "".join(CYR.get(c, c) for c in s)
    for a, b in (("ç", "c"), ("ş", "s"), ("ğ", "g"), ("ı", "i"), ("ö", "o"), ("ü", "u"), ("ə", "e"), ("ch", "c"), ("sh", "s"), ("zh", "j"),
                 ("dj", "c"), ("dzh", "c"), ("q", "g"), ("x", "h"), ("kh", "h"), ("w", "v"), ("-", " ")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


SCHOOL_KEYS = [("karabag", "karabakh"), ("karabak", "karabakh"), ("karabah", "karabakh"), ("shusha", "karabakh"), ("susa", "karabakh"), ("lampa", "karabakh"),
               ("gazah", "ganja-gazakh"), ("kazak", "ganja-gazakh"), ("kazah", "ganja-gazakh"), ("ganja", "ganja-gazakh"), ("genje", "ganja-gazakh"), ("gence", "ganja-gazakh"),
               ("gendje", "ganja-gazakh"), ("gyanja", "ganja-gazakh"), ("borcal", "ganja-gazakh"), ("bordjal", "ganja-gazakh"), ("fahral", "ganja-gazakh"), ("celeberd", "ganja-gazakh"),
               ("kuba", "guba-shirvan"), ("guba", "guba-shirvan"), ("shirvan", "guba-shirvan"), ("sirvan", "guba-shirvan"), ("baku", "guba-shirvan"), ("baki", "guba-shirvan"),
               ("derbend", "guba-shirvan"), ("dagestan", "guba-shirvan"), ("daghestan", "guba-shirvan"), ("zakatal", "guba-shirvan"), ("talis", "guba-shirvan"),
               ("lenkoran", "guba-shirvan"), ("mugan", "guba-shirvan"), ("moghan", "guba-shirvan"), ("hizi", "guba-shirvan"), ("khyzy", "guba-shirvan"),
               ("perepedil", "guba-shirvan"), ("pirebedil", "guba-shirvan"), ("seichur", "guba-shirvan"), ("chichi", "guba-shirvan"), ("cici", "guba-shirvan"), ("marasal", "guba-shirvan"),
               ("tabriz", "tabriz"), ("tebriz", "tabriz"), ("heriz", "tabriz"), ("serapi", "tabriz"), ("ardabil", "tabriz"), ("ardebil", "tabriz"), ("karaja", "tabriz"),
               ("sarab", "tabriz"), ("meshkin", "tabriz"), ("nakhchivan", "nakhchivan"), ("nahcivan", "nakhchivan"), ("nahicevan", "nakhchivan")]
LEG_SCHOOL = {"кар": "karabakh", "к ш": "guba-shirvan", "кш": "guba-shirvan", "г к": "ganja-gazakh", "гк": "ganja-gazakh", "теб": "tabriz"}


def skel(s):
    """Согласный «скелет» (Хила = xilə, Сараб = sərab): гласные в разных транскрипциях пишутся по-разному."""
    return re.sub(r"[aeiouy ]", "", fold(s))


def school_guess(*texts):
    """Школа по словам в названии / месте / описании / имени папки → (код, слово) или ("", "")."""
    t = fold(" ".join(x or "" for x in texts))
    for k, v in SCHOOL_KEYS:
        if k in t: return v, k
    return "", ""


def text_facts(t):
    """Век и размер из описания (azerbaijanrugs и т.п.)."""
    out = {}
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)[ -]centur(?:y|ies)", t or "", re.I) or re.search(r"\b([XVI]{1,5})\s*(?:в\.|век)", t or "")
    if m: out["date"] = m.group(0)
    m = re.search(r"(\d{2,3})\s*[x×х]\s*(\d{2,3})\s*cm", t or "", re.I) or re.search(r"(\d{2,3})\s*[x×х]\s*(\d{2,3})\s*см", t or "")
    if m: out["size"] = f"{m.group(1)}×{m.group(2)} см"
    return out


def install(app, DATA, D_WORK):
    D_RUGS = os.path.join(DATA, "rugs"); D_IMG = os.path.join(DATA, "rug_img"); D_TH = os.path.join(DATA, "rug_thumbs")
    for d in (D_RUGS, D_IMG, D_TH): os.makedirs(d, exist_ok=True)
    DICT = json.load(open(os.path.join(HERE, "models", "dictionary.json"), encoding="utf-8"))["items"]
    BYID = {d["id"]: d for d in DICT}
    ALIAS = {}
    for d in DICT:
        for n in [d["id"]] + d["names"]: ALIAS.setdefault(n.lower(), d["id"])
    figcache = {"t": 0, "by": {}, "exp": {}, "vote": {}}
    D_CROPS = os.path.join(DATA, "crops")
    COMP = {}
    for d in DICT:
        if d["type"] == "composition":
            for n in [d["az"], d["ru"]] + d["names"]:
                COMP.setdefault(fold(n), d["id"]); COMP.setdefault("~" + skel(n), d["id"])

    def legend():
        p = os.path.join(DATA, "legend.csv"); out = {}
        if os.path.exists(p):
            import csv
            for r in csv.DictReader(open(p, encoding="utf-8")):
                out[(str(r.get("table") or "").strip(), str(r.get("fig") or "").strip())] = r
        return out
    FOLD = {d["id"]: " | ".join(fold(x) for x in [d["az"], d["ru"], d["tr"]] + d["names"]) for d in DICT}

    def user(req): return getattr(req.state, "user", None) or "local"

    def load(rid):
        p = os.path.join(D_RUGS, rid + ".json")
        if not os.path.exists(p): return None
        with open(p, encoding="utf-8") as f: return json.load(f)

    def save(r):
        tmp = os.path.join(D_RUGS, r["id"] + ".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f: json.dump(r, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(D_RUGS, r["id"] + ".json"))

    def new_id(seed):
        base = re.sub(r"[^\w\-]+", "_", seed.lower())[:40].strip("_") or "rug"
        rid, n = base, 1
        while os.path.exists(os.path.join(D_RUGS, rid + ".json")): n += 1; rid = f"{base}_{n}"
        return rid

    def store_image(rid, data):
        a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if a is None: raise ValueError("не картинка")
        h, w = a.shape[:2]; s = MAXSIDE / max(h, w)
        if s < 1: a = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(D_IMG, rid + ".jpg"), a, [cv2.IMWRITE_JPEG_QUALITY, 88])
        th = os.path.join(D_TH, rid + ".jpg")
        if os.path.exists(th): os.remove(th)
        return a.shape[1], a.shape[0]

    def blank(rid, **kw):
        r = {"id": rid, "title": "", "source": "", "museum": "", "inv": "", "url": "", "license": "", "place": "", "date": "",
             "school": "", "composition": "", "technique": "", "size": "", "notes": "", "status": "new", "w": 0, "h": 0,
             "marks": [], "created": int(time.time()), "updated": int(time.time())}
        r.update({k: v for k, v in kw.items() if k in r and v is not None})
        return r

    def figures_by_concept():
        """Схемы студии (принятые и в работе), привязанные к узорам словаря через meta.site_id / название."""
        if time.time() - figcache["t"] < 60: return figcache["by"]
        by = {}; exp = {}; vote = {}; LEG = legend()
        for f in os.listdir(D_WORK):
            if not f.endswith(".json"): continue
            try:
                with open(os.path.join(D_WORK, f), encoding="utf-8") as fh: w = json.load(fh)
            except Exception: continue
            if not w.get("matrix") or w.get("kind") in ("photo", "junk"): continue
            m = w.get("meta") or {}
            cid = ALIAS.get((m.get("site_id") or "").lower()) or ALIAS.get((m.get("name_az") or "").lower()) or ALIAS.get((m.get("name") or "").lower())
            fid = f[:-5]
            if cid: by.setdefault(cid, []).append({"id": fid, "name": m.get("name") or "", "done": bool(w.get("done"))})
            # Керимов: у какого типа ковра встречается этот элемент (столбец «ковёр» легенды) и какая школа (столбец «тип»)
            carpet, typ = m.get("carpet") or "", m.get("type") or ""
            mt = re.match(r"t(\d+)_f0*(\d+)$", fid)
            if mt and (not carpet or not typ):
                lr = LEG.get((mt.group(1), mt.group(2))) or {}
                carpet = carpet or lr.get("carpet") or ""; typ = typ or lr.get("type") or ""
            carpet = carpet.strip().strip("|").strip()
            comp = (COMP.get(fold(carpet)) or COMP.get(fold(carpet.split()[0])) or COMP.get("~" + skel(carpet))) if carpet else None
            if comp:
                exp.setdefault(comp, []).append({"id": fid, "name": m.get("name") or "", "motif": cid or "", "done": bool(w.get("done"))})
                sc = LEG_SCHOOL.get(re.sub(r"[^а-яё ]", " ", typ.lower()).strip().replace("  ", " "))
                if sc: vote.setdefault(comp, {}).setdefault(sc, 0); vote[comp][sc] += 1
        figcache.update(t=time.time(), by=by, exp=exp, vote=vote)
        return by

    def summary(r):
        mk = r.get("marks") or []
        return {"id": r["id"], "title": r.get("title") or r["id"], "school": r.get("school", ""), "museum": r.get("museum", ""),
                "status": r.get("status", "new"), "n": len(mk), "confirmed": sum(1 for m in mk if m.get("status") == "confirmed"),
                "motifs": len({m.get("motif") for m in mk if m.get("motif")}), "updated": r.get("updated", 0), "source": r.get("source", "")}

    # ---------------- словарь ----------------
    @app.get("/api/dict")
    def api_dict(q: str = "", type: str = "", limit: int = 40):
        ql = fold(q); figs = figures_by_concept(); out = []
        for d in DICT:
            if type and d["type"] != type: continue
            f = FOLD[d["id"]]
            if ql and ql not in f: continue
            rank = 0 if not ql else (0 if fold(d["az"]).startswith(ql) or fold(d["ru"]).startswith(ql) else (1 if (" | " + ql) in (" | " + f) or (" " + ql) in f else 2))
            out.append((rank, -len(figs.get(d["id"], [])), {**{k: d[k] for k in ("id", "type", "az", "ru", "tr", "role", "schools")}, "schemes": len(figs.get(d["id"], []))}))
        out.sort(key=lambda t: (t[0], t[1]))
        return [t[2] for t in out[:limit]]

    @app.get("/api/dict/{cid}")
    def api_dict_one(cid: str):
        d = BYID.get(cid)
        if not d: return JSONResponse({"error": "нет такого"}, 404)
        rugs = []
        for f in os.listdir(D_RUGS):
            if not f.endswith(".json"): continue
            r = load(f[:-5]) or {}
            n = sum(1 for m in r.get("marks", []) if m.get("motif") == cid and m.get("status") != "rejected")
            if n: rugs.append({"id": r["id"], "title": r.get("title") or r["id"], "n": n})
        figs = figures_by_concept(); v = figcache["vote"].get(cid, {})
        return {**d, "figures": figs.get(cid, []), "rugs": rugs, "expected": figcache["exp"].get(cid, []),
                "school_vote": max(v, key=v.get) if v else ""}

    # ---------------- ковры ----------------
    @app.get("/api/rugs")
    def api_rugs():
        out = []
        for f in sorted(os.listdir(D_RUGS)):
            if f.endswith(".json"):
                r = load(f[:-5])
                if r: out.append(summary(r))
        out.sort(key=lambda s: (s["status"] == "done", -s["updated"]))
        return {"rugs": out, "import": imp, "zones": ZONES, "schools": SCHOOLS}

    @app.get("/api/rugs/{rid}")
    def api_rug(rid: str):
        r = load(rid)
        if not r: return JSONResponse({"error": "нет такого ковра"}, 404)
        for m in r.get("marks", []):
            d = BYID.get(m.get("motif") or "")
            m["_motif"] = {"az": d["az"], "ru": d["ru"], "tr": d["tr"]} if d else None
        c = BYID.get(r.get("composition") or "")
        r["_composition"] = {"az": c["az"], "ru": c["ru"]} if c else None
        return r

    @app.post("/api/rugs/{rid}")
    async def api_rug_save(rid: str, request: Request):
        b = await request.json(); who = user(request); now = int(time.time())
        with _lock:
            r = load(rid)
            if not r: return JSONResponse({"error": "нет такого ковра"}, 404)
            for k in ("title", "museum", "inv", "url", "license", "place", "date", "school", "composition", "technique", "size", "notes", "status"):
                if k in b: r[k] = str(b[k] or "")[:2000]
            if b.get("school_suggest") is None and "school_suggest" in b: r.pop("school_suggest", None)
            if "layout" in b: r["layout"] = b["layout"] if isinstance(b["layout"], dict) else None
            if "marks" in b:
                old = {m["id"]: m for m in r.get("marks", [])}; marks = []
                for m in b["marks"]:
                    try: box = [max(0.0, min(1.0, float(v))) for v in m["box"]][:4]
                    except Exception: continue
                    if len(box) != 4 or box[2] < 0.003 or box[3] < 0.003: continue
                    mid = str(m.get("id") or f"m{now}{len(marks)}")
                    keep = {k: m.get(k) for k in ("zone", "motif", "motif_text", "scheme", "group", "certainty", "status", "note", "angle")}
                    keep = {k: ("" if v is None else v) for k, v in keep.items()}
                    if keep["zone"] not in ZONES: keep["zone"] = "other"
                    if keep["status"] not in STATUSES: keep["status"] = "proposed"
                    if keep["motif"] and keep["motif"] not in BYID: keep["motif_text"] = keep["motif_text"] or keep["motif"]; keep["motif"] = ""
                    o = old.get(mid, {})
                    nm = {"id": mid, "box": [round(v, 5) for v in box], **keep,
                          "by": o.get("by") or who, "at": o.get("at") or now,
                          "confirmed_by": o.get("confirmed_by", ""), "confirmed_at": o.get("confirmed_at", 0)}
                    changed = not o or any(o.get(k) != nm.get(k) for k in ("box", "zone", "motif", "motif_text", "scheme", "group"))
                    if changed and o: nm["edited_by"], nm["edited_at"] = who, now
                    if nm["status"] in ("confirmed", "rejected") and (o.get("status") != nm["status"] or changed):
                        nm["confirmed_by"], nm["confirmed_at"] = who, now
                    if nm["status"] == "proposed": nm["confirmed_by"], nm["confirmed_at"] = "", 0
                    marks.append(nm)
                r["marks"] = marks
            r["updated"], r["updated_by"] = now, who
            save(r)
        return api_rug(rid)

    @app.delete("/api/rugs/{rid}")
    def api_rug_del(rid: str):
        for p in (os.path.join(D_RUGS, rid + ".json"), os.path.join(D_IMG, rid + ".jpg"), os.path.join(D_TH, rid + ".jpg")):
            if os.path.exists(p): os.remove(p)
        return {"ok": True}

    @app.post("/api/rugs_upload")
    async def api_rugs_upload(request: Request, files: list[UploadFile] = File(...)):
        made = []
        for f in files:
            name = os.path.basename(f.filename or "фото")
            if os.path.splitext(name)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"): continue
            rid = new_id(os.path.splitext(name)[0])
            try: w, h = store_image(rid, await f.read())
            except Exception: continue
            save(blank(rid, title=os.path.splitext(name)[0], source="загрузка: " + user(request), w=w, h=h)); made.append(rid)
        return {"made": made}

    @app.post("/api/rugs_import")
    async def api_rugs_import(request: Request):
        """Каталог (catalog.json из museum_open_colab / museum_photos): [{id, title, museum, inv, place, date, url, img, license}] — фото качает сервер."""
        items = await request.json()
        if isinstance(items, dict): items = items.get("items") or []
        have = set()
        for f in os.listdir(D_RUGS):
            if f.endswith(".json"): have.add((load(f[:-5]) or {}).get("src_id"))
        todo = [it for it in items if isinstance(it, dict) and it.get("img") and it.get("id") not in have]
        if imp["running"]: return JSONResponse({"error": "импорт уже идёт"}, 409)
        who = user(request)

        def run():
            imp.update(running=True, done=0, total=len(todo), errors=0, last="")
            for it in todo:
                try:
                    req = urllib.request.Request(it["img"], headers={"User-Agent": "CarpetDNA-CPS/1.0 (research catalogue)"})
                    data = urllib.request.urlopen(req, timeout=60).read()
                    rid = new_id(it.get("id") or it.get("title") or "rug")
                    w, h = store_image(rid, data)
                    r = blank(rid, title=it.get("title") or "", museum=it.get("museum") or "", inv=it.get("inv") or "", url=it.get("url") or "",
                              license=it.get("license") or "", place=it.get("place") or "", date=it.get("date") or "", source="импорт: " + who, w=w, h=h)
                    r["src_id"] = it.get("id"); r["img_url"] = it["img"]
                    sc, why = school_guess(it.get("title"), it.get("place"), it.get("type"))
                    if sc: r["school_suggest"] = {"school": sc, "why": f"слово «{why}» в подписи музея"}
                    save(r); imp["last"] = r["title"]
                except Exception as e:
                    imp["errors"] += 1; imp["last"] = f"ошибка: {it.get('id')}: {str(e)[:80]}"
                imp["done"] += 1
            imp["running"] = False
        threading.Thread(target=run, daemon=True).start()
        return {"started": len(todo), "skipped": len(items) - len(todo)}

    @app.post("/api/rugs/{rid}/layout")
    def api_rug_layout(rid: str):
        """Автоматически найти край ковра, полосы кайм и поле (предложение; человек правит)."""
        import rug_layout
        r = load(rid); a = cv2.imread(os.path.join(D_IMG, rid + ".jpg"))
        if not r or a is None: return JSONResponse({"error": "нет такого ковра"}, 404)
        lay = rug_layout.detect(a); lay["status"] = "auto"
        return {"layout": lay}

    @app.post("/api/rugs_from_figures")
    def api_rugs_from_figures(request: Request):
        """Фото ковров из книг (рисунки, помеченные в студии как «фото ковра») → ковры для разметки."""
        have = {(load(f[:-5]) or {}).get("src_id") for f in os.listdir(D_RUGS) if f.endswith(".json")}
        made = 0
        for f in sorted(os.listdir(D_WORK)):
            if not f.endswith(".json"): continue
            fid = f[:-5]
            try:
                with open(os.path.join(D_WORK, f), encoding="utf-8") as fh: w = json.load(fh)
            except Exception: continue
            if w.get("kind") != "photo" or ("fig:" + fid) in have: continue
            src = os.path.join(D_CROPS, fid + ".png")
            if not os.path.exists(src): continue
            sheet = w.get("sheet") or ""
            mp = re.search(r"page_(\d+)", sheet); book = sheet.split("/")[0]
            title = {"kerimov_vol1_patterns": "Керимов, т. I", "az_carpets_magazine_33_patterns": "Журнал «Azərbaycan xalçaları» №33"}.get(book, book)
            mt = re.match(r"t(\d+)_f0*(\d+)$", fid)
            title += (f", табл. {mt.group(1)}, фото {mt.group(2)}" if mt else "") + (f" (стр. скана {int(mp.group(1))})" if mp else "")
            rid = new_id("k_" + fid if mt else fid)
            try:
                with open(src, "rb") as fh: wd, ht = store_image(rid, fh.read())
            except Exception: continue
            r = blank(rid, title=title, source="фото из книги: " + fid, w=wd, h=ht)
            r["src_id"] = "fig:" + fid; r["notes"] = (w.get("meta") or {}).get("note") or ""
            save(r); made += 1
        return {"made": made}

    @app.post("/api/rugs_zip")
    async def api_rugs_zip(request: Request, file: UploadFile = File(...)):
        """Архив фото ковров: картинки + одноимённые .txt (описание). Школа предлагается по папке / имени / описанию."""
        import zipfile
        data = await file.read(); who = user(request)
        try: z = zipfile.ZipFile(io.BytesIO(data))
        except Exception: return JSONResponse({"error": "это не zip"}, 400)
        names = z.namelist(); low = {n.lower(): n for n in names}
        have = {(load(f[:-5]) or {}).get("src_id") for f in os.listdir(D_RUGS) if f.endswith(".json")}
        made, skipped = 0, 0
        for n in names:
            base, ext = os.path.splitext(n)
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp") or "/." in "/" + n or n.startswith("__MACOSX"): continue
            sid = "zip:" + n
            if sid in have: skipped += 1; continue
            txt = ""
            for cand in (base + ".txt", base + ".TXT"):
                if cand.lower() in low:
                    txt = z.read(low[cand.lower()]).decode("utf-8", "ignore").strip(); break
            folder = os.path.dirname(n); stem = os.path.basename(base)
            rid = new_id(stem)
            try: wd, ht = store_image(rid, z.read(n))
            except Exception: continue
            title = (txt.split(".")[0][:90] if txt else stem.replace("_", " "))
            r = blank(rid, title=title, source="архив: " + who + (" / " + folder if folder else ""), w=wd, h=ht, notes=txt[:2000], **text_facts(txt))
            sc, why = school_guess(stem, folder, txt[:300])
            if sc: r["school_suggest"] = {"school": sc, "why": f"слово «{why}» в имени/папке/описании"}
            r["src_id"] = sid; save(r); made += 1
        return {"made": made, "skipped": skipped}

    # ---------------- картинки ----------------
    @app.get("/rugimg/{rid}.jpg")
    def rug_img(rid: str):
        p = os.path.join(D_IMG, os.path.basename(rid) + ".jpg")
        return FileResponse(p) if os.path.exists(p) else Response(status_code=404)

    @app.get("/rugthumb/{rid}.jpg")
    def rug_thumb(rid: str):
        rid = os.path.basename(rid); p = os.path.join(D_TH, rid + ".jpg"); src = os.path.join(D_IMG, rid + ".jpg")
        if not os.path.exists(p) and os.path.exists(src):
            a = cv2.imread(src); s = 220 / max(a.shape[:2]); a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(p, a, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return FileResponse(p, headers={"Cache-Control": "max-age=3600"}) if os.path.exists(p) else Response(status_code=404)

    @app.get("/api/rugs/{rid}/crop")
    def rug_crop(rid: str, b: str = Query(...), s: int = 160):
        a = cv2.imread(os.path.join(D_IMG, os.path.basename(rid) + ".jpg"))
        if a is None: return Response(status_code=404)
        try: x, y, w, h = [float(v) for v in b.split(",")]
        except Exception: return Response(status_code=400)
        H, W = a.shape[:2]; x0, y0 = int(x * W), int(y * H); x1, y1 = max(x0 + 2, int((x + w) * W)), max(y0 + 2, int((y + h) * H))
        c = a[y0:y1, x0:x1]; k = min(4.0, max(1, s) / max(c.shape[:2]))
        c = cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_NEAREST)
        return Response(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), media_type="image/jpeg")
