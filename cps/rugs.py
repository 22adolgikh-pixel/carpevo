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
from starlette.concurrency import run_in_threadpool

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = {"field": "поле", "medallion": "медальон", "spandrel": "угол поля", "border": "главная кайма",
         "minor": "малая кайма", "guard": "бордюрная полоска", "end": "торцевая полоса", "other": "другое"}
STATUSES = ("proposed", "confirmed", "rejected")
SCHOOLS = {"guba-shirvan": "Губа-Ширван", "ganja-gazakh": "Гянджа-Газах", "karabakh": "Карабах", "tabriz": "Тебриз", "nakhchivan": "Нахчыван", "": "—"}
MAXSIDE = 4000   # v10.27: хайрез загружается почти без уменьшения (поиск повторов и совмещение всё равно уменьшают фото у себя)
SAME_FIELDS = ("school", "composition", "technique", "museum", "inv", "place", "date", "size")   # что переносится с двойника в пустые поля
PASS_FIELDS = SAME_FIELDS + ("url", "license", "notes")   # v10.26: что части разворота наследуют от исходного фото
_lock = threading.Lock()
imp = {"running": False, "done": 0, "total": 0, "errors": 0, "last": ""}
sync = {"running": False, "stage": "", "done": 0, "total": 0, "error": "", "last_ok": None, "made": 0, "updated": 0}


def drive_fetch(file_id, dest, progress=None):
    """Скачать публичный («по ссылке») файл Google Drive потоком в файл (большие файлы — через подтверждение антивирус-страницы)."""
    import http.cookiejar
    cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    H = {"User-Agent": "Mozilla/5.0 (compatible; cps-rugs/1.0)"}
    urls = [f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"]
    last = ""
    for u in urls:
        with op.open(urllib.request.Request(u, headers=H), timeout=120) as r:
            if "text/html" in r.headers.get("Content-Type", ""):
                last = r.read(4000).decode("utf-8", "ignore"); continue
            total = int(r.headers.get("Content-Length") or 0); got = 0
            with open(dest, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); got += len(b)
                    if progress: progress(got, total)
            return
    raise RuntimeError("Диск не отдал файл (нет доступа «по ссылке»?) " + re.sub(r"<[^>]+>", " ", last)[:200])


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
                "motifs": len({m.get("motif") for m in mk if m.get("motif")}), "updated": r.get("updated", 0), "source": r.get("source", ""),
                "src": (r.get("cat") or {}).get("src") or ("upload" if not r.get("src_id") else (r.get("src_id") or "").split(":")[0]),
                "group": r.get("same_group", ""), "same": len(r.get("same") or []), "fragment": (r.get("cat") or {}).get("fragment", False),
                "composition": r.get("composition", ""), "kind": r.get("technique", ""),
                "multi": len(r.get("multi") or []), "split_into": r.get("split_into") or [], "split_from": r.get("split_from", ""),
                "machine": sum(1 for m in mk if m.get("machine") and m.get("status") == "proposed"),
                "w": r.get("w") or 0, "h": r.get("h") or 0,
                "same_no": [k for k, v in (r.get("same_ok") or {}).items() if v and not v.get("ok")],
                "same_ids": [x.get("id") for x in (r.get("same") or []) if x.get("id")],
                "part": (r.get("cat") or {}).get("part", 0)}

    D_LIB = os.path.join(DATA, "rug_lib"); SETF = os.path.join(DATA, "rugs_settings.json")

    def settings():
        try:
            with open(SETF, encoding="utf-8") as f: return json.load(f)
        except Exception: return {"bundle_id": os.environ.get("CPS_RUGS_BUNDLE_ID", "")}

    def comp_id(name):
        if not name: return ""
        if name in BYID: return name
        return COMP.get(fold(name)) or COMP.get("~" + skel(name)) or ""

    CAT_FIELDS = ("title", "museum", "inv", "url", "license", "place", "date", "size", "technique", "school", "composition")

    def run_sync(bid, who):
        import zipfile, shutil
        sync.update(running=True, stage="скачиваю пакет с Диска", done=0, total=0, error="", made=0, updated=0)
        try:
            os.makedirs(D_LIB, exist_ok=True); tmp = os.path.join(D_LIB, "_bundle.zip")
            if os.environ.get("CPS_RUGS_TEST") and os.path.exists(bid): shutil.copy(bid, tmp)
            else: drive_fetch(bid, tmp, lambda g, t: sync.update(done=g // (1 << 20), total=t // (1 << 20)))
            sync.update(stage="распаковываю")
            with zipfile.ZipFile(tmp) as z:
                cat = json.loads(z.read("catalog.json").decode("utf-8"))
                for n in z.namelist():
                    if n.startswith("img/"): z.extract(n, D_LIB)
            os.remove(tmp)
            groups = {g["id"]: g for g in cat.get("groups", [])}
            items = cat.get("items", []); sync.update(stage="обновляю карточки ковров", done=0, total=len(items))
            for n, it in enumerate(items):
                rid = re.sub(r"[^\w\-]+", "_", it["id"])[:80]
                with _lock:
                    r = load(rid); new = r is None
                    if new: r = blank(rid, source="каталог: " + it.get("src", ""))
                    old = r.get("cat") or {}
                    val = {"title": it.get("title", ""), "museum": it.get("museum", ""), "inv": it.get("inv", ""), "url": it.get("url", ""),
                           "license": it.get("license", ""), "place": it.get("place", ""), "date": it.get("date", ""), "size": it.get("size", ""),
                           "technique": it.get("technique", ""), "school": it.get("school", ""), "composition": comp_id(it.get("composition", ""))}
                    for k in CAT_FIELDS:                                 # правка человека важнее каталога: обновляем, только если поле не трогали
                        if not r.get(k) or r.get(k) == old.get(k): r[k] = val[k]
                    if not r.get("notes") or r.get("notes") == old.get("text"): r["notes"] = it.get("text", "")
                    r["cat"] = {**val, "text": it.get("text", ""), "names": it.get("names", []), "src": it.get("src", ""), "school_why": it.get("school_why", ""),
                                "composition_name": it.get("composition", ""), "fragment": bool(it.get("fragment")), "object": it.get("object", ""),
                                "ambiguous": bool(it.get("ambiguous")), "folder": it.get("folder", ""),
                                "page_figs": it.get("page_figs") or [], "page_box": it.get("page_box") or 0}
                    r["src_id"] = "cat:" + it["id"]; r["lib_img"] = it.get("img", "")
                    g = groups.get(it.get("same_group") or "")
                    if g:
                        r["same_group"] = g["id"]
                        sc = {}
                        for a, b, v, cs in g.get("pairs", []):
                            if it["id"] in (a, b): sc[b if a == it["id"] else a] = v
                        r["same"] = [{"id": re.sub(r"[^\w\-]+", "_", m)[:80], "score": sc.get(m, 0)} for m in g["members"] if m != it["id"]]
                    else: r.pop("same_group", None); r["same"] = []
                    r.setdefault("same_ok", {})
                    p = os.path.join(D_LIB, r["lib_img"])
                    if os.path.exists(p) and not r.get("w"):
                        a = cv2.imread(p)
                        if a is not None: r["h"], r["w"] = a.shape[:2]
                    th = os.path.join(D_TH, rid + ".jpg")
                    if os.path.exists(th) and new is False and r.get("_lib_t") != cat.get("built"): os.remove(th)
                    r["_lib_t"] = cat.get("built")
                    save(r)
                sync["made" if new else "updated"] += 1; sync["done"] = n + 1
            st = settings(); st.update(bundle_id=bid, last_sync=time.strftime("%Y-%m-%d %H:%M"), built=cat.get("built"), by=who)
            with open(SETF, "w", encoding="utf-8") as f: json.dump(st, f, ensure_ascii=False)
            sync.update(stage="готово", last_ok=time.strftime("%Y-%m-%d %H:%M"))
        except Exception as e:
            sync.update(error=str(e)[:300], stage="ошибка")
        sync["running"] = False

    @app.get("/api/rugs_sync")
    def api_rugs_sync_state():
        st = settings(); return {**sync, "bundle_id": st.get("bundle_id", ""), "last_sync": st.get("last_sync"), "built": st.get("built")}

    @app.post("/api/rugs_sync")
    async def api_rugs_sync(request: Request):
        b = await request.json() if (await request.body()) else {}
        bid = (b.get("bundle_id") or settings().get("bundle_id") or "").strip()
        if not (os.environ.get("CPS_RUGS_TEST") and os.path.exists(bid)):          # для проверки: путь к локальному пакету
            m = re.search(r"[-\w]{25,}", bid); bid = m.group(0) if m else ""
        if not bid: return JSONResponse({"error": "нужен id пакета (или ссылка на rugs_bundle.zip)"}, 400)
        if sync["running"]: return JSONResponse({"error": "загрузка уже идёт"}, 409)
        threading.Thread(target=run_sync, args=(bid, user(request)), daemon=True).start()
        return {"started": True, "bundle_id": bid}

    @app.post("/api/rugs/{rid}/same")
    async def api_rug_same(rid: str, request: Request):
        """Подтвердить / отклонить «это тот же ковёр» (отметка ставится у обоих)."""
        b = await request.json(); other = b.get("other"); ok = b.get("ok")
        who = user(request)
        filled = {}
        with _lock:
            ra, rb = load(rid), load(other)
            for a, c, r, o in ((rid, other, ra, rb), (other, rid, rb, ra)):
                if not r: continue
                r.setdefault("same_ok", {})
                if ok is not None and c not in {x.get("id") for x in r.get("same") or []}:
                    r.setdefault("same", []).append({"id": c, "score": 0})
                if ok is None: r["same_ok"].pop(c, None)
                else: r["same_ok"][c] = {"ok": bool(ok), "by": who, "at": int(time.time())}
                # подтвердили «тот же ковёр» → пустые поля паспорта берём у двойника; заполненные НЕ трогаем
                # (если значения расходятся — обе карточки сохраняют своё, расхождение видно в паспорте)
                if ok and o:
                    got = []
                    for k in SAME_FIELDS:
                        if not r.get(k) and o.get(k):
                            r[k] = o[k]; got.append(k)
                            r.setdefault("from_twin", {})[k] = c
                    if got: filled[a] = got
                save(r)
        out = api_rug(rid); out["filled"] = filled
        return out

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
        for x in r.get("same") or []:
            o = load(x["id"]) or {}
            x.update(title=o.get("title") or x["id"], src=(o.get("cat") or {}).get("src", ""), school=o.get("school", ""), museum=o.get("museum", ""),
                     ok=(r.get("same_ok") or {}).get(x["id"]), n=len([m for m in o.get("marks", []) if m.get("status") != "rejected"]),
                     wh=img_size(x["id"], o))
        r["wh"] = img_size(rid, r)
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
            if b.get("multi_no"): r["multi"] = []                     # «это один ковёр, не делить»
            if "marks" in b:
                old = {m["id"]: m for m in r.get("marks", [])}; marks = []
                for m in b["marks"]:
                    try: box = [max(0.0, min(1.0, float(v))) for v in m["box"]][:4]
                    except Exception: continue
                    if len(box) != 4 or box[2] < 0.003 or box[3] < 0.003: continue
                    mid = str(m.get("id") or f"m{now}{len(marks)}")
                    keep = {k: m.get(k) for k in ("zone", "motif", "motif_text", "scheme", "group", "certainty", "status", "note", "angle", "flip", "machine", "score", "xfer")}
                    keep = {k: ("" if v is None else v) for k, v in keep.items()}
                    if keep["zone"] not in ZONES: keep["zone"] = "other"
                    if keep["status"] not in STATUSES: keep["status"] = "proposed"
                    keep["angle"] = int(keep["angle"] or 0) % 360 if str(keep["angle"] or 0).lstrip("-").isdigit() else 0
                    keep["flip"] = 1 if keep["flip"] else 0
                    if keep["motif"] and keep["motif"] not in BYID: keep["motif_text"] = keep["motif_text"] or keep["motif"]; keep["motif"] = ""
                    o = old.get(mid, {})
                    nm = {"id": mid, "box": [round(v, 5) for v in box], **keep,
                          "by": o.get("by") or (("машина → " + who) if keep.get("machine") else who), "at": o.get("at") or now,
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
        r = load(rid); p = img_path(rid); a = cv2.imread(p) if p else None
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

    # ---------------- машинные предложения (v10.23) ----------------
    D_OUT = os.path.join(DATA, "out")

    def marks_boxes(r): return [m["box"] for m in r.get("marks", []) if m.get("status") != "rejected"]

    @app.post("/api/rugs/{rid}/repeats")
    async def api_rug_repeats(rid: str, request: Request):
        """Человек обвёл элемент → ищем такие же на этом фото (с поворотами и отражениями). Ответ — кандидаты, сохраняет клиент."""
        import rug_match
        b = await request.json(); r = load(rid); p = img_path(rid); a = cv2.imread(p) if p else None
        if not r or a is None: return JSONResponse({"error": "нет такого ковра"}, 404)
        box = b.get("box")
        if not box:
            m = next((m for m in r.get("marks", []) if m["id"] == b.get("mark")), None)
            if not m: return JSONResponse({"error": "нет такой рамки"}, 404)
            box = m["box"]
        rejected = [m["box"] for m in r.get("marks", []) if m.get("status") == "rejected"]
        # тяжёлый расчёт — в отдельном потоке: иначе, пока он идёт, сервер не отвечает никому (пропадают миниатюры и т. п.)
        c = await run_in_threadpool(rug_match.find_repeats, a, box, marks_boxes(r) + rejected)
        best = max([x["score"] for x in c] or [0])
        for x in c: x["sure"] = x["score"] >= max(0.68, best * 0.86)     # «точно такой же» / «похоже (другой цвет или близкий узор)»
        return {"cands": c}

    def scale_of(r, W, H):
        """Сколько пикселей фото на один узел: по любой рамке со схемой (лучше подтверждённой)."""
        ms = sorted([m for m in r.get("marks", []) if m.get("scheme") and m.get("status") != "rejected"], key=lambda m: m.get("status") != "confirmed")
        for m in ms:
            q = cv2.imread(os.path.join(D_OUT, m["scheme"] + ".png"), cv2.IMREAD_UNCHANGED)
            if q is None: continue
            kh, kw = q.shape[:2]
            if (m.get("angle") or 0) in (90, 270): kh, kw = kw, kh
            return m["box"][2] * W / kw, m["box"][3] * H / kh, m
        return None

    @app.post("/api/rugs/{rid}/scheme_find")
    async def api_rug_scheme_find(rid: str, request: Request):
        """Ищем на фото элемент по схеме студии; масштаб — по уже отмеченному элементу со схемой.
        at=[x,y] — элемент перетащили на ковёр в эту точку: ищем только рядом, а если масштаба ещё нет — ставим рамку примерно."""
        import rug_match
        b = await request.json(); r = load(rid); p = img_path(rid); a = cv2.imread(p) if p else None
        if not r or a is None: return JSONResponse({"error": "нет такого ковра"}, 404)
        q = cv2.imread(os.path.join(D_OUT, os.path.basename(b.get("scheme") or "") + ".png"), cv2.IMREAD_UNCHANGED)
        if q is None: return JSONResponse({"error": "у этой схемы нет картинки (рисунок не закончен в студии)"}, 404)
        H, W = a.shape[:2]; kh, kw = q.shape[:2]; at = b.get("at")
        sc = scale_of(r, W, H)
        if not sc and not at:
            return JSONResponse({"error": "сначала обведите на этом ковре хотя бы один элемент и выберите ему схему — по нему машина поймёт масштаб (сколько пикселей в узле)"}, 400)
        if sc: kx, ky, ref = sc
        else: kx = ky = min(W, H) / 7 / max(kw, kh); ref = None
        guess = None; region = None
        if at:
            gw, gh = kw * kx / W, kh * ky / H
            guess = [max(0, min(1 - gw, at[0] - gw / 2)), max(0, min(1 - gh, at[1] - gh / 2)), gw, gh]
            region = [max(0, at[0] - gw * 1.3), max(0, at[1] - gh * 1.3), 0, 0]
            region[2] = min(1 - region[0], gw * 2.6); region[3] = min(1 - region[1], gh * 2.6)
        c = await run_in_threadpool(lambda: rug_match.find_scheme(a, q, kx, ky, exclude=[] if at else marks_boxes(r), limit=1 if at else 3, region=region))
        return {"cands": c, "scale": [round(kx, 2), round(ky, 2)], "ref": ref["id"] if ref else "", "guess": guess, "scale_known": bool(sc)}

    _sizes = {}

    def img_size(rid, r=None):
        """размер фото (ширина, высота) — чтобы видеть, у какого двойника фото крупнее"""
        p = img_path(rid)
        if not p: return None
        k = (p, os.path.getmtime(p))
        if k not in _sizes:
            a = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            _sizes[k] = None if a is None else [a.shape[1] * 8, a.shape[0] * 8]
            if a is not None:
                full = cv2.imread(p); _sizes[k] = [full.shape[1], full.shape[0]] if full is not None else _sizes[k]
        return _sizes[k]

    _H = {}

    @app.post("/api/rugs/{rid}/transfer")
    async def api_rug_transfer(rid: str, request: Request):
        """Перенести разметку с фото rid на фото двойника (тот же ковёр, другой источник): фото совмещаются по точкам (SIFT),
        рамки переводятся в координаты другого фото. Перенесённые рамки — предложения (пунктир «?»), их проверяет человек."""
        import rug_match
        b = await request.json(); dst = b.get("to") or ""; who = user(request)
        r = load(rid); o = load(dst)
        pa, pb = img_path(rid), img_path(dst)
        if not r or not o or not pa or not pb: return JSONResponse({"error": "нет такого ковра или фото"}, 404)
        key = (pa, pb, os.path.getmtime(pa), os.path.getmtime(pb))
        if key not in _H:
            A, B = cv2.imread(pa), cv2.imread(pb)
            _H[key] = await run_in_threadpool(rug_match.align, A, B)
        H, n = _H[key]
        if H is None:
            return JSONResponse({"error": f"не получилось совместить фото (общих точек: {n}) — возможно, это всё-таки разные ковры или фото слишком разные"}, 400)
        made = skipped = 0
        with _lock:
            o = load(dst); have = {m["id"] for m in o.get("marks", [])}
            def iou(p, q):
                ix = max(0, min(p[0] + p[2], q[0] + q[2]) - max(p[0], q[0])); iy = max(0, min(p[1] + p[3], q[1] + q[3]) - max(p[1], q[1]))
                u = p[2] * p[3] + q[2] * q[3] - ix * iy
                return ix * iy / u if u > 0 else 0
            for m in r.get("marks", []):
                if m.get("status") == "rejected": continue
                nid = "x" + m["id"]
                if nid in have: skipped += 1; continue
                box, ang = rug_match.map_box(H, m["box"])
                if not box or any(iou(box, q["box"]) > 0.5 for q in o.get("marks", [])): skipped += 1; continue
                g = m.get("group") or ""
                o.setdefault("marks", []).append({**{k: m.get(k, "") for k in ("zone", "motif", "motif_text", "scheme", "certainty", "note")},
                    "id": nid, "box": box, "group": ("x" + g) if g else "", "angle": ((m.get("angle") or 0) + ang) % 360, "flip": m.get("flip") or 0,
                    "status": "proposed", "machine": 1, "score": 0, "xfer": rid,
                    "by": "перенесено с двойника → " + who, "at": int(time.time()), "confirmed_by": "", "confirmed_at": 0})
                made += 1
            o["updated"], o["updated_by"] = int(time.time()), who; save(o)
        return {"made": made, "skipped": skipped, "points": n}

    @app.post("/api/rugs/{rid}/split_detect")
    def api_rug_split_detect(rid: str):
        import rug_match
        p = img_path(rid); a = cv2.imread(p) if p else None
        if a is None: return JSONResponse({"error": "нет фото"}, 404)
        return {"parts": rug_match.find_carpets(a)}

    def part_title(r, fig, i, n):
        """название части разворота: подпись рисунка из книги, иначе «… · ковёр i из n»"""
        if fig and (fig.get("title") or fig.get("caption") or fig.get("label")):
            return fig.get("title") or ", ".join(x for x in (fig.get("label"), fig.get("caption")) if x)
        base = re.sub(r" · ковёр \d+ из \d+$", "", r.get("title") or r["id"])
        return base + f" · ковёр {i + 1} из {n}"

    def apply_part(n, r, fig, i, cnt, box):
        """паспорт части: всё, что знали о развороте, + подпись своего рисунка (если книга дала подписи по порядку).
        Заполненные человеком поля части не трогаем."""
        for k in PASS_FIELDS:
            if not n.get(k) and r.get(k): n[k] = r[k]
        cat = {**(r.get("cat") or {}), "part": i + 1, "parts": cnt}
        cat.pop("page_figs", None)
        if fig:
            for k, v in (("school", fig.get("school")), ("composition", comp_id(fig.get("composition") or "")), ("date", fig.get("date")),
                         ("technique", fig.get("technique"))):
                if v and (not n.get(k) or n.get(k) == (n.get("cat") or {}).get(k) or n.get(k) == r.get(k)): n[k] = v
            if fig.get("text") and (not n.get("notes") or n.get("notes") == r.get("notes")): n["notes"] = fig["text"]
            cat.update(names=[x for x in (fig.get("composition"), fig.get("caption")) if x], school_why=fig.get("school_why", ""),
                       composition_name=fig.get("composition", ""), text=fig.get("text", ""), fragment=bool(fig.get("fragment")),
                       fig_label=fig.get("label", ""), ambiguous=False, **{k: n.get(k, "") for k in ("school", "composition", "date", "technique")})
        else: cat["ambiguous"] = True
        n["cat"] = cat
        auto = not n.get("title") or re.search(r" · ковёр \d+ из \d+$", n.get("title") or "") or n.get("title") == (n.get("_auto_title") or "")
        if auto: n["title"] = n["_auto_title"] = part_title(r, fig, i, cnt)
        n["split_from"] = r["id"]; n["split_box"] = [round(float(v), 5) for v in box]; n["split_n"] = i
        n["src_id"] = (r.get("src_id") or r["id"]) + f"#{i + 1}"
        # двойники разворота — кандидаты и для части (человек отметит ✓ / ✗ у той части, где этот ковёр)
        have = {x.get("id") for x in n.get("same") or []}
        for x in r.get("same") or []:
            if x.get("id") and x["id"] not in have: n.setdefault("same", []).append({"id": x["id"], "score": x.get("score", 0), "via": r["id"]})

    def remap(m, ob, nb):
        """рамка в координатах старой вырезки → в координатах новой (при правке разделения)"""
        bx, by, bw, bh = m["box"]
        ax, ay = ob[0] + bx * ob[2], ob[1] + by * ob[3]; aw, ah = bw * ob[2], bh * ob[3]
        x0, y0 = max(ax, nb[0]), max(ay, nb[1]); x1, y1 = min(ax + aw, nb[0] + nb[2]), min(ay + ah, nb[1] + nb[3])
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) < 0.5 * aw * ah: return None
        return [round((x0 - nb[0]) / nb[2], 5), round((y0 - nb[1]) / nb[3], 5), round((x1 - x0) / nb[2], 5), round((y1 - y0) / nb[3], 5)]

    @app.post("/api/rugs/{rid}/split")
    async def api_rug_split(rid: str, request: Request):
        """Фото с несколькими коврами (разворот книги) → отдельные карточки, в порядке, который задал человек.
        Каждая часть получает паспорт разворота и, если есть, подпись своего рисунка из книги (figs — номер подписи для каждой части).
        Повторный вызов = правка разделения: части с указанным id перерезаются (их рамки переносятся), лишние удаляются.
        Исходная карточка скрывается, не удаляется."""
        b = await request.json(); parts = b.get("parts") or []; ids = b.get("ids") or []; fsel = b.get("figs") or []; who = user(request)
        r = load(rid); p = img_path(rid); a = cv2.imread(p) if p else None
        if not r or a is None: return JSONResponse({"error": "нет такого ковра"}, 404)
        if len(parts) < 1: return JSONResponse({"error": "нет частей"}, 400)
        H, W = a.shape[:2]; made = []; pf = (r.get("cat") or {}).get("page_figs") or []
        with _lock:
            old = list(r.get("split_into") or []); keep = set()
            for i, box in enumerate(parts):
                x, y, w, h = [float(v) for v in box]
                x0, y0 = max(0, int(x * W)), max(0, int(y * H)); x1, y1 = min(W, int((x + w) * W)), min(H, int((y + h) * H))
                if x1 - x0 < 20 or y1 - y0 < 20: continue
                fi = fsel[i] if i < len(fsel) else None
                fig = pf[fi] if isinstance(fi, int) and 0 <= fi < len(pf) else (pf[i] if fi is None and len(pf) == len(parts) else None)
                cid = ids[i] if i < len(ids) else None
                n = load(cid) if cid and cid in old else None
                nid = n["id"] if n else new_id(rid + f"_{i + 1}")
                wd, ht = store_image(nid, cv2.imencode(".jpg", a[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
                if n:
                    ob = n.get("split_box"); nb = [x, y, w, h]
                    if ob:
                        ms = []
                        for m in n.get("marks", []):
                            bb = remap(m, ob, nb)
                            if bb: ms.append({**m, "box": bb})
                        n["marks"] = ms
                    n["w"], n["h"] = wd, ht
                else:
                    n = blank(nid, source=r.get("source", ""), w=wd, h=ht)
                    # рамки исходного фото, целиком попавшие в эту часть, переносим (в координатах части)
                    for m in r.get("marks", []):
                        bx, by, bw, bh = m["box"]
                        if bx >= x and by >= y and bx + bw <= x + w and by + bh <= y + h:
                            n["marks"].append({**m, "box": [round((bx - x) / w, 5), round((by - y) / h, 5), round(bw / w, 5), round(bh / h, 5)]})
                apply_part(n, r, fig, len(made), len(parts), [x, y, w, h])
                n["updated"] = int(time.time()); save(n); made.append(nid); keep.add(nid)
            for c in old:
                if c not in keep: api_rug_del(c)
            r["split_into"] = made; r["split_by"] = who; r["updated"] = int(time.time()); save(r)
        return {"made": made}

    @app.get("/api/rugs/{rid}/split_parts")
    def api_rug_split_parts(rid: str):
        """текущее разделение (для правки): рамки частей в порядке, их id, подписи рисунков страницы"""
        r = load(rid)
        if not r: return JSONResponse({"error": "нет такого ковра"}, 404)
        out = []
        for c in r.get("split_into") or []:
            n = load(c) or {}
            out.append({"id": c, "box": n.get("split_box"), "title": n.get("title", ""), "n": len(n.get("marks", [])), "fig": (n.get("cat") or {}).get("fig_label", "")})
        return {"parts": out, "page_figs": (r.get("cat") or {}).get("page_figs") or []}

    def repair_parts():
        """v10.26: у частей, разрезанных до этой версии, паспорт был почти пустой, а место на развороте не записано.
        Дополняем пустые поля из исходного фото и находим, где часть лежала на развороте (по совпадающим точкам)."""
        import rug_match
        for f in os.listdir(D_RUGS):
            if not f.endswith(".json"): continue
            try:
                n = load(f[:-5])
                if not n or not n.get("split_from") or n.get("_repaired"): continue
                r = load(n["split_from"])
                if not r: continue
                kids = r.get("split_into") or []
                i = kids.index(n["id"]) if n["id"] in kids else 0
                box = n.get("split_box")
                if not box:
                    pa, pb = img_path(n["id"]), img_path(r["id"])
                    A, B = (cv2.imread(pa) if pa else None), (cv2.imread(pb) if pb else None)
                    if A is not None and B is not None:
                        Hm, _ = rug_match.align(A, B)
                        if Hm is not None: box, _a = rug_match.map_box(Hm, [0, 0, 1, 1])
                with _lock:
                    n = load(n["id"])
                    pf = (r.get("cat") or {}).get("page_figs") or []
                    apply_part(n, r, pf[i] if len(pf) == len(kids) else None, i, len(kids), box or [0, 0, 1, 1])
                    if not box: n.pop("split_box", None)
                    n["_repaired"] = 1; save(n)
            except Exception as e:
                print("repair_parts:", f, e)

    @app.post("/api/rugs/{rid}/unsplit")
    def api_rug_unsplit(rid: str):
        """Отменить разделение: части удаляются, исходная карточка снова видна."""
        r = load(rid)
        if not r: return JSONResponse({"error": "нет такого ковра"}, 404)
        for c in r.get("split_into") or []: api_rug_del(c)
        r.pop("split_into", None); save(r)
        return {"ok": True}

    scan = {"running": False, "done": 0, "total": 0, "found": 0}

    @app.get("/api/rugs_multi")
    def api_rugs_multi_state(): return scan

    @app.post("/api/rugs_multi")
    def api_rugs_multi():
        """Пройти все фото и пометить те, где похоже на несколько ковров (развороты книг)."""
        import rug_match
        if scan["running"]: return scan
        ids = [f[:-5] for f in os.listdir(D_RUGS) if f.endswith(".json")]

        def run():
            scan.update(running=True, done=0, total=len(ids), found=0)
            for rid in ids:
                try:
                    r = load(rid)
                    if r and "multi" not in r and not r.get("split_into") and not r.get("split_from"):
                        p = img_path(rid); a = cv2.imread(p) if p else None
                        if a is not None:
                            parts = rug_match.find_carpets(a)
                            with _lock:
                                r = load(rid); r["multi"] = parts; save(r)
                    if r and r.get("multi"): scan["found"] += 1
                except Exception: pass
                scan["done"] += 1
            scan["running"] = False
        threading.Thread(target=run, daemon=True).start()
        return scan

    # ---------------- картинки ----------------
    def img_path(rid):
        p = os.path.join(D_IMG, rid + ".jpg")
        if os.path.exists(p): return p
        r = load(rid) or {}
        q = os.path.join(D_LIB, r.get("lib_img") or ("img/" + rid + ".jpg"))
        return q if os.path.exists(q) else None

    @app.get("/rugimg/{rid}.jpg")
    def rug_img(rid: str):
        p = img_path(os.path.basename(rid))
        return FileResponse(p) if p else Response(status_code=404)

    @app.get("/rugthumb/{rid}.jpg")
    def rug_thumb(rid: str):
        rid = os.path.basename(rid); p = os.path.join(D_TH, rid + ".jpg"); src = img_path(rid)
        if not os.path.exists(p) and src:
            a = cv2.imread(src); s = 220 / max(a.shape[:2]); a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(p, a, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return FileResponse(p, headers={"Cache-Control": "max-age=3600"}) if os.path.exists(p) else Response(status_code=404)

    @app.get("/api/rugs/{rid}/crop")
    def rug_crop(rid: str, b: str = Query(...), s: int = 160):
        p = img_path(os.path.basename(rid)); a = cv2.imread(p) if p else None
        if a is None: return Response(status_code=404)
        try: x, y, w, h = [float(v) for v in b.split(",")]
        except Exception: return Response(status_code=400)
        H, W = a.shape[:2]; x0, y0 = int(x * W), int(y * H); x1, y1 = max(x0 + 2, int((x + w) * W)), max(y0 + 2, int((y + h) * H))
        c = a[y0:y1, x0:x1]; k = min(4.0, max(1, s) / max(c.shape[:2]))
        c = cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_NEAREST)
        return Response(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), media_type="image/jpeg")

    threading.Thread(target=repair_parts, daemon=True).start()   # после того как определены img_path и остальное
