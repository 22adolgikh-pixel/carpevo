"""v11.1 — единая палитра красителей.

Библиотека красителей одна на всю студию: data/dyes.json (первый раз копируется из dyes.json рядом с кодом).
У красителя: id (латиница, не меняется), az, ru,
  hex  — как краситель показывается везде в студии и в выгрузках (его меняет человек на экране «Красители»);
  scan — как он выглядит в выцветшей печати книги (по нему автооцифровка узнаёт краситель на скане).

В рисунке (data/work/<id>.json) у каждого цвета палитры:
  palette[i]       — итоговый цвет (по нему рисуются выгрузки; для связанных цветов вычисляется из библиотеки);
  palette_dyes[i]  — id красителя или null;
  palette_link[i]  — true: цвет берётся из библиотеки (меняется вместе с ней); false: свой цвет / только предложение автомата;
  palette_scan[i]  — цвет, замеренный на скане (остаётся как замер, его видно в подсказке).
Если в одном рисунке один краситель встречается несколько раз (два оттенка «zoğalı»), оттенки получаются из цвета
библиотеки сдвигом светлоты (L*) в том же порядке, в каком они светлее/темнее на скане. Тот же расчёт — в index.html (resolvePal).
"""
import os, json, re, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "dyes.json")
STEP_L = 10.0          # шаг светлоты между оттенками одного красителя в одном рисунке
_lock = threading.RLock()
state = {"running": False, "done": 0, "total": 0, "dye": "", "last": ""}
_cfg = {}


# ---------------------------------------------------------------- цвет
def _hex_rgb(h):
    h = str(h or "").strip().lstrip("#")
    if len(h) == 3: h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h): return None
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def norm_hex(h):
    rgb = _hex_rgb(h)
    return "#%02x%02x%02x" % rgb if rgb else None


def _lin(v):
    v /= 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _gam(v):
    v = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return max(0, min(255, int(v * 255 + 0.5)))


def hex2lab(h):
    r, g, b = (_lin(v) for v in _hex_rgb(h))
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx = f((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047)
    fy = f(0.2126 * r + 0.7152 * g + 0.0722 * b)
    fz = f((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883)
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]


def lab2hex(L, a, b):
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    inv = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    X, Y, Z = inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bb = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return "#%02x%02x%02x" % (_gam(max(0, r)), _gam(max(0, g)), _gam(max(0, bb)))


def shade(hex_, k):
    """Оттенок k (0 — сам цвет): светлее при k>0, темнее при k<0."""
    if not k: return norm_hex(hex_)
    L, a, b = hex2lab(hex_)
    return lab2hex(min(97.0, max(3.0, L + k * STEP_L)), a, b)


# ---------------------------------------------------------------- библиотека
def path():
    return os.path.join(_cfg.get("data") or os.path.join(HERE, "data"), "dyes.json")


def _seed():
    try: d = json.load(open(SEED, encoding="utf-8"))["dyes"]
    except Exception: d = []
    return [{"id": x["id"], "az": x["az"], "ru": x.get("ru", ""), "hex": norm_hex(x["hex"]), "scan": norm_hex(x["hex"])} for x in d]


def load():
    with _lock:
        try:
            d = json.load(open(path(), encoding="utf-8"))
            lst = d["dyes"] if isinstance(d, dict) else d
        except Exception:
            lst = _seed()
        out = []
        for x in lst:
            h = norm_hex(x.get("hex"))
            if not (x.get("id") and h): continue
            d = {"id": x["id"], "az": x.get("az") or x["id"], "ru": x.get("ru", ""), "hex": h, "scan": norm_hex(x.get("scan")) or h}
            if x.get("note"): d["note"] = x["note"]
            out.append(d)
        return out


def save(lst):
    with _lock:
        p = path(); os.makedirs(os.path.dirname(p), exist_ok=True)
        doc = {"_doc": "Единая палитра красителей студии (v11.1). hex — как краситель показывается везде; scan — как он выглядит в выцветшей печати "
                       "книги (по нему автооцифровка узнаёт краситель). id не менять. Правится на экране «Красители».", "dyes": lst}
        tmp = f"{p}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(doc, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)


def lib():
    return {d["id"]: d for d in load()}


_TR = str.maketrans({"ə": "e", "ı": "i", "ö": "o", "ü": "u", "ğ": "g", "ş": "s", "ç": "c", "Ə": "e", "I": "i", "İ": "i"})


def new_id(az, taken):
    base = re.sub(r"[^a-z0-9]+", "_", str(az or "").translate(_TR).lower()).strip("_") or "dye"
    i, cand = 1, base
    while cand in taken: i += 1; cand = f"{base}_{i}"
    return cand


# ---------------------------------------------------------------- палитра рисунка
def resolve(work, L=None):
    """Пересчитать связанные цвета палитры рисунка по библиотеке. Возвращает True, если палитра или подписи изменились."""
    pal = list(work.get("palette") or [])
    if not pal: return False
    n = len(pal); L = L if L is not None else lib()
    dy = (list(work.get("palette_dyes") or []) + [None] * n)[:n]
    ln = [bool(x) for x in (list(work.get("palette_link") or []) + [False] * n)[:n]]
    sc = (list(work.get("palette_scan") or []) + [None] * n)[:n]
    names = (list(work.get("palette_names") or []) + [""] * n)[:n]
    ru = work.get("palette_names_ru")
    ru = (list(ru) + [""] * n)[:n] if isinstance(ru, list) else None
    groups = {}
    for i in range(n):
        if ln[i] and dy[i] in L: groups.setdefault(dy[i], []).append(i)
        elif ln[i]: ln[i] = False                                   # краситель удалён из библиотеки — цвет остаётся как был
    new = list(pal)
    for d, idx in groups.items():
        base = L[d]["hex"]
        if len(idx) == 1:
            new[idx[0]] = base
        else:
            key = lambda i: (hex2lab(sc[i])[0] if _hex_rgb(sc[i]) else 50.0, i)
            order = sorted(idx, key=key); mid = (len(order) - 1) // 2
            for pos, i in enumerate(order): new[i] = shade(base, pos - mid)
        for k, i in enumerate(idx):                                 # подписи: «zoğalı», «zoğalı 2» — по порядку цветов
            names[i] = L[d]["az"] if k == 0 else f"{L[d]['az']} {k + 1}"
            if ru is not None: ru[i] = L[d]["ru"] if k == 0 else f"{L[d]['ru']} {k + 1}"
    changed = new != pal or names != (work.get("palette_names") or [])[:n] or ln != list(work.get("palette_link") or [])[:n]
    work["palette"] = new; work["palette_dyes"] = dy; work["palette_link"] = ln; work["palette_scan"] = sc; work["palette_names"] = names
    if ru is not None: work["palette_names_ru"] = ru
    return changed


def migrate_work(work):
    """Один раз на рисунок: замер со скана — в palette_scan; краситель, который подтвердил человек (рисунок принят или палитру правили),
    становится связанным. Предложения автомата в непринятых рисунках остаются предложениями (видны знаком «?»)."""
    if work.get("palette_v") == 1: return False
    pal = list(work.get("palette") or []); n = len(pal)
    dy = (list(work.get("palette_dyes") or []) + [None] * n)[:n]
    if not any(dy): return False                                  # ч/б схема без красителей — не трогаем (файл не переписывается)
    work["palette_v"] = 1
    if not work.get("palette_scan"):
        work["palette_scan"] = list(pal) if (work.get("color_mode") or (work.get("auto") or {}).get("mode") == "color") else [None] * n
    if "palette_link" not in work:
        ok = bool(work.get("done") or work.get("palette_touched"))
        work["palette_link"] = [bool(d) and ok for d in dy]
    resolve(work)
    return True


def _usage(work_dir):
    """Сколько рисунков связаны с каждым красителем и сколько предложены автоматом (без связи)."""
    use, sug, figs = {}, {}, {}
    for fn in os.listdir(work_dir):
        if not fn.endswith(".json"): continue
        try: w = json.load(open(os.path.join(work_dir, fn), encoding="utf-8"))
        except Exception: continue
        dy = w.get("palette_dyes") or []; ln = w.get("palette_link") or []
        for d in {x for x in dy if x}:
            linked = any(i < len(ln) and ln[i] for i, x in enumerate(dy) if x == d)
            tgt = use if linked else sug; tgt[d] = tgt.get(d, 0) + 1
            figs.setdefault(d, {})[fn[:-5]] = linked
    return use, sug, figs


def recolor(dye_ids, who=""):
    """Перекрасить все рисунки, где связаны эти красители, и перерисовать их выгрузки (в фоне)."""
    wd, render = _cfg["work"], _cfg["render"]
    L = lib(); dye_ids = set(dye_ids)
    todo = []
    for fn in sorted(os.listdir(wd)):
        if not fn.endswith(".json"): continue
        try: w = json.load(open(os.path.join(wd, fn), encoding="utf-8"))
        except Exception: continue
        if any(d in dye_ids for d in (w.get("palette_dyes") or [])): todo.append(fn)
    state.update(running=True, done=0, total=len(todo), dye=", ".join(sorted(dye_ids)))
    n = 0
    try:
        for fn in todo:
            p = os.path.join(wd, fn)
            with _lock:
                w = _cfg["jload"](p, {}) or {}
                if resolve(w, L):
                    _cfg["jsave"](p, w); n += 1
                    if w.get("matrix") and w.get("palette"):
                        try: render(fn[:-5], w["matrix"], w["palette"], w.get("transparent_bg", True), w.get("rapport"))
                        except Exception as e: print("dyes recolor", fn, e)
            state["done"] += 1
    finally:
        state.update(running=False, last=f"{time.strftime('%H:%M')}: перекрашено рисунков {n} ({state['dye']})")
        try:
            import events
            if n: events.add(f"Краситель изменён: {state['dye']}", f"перекрашено рисунков {n}, выгрузки перерисованы", link={"to": "dyes"})
        except Exception: pass
    return n


def migrate_all():
    wd, render = _cfg["work"], _cfg["render"]
    if not os.path.exists(path()): save(load())                    # первый запуск: библиотека переезжает в data/ (её правят люди)
    L = lib(); n = r = 0
    for fn in sorted(os.listdir(wd)):
        if not fn.endswith(".json"): continue
        p = os.path.join(wd, fn)
        with _lock:
            w = _cfg["jload"](p, {}) or {}
            if not w or w.get("palette_v") == 1: continue
            before = list(w.get("palette") or [])
            if migrate_work(w):
                _cfg["jsave"](p, w); n += 1
                if w.get("palette") != before and w.get("matrix"):
                    try: render(fn[:-5], w["matrix"], w["palette"], w.get("transparent_bg", True), w.get("rapport")); r += 1
                    except Exception as e: print("dyes migrate", fn, e)
    if n: print(f"dyes: палитры {n} рисунков переведены на ссылки на красители, перекрашено {r}")
    state["last"] = state["last"] or (f"переход на единую палитру: перекрашено рисунков {r}" if r else "")


def install(app, data_dir, work_dir, render, jload, jsave, on_change=None):
    from fastapi import Body, Query
    from fastapi.responses import JSONResponse
    _cfg.update(data=data_dir, work=work_dir, render=render, jload=jload, jsave=jsave, on_change=on_change)

    @app.on_event("startup")
    def _dyes_start():
        threading.Thread(target=migrate_all, daemon=True).start()

    @app.get("/api/dyes")
    def api_dyes(usage: int = Query(0)):
        out = {"dyes": load(), "recolor": state}
        if usage:
            use, sug, figs = _usage(work_dir)
            for d in out["dyes"]:
                d["used"] = use.get(d["id"], 0); d["suggested"] = sug.get(d["id"], 0)
        return out

    @app.get("/api/dyes/{did}/figures")
    def api_dye_figs(did: str):
        _, _, figs = _usage(work_dir)
        f = figs.get(did, {})
        return {"id": did, "figures": [{"id": k, "linked": v} for k, v in sorted(f.items(), key=lambda kv: (not kv[1], kv[0]))]}

    @app.post("/api/dyes")
    def api_dyes_save(body: dict = Body(...)):
        """Добавить краситель или поменять его название / цвет. Новый цвет сразу расходится по всем связанным рисункам (в фоне)."""
        with _lock:
            lst = load(); by = {d["id"]: d for d in lst}
            did = str(body.get("id") or "").strip()
            if did and did not in by: return JSONResponse({"error": "нет такого красителя"}, 404)
            if not did:
                az = str(body.get("az") or "").strip()
                if not az: return JSONResponse({"error": "нужно название (az)"}, 400)
                h = norm_hex(body.get("hex"))
                if not h: return JSONResponse({"error": "цвет в виде #rrggbb"}, 400)
                d = {"id": new_id(az, set(by)), "az": az, "ru": str(body.get("ru") or "").strip(), "hex": h, "scan": norm_hex(body.get("scan")) or h}
                lst.append(d); save(lst)
                if on_change: on_change()
                return {"ok": True, "dye": d, "recolor": 0}
            d = by[did]; old = d["hex"]
            for k in ("az", "ru"):
                if k in body and str(body[k]).strip(): d[k] = str(body[k]).strip()
            for k in ("hex", "scan"):
                if k in body:
                    h = norm_hex(body[k])
                    if not h: return JSONResponse({"error": "цвет в виде #rrggbb"}, 400)
                    d[k] = h
            save(lst)
        if on_change: on_change()
        if state["running"]: return {"ok": True, "dye": d, "recolor": "busy"}
        if d["hex"] != old or "az" in body or "ru" in body:
            threading.Thread(target=recolor, args=([did],), daemon=True).start()
            return {"ok": True, "dye": d, "recolor": "started"}
        return {"ok": True, "dye": d, "recolor": 0}

    @app.post("/api/dyes/resolve")
    def api_dyes_resolve(body: dict = Body(...)):
        """Палитра рисунка по библиотеке — для проверки (тот же расчёт, что в редакторе)."""
        w = {k: body.get(k) for k in ("palette", "palette_dyes", "palette_link", "palette_scan", "palette_names")}
        resolve(w)
        return {k: w[k] for k in ("palette", "palette_names", "palette_link")}
