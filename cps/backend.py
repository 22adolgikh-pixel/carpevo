# backend.py — Carpet Pattern Studio v3
# Запуск:  pip install fastapi uvicorn python-multipart opencv-python-headless numpy
#          uvicorn backend:app --port 8000        → http://localhost:8000
# Папку со сканами можно указать через переменную окружения, например Google Drive:
#          CPS_SCANS="$HOME/Library/CloudStorage/GoogleDrive-.../My Drive/scans" uvicorn backend:app --port 8000
import os, io, csv, json, re, time
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Body, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from az_names import az_draft, site_suggest
import autogrid
import autocolor
import rapport
import risk
import learn
import hard
import dupes
import threading
import cloud
import backup
import restore
try:
    import pymupdf as fitz  # PyMuPDF (new import name; falls back to legacy)
except Exception:
    try:
        import fitz
    except Exception:
        fitz = None

HERE = os.path.dirname(os.path.abspath(__file__))
SCANS = os.path.abspath(os.environ.get("CPS_SCANS", os.path.join(HERE, "scans")))
DATA = os.path.join(HERE, "data")
D_THUMBS = os.path.join(DATA, "thumbs")
D_SHEETS, D_CROPS, D_WORK, D_OUT = (os.path.join(DATA, d) for d in ("sheets", "crops", "work", "out"))
for d in (SCANS, DATA, D_SHEETS, D_CROPS, D_WORK, D_OUT): os.makedirs(d, exist_ok=True)
LEGEND = os.path.join(DATA, "legend.csv")
SITE_INDEX = os.path.join(DATA, "site_index.json")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
CROP_PAD = 0.08          # запас вокруг рамки рисунка (доля), чтобы было куда тянуть углы
META_KEYS = ("section", "name_az", "name", "translation", "carpet", "type", "note")
DIFFICULTY = ("simple", "medium", "complex", "ultra")   # простой / средний / сложный / ультра
OUT_SUFFIXES = (".png", "_x12.png", "_grid.png", "_bg.png", ".svg", "_unit.png")   # что лежит в data/out на каждый рисунок

app = FastAPI(title="Carpet Pattern Studio")
cloud.install(app)    # пароль для входа по ссылке (share.command), если задан CPS_PASSWORD

# ---------------- утилиты ----------------
def _safe_rel(rel):
    p = os.path.abspath(os.path.join(SCANS, rel))
    if not p.startswith(SCANS + os.sep) and p != SCANS: raise ValueError("bad path")
    return p

def _sheet_key(rel): return re.sub(r"[^\w\-]+", "_", rel)

def _fid_ok(fid): return re.fullmatch(r"[\w\-]+", fid) is not None

def imread_any(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(path, np.uint8)          # работает с кириллицей в пути
    return cv2.imdecode(buf, flags)

def imwrite_any(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok: buf.tofile(path)
    return ok

def jload(p, default=None):
    try: return json.load(open(p, encoding="utf-8"))
    except Exception: return default

def jsave(p, obj):
    """Атомарная запись: пишем во временный файл и подменяем — оборванная запись (перезапуск, параллельное чтение) не оставит битый json."""
    tmp = f"{p}.tmp{os.getpid()}_{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)

# ---------------- легенда (таблица из книги) ----------------
def legend_rows():
    if not os.path.exists(LEGEND): return []
    with open(LEGEND, encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def legend_lookup(table, fig):
    for r in legend_rows():
        if str(r.get("table")).strip() == str(table) and str(r.get("fig")).strip() == str(fig): return r
    return None

def meta_from_legend(table, fig):
    lg = legend_lookup(table, fig) or {}
    m = {k: (lg.get(k) or "").strip() for k in META_KEYS}
    if m["name"] == "?": m["name"] = ""
    if not m["name_az"] and m["name"]:
        m["name_az"] = az_draft(m["name"]); m["name_az_auto"] = True   # черновик из кириллицы книги — проверить
    return m

def with_az(meta):
    """Дополняет meta черновиком name_az, если его ещё нет (не сохраняет на диск)."""
    m = dict(meta or {})
    if not (m.get("name_az") or "").strip() and (m.get("name") or "").strip() not in ("", "?"):
        m["name_az"] = az_draft(m["name"]); m["name_az_auto"] = True
    return m

def norm_diff(v): return v if v in DIFFICULTY else None

# ---------------- поиск рамок рисунков на листе ----------------
def segment_sheet(g):
    """Ищет прямоугольные поля миллиметровки на белом листе. Перебирает параметры и
    берёт вариант, где больше всего «чистых» прямоугольников (слипшиеся рамки отбрасываются)."""
    H, W = g.shape
    min_area = (W * H) * 0.004
    best, best_score = [], -1e9
    for thr in (244, 242, 238, 234, 228):
        for k in (5, 9):
            b = cv2.blur(g, (k, k))
            m = (b < thr).astype(np.uint8) * 255
            ko = max(9, int(min(W, H) * 0.018)) | 1
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes, score = [], 0.0
            for c in cs:
                x, y, w, h = cv2.boundingRect(c)
                if w * h < min_area or w > W * 0.97 or h > H * 0.97: continue
                fill = cv2.contourArea(c) / float(w * h)
                if fill >= 0.88: boxes.append([x, y, w, h]); score += 1
                else: score -= 1.5
            if score > best_score: best, best_score = boxes, score
    # порядок чтения: ряды — по вертикальному перекрытию рамок (большая и маленькая рамка в одном ряду
    # не разъезжаются), внутри ряда — слева направо
    n = len(best); parent = list(range(n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, c = best[i], best[j]
            ov = min(a[1] + a[3], c[1] + c[3]) - max(a[1], c[1])
            if ov > 0.4 * min(a[3], c[3]): parent[find(i)] = find(j)
    groups = {}
    for i, b in enumerate(best): groups.setdefault(find(i), []).append(b)
    rows = sorted(groups.values(), key=lambda r: sum(b[1] + b[3] / 2 for b in r) / len(r))
    ordered = [b for r in rows for b in sorted(r, key=lambda b: b[0])]
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h), "fig": i + 1} for i, (x, y, w, h) in enumerate(ordered)]

def fig_id(table, fig, sheet_key):
    t = str(table).strip() if table not in (None, "") else ""
    return (f"t{int(t):02d}_f{int(fig):02d}" if t.isdigit() else f"{sheet_key}_f{int(fig):02d}")

def _guess_table(rel):
    """Номер таблицы по имени файла листа (как guessTable в интерфейсе)."""
    rel = rel.replace("\\", "/")
    try: pt = json.load(open(os.path.join(HERE, "plate_tables.json"), encoding="utf-8"))
    except Exception: pt = {}
    if rel in pt: return pt[rel]
    kp = re.search(r"kerimov_vol1_patterns/page_(\d+)\.", rel)
    if kp and int(kp.group(1)) >= 133: return int(kp.group(1)) - 119
    m = re.search(r"(?:t|табл|tabl|table)[ ._-]?(\d{1,3})", rel, re.I)
    return int(m.group(1)) if m else ""

# ---------------- API: листы ----------------
@app.get("/api/config")
def api_config():
    return {"scans": SCANS, "data": DATA}

@app.get("/api/sheets")
def api_sheets():
    out = []
    for root, _, files in os.walk(SCANS):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in IMG_EXT: continue
            rel = os.path.relpath(os.path.join(root, fn), SCANS)
            meta = jload(os.path.join(D_SHEETS, _sheet_key(rel) + ".json"), {}) or {}
            figs = meta.get("figures", [])
            works = [jload(os.path.join(D_WORK, f + ".json"), {}) or {} for f in figs]
            done = sum(1 for w in works if w.get("done"))
            diff = {}
            for w in works:
                d = norm_diff(w.get("difficulty"))
                if d: diff[d] = diff.get(d, 0) + 1
            parts = rel.replace("\\", "/").split("/")
            src = parts[0] if len(parts) > 1 else "(без папки)"
            sub = parts[1] if len(parts) > 2 else ""          # второй уровень папок (глубже — внутри него)
            out.append({"path": rel, "name": fn, "source": src, "sub": sub, "table": meta.get("table"), "n": len(figs), "done": done, "diff": diff, "no_figures": bool(meta.get("no_figures"))})
    out.sort(key=lambda s: s["path"])
    return out

@app.get("/api/sheet_thumb")
def api_sheet_thumb(path: str = Query(...), w: int = Query(160)):
    """Маленькое превью скана для списка листов (кэш в data/thumbs)."""
    try: p = _safe_rel(path)
    except ValueError: return JSONResponse({"error": "bad path"}, 400)
    if not os.path.exists(p): return JSONResponse({"error": "nf"}, 404)
    w = max(60, min(int(w), 400))
    td = os.path.join(DATA, "thumbs"); os.makedirs(td, exist_ok=True)
    tp = os.path.join(td, f"{_sheet_key(path)}_{w}_{int(os.path.getmtime(p))}.jpg")
    if not os.path.exists(tp):
        img = imread_any(p, cv2.IMREAD_REDUCED_COLOR_4 if p.lower().endswith((".jpg", ".jpeg")) else cv2.IMREAD_COLOR)
        if img is None: return JSONResponse({"error": "decode"}, 415)
        h = max(1, round(img.shape[0] * w / img.shape[1]))
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok: buf.tofile(tp)
    return FileResponse(tp, headers={"Cache-Control": "public, max-age=86400"})

@app.get("/api/sheet_image")
def api_sheet_image(path: str = Query(...)):
    try: p = _safe_rel(path)
    except ValueError: return JSONResponse({"error": "bad path"}, 400)
    return FileResponse(p) if os.path.exists(p) else JSONResponse({"error": "nf"}, 404)

@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        name = os.path.basename(f.filename or "")
        if os.path.splitext(name)[1].lower() not in IMG_EXT: continue
        with open(os.path.join(SCANS, name), "wb") as fh: fh.write(await f.read())
        saved.append(name)
    return {"saved": saved}

@app.get("/api/sheet")
def api_sheet(path: str = Query(...)):
    sh = jload(os.path.join(D_SHEETS, _sheet_key(path) + ".json"), {"path": path, "table": None, "boxes": []})
    for b in sh.get("boxes", []):      # сложность могли поменять уже на экране пикселей
        w = jload(os.path.join(D_WORK, (b.get("id") or "") + ".json"), {}) or {}
        if norm_diff(w.get("difficulty")): b["difficulty"] = w["difficulty"]
    return sh

@app.post("/api/segment")
def api_segment(body: dict = Body(...)):
    img = imread_any(_safe_rel(body["path"]), cv2.IMREAD_GRAYSCALE)
    if img is None: return JSONResponse({"error": "не читается изображение"}, 400)
    return {"boxes": segment_sheet(img), "w": img.shape[1], "h": img.shape[0]}

@app.post("/api/sheet_save")
def api_sheet_save(body: dict = Body(...)):
    """Сохраняет разметку листа и нарезает рисунки в data/crops/<id>.png (с запасом по краям)."""
    rel = body["path"]; key = _sheet_key(rel)
    col = imread_any(_safe_rel(rel))
    if col is None: return JSONResponse({"error": "не читается изображение"}, 400)
    H, W = col.shape[:2]
    table = body.get("table")
    figs, boxes = [], []
    for b in body.get("boxes", []):
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        t = b.get("table") or table
        fid = fig_id(t, b.get("fig", len(figs) + 1), key)
        px, py = int(w * CROP_PAD), int(h * CROP_PAD)
        x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
        imwrite_any(os.path.join(D_CROPS, fid + ".png"), col[y0:y1, x0:x1])
        wp = os.path.join(D_WORK, fid + ".json")
        work = jload(wp, {}) or {}
        # рамка рисунка внутри кропа — стартовые углы для геометрии
        work.update({"id": fid, "sheet": rel, "table": t, "fig": b.get("fig"),
                     "crop_origin": [x0, y0], "frame": [x - x0, y - y0, w, h]})
        if "difficulty" in b: work["difficulty"] = norm_diff(b.get("difficulty"))
        if b.get("frame_changed") or "quad" not in work:
            work["quad"] = [[x - x0, y - y0], [x - x0 + w, y - y0], [x - x0 + w, y - y0 + h], [x - x0, y - y0 + h]]
            work.pop("grid", None)
        if "meta" not in work:
            work["meta"] = meta_from_legend(t, b.get("fig"))
        jsave(wp, work)
        boxes.append({"x": x, "y": y, "w": w, "h": h, "fig": b.get("fig"), "table": b.get("table") or None, "id": fid,
                      "difficulty": norm_diff(b.get("difficulty"))})
        figs.append(fid)
    jsave(os.path.join(D_SHEETS, key + ".json"), {"path": rel, "table": table, "boxes": boxes, "figures": figs})
    return {"figures": figs}

@app.post("/api/sheet_empty")
def api_sheet_empty(body: dict = Body(...)):
    """Отмечает лист как «не содержит искомых изображений» (например, текстовая
    страница, обложка, реклама) — чтобы не пытаться на нём искать рамки рисунков."""
    rel = body["path"]; key = _sheet_key(rel)
    p = os.path.join(D_SHEETS, key + ".json")
    sh = jload(p, {"path": rel, "table": None, "boxes": [], "figures": []}) or {}
    sh["no_figures"] = bool(body.get("value", True))
    jsave(p, sh)
    return {"ok": True, "no_figures": sh["no_figures"]}

# ---------------- API: PDF-книги (раскладка страниц на сканы) ----------------
PDF_MAX_PAGES_PER_CALL = 80

def _find_pdfs():
    found = []
    for root, _, files in os.walk(SCANS):
        for fn in sorted(files):
            if fn.lower().endswith(".pdf"):
                found.append(os.path.relpath(os.path.join(root, fn), SCANS))
    return sorted(found)

@app.get("/api/pdfs")
def api_pdfs():
    if fitz is None:
        return JSONResponse({"error": "на сервере не установлен pymupdf (pip install pymupdf)"}, 500)
    out = []
    for rel in _find_pdfs():
        p = _safe_rel(rel)
        try:
            doc = fitz.open(p)
            npages = doc.page_count
            doc.close()
        except Exception as e:
            npages = None
        stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(os.path.basename(rel))[0])
        outdir = os.path.join(SCANS, stem)
        done = 0
        if os.path.isdir(outdir):
            done = len([f for f in os.listdir(outdir) if f.lower().endswith(".png")])
        out.append({"path": rel, "name": os.path.basename(rel), "pages": npages,
                    "size_mb": round(os.path.getsize(p) / 1048576, 1), "out_dir": stem, "extracted": done})
    return out

@app.post("/api/split_pdf")
def api_split_pdf(body: dict = Body(...)):
    """Рендерит диапазон страниц PDF в PNG-файлы в scans/<stem>/page_NNNN.png,
    чтобы дальше они появились в списке листов как обычные сканы."""
    if fitz is None:
        return JSONResponse({"error": "на сервере не установлен pymupdf (pip install pymupdf)"}, 500)
    rel = body.get("path")
    if not rel: return JSONResponse({"error": "не указан path"}, 400)
    try: p = _safe_rel(rel)
    except ValueError: return JSONResponse({"error": "bad path"}, 400)
    if not os.path.exists(p): return JSONResponse({"error": "nf"}, 404)
    page_from = max(1, int(body.get("from", 1)))
    page_to = int(body.get("to", page_from))
    zoom = float(body.get("zoom", 3.0))  # 3.0 ≈ 216 dpi
    if page_to < page_from: page_from, page_to = page_to, page_from
    if page_to - page_from + 1 > PDF_MAX_PAGES_PER_CALL:
        page_to = page_from + PDF_MAX_PAGES_PER_CALL - 1
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(os.path.basename(rel))[0])
    outdir = os.path.join(SCANS, stem)
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(p)
    n = doc.page_count
    page_from = max(1, page_from); page_to = min(n, page_to)
    mat = fitz.Matrix(zoom, zoom)
    saved = []
    for i in range(page_from - 1, page_to):
        pg = doc.load_page(i)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        fn = f"page_{i+1:04d}.png"
        pix.save(os.path.join(outdir, fn))
        saved.append(f"{stem}/{fn}")
    doc.close()
    return {"ok": True, "saved": saved, "pages_total": n, "from": page_from, "to": page_to}

# ---------------- API: рисунки ----------------
@app.post("/api/figure_upload")
async def api_figure_upload(files: list[UploadFile] = File(...)):
    """Загрузить уже вырезанный рисунок напрямую (без листа)."""
    saved = []
    for f in files:
        stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(os.path.basename(f.filename or "fig"))[0])
        img = cv2.imdecode(np.frombuffer(await f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        imwrite_any(os.path.join(D_CROPS, stem + ".png"), img)
        h, w = img.shape[:2]
        wp = os.path.join(D_WORK, stem + ".json")
        work = jload(wp, {}) or {}
        m = re.match(r"t(\d+)_f(\d+)", stem)
        work.update({"id": stem, "table": int(m.group(1)) if m else None, "fig": int(m.group(2)) if m else None,
                     "quad": work.get("quad") or [[0, 0], [w, 0], [w, h], [0, h]]})
        if "meta" not in work:
            work["meta"] = meta_from_legend(work["table"], work["fig"]) if m else {k: "" for k in META_KEYS}
        jsave(wp, work); saved.append(stem)
    return {"saved": saved}

def _is_border(w):
    """кайма / бордюр: раздел в подписи, найденный раппорт или вытянутая полоса"""
    sect = ((w.get("meta") or {}).get("section") or "").lower()
    if any(k in sect for k in ("кайм", "бордюр", "haşiy", "border")): return True
    if (w.get("rapport") or {}).get("found"): return True
    m = w.get("matrix")
    return bool(m and max(m["w"], m["h"]) >= 1.9 * max(1, min(m["w"], m["h"])))

def _hard_left(w):
    """Сколько трудных кусков осталось проверить (по кэшу; None — ещё не считалось или сетка менялась)."""
    if not w.get("hard"): return None
    try:
        r = hard.remaining(w)
        return None if r is None else len(r)
    except Exception:
        return None


@app.get("/api/figures")
def api_figures():
    out = []
    for fn in sorted(os.listdir(D_WORK)):
        if not fn.endswith(".json"): continue
        w = jload(os.path.join(D_WORK, fn), {}) or {}
        m = with_az(w.get("meta", {}))
        out.append({"id": w.get("id", fn[:-5]), "table": w.get("table"), "fig": w.get("fig"), "sheet": w.get("sheet"),
                    "name_az": m.get("name_az", ""), "name_az_auto": bool(m.get("name_az_auto")),
                    "name": m.get("name", ""), "translation": m.get("translation", ""), "carpet": m.get("carpet", ""),
                    "site_id": m.get("site_id", ""), "done": bool(w.get("done")), "difficulty": norm_diff(w.get("difficulty")),
                    "review": bool(w.get("review")),
                    "auto": bool(w.get("auto")) and not (w.get("auto") or {}).get("reviewed"),
                    "auto_conf": (w.get("auto") or {}).get("confidence"),
                    "risk": (w.get("auto") or {}).get("risk"),
                    "diff_est": (w.get("auto") or {}).get("difficulty_est"),
                    "sym": ((w.get("auto") or {}).get("symmetry") or {}).get("kind"),
                    "n_sus": ((w.get("auto") or {}).get("symmetry") or {}).get("n_suspects"),
                    "size": [w["matrix"]["w"], w["matrix"]["h"]] if w.get("matrix") else None,
                    "colors": len(set("".join(w["matrix"]["rows"])) - {"."}) if w.get("matrix") else None,
                    "color": bool(w.get("color_mode")), "photo": w.get("kind") == "photo",
                    "border": _is_border(w), "rapport": bool((w.get("rapport") or {}).get("found")),
                    "hard_left": _hard_left(w),
                    "t": os.path.getmtime(os.path.join(D_WORK, fn))})
    def k(f):
        try: return (0, int(f["table"] or 0), int(f["fig"] or 0), f["id"])
        except Exception: return (1, 0, 0, f["id"])
    return sorted(out, key=k)

@app.get("/api/orig_aligned/{fid}")
def api_orig_aligned(fid: str, s: int = 12):
    """Оригинал, выпрямленный по сетке и обрезанный ровно по матрице — как «подложка» при пробеле: 1 узел = s px, совпадает с _bg.png."""
    wp = os.path.join(D_WORK, fid + ".json"); cp = os.path.join(D_CROPS, fid + ".png")
    if not _fid_ok(fid) or not os.path.exists(wp) or not os.path.exists(cp): return JSONResponse({"error": "nf"}, 404)
    w = jload(wp, {}) or {}
    if not (w.get("matrix") and w.get("quad") and w.get("grid")): return JSONResponse({"error": "no grid"}, 404)
    cache = os.path.join(D_THUMBS, f"orig_{fid}_{s}.png")
    if os.path.exists(cache) and os.path.getmtime(cache) >= max(os.path.getmtime(wp), os.path.getmtime(cp)):
        return FileResponse(cache, headers={"Cache-Control": "no-store"})
    img = imread_any(cp, cv2.IMREAD_COLOR)
    g, m = w["grid"], w["matrix"]; cols, rows = int(g["cols"]), int(g["rows"])
    Wd, Hd = cols * s, rows * s
    Hm = cv2.getPerspectiveTransform(np.array(w["quad"], np.float32), np.array([[0, 0], [Wd, 0], [Wd, Hd], [0, Hd]], np.float32))
    warp = cv2.warpPerspective(img, Hm, (Wd, Hd), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    ox, oy = (m.get("origin") or [0, 0])[:2]
    out = warp[max(0, oy * s):(oy + m["h"]) * s, max(0, ox * s):(ox + m["w"]) * s]
    os.makedirs(D_THUMBS, exist_ok=True); cv2.imwrite(cache, out)
    return FileResponse(cache, headers={"Cache-Control": "no-store"})


@app.get("/api/crop/{fid}")
def api_crop(fid: str):
    p = os.path.join(D_CROPS, fid + ".png")
    if not _fid_ok(fid) or not os.path.exists(p): return JSONResponse({"error": "nf"}, 404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})

@app.get("/api/work/{fid}")
def api_work_get(fid: str):
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    w = jload(os.path.join(D_WORK, fid + ".json"), {"id": fid})
    w["meta"] = with_az(w.get("meta", {}))
    return w

GRID_S = 12        # масштаб PNG ×12
GRID_THIN = (150, 198, 240, 110)    # RGBA: светло-голубая линия каждого узла
GRID_BOLD = (74, 146, 214, 220)     # каждые 10 узлов (от левого верхнего угла рисунка)
GRID_MID = (74, 146, 214, 255)      # маленькие метки середины на краях

def render_grid_layer(w, h, S=GRID_S):
    """Сетка отдельным слоем (PNG той же величины, что ×12): одинаковая у всех рисунков —
    тонкая светло-голубая линия на каждый узел, жирная каждые 10 от левого верхнего угла, треугольные метки середины."""
    g = np.zeros((h * S, w * S, 4), np.uint8)
    thin = GRID_THIN[2::-1] + GRID_THIN[3:]; bold = GRID_BOLD[2::-1] + GRID_BOLD[3:]; mid = GRID_MID[2::-1] + GRID_MID[3:]
    for i in range(w + 1):
        x = min(i * S, w * S - 1); g[:, x] = bold if i % 10 == 0 or i == w else thin
        if i % 10 == 0 or i == w: g[:, max(0, x - 1)] = bold
    for j in range(h + 1):
        y = min(j * S, h * S - 1); g[y, :] = bold if j % 10 == 0 or j == h else thin
        if j % 10 == 0 or j == h: g[max(0, y - 1), :] = bold
    cx, cy, k = w * S // 2, h * S // 2, max(3, S // 3)       # ▼▲◀▶ метки середины снаружи сетки не помещаются — рисуем по краям внутрь
    for d in range(k):
        g[d, max(0, cx - (k - d)):cx + (k - d)] = mid; g[h * S - 1 - d, max(0, cx - (k - d)):cx + (k - d)] = mid
        g[max(0, cy - (k - d)):cy + (k - d), d] = mid; g[max(0, cy - (k - d)):cy + (k - d), w * S - 1 - d] = mid
    return g

def render_outputs(fid, mat, palette, transparent_bg=True, rap=None):
    """Выгрузки для любых дальнейших задач (v6):
    .png — 1 узел = 1 px, фон прозрачный; _x12.png — то же ×12, без сетки; _grid.png — сетка отдельным слоем;
    _bg.png — ×12 с заливкой фона (для предпросмотра/печати); .svg — по слою на каждый цвет (<g id="color-N">),
    фон — отдельный слой (скрыт), сетка — отдельный скрытый слой. Источник правды — матрица в data/work."""
    w, h, rows = int(mat["w"]), int(mat["h"]), mat["rows"]
    pal = [tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for c in palette]
    K = np.zeros((h, w), np.int16)
    for j, row in enumerate(rows[:h]):
        for i, ch in enumerate(row[:w]): K[j, i] = 0 if ch == "." else int(ch, 36)
    rgba = np.zeros((h, w, 4), np.uint8)
    for k in np.unique(K):
        r, g, b = pal[k] if k < len(pal) else (255, 0, 255)
        rgba[K == k] = (b, g, r, 0 if k == 0 else 255)
    imwrite_any(os.path.join(D_OUT, fid + ".png"), rgba)
    S = GRID_S
    big = cv2.resize(rgba, (w * S, h * S), interpolation=cv2.INTER_NEAREST)
    imwrite_any(os.path.join(D_OUT, fid + "_x12.png"), big)          # чистый рисунок, без сетки
    imwrite_any(os.path.join(D_OUT, fid + "_grid.png"), render_grid_layer(w, h, S))
    bgc = pal[0] if pal else (255, 255, 255)
    solid = big.copy(); solid[big[..., 3] == 0] = (bgc[2], bgc[1], bgc[0], 255)
    imwrite_any(os.path.join(D_OUT, fid + "_bg.png"), solid)
    layers = []
    for k in sorted(int(x) for x in np.unique(K) if x != 0):
        rects = []
        for j in range(h):
            i = 0
            while i < w:
                if K[j, i] != k: i += 1; continue
                n = 1
                while i + n < w and K[j, i + n] == k: n += 1
                rects.append(f'<rect x="{i}" y="{j}" width="{n}" height="1"/>'); i += n
        col = palette[k] if k < len(palette) else "#f0f"
        layers.append(f'<g id="color-{k}" data-color="{col}" fill="{col}">' + "".join(rects) + "</g>")
    gl = "".join(f'<line x1="{i}" y1="0" x2="{i}" y2="{h}" stroke-width="{0.12 if i % 10 == 0 or i == w else 0.04}"/>' for i in range(w + 1)) + \
         "".join(f'<line x1="0" y1="{j}" x2="{w}" y2="{j}" stroke-width="{0.12 if j % 10 == 0 or j == h else 0.04}"/>' for j in range(h + 1))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w * 10}" height="{h * 10}" shape-rendering="crispEdges">'
           f'<g id="background" fill="{palette[0] if palette else "#fff"}" display="none"><rect width="{w}" height="{h}"/></g>'
           + "".join(layers) +
           f'<g id="grid" stroke="#4a92d6" stroke-opacity=".6" display="none">{gl}</g></svg>')
    open(os.path.join(D_OUT, fid + ".svg"), "w", encoding="utf-8").write(svg)
    up = os.path.join(D_OUT, fid + "_unit.png")
    if rap and rap.get("found") and rap.get("unit"):          # раппорт каймы: один повтор, ×12, фон залит
        U = rapport.matrix_to_array(rap["unit"])
        img = np.zeros(U.shape + (3,), np.uint8)
        for k in np.unique(U):
            r_, g_, b_ = pal[k] if k < len(pal) else (255, 0, 255); img[U == k] = (b_, g_, r_)
        imwrite_any(up, cv2.resize(img, (U.shape[1] * S, U.shape[0] * S), interpolation=cv2.INTER_NEAREST))
    elif os.path.exists(up):
        os.remove(up)

@app.post("/api/work/{fid}")
def api_work_save(fid: str, body: dict = Body(...)):
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    wp = os.path.join(D_WORK, fid + ".json")
    work = jload(wp, {}) or {}
    work.update(body); work["id"] = fid; work["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if "matrix" in body and isinstance(work.get("auto"), dict): work["auto"]["edited"] = True      # человек правил — перепрогон это не тронет
    if work.get("done") and isinstance(work.get("auto"), dict): work["auto"]["reviewed"] = True
    jsave(wp, work)
    if work.get("matrix") and work.get("palette"):
        if "matrix" in body:                                   # пиксели правили — раппорт каймы пересчитать
            work["rapport"] = rapport.for_work(work); jsave(wp, work)
        render_outputs(fid, work["matrix"], work["palette"], work.get("transparent_bg", True), work.get("rapport"))
    return {"ok": True, "saved_at": work["saved_at"], "rapport": work.get("rapport")}

@app.post("/api/rename/{fid}")
def api_rename(fid: str, body: dict = Body(...)):
    new = body.get("new_id", "")
    if not (_fid_ok(fid) and _fid_ok(new)): return JSONResponse({"error": "bad id"}, 400)
    if os.path.exists(os.path.join(D_WORK, new + ".json")): return JSONResponse({"error": "такой id уже есть"}, 409)
    for d, ext in ((D_WORK, ".json"), (D_CROPS, ".png")) + tuple((D_OUT, x) for x in OUT_SUFFIXES):
        p = os.path.join(d, fid + ext)
        if os.path.exists(p): os.rename(p, os.path.join(d, new + ext))
    w = jload(os.path.join(D_WORK, new + ".json"), {}); w["id"] = new; jsave(os.path.join(D_WORK, new + ".json"), w)
    return {"ok": True}

@app.delete("/api/work/{fid}")
def api_delete(fid: str):
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    for d, ext in ((D_WORK, ".json"), (D_CROPS, ".png")) + tuple((D_OUT, x) for x in OUT_SUFFIXES):
        p = os.path.join(d, fid + ext)
        if os.path.exists(p): os.remove(p)
    return {"ok": True}

@app.get("/out/{name}")
def api_out(name: str):
    p = os.path.join(D_OUT, os.path.basename(name))
    return FileResponse(p, headers={"Cache-Control": "no-store"}) if os.path.exists(p) else JSONResponse({"error": "nf"}, 404)

# ---------------- авто-оцифровка: сетка + пиксели ----------------
def _auto_mode(crop_bgr, frame, mode):
    if mode in ("bw", "color"): return mode
    return "color" if autocolor.is_colorful(crop_bgr, frame) else "bw"


def _difficulty_est(work):
    """Оценка сложности по числу закрашенных клеток и цветов (не заменяет ручную пометку difficulty).
    simple < 400 клеток, medium < 1500, complex < 5000, иначе ultra; много цветов (≥6) — на ступень выше."""
    m = work.get("matrix")
    if not m: return None
    ink = sum(c != "." for r in m["rows"] for c in r)
    lvl = 0 if ink < 400 else 1 if ink < 1500 else 2 if ink < 5000 else 3
    if len(set("".join(m["rows"])) - {"."}) >= 6 and lvl < 3: lvl += 1
    return DIFFICULTY[lvl]


def _apply_risk(work):
    """Симметрия + риск-скор → work['auto'] (подозрительные клетки в координатах матрицы)."""
    a = work.get("auto")
    if not isinstance(a, dict) or not work.get("matrix"): return
    try:
        sym = risk.symmetry(rapport.matrix_to_array(work["matrix"]))
    except Exception:
        sym = None
    if sym:
        a["symmetry"] = {"kind": sym["kind"], "score": sym["score"], "suspects": sym["suspects"][:200], "n_suspects": len(sym["suspects"])}
        if sym["kind"] and sym["suspects"]:
            a["flags"] = [f for f in a.get("flags", []) if not f.startswith("симметрия")] + [f"симметрия ({sym['kind']}): {len(sym['suspects'])} клеток нарушают — см. подсветку"]
    a["risk"] = risk.risk_score(a, sym)
    a["difficulty_est"] = _difficulty_est(work)


def run_auto(fid, force=False, dark_only=False, mode="auto", pitch_hint=None, ncolors=None, dyes=None, canon=False, cols=None, rows=None):
    """mode: auto (цвет определяется сам) / bw (ЧБ-тушь: контур + серая заливка) / color (цветная схема)."""
    wp = os.path.join(D_WORK, fid + ".json")
    work = jload(wp, {}) or {}
    if work.get("matrix") and not force:
        return {"id": fid, "skipped": "уже есть пиксели"}
    crop_c = imread_any(os.path.join(D_CROPS, fid + ".png"), cv2.IMREAD_COLOR)
    if crop_c is None: return {"id": fid, "error": "нет кропа"}
    frame = work.get("frame") or [0, 0, crop_c.shape[1], crop_c.shape[0]]
    mode = "color" if cols else _auto_mode(crop_c, frame, mode)     # cols — режим «фото»: сетку задаёт человек
    try:
        if mode == "color":
            G, C, M, info = autocolor.auto_figure_color(crop_c, frame, pitch_hint=pitch_hint, ncolors=ncolors, dyes=dyes or None, canon=bool(canon), force_cols=cols, force_rows=rows)
        else:
            G, C, M, info = autogrid.auto_figure(cv2.cvtColor(crop_c, cv2.COLOR_BGR2GRAY), frame, dark_only=dark_only)
    except Exception as e:
        return {"id": fid, "error": f"сетка не найдена: {e}"}
    if mode == "color" and min(G.get("R", [1, 1])) < autocolor.PHOTO_R:
        # сетки нет: это фото ковра (напр. табл. 204 «Джек»), а не схема — пиксели не делаем, помечаем
        work["kind"] = "photo"; work["auto"] = {"mode": "color", "photo": True, "R": G.get("R"), "reviewed": False,
                                                "flags": ["похоже на фото ковра, а не на схему — оцифровка не делалась"]}
        work["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S"); jsave(wp, work)
        return {"id": fid, "photo": True, "R": G.get("R"), "mode": mode}
    upd = autocolor.to_work_color(G, M, info) if mode == "color" else autogrid.to_work(G, M, info)
    if mode != "color" and work.get("palette_touched") and work.get("palette"):   # свою палитру не трогаем (ЧБ), только дополняем
        pal, names = list(work["palette"]), list(work.get("palette_names") or [])
        while len(pal) < len(upd["palette"]): pal.append(upd["palette"][len(pal)]); names.append(upd["palette_names"][len(names)])
        upd["palette"], upd["palette_names"] = pal, names; upd.pop("palette_touched")
    if mode != "color":
        for k in ("color_mode", "palette_names_ru"): work.pop(k, None)
    work.pop("kind", None)
    if cols: work["source_kind"] = "photo"
    work.update(upd); work["id"] = fid; work["done"] = False
    work["adjust"] = {"brightness": 0, "contrast": 1, "gamma": 1, "invert": False}
    work["rapport"] = rapport.for_work(work)
    rp = work["rapport"]
    if rp and rp.get("found") and not rp.get("check") and rp.get("agreement", 0) >= 0.9:      # кайма = повторяющийся узор: исправляем все повторы по большинству
        K2, nch = rapport.tile(rapport.matrix_to_array(work["matrix"]), rp)
        if nch:
            work["matrix"]["rows"] = rapport._rows_to_str(K2)
            work["rapport"] = rapport.for_work(work)
            work["auto"]["flags"] = list(work["auto"].get("flags") or []) + [f"кайма: раппорт применён ко всей полосе (исправлено клеток: {nch})"]
    _apply_risk(work)
    work.pop("hard", None)
    if mode != "color":                                           # v10.13: трудные куски сразу (сетка новая — старые отметки недействительны)
        P = learn.load_model()
        if P is not None:
            try: hard.ensure(work, cv2.cvtColor(crop_c, cv2.COLOR_BGR2GRAY), P)
            except Exception: pass
    work["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jsave(wp, work)
    render_outputs(fid, work["matrix"], work["palette"], work.get("transparent_bg", True), work.get("rapport"))
    return {"id": fid, "risk": work["auto"].get("risk"), "size": [work["matrix"]["w"], work["matrix"]["h"]], "mode": mode,
            "pitch": [round(G["px"], 3), round(G["py"], 3)], "R": G.get("R"),
            "confidence": work["auto"]["confidence"], "flags": work["auto"]["flags"],
            "rapport": _rapport_brief(work.get("rapport"))}


def _rapport_brief(r):
    if not r: return None
    if not r.get("found"): return {"found": False}
    return {k: r[k] for k in ("found", "axis", "period", "match", "repeats", "check", "is_mirror", "is_glide")}


@app.post("/api/auto/{fid}")
def api_auto(fid: str, body: dict = Body(default={})):
    """Автоматически натянуть сетку и перевести рисунок в пиксели (черновик — проверить на экране «Пиксели»).
    v6: по умолчанию контур + серая заливка (замер на 134 готовых рисунках: 95.6% клеток верно против 87% у «только контура»);
    fill=false — только тёмный контур. v7: mode = auto | bw | color (цветные схемы, см. autocolor.py);
    ncolors — оставить столько цветов (если авто разделило один цвет на два)."""
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    b = body or {}
    hint = None
    w = jload(os.path.join(D_WORK, fid + ".json"), {}) or {}
    if w.get("sheet"):                                  # шаг, уже уверенно найденный на других рисунках листа
        hint = _sheet_pitch_hint(w["sheet"], exclude=fid)
    nc = b.get("ncolors")
    return run_auto(fid, bool(b.get("force", True)), dark_only=not bool(b.get("fill", True)),
                    mode=b.get("mode", "auto"), pitch_hint=hint, ncolors=int(nc) if nc else None,
                    dyes=b.get("dyes") or None, canon=b.get("canon", False),
                    cols=int(b["cols"]) if b.get("cols") else None, rows=int(b["rows"]) if b.get("rows") else None)


def _sheet_pitch_hint(sheet_rel, exclude=None):
    meta = jload(os.path.join(D_SHEETS, _sheet_key(sheet_rel) + ".json"), {}) or {}
    ps = []
    for f in meta.get("figures", []):
        if f == exclude: continue
        a = (jload(os.path.join(D_WORK, f + ".json"), {}) or {}).get("auto") or {}
        if a.get("mode") == "color" and a.get("R") and min(a["R"]) >= autocolor.GOOD_R and a.get("pitch"): ps.append(a["pitch"])
    if not ps: return None
    return (float(np.median([p[0] for p in ps])), float(np.median([p[1] for p in ps])))


@app.post("/api/auto_sheet")
def api_auto_sheet(body: dict = Body(...)):
    """Все рисунки листа. По умолчанию пропускает те, где пиксели уже есть (force — пересчитать всё).
    Цветные листы — в два прохода: сначала все рисунки сами по себе, затем неуверенные (узкие каймы)
    пересчитываются с шагом, найденным на уверенных рисунках того же листа."""
    meta = jload(os.path.join(D_SHEETS, _sheet_key(body["path"]) + ".json"), {}) or {}
    figs = meta.get("figures", [])
    if not figs: return JSONResponse({"error": "лист ещё не нарезан — сначала «Сохранить и нарезать»"}, 400)
    dark_only = not bool(body.get("fill", True)); mode = body.get("mode", "auto"); force = bool(body.get("force"))
    dyes = body.get("dyes") or None; canon = bool(body.get("canon"))
    res = [run_auto(f, force, dark_only=dark_only, mode=mode, dyes=dyes, canon=canon) for f in figs]
    good = [r for r in res if r.get("mode") == "color" and r.get("R") and min(r["R"]) >= autocolor.GOOD_R]
    if good:
        hint = (float(np.median([r["pitch"][0] for r in good])), float(np.median([r["pitch"][1] for r in good])))
        for i, r in enumerate(res):
            if r.get("mode") == "color" and r.get("R") and min(r["R"]) < autocolor.GOOD_R:
                r2 = run_auto(figs[i], True, mode="color", pitch_hint=hint, dyes=dyes, canon=canon)
                if r2.get("R") and (r.get("photo") or sum(r2["R"]) > sum(r["R"])):
                    res[i] = r2
                elif not r.get("photo"):
                    res[i] = run_auto(figs[i], True, mode="color", dyes=dyes, canon=canon)      # первый вариант был лучше — вернуть его
    return {"results": res}


@app.get("/api/model")
def api_model():
    p = learn.MODEL_PATH
    if not os.path.exists(p): return {"present": False}
    meta = (jload(p, {}) or {}).get("meta", {})
    return {"present": True, "trained": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p))), **meta}


MODEL_JOB = {"running": False, "started": None, "finished": None, "ok": None, "out": "", "err": ""}


def _model_train_run():
    import subprocess, sys
    J = MODEL_JOB; J.update(running=True, started=time.strftime("%Y-%m-%d %H:%M:%S"), finished=None, ok=None, out="", err="")
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "learn.py"), "train"], capture_output=True, text=True, timeout=3600,
                           env={**os.environ, "CPS_DATA": DATA})
        J.update(ok=r.returncode == 0, out=(r.stdout or "")[-1500:], err=(r.stderr or "")[-600:])
    except Exception as e:
        J.update(ok=False, err=str(e))
    finally:
        J.update(running=False, finished=time.strftime("%Y-%m-%d %H:%M:%S"))


@app.post("/api/model/train")
def api_model_train():
    """Обучить классификатор клеток на готовых ЧБ-рисунках — в фоне (минуты); ход виден в статус-баре."""
    if MODEL_JOB["running"]: return JSONResponse({"error": "уже обучается"}, 409)
    threading.Thread(target=_model_train_run, daemon=True).start()
    return {"started": True}


@app.get("/api/model/job")
def api_model_job():
    return MODEL_JOB


@app.get("/api/broken")
def api_broken(fix: int = 0):
    """Битые json в data/work и data/sheets (обрыв записи). fix=1 — убрать их в data/broken, чтобы не мешали обучению и спискам."""
    out = []
    for d in (D_WORK, D_SHEETS):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"): continue
            p = os.path.join(d, fn)
            try: json.load(open(p, encoding="utf-8"))
            except Exception as e:
                out.append({"file": os.path.relpath(p, DATA), "error": str(e)[:80]})
                if fix:
                    os.makedirs(os.path.join(DATA, "broken"), exist_ok=True); os.replace(p, os.path.join(DATA, "broken", fn))
    return {"broken": out, "fixed": bool(fix)}


@app.get("/api/lostgray")
def api_lostgray(fix: int = 0):
    """Принятые ч/б рисунки без серого, где автомат уверенно видит заливку (старый прогон без заливки её стирал).
    fix=1 — вернуть серое (контур и сетка человека сохраняются) и отправить рисунок обратно в проверку."""
    import lostgray
    out = []
    for fn in sorted(os.listdir(D_WORK)):
        if not fn.endswith(".json"): continue
        wp = os.path.join(D_WORK, fn); w = jload(wp, {}) or {}
        a = w.get("auto") or {}
        if not (w.get("matrix") and w.get("done") and a.get("reviewed") and not w.get("color_mode") and w.get("kind") != "photo" and w.get("quad") and w.get("grid")): continue
        if "2" in "".join(w["matrix"]["rows"]) or a.get("gray_restored"): continue
        cp = os.path.join(D_CROPS, fn[:-5] + ".png")
        g = cv2.imread(cp, 0) if os.path.exists(cp) else None
        if g is None: continue
        try: r = lostgray.propose(w, g)
        except Exception: continue
        if not r: continue
        out.append({"id": fn[:-5], "sheet": w.get("sheet"), "share": round(r[1], 3)})
        if fix:
            jsave(wp, lostgray.apply(w, r[0], r[1]))
    return {"count": len(out), "fixed": bool(fix), "items": out}


@app.post("/api/accept")
def api_accept(body: dict = Body(...)):
    """Массово принять авто-результаты (done + reviewed) — для очереди проверки."""
    n = 0
    for fid in body.get("ids", []):
        if not _fid_ok(fid): continue
        wp = os.path.join(D_WORK, fid + ".json"); w = jload(wp, {}) or {}
        if not w.get("matrix"): continue
        val = bool(body.get("value", True))                 # value=false — отменить принятие (быстрая проверка: «↶»)
        w["done"] = val
        if isinstance(w.get("auto"), dict): w["auto"]["reviewed"] = val
        w["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S"); jsave(wp, w); n += 1
    return {"accepted": n}


@app.post("/api/review_mark")
def api_review_mark(body: dict = Body(...)):
    """Массово отметить рисунки «на доработку» (review) или снять отметку."""
    n = 0; val = bool(body.get("value", True))
    for fid in body.get("ids", []):
        if not _fid_ok(fid): continue
        wp = os.path.join(D_WORK, fid + ".json"); w = jload(wp, {}) or {}
        if not w: continue
        w["review"] = val; jsave(wp, w); n += 1
    return {"marked": n}


# ---------------- v10.13: «трудные места» — проверка только кусков, где пороги и модель расходятся ----------------
def _hard_load(fid):
    wp = os.path.join(D_WORK, fid + ".json"); w = jload(wp, {}) or {}
    if not hard.eligible(w): return wp, w, None, "рисунок не подходит (нужен ЧБ-автомат с сеткой)"
    P = learn.load_model()
    if P is None: return wp, w, None, "модель клеток не обучена — сначала «обучить на готовых» (конвейер, шаг 5)"
    g = imread_any(os.path.join(D_CROPS, fid + ".png"), cv2.IMREAD_GRAYSCALE)
    if g is None: return wp, w, None, "нет кропа"
    try:
        ch = hard.ensure_strength(w, g)
        if hard.ensure(w, g, P) or ch: jsave(wp, w)
    except Exception as e:
        return wp, w, None, f"не удалось посчитать: {e}"
    return wp, w, hard.remaining(w) or [], None


@app.get("/api/hard/{fid}")
def api_hard(fid: str):
    """Непроверенные трудные куски рисунка (самые спорные первыми) + матрица и палитра для экрана проверки."""
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    wp, w, rem, err = _hard_load(fid)
    if err: return JSONResponse({"error": err}, 400)
    hd = w["hard"]
    return {"id": fid, "tiles": rem, "total": len(hd["tiles"]), "verified": len(hd.get("verified", [])),
            "too_many": hard.too_many(w), "weak": hard.weak(w), "strength": (w.get("auto") or {}).get("grid_strength"), "ntiles": hd.get("ntiles"), "tile": hard.TILE, "done": bool(w.get("done")),
            "matrix": w["matrix"], "palette": w.get("palette"), "t": os.path.getmtime(wp)}


@app.post("/api/hard/{fid}/kind")
def api_hard_kind(fid: str, body: dict = Body(default={})):
    """Один клик по карточке «фото / схема» (слабая сетка): photo — убрать из очереди схем как фото ковра;
    scheme — подтвердить, что это схема (карточка больше не показывается, дальше трудные места)."""
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    wp = os.path.join(D_WORK, fid + ".json"); w = jload(wp, {}) or {}
    if not w: return JSONResponse({"error": "нет рисунка"}, 404)
    k = (body or {}).get("kind")
    if k == "photo":
        w["kind"] = "photo"; w["done"] = False
        if isinstance(w.get("auto"), dict): w["auto"]["photo"] = True; w["auto"]["marked_photo"] = True
    elif k == "scheme":
        if not isinstance(w.get("auto"), dict): w["auto"] = {}
        w["auto"]["scheme_ok"] = True
    else: return JSONResponse({"error": "kind: photo | scheme"}, 400)
    w["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S"); jsave(wp, w)
    return {"kind": k}


@app.post("/api/hard/{fid}/verify")
def api_hard_verify(fid: str, body: dict = Body(default={})):
    """Отметить кусок {x,y} проверенным; cells — правки [[x,y,v]] (0 фон, 1 контур, 2 серое).
    Когда непроверенных кусков не осталось — рисунок принимается (done), остальное подтверждено согласием двух методов."""
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    wp, w, rem, err = _hard_load(fid)
    if err: return JSONResponse({"error": err}, 400)
    b = body or {}
    changed = hard.apply_cells(w, b.get("cells") or [])
    t = b.get("tile")
    if t is not None:
        key = [int(t["x"]), int(t["y"])]
        if key not in w["hard"].setdefault("verified", []): w["hard"]["verified"].append(key)
    if changed and isinstance(w.get("auto"), dict): w["auto"]["edited"] = True
    left = hard.remaining(w) or []
    accepted = False
    if not left and not w.get("done"):
        w["done"] = True; accepted = True
        if isinstance(w.get("auto"), dict): w["auto"]["reviewed"] = True; w["auto"]["accepted_via"] = "hard"
    w["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S"); jsave(wp, w)
    if changed:
        w["rapport"] = rapport.for_work(w); jsave(wp, w)
        render_outputs(fid, w["matrix"], w["palette"], w.get("transparent_bg", True), w.get("rapport"))
    return {"left": len(left), "changed": changed, "accepted": accepted}


@app.post("/api/risk_backfill")
def api_risk_backfill(body: dict = Body(default={})):
    """Посчитать симметрию и риск для авто-рисунков, у которых их ещё нет."""
    n = 0
    for fn in os.listdir(D_WORK):
        if not fn.endswith(".json"): continue
        wp = os.path.join(D_WORK, fn); w = jload(wp, {}) or {}
        a = w.get("auto")
        if isinstance(a, dict) and w.get("matrix") and ("risk" not in a or (body or {}).get("force", True)):
            if sum(c != "." for r in w["matrix"]["rows"] for c in r) < 3:
                a["confidence"] = 0.0; a["flags"] = [f for f in a.get("flags", []) if "не найден" not in f] + ["рисунок не найден"]
            _apply_risk(w); jsave(wp, w); n += 1
    return {"updated": n}


# ---------------- ночной прогон («разведчик») ----------------
PROWL = {"running": False, "stop": False, "total": 0, "done": 0, "errors": 0, "current": "", "started": None, "finished": None, "log": []}


def _prowl_targets(only_sheets=None):
    """Рисунки без пикселей на нарезанных листах (в порядке листов)."""
    out = []
    for fn in sorted(os.listdir(D_SHEETS)):
        if not fn.endswith(".json"): continue
        meta = jload(os.path.join(D_SHEETS, fn), {}) or {}
        if only_sheets and meta.get("path") not in only_sheets: continue
        for f in meta.get("figures", []):
            w = jload(os.path.join(D_WORK, f + ".json"), {}) or {}
            if not w.get("matrix") and w.get("kind") != "photo" and os.path.exists(os.path.join(D_CROPS, f + ".png")):
                out.append((meta.get("path"), f))
    return out


def _regray_targets():
    """ЧБ-автоматы, не проверенные человеком, где нет серой заливки (прогон без заливки терял её) — их можно пересчитать без потерь."""
    out = []
    for fn in sorted(os.listdir(D_WORK)):
        if not fn.endswith(".json"): continue
        w = jload(os.path.join(D_WORK, fn), {}) or {}
        a = w.get("auto") or {}
        if not (w.get("matrix") and a and not a.get("reviewed") and not w.get("done") and not w.get("color_mode") and w.get("kind") != "photo"): continue
        if "2" in "".join(w["matrix"]["rows"]): continue
        if os.path.exists(os.path.join(D_CROPS, fn[:-5] + ".png")): out.append((w.get("sheet"), fn[:-5]))
    return out


def _redo_targets():
    """Автоматы, которых человек не касался (не принят, не правился, палитра не тронута): их можно пересчитать свежим алгоритмом."""
    out = []
    for fn in sorted(os.listdir(D_WORK)):
        if not fn.endswith(".json"): continue
        w = jload(os.path.join(D_WORK, fn), {}) or {}
        a = w.get("auto") or {}
        if not (w.get("matrix") and a and not a.get("reviewed") and not a.get("edited") and not w.get("done") and not w.get("palette_touched") and w.get("kind") != "photo" and not a.get("photo")): continue
        if a.get("mode") == "color" and w.get("source_kind") == "photo": continue
        if os.path.exists(os.path.join(D_CROPS, fn[:-5] + ".png")): out.append((w.get("sheet"), fn[:-5], bool(w.get("color_mode"))))
    return out


def _hard_targets():
    """ЧБ-автоматы в очереди (не приняты), у которых трудные куски ещё не посчитаны или устарели (сетка менялась)."""
    out = []
    for fn in sorted(os.listdir(D_WORK)):
        if not fn.endswith(".json"): continue
        w = jload(os.path.join(D_WORK, fn), {}) or {}
        if w.get("done") or not hard.eligible(w): continue
        if (w.get("hard") or {}).get("sig") == hard.sig(w): continue
        if os.path.exists(os.path.join(D_CROPS, fn[:-5] + ".png")): out.append((w.get("sheet"), fn[:-5]))
    return out


def _segment_pending():
    """Листы без разметки (нет файла в data/sheets)."""
    out = []
    for root, _, files in os.walk(SCANS):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in IMG_EXT: continue
            rel = os.path.relpath(os.path.join(root, fn), SCANS)
            if not os.path.exists(os.path.join(D_SHEETS, _sheet_key(rel) + ".json")): out.append(rel)
    return sorted(out)


def _segment_one(rel):
    img = imread_any(_safe_rel(rel), cv2.IMREAD_GRAYSCALE)
    if img is None: raise RuntimeError("не читается")
    boxes = segment_sheet(img)
    key = _sheet_key(rel)
    if not boxes:
        jsave(os.path.join(D_SHEETS, key + ".json"), {"path": rel, "table": _guess_table(rel), "boxes": [], "figures": [], "no_figures": True, "auto_segmented": True})
        return 0
    r = api_sheet_save({"path": rel, "table": _guess_table(rel), "boxes": boxes})
    mp = os.path.join(D_SHEETS, key + ".json"); meta = jload(mp, {}) or {}; meta["auto_segmented"] = True; jsave(mp, meta)
    return len(boxes)


def _prowl_run(stages, only_sheets=None):
    P = PROWL; P.update(running=True, stop=False, stage="", total=0, done=0, errors=0, started=time.strftime("%Y-%m-%d %H:%M:%S"), finished=None, log=[])
    def err(msg): P["errors"] += 1; P["log"] = (P["log"] + [msg])[-50:]
    try:
        if "segment" in stages:
            P.update(stage="segment", done=0, total=0); t = _segment_pending(); P["total"] = len(t)
            for rel in t:
                if P["stop"]: break
                P["current"] = rel
                try: _segment_one(rel)
                except Exception as e: err(f"{rel}: {e}")
                P["done"] += 1
        if "redo" in stages and not P["stop"]:
            t = _redo_targets(); P.update(stage="redo", done=0, total=len(t))
            for path, f, is_color in t:
                if P["stop"]: break
                P["current"] = f
                try:
                    hint = _sheet_pitch_hint(path, exclude=f) if (path and is_color) else None
                    r = run_auto(f, True, dark_only=False, mode="color" if is_color else "bw", pitch_hint=hint)
                    if r.get("error"): err(f"{f}: {r['error']}")
                except Exception as e: err(f"{f}: {e}")
                P["done"] += 1
        if "regray" in stages and not P["stop"]:
            t = _regray_targets(); P.update(stage="auto", done=0, total=len(t))
            for path, f in t:
                if P["stop"]: break
                P["current"] = f
                try:
                    r = run_auto(f, True, dark_only=False, mode="bw")
                    if r.get("error"): err(f"{f}: {r['error']}")
                except Exception as e: err(f"{f}: {e}")
                P["done"] += 1
        if "auto" in stages and not P["stop"]:
            t = _prowl_targets(only_sheets); P.update(stage="auto", done=0, total=len(t))
            for path, f in t:
                if P["stop"]: break
                P["current"] = f
                try:
                    hint = _sheet_pitch_hint(path, exclude=f) if path else None
                    r = run_auto(f, False, dark_only=False, mode="auto", pitch_hint=hint)
                    if r.get("error"): err(f"{f}: {r['error']}")
                except Exception as e: err(f"{f}: {e}")
                P["done"] += 1
        if "hard" in stages and not P["stop"]:
            t = _hard_targets(); P.update(stage="hard", done=0, total=len(t))
            if t and learn.load_model() is None: err("модель клеток не обучена — трудные места не посчитать"); t = []
            for path, f in t:
                if P["stop"]: break
                P["current"] = f
                _, _, _, e = _hard_load(f)
                if e: err(f"{f}: {e}")
                P["done"] += 1
    finally:
        P.update(running=False, current="", stage="", finished=time.strftime("%Y-%m-%d %H:%M:%S"))


@app.post("/api/prowl/start")
def api_prowl_start(body: dict = Body(default={})):
    if PROWL["running"]: return JSONResponse({"error": "уже идёт"}, 409)
    stages = [s for s in ((body or {}).get("stages") or ["segment", "auto"]) if s in ("segment", "auto", "regray", "redo", "hard")]
    only = (body or {}).get("sheets")
    n = (len(_segment_pending()) if "segment" in stages else 0) + (len(_prowl_targets(only)) if "auto" in stages else 0) + (len(_regray_targets()) if "regray" in stages else 0) + (len(_redo_targets()) if "redo" in stages else 0) + (len(_hard_targets()) if "hard" in stages else 0)
    if not n: return {"started": False, "total": 0}
    threading.Thread(target=_prowl_run, args=(stages, only), daemon=True).start()
    return {"started": True, "total": n}


@app.post("/api/prowl/stop")
def api_prowl_stop():
    PROWL["stop"] = True
    return {"ok": True}


@app.get("/api/prowl")
def api_prowl(light: int = 0):
    d = dict(PROWL)
    if light: return d
    d["pending"] = len(_prowl_targets()) if not PROWL["running"] else None
    d["pending_segment"] = len(_segment_pending()) if not PROWL["running"] else None
    d["pending_regray"] = len(_regray_targets()) if not PROWL["running"] else None
    d["pending_redo"] = len(_redo_targets()) if not PROWL["running"] else None
    d["pending_hard"] = len(_hard_targets()) if not PROWL["running"] else None
    return d


@app.get("/api/dupes")
def api_dupes(min_score: float = 0.93):
    """Повторы узоров (ЧБ-схема и цветная того же узора, дубли между листами). Только кандидаты — решает человек."""
    items = dupes.load(D_WORK)
    return {"n": len(items), "pairs": dupes.find(items, min_score=min_score)[:300]}


@app.get("/api/pipeline")
def api_pipeline():
    """Сводка по конвейеру: листы → оцифровка → очередь проверки → готово → модель."""
    sheets = {"total": 0, "cut": 0, "empty": 0, "uncut": 0, "auto_cut": 0}
    for root, _, files in os.walk(SCANS):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in IMG_EXT: continue
            rel = os.path.relpath(os.path.join(root, fn), SCANS); sheets["total"] += 1
            mp = os.path.join(D_SHEETS, _sheet_key(rel) + ".json")
            if not os.path.exists(mp): sheets["uncut"] += 1; continue
            m = jload(mp, {}) or {}
            if m.get("no_figures"): sheets["empty"] += 1
            elif m.get("figures"):
                sheets["cut"] += 1; sheets["auto_cut"] += bool(m.get("auto_segmented"))
    f = {"total": 0, "undigitized": 0, "photo": 0, "queue": 0, "queue_low": 0, "queue_mid": 0, "queue_high": 0, "done": 0, "review": 0, "done_bw": 0, "new_done_bw": 0, "hard_ready": 0, "hard_tiles": 0, "hard_zero": 0}
    diff = {}
    mp = learn.MODEL_PATH; mt = os.path.getmtime(mp) if os.path.exists(mp) else 0
    for fn in os.listdir(D_WORK):
        if not fn.endswith(".json"): continue
        w = jload(os.path.join(D_WORK, fn), {}) or {}; f["total"] += 1
        a = w.get("auto") or {}
        if w.get("kind") == "photo": f["photo"] += 1; continue
        if not w.get("matrix"): f["undigitized"] += 1; continue
        if w.get("done"):
            f["done"] += 1
            if not w.get("color_mode"):
                f["done_bw"] += 1; f["new_done_bw"] += os.path.getmtime(os.path.join(D_WORK, fn)) > mt
        elif a and not a.get("reviewed"):
            f["queue"] += 1; r = a.get("risk"); r = (1 - (a.get("confidence") or 0)) if r is None else r
            f["queue_low" if r <= 0.25 else "queue_mid" if r <= 0.55 else "queue_high"] += 1
            hl = _hard_left(w)
            if hl is not None:
                f["hard_ready"] += 1; f["hard_tiles"] += hl; f["hard_zero"] += hl == 0
            d = a.get("difficulty_est"); diff[d] = diff.get(d, 0) + 1
        if w.get("review"): f["review"] += 1
    model = {"present": bool(mt)}
    if mt: model.update({"trained": time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)), **((jload(mp, {}) or {}).get("meta") or {})})
    return {"sheets": sheets, "figures": f, "queue_difficulty": diff, "model": model, "prowl": api_prowl()}


@app.post("/api/rapport/{fid}")
def api_rapport(fid: str):
    """Пересчитать раппорт каймы по текущей матрице (после ручных правок)."""
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    wp = os.path.join(D_WORK, fid + ".json"); work = jload(wp, {}) or {}
    if not work.get("matrix"): return JSONResponse({"error": "нет пикселей"}, 400)
    m = work["matrix"]; K = rapport.matrix_to_array(m)
    r = rapport.find_rapport(K)
    if r is not None: r["computed_at_matrix"] = [int(m["w"]), int(m["h"])]
    work["rapport"] = r; jsave(wp, work)
    render_outputs(fid, work["matrix"], work["palette"], work.get("transparent_bg", True), r)
    return r or {"found": False}


# ---------------- легенда / индекс сайта ----------------
@app.get("/api/legend")
def api_legend():
    rows = legend_rows()
    for r in rows:      # черновик латиницы там, где name_az ещё не заполнен
        if not (r.get("name_az") or "").strip() and (r.get("name") or "").strip() not in ("", "?"):
            r["name_az"] = az_draft(r["name"])
    return rows

@app.post("/api/legend")
def api_legend_save(body: dict = Body(...)):
    text = body.get("csv", "")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "table" not in rows[0] or "fig" not in rows[0]:
        return JSONResponse({"error": "нужен CSV с колонками table,fig,name_az,name,translation,carpet,type,note"}, 400)
    open(LEGEND, "w", encoding="utf-8").write(text)
    return {"ok": True, "rows": len(rows)}

@app.get("/api/site_index")
def api_site_index(): return jload(SITE_INDEX, [])

@app.post("/api/site_suggest")
def api_site_suggest(body: dict = Body(...)):
    """Подбор страниц сайта по азербайджанскому названию (латиница/кириллица) и русскому переводу."""
    names = [n for n in body.get("names", []) if n]
    return site_suggest(SITE_INDEX, names, body.get("translation", ""), int(body.get("limit", 8)))

@app.get("/api/dyes")
def api_dyes():
    """Библиотека красителей (dyes.json): одни и те же названия и опорные цвета для всех рисунков."""
    return {"dyes": [{"id": i, "az": a, "ru": r, "hex": h} for a, r, h, i in autocolor.DYES]}

@app.get("/api/az_draft")
def api_az_draft(text: str = ""):
    return {"text": text, "az": az_draft(text)}

# ---------------- пакет для сайта ----------------

# ---------------- три языка для сайта «Ковровое ДНК» ----------------
SCHOOL_BY_TYPE = {"Кар.": "karabakh", "Г.-К.": "ganja-gazakh", "Г.-К": "ganja-gazakh", "Г.—К": "ganja-gazakh",
                  "К.-Ш.": "guba-shirvan", "К.-Ш": "guba-shirvan", "К.—Ш": "guba-shirvan"}
I18N_FILE = os.path.join(DATA, "i18n.json")   # словарь переводов (значения, ковры, примечания) — пополняется

def i18n_fields(w, m, sids, site):
    """names az/ru/en: az — canonical_az первой привязанной страницы сайта, ru — перевод из книги,
    en — из словаря data/i18n.json или en_meaning страницы. Непереведённое остаётся пустым."""
    d = jload(I18N_FILE, {}) or {}
    prim = site.get(sids[0]) if sids else None
    tr = re.split(r"\s+(Карабах|Казах|Кубин|Губ|Гянд|Ширв|Баку|Бакин)", m.get("translation", ""))[0].strip(" .")
    tr = "" if tr == "?" else tr
    ru = tr or (prim or {}).get("ru_meaning") or ""
    en = (d.get("meaning_en") or {}).get(ru) or ((prim or {}).get("en_meaning") if prim and not tr else "") or ""
    az = (prim or {}).get("canonical_az") or m.get("name_az", "")
    carpet = (m.get("carpet") or "").strip().rstrip(" |")
    c3 = (d.get("carpet") or {}).get(carpet)
    note = (m.get("note") or "").strip(); n3 = (d.get("notes") or {}).get(note)
    t, f = w.get("table"), w.get("fig")
    return {"names": {"az": az, "ru": ru, "en": en},
            "carpet_i18n": ({"ru": carpet, "az": c3[0], "en": c3[1]} if c3 else ({"ru": carpet} if carpet else None)),
            "note_i18n": ({"ru": note, **n3} if n3 else ({"ru": note} if note else None)),
            "source_i18n": ({"ru": f"Табл. {t}, рис. {f}", "az": f"Cədvəl {t}, şəkil {f}", "en": f"Table {t}, fig. {f}"} if t else None),
            "school": SCHOOL_BY_TYPE.get(m.get("type", "")), "grid_origin": "hand_drawn",
            "i18n_status": "az/en: машинный перевод, человеком не проверен"}

@app.get("/api/bundle")
def api_bundle(all: int = 0):
    items = []
    site = {x["id"]: x for x in jload(SITE_INDEX, [])}
    for fn in sorted(os.listdir(D_WORK)):
        w = jload(os.path.join(D_WORK, fn), {}) or {}
        if not w.get("matrix") or not (w.get("done") or all): continue
        m = with_az(w.get("meta", {}))
        sids = [s.strip() for s in (m.get("site_id") or "").split(",") if s.strip()]
        items.append(i18n_fields(w, m, sids, site))
        items[-1].update({"id": w["id"], "table": w.get("table"), "fig": w.get("fig"),
                      "section": m.get("section", ""),
                      "name": m.get("name_az") or m.get("name", ""),        # основное имя — азербайджанское (латиница)
                      "name_az": m.get("name_az", ""), "name_book": m.get("name", ""),
                      "difficulty": norm_diff(w.get("difficulty")), "colors": len(set("".join(w["matrix"]["rows"])) - {"."}), "translation": m.get("translation", ""),
                      "carpet": m.get("carpet", ""), "type": m.get("type", ""), "note": m.get("note", ""),
                      "site_ids": [s.strip() for s in (m.get("site_id") or "").split(",") if s.strip()],
                      "source": m.get("source", "") or (f"Табл. {w.get('table')}, рис. {w.get('fig')}" if w.get("table") else ""),
                      "w": w["matrix"]["w"], "h": w["matrix"]["h"], "rows": w["matrix"]["rows"],
                      "palette": w["palette"], "palette_names": w.get("palette_names", []),
                      "updated": w.get("saved_at", "")})
    data = json.dumps({"kind": "carpet_pixel_schemes", "version": 1, "exported": time.strftime("%Y-%m-%d %H:%M"),
                       "items": items}, ensure_ascii=False)
    return Response(data, media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="pixel_schemes_bundle.json"'})

def regen_missing_crops():
    """data/crops не бэкапится и не восстанавливается (в git не идёт — сканы под
    авторским правом) — пересобирается из data/sheets (рамки) + scans (сам скан),
    той же логикой, что и «Сохранить и нарезать». Вызывается один раз при
    старте на хостинге, где после restore.py есть sheets+scans, но нет crops."""
    n_ok = n_missing_scan = n_err = 0
    for fn in os.listdir(D_SHEETS):
        if not fn.endswith(".json"):
            continue
        sh = jload(os.path.join(D_SHEETS, fn), {}) or {}
        rel = sh.get("path")
        boxes = sh.get("boxes") or []
        need = [b for b in boxes if b.get("id") and not os.path.exists(os.path.join(D_CROPS, b["id"] + ".png"))]
        if not need or not rel:
            continue
        try:
            col = imread_any(_safe_rel(rel))
        except Exception:
            col = None
        if col is None:
            n_missing_scan += len(need)
            continue
        H, W = col.shape[:2]
        for b in need:
            try:
                x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
                px, py = int(w * CROP_PAD), int(h * CROP_PAD)
                x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
                imwrite_any(os.path.join(D_CROPS, b["id"] + ".png"), col[y0:y1, x0:x1])
                n_ok += 1
            except Exception:
                n_err += 1
    if n_ok or n_missing_scan or n_err:
        print(f"regen_crops: восстановлено {n_ok}, скан не найден для {n_missing_scan}, ошибок {n_err}")


@app.on_event("startup")
def backup_start():
    restore.restore_all(HERE, DATA, SCANS)  # на «чистом» хостинге — подтянуть данные и сканы; на Mac — no-op
    regen_missing_crops()
    backup.start(HERE, DATA)

@app.on_event("shutdown")
def backup_stop():
    if backup.state["enabled"]: backup.backup_once(HERE, DATA, "остановка студии")

@app.get("/api/backup")
def api_backup(): return backup.state

@app.post("/api/regen_crops")
def api_regen_crops():
    """Ручной запуск пересборки недостающих data/crops из data/sheets+scans
    (обычно не нужен — делается сам при старте; пригодится, если сканы
    подложили уже после старта студии)."""
    regen_missing_crops()
    return {"ok": True}

@app.post("/api/backup")
def api_backup_now(): return backup.backup_once(HERE, DATA, "вручную")

@app.on_event("startup")
def migrate_v7():
    """v7: сетка светло-голубая, метки середины меньше — перерисовать _grid.png и SVG один раз (после migrate_v6)."""
    migrate_v6()
    flag = os.path.join(DATA, ".v7_rendered")
    if os.path.exists(flag): return
    for fn in os.listdir(D_WORK):
        if not fn.endswith(".json"): continue
        w = jload(os.path.join(D_WORK, fn), {}) or {}
        if w.get("matrix") and w.get("palette"):
            try: render_outputs(fn[:-5], w["matrix"], w["palette"])
            except Exception as e: print("migrate v7", fn, e)
    open(flag, "w").write(time.strftime("%Y-%m-%d %H:%M"))

def migrate_v6():
    """v6: старый серый цвет 2 (#8a8a8a) → светлее (#b8b8b8); картинки перерисовываются (новая сетка, слои SVG)."""
    flag = os.path.join(DATA, ".v6_migrated")
    if os.path.exists(flag): return
    for fn in os.listdir(D_WORK):
        if not fn.endswith(".json"): continue
        p = os.path.join(D_WORK, fn); w = jload(p, {}) or {}
        pal = w.get("palette") or []
        if any(str(c).lower() == "#8a8a8a" for c in pal):
            w["palette"] = ["#b8b8b8" if str(c).lower() == "#8a8a8a" else c for c in pal]; jsave(p, w)
        if w.get("matrix") and w.get("palette"):
            try: render_outputs(fn[:-5], w["matrix"], w["palette"])
            except Exception as e: print("migrate v6", fn, e)
    open(flag, "w").write(time.strftime("%Y-%m-%d %H:%M"))

@app.on_event("startup")
def migrate_outputs():
    """v4: PNG ×12 теперь без сетки, а сетка — отдельным слоем _grid.png. Перерисовываем старые выгрузки один раз."""
    for fn in os.listdir(D_WORK):
        if not fn.endswith(".json"): continue
        fid = fn[:-5]
        if os.path.exists(os.path.join(D_OUT, fid + "_grid.png")): continue
        w = jload(os.path.join(D_WORK, fn), {}) or {}
        if w.get("matrix") and w.get("palette") and os.path.exists(os.path.join(D_OUT, fid + "_x12.png")):
            try: render_outputs(fid, w["matrix"], w["palette"], w.get("transparent_bg", True))
            except Exception as e: print("migrate", fid, e)

@app.get("/")
def index(): return FileResponse(os.path.join(HERE, "index.html"), headers={"Cache-Control": "no-store"})
