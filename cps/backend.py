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
import cloud
import backup
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
D_SHEETS, D_CROPS, D_WORK, D_OUT = (os.path.join(DATA, d) for d in ("sheets", "crops", "work", "out"))
for d in (SCANS, DATA, D_SHEETS, D_CROPS, D_WORK, D_OUT): os.makedirs(d, exist_ok=True)
LEGEND = os.path.join(DATA, "legend.csv")
SITE_INDEX = os.path.join(DATA, "site_index.json")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
CROP_PAD = 0.08          # запас вокруг рамки рисунка (доля), чтобы было куда тянуть углы
META_KEYS = ("section", "name_az", "name", "translation", "carpet", "type", "note")
DIFFICULTY = ("simple", "medium", "complex", "ultra")   # простой / средний / сложный / ультра
OUT_SUFFIXES = (".png", "_x12.png", "_grid.png", "_bg.png", ".svg")   # что лежит в data/out на каждый рисунок

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

def jsave(p, obj): json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

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
            src = rel.replace("\\", "/").split("/")[0] if "/" in rel.replace("\\", "/") else "(без папки)"
            out.append({"path": rel, "name": fn, "source": src, "table": meta.get("table"), "n": len(figs), "done": done, "diff": diff})
    out.sort(key=lambda s: s["path"])
    return out

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
                    "size": [w["matrix"]["w"], w["matrix"]["h"]] if w.get("matrix") else None,
                    "colors": len(set("".join(w["matrix"]["rows"])) - {"."}) if w.get("matrix") else None,
                    "t": os.path.getmtime(os.path.join(D_WORK, fn))})
    def k(f):
        try: return (0, int(f["table"] or 0), int(f["fig"] or 0), f["id"])
        except Exception: return (1, 0, 0, f["id"])
    return sorted(out, key=k)

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
GRID_THIN = (236, 150, 186, 90)     # RGBA: светло-розовая линия каждого узла
GRID_BOLD = (214, 64, 124, 200)     # каждые 10 узлов (от левого верхнего угла рисунка)
GRID_MID = (214, 64, 124, 255)      # метки середины на краях

def render_grid_layer(w, h, S=GRID_S):
    """Сетка отдельным слоем (PNG той же величины, что ×12): одинаковая у всех рисунков —
    тонкая розовая линия на каждый узел, жирная каждые 10 от левого верхнего угла, треугольные метки середины."""
    g = np.zeros((h * S, w * S, 4), np.uint8)
    thin = GRID_THIN[2::-1] + GRID_THIN[3:]; bold = GRID_BOLD[2::-1] + GRID_BOLD[3:]; mid = GRID_MID[2::-1] + GRID_MID[3:]
    for i in range(w + 1):
        x = min(i * S, w * S - 1); g[:, x] = bold if i % 10 == 0 or i == w else thin
        if i % 10 == 0 or i == w: g[:, max(0, x - 1)] = bold
    for j in range(h + 1):
        y = min(j * S, h * S - 1); g[y, :] = bold if j % 10 == 0 or j == h else thin
        if j % 10 == 0 or j == h: g[max(0, y - 1), :] = bold
    cx, cy, k = w * S // 2, h * S // 2, max(4, S // 2)       # ▼▲◀▶ метки середины снаружи сетки не помещаются — рисуем по краям внутрь
    for d in range(k):
        g[d, max(0, cx - (k - d)):cx + (k - d)] = mid; g[h * S - 1 - d, max(0, cx - (k - d)):cx + (k - d)] = mid
        g[max(0, cy - (k - d)):cy + (k - d), d] = mid; g[max(0, cy - (k - d)):cy + (k - d), w * S - 1 - d] = mid
    return g

def render_outputs(fid, mat, palette, transparent_bg=True):
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
           f'<g id="grid" stroke="#d6407c" stroke-opacity=".6" display="none">{gl}</g></svg>')
    open(os.path.join(D_OUT, fid + ".svg"), "w", encoding="utf-8").write(svg)

@app.post("/api/work/{fid}")
def api_work_save(fid: str, body: dict = Body(...)):
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    wp = os.path.join(D_WORK, fid + ".json")
    work = jload(wp, {}) or {}
    work.update(body); work["id"] = fid; work["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if work.get("done") and isinstance(work.get("auto"), dict): work["auto"]["reviewed"] = True
    jsave(wp, work)
    if work.get("matrix") and work.get("palette"):
        render_outputs(fid, work["matrix"], work["palette"], work.get("transparent_bg", True))
    return {"ok": True, "saved_at": work["saved_at"]}

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
def run_auto(fid, force=False, dark_only=True):
    wp = os.path.join(D_WORK, fid + ".json")
    work = jload(wp, {}) or {}
    if work.get("matrix") and not force:
        return {"id": fid, "skipped": "уже есть пиксели"}
    crop = imread_any(os.path.join(D_CROPS, fid + ".png"), cv2.IMREAD_GRAYSCALE)
    if crop is None: return {"id": fid, "error": "нет кропа"}
    frame = work.get("frame") or [0, 0, crop.shape[1], crop.shape[0]]
    try:
        G, C, M, info = autogrid.auto_figure(crop, frame, dark_only=dark_only)
    except Exception as e:
        return {"id": fid, "error": f"сетка не найдена: {e}"}
    upd = autogrid.to_work(G, M, info)
    if work.get("palette_touched") and work.get("palette"):      # свою палитру не трогаем, только дополняем
        pal, names = list(work["palette"]), list(work.get("palette_names") or [])
        while len(pal) < len(upd["palette"]): pal.append(upd["palette"][len(pal)]); names.append(upd["palette_names"][len(names)])
        upd["palette"], upd["palette_names"] = pal, names; upd.pop("palette_touched")
    work.update(upd); work["id"] = fid; work["done"] = False
    work["adjust"] = {"brightness": 0, "contrast": 1, "gamma": 1, "invert": False}
    work["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jsave(wp, work)
    render_outputs(fid, work["matrix"], work["palette"], work.get("transparent_bg", True))
    return {"id": fid, "size": [work["matrix"]["w"], work["matrix"]["h"]],
            "confidence": work["auto"]["confidence"], "flags": work["auto"]["flags"]}

@app.post("/api/auto/{fid}")
def api_auto(fid: str, body: dict = Body(default={})):
    """Автоматически натянуть сетку и перевести рисунок в пиксели (черновик — проверить на экране «Пиксели»).
    v6: по умолчанию контур + серая заливка (замер на 134 готовых рисунках: 95.6% клеток верно против 87% у «только контура»);
    fill=false — только тёмный контур."""
    if not _fid_ok(fid): return JSONResponse({"error": "bad id"}, 400)
    return run_auto(fid, bool((body or {}).get("force", True)), dark_only=not bool((body or {}).get("fill", True)))

@app.post("/api/auto_sheet")
def api_auto_sheet(body: dict = Body(...)):
    """Все рисунки листа. По умолчанию пропускает те, где пиксели уже есть (force — пересчитать всё)."""
    meta = jload(os.path.join(D_SHEETS, _sheet_key(body["path"]) + ".json"), {}) or {}
    figs = meta.get("figures", [])
    if not figs: return JSONResponse({"error": "лист ещё не нарезан — сначала «Сохранить и нарезать»"}, 400)
    dark_only = not bool(body.get("fill", True))
    return {"results": [run_auto(f, bool(body.get("force")), dark_only=dark_only) for f in figs]}

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

@app.on_event("startup")
def backup_start(): backup.start(HERE, DATA)

@app.on_event("shutdown")
def backup_stop():
    if backup.state["enabled"]: backup.backup_once(HERE, DATA, "остановка студии")

@app.get("/api/backup")
def api_backup(): return backup.state

@app.post("/api/backup")
def api_backup_now(): return backup.backup_once(HERE, DATA, "вручную")

@app.on_event("startup")
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
