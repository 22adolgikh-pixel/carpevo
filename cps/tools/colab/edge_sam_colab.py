# edge_sam_colab.py — v12.59: проверка «край по маске SAM 2» против правок человека (A/B, ничего не меняет на сервере).
#
# Что делает:
#   1. Берёт с сервера /api/open/edge_pack.zip: ковры, где человек правил край в карточке «край». Для каждого — фото, выпрямленное по 4 углам
#      с запасом 15 %, правка человека (hum) и край машины (mach, нынешний edge_full) — смещения края внутрь от линии углов, px, 64 станции.
#   2. SAM 2 (подсказка: рамка по углам + точки внутри ковра и снаружи) → маска ковра → на тех же 64 станциях каждой стороны снимается край.
#   3. Сравнивает три варианта с человеком: «линия углов» (ничего не двигать), «машина» (как сейчас), «SAM». Ошибка — в % от меньшей стороны ковра.
#   4. Итог: таблица, сколько ковров укладываются в 1 % и 2 % на всех сторонах; картинка edge_sam_sheet.jpg (зелёный — человек, синий — машина,
#      красный — SAM); edge_sam_eval.json. Всё в /content (и на Drive, если подключён).
#
# Запуск (Colab, лучше T4; на CPU тоже пойдёт, ~20 с на ковёр):
#   !pip -q install "git+https://github.com/facebookresearch/sam2.git"
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env CPS_KEY=<ключ из студии: пользователи → «ключ»> (можно всю ссылку с ?key=…)
#   %run /content/carpevo/cps/tools/colab/edge_sam_colab.py
# Настройки (env): ES_MODEL=facebook/sam2.1-hiera-large (или …-base-plus / -small / -tiny) · ES_SEARCH=0.15 (насколько далеко от линии углов искать край, доля стороны)
import os, io, json, time, zipfile, urllib.request, urllib.parse
import numpy as np, cv2

URL = os.environ.get("CPS_URL", "https://studio.carpets.wiki").rstrip("/")
KEY = os.environ.get("CPS_KEY", "").strip()
if "key=" in KEY: KEY = urllib.parse.parse_qs(urllib.parse.urlparse(KEY).query).get("key", [""])[0] or KEY.split("key=")[-1]
MODEL = os.environ.get("ES_MODEL", "facebook/sam2.1-hiera-large"); SEARCH = float(os.environ.get("ES_SEARCH", "0.15"))
assert KEY, "задайте ключ: %env CPS_KEY=…"
OUT = "/content/drive/MyDrive/cps_edge" if os.path.isdir("/content/drive/MyDrive") else "/content"
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(*a): print("[%4.0fs]" % (time.time() - T0), *a, flush=True)

log("качаю ковры с правкой края…")
with urllib.request.urlopen(URL + "/api/open/edge_pack.zip?key=" + urllib.parse.quote(KEY), timeout=1800) as r: zb = r.read()
Z = zipfile.ZipFile(io.BytesIO(zb)); R = []
for n in Z.namelist():
    if n.endswith(".json"):
        d = json.loads(Z.read(n))
        if d.get("hum"): d["img"] = cv2.imdecode(np.frombuffer(Z.read(n[:-5] + ".jpg"), np.uint8), cv2.IMREAD_COLOR); R.append(d)
    elif n.endswith(".err"): log("  ошибка на сервере:", n, Z.read(n).decode()[:120])
log("ковров с правкой:", len(R))
assert R, "нет ковров с правкой края"

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
dev = "cuda" if torch.cuda.is_available() else "cpu"
P = SAM2ImagePredictor.from_pretrained(MODEL, device=dev); log("SAM 2:", MODEL, dev)

NST = 64; TT = np.linspace(0, 1, NST)

def sam_mask(img, g):
    W, H, px, py = g["W"], g["H"], g["px"], g["py"]; ch, cw = img.shape[:2]
    m = 0.04
    box = np.array([px - m * W, py - m * H, px + W + m * W, py + H + m * H], np.float32)
    pin = [[px + W * a, py + H * b] for a, b in ((.5, .5), (.25, .25), (.75, .25), (.25, .75), (.75, .75))]
    pout = [[max(2, px * 0.3), max(2, py * 0.3)], [min(cw - 3, cw - px * 0.3), max(2, py * 0.3)], [max(2, px * 0.3), min(ch - 3, ch - py * 0.3)], [min(cw - 3, cw - px * 0.3), min(ch - 3, ch - py * 0.3)]]
    with torch.inference_mode():
        P.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ms, sc, _ = P.predict(point_coords=np.array(pin + pout, np.float32), point_labels=np.array([1] * len(pin) + [0] * len(pout)), box=box, multimask_output=True)
    rect = np.zeros((ch, cw), bool); rect[py:py + H, px:px + W] = True
    best, bs = None, -1
    for mk, s in zip(ms, sc):                                            # из трёх масок — та, что лучше всего совпадает с рамкой углов
        mk = mk > 0.5; iou = (mk & rect).sum() / max(1, (mk | rect).sum())
        if iou > bs: best, bs = mk, iou
    mk = best.astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(mk)
    if n > 1: mk = (cc == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); f = np.zeros_like(mk)
    if cs: cv2.drawContours(f, [max(cs, key=cv2.contourArea)], -1, 1, -1)
    return f.astype(bool), float(bs)

def sam_curves(mk, g):
    W, H, px, py = g["W"], g["H"], g["px"], g["py"]; ch, cw = mk.shape; out = {}
    sy, sx = int(SEARCH * H), int(SEARCH * W); band = 3
    def col(x, y0, y1, first):
        x0, x1 = max(0, int(x) - band), min(cw, int(x) + band + 1); y0, y1 = max(0, y0), min(ch, y1)
        c = mk[y0:y1, x0:x1].any(1); idx = np.nonzero(c)[0]
        return np.nan if not len(idx) else (y0 + (idx[0] if first else idx[-1] + 1))
    def row(y, x0, x1, first):
        y0_, y1_ = max(0, int(y) - band), min(ch, int(y) + band + 1); x0, x1 = max(0, x0), min(cw, x1)
        c = mk[y0_:y1_, x0:x1].any(0); idx = np.nonzero(c)[0]
        return np.nan if not len(idx) else (x0 + (idx[0] if first else idx[-1] + 1))
    xs = px + TT * (W - 1); ys = py + TT * (H - 1)
    out["top"] = np.array([col(x, py - sy, py + sy, True) - py for x in xs])
    out["bot"] = np.array([(py + H) - col(x, py + H - sy, py + H + sy, False) for x in xs])
    out["left"] = np.array([row(y, px - sx, px + sx, True) - px for y in ys])
    out["right"] = np.array([(px + W) - row(y, px + W - sx, px + W + sx, False) for y in ys])
    for k, v in out.items():
        ok = ~np.isnan(v)
        v = np.interp(np.arange(NST), np.nonzero(ok)[0], v[ok]) if ok.sum() >= 4 else np.zeros(NST)
        v = cv2.GaussianBlur(cv2.medianBlur(v.astype(np.float32).reshape(1, -1), 5), (9, 1), 2).reshape(-1)
        out[k] = v
    return out

def err(a, b, sh): e = np.abs(np.asarray(a, float) - np.asarray(b, float)) / sh * 100; return float(e.mean()), float(e.max())

VAR = ("line", "mach", "sam", "best")                                   # best — по каждой стороне лучшее из «машина» и «SAM» (потолок для гибрида)
res = []; sheet = []
for i, d in enumerate(R):
    g = d["geo"]; sh = min(g["W"], g["H"])
    mk, iou = sam_mask(d["img"], g); sc = sam_curves(mk, g)
    row_ = {"id": d["id"], "iou_rect": round(iou, 3), "sides": {}}
    for k in ("top", "bot", "left", "right"):
        h = d["hum"][k]; z = {"line": err(np.zeros(NST), h, sh), "sam": err(sc[k], h, sh)}
        if d.get("mach") and k in d["mach"]: z["mach"] = err(d["mach"][k], h, sh)
        z["best"] = min((z[v] for v in ("mach", "sam") if v in z), key=lambda t: t[1])
        row_["sides"][k] = {v: [round(a, 2), round(b, 2)] for v, (a, b) in z.items()}
    res.append(row_)
    a = d["img"].copy(); W, H, px, py = g["W"], g["H"], g["px"], g["py"]
    def draw(c, col_):
        for k in ("top", "bot", "left", "right"):
            v = np.asarray(c[k], float)
            if k == "top": pts = np.stack([px + TT * W, py + v], 1)
            elif k == "bot": pts = np.stack([px + TT * W, py + H - v], 1)
            elif k == "left": pts = np.stack([px + v, py + TT * H], 1)
            else: pts = np.stack([px + W - v, py + TT * H], 1)
            cv2.polylines(a, [pts.astype(np.int32)], False, col_, 3)
    if d.get("mach"): draw(d["mach"], (255, 120, 0))
    draw(sc, (0, 0, 255)); draw(d["hum"], (0, 220, 0))
    mx = max(v["sam"][1] for v in row_["sides"].values())
    s = 360 / max(a.shape[:2]); t = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = cv2.copyMakeBorder(t, 0, 360 - t.shape[0], 0, 360 - t.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(t, "%s  SAM max %.1f%%" % (d["id"][:22], mx), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    sheet.append((mx, t))
    if (i + 1) % 5 == 0: log(f"  {i + 1}/{len(R)}")

def share(var, thr, stat=1):
    ok = [all(r["sides"][k].get(var, [99, 99])[stat] <= thr for k in r["sides"]) for r in res if all(var in r["sides"][k] for k in r["sides"])]
    return sum(ok), len(ok)
print("\n=== Ковры, где край укладывается в допуск на ВСЕХ четырёх сторонах (наибольшее расхождение) ===")
print("%-24s %10s %10s %10s" % ("вариант", "≤ 1 %", "≤ 2 %", "≤ 3 %"))
names = {"line": "линия углов (не двигать)", "mach": "машина (как сейчас)", "sam": "SAM 2", "best": "лучшее из двух по сторонам"}
for v in VAR:
    c = [share(v, t) for t in (1, 2, 3)]
    print("%-24s %10s %10s %10s" % (names[v], *("%d/%d" % x for x in c)))
print("\n=== По сторонам: доля ковров с наибольшим расхождением ≤ 1 % (среднее расхождение, %) ===")
for k in ("top", "bot", "left", "right"):
    line = []
    for v in VAR:
        xs = [r["sides"][k][v] for r in res if v in r["sides"][k]]
        if xs: line.append("%s %d/%d (ср %.2f)" % (v, sum(x[1] <= 1 for x in xs), len(xs), np.mean([x[0] for x in xs])))
    print("%-6s" % k, " · ".join(line))
sheet.sort(key=lambda x: x[0]); tiles = [t for _, t in sheet]
while len(tiles) % 6: tiles.append(np.full_like(tiles[0], 20))
img = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
cv2.imwrite(os.path.join(OUT, "edge_sam_sheet.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump({"model": MODEL, "search": SEARCH, "n": len(res), "rugs": res}, open(os.path.join(OUT, "edge_sam_eval.json"), "w"), ensure_ascii=False, indent=1)
log("готово:", os.path.join(OUT, "edge_sam_sheet.jpg"), "(сверху — лучшие, внизу — худшие; зелёный — человек, синий — машина, красный — SAM) и edge_sam_eval.json")
log("пришлите сюда таблицу выше (и картинку, если не жалко)")
