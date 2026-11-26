# xfind_night_colab.py — v12.36: ночной поиск «где ещё встречается этот элемент» по всему каталогу (Colab, T4 GPU).
#
# Для каждой группы подтверждённых рамок (элемент, обведённый на ковре) ищем такие же места на всех остальных коврах:
#   шаг 1 — сетка признаков DINOv3-S (640 px, PCA 128) + окна той же доли фото в 5 размерах (как motif_search): лучшие ковры;
#   шаг 2 — первые XF_NR ковров (по 3 окна, у каждого 7 сдвигов/размеров) перепроверяются овальной вырезкой DINOv3-S
#           (как element_embed «match»), образец — до 4 рамок группы × 8 положений.
# Замер 25.11 (119 названных элементов, 480 ковров): верное место на 1-м ковре 28 %, среди первых 5 — 56 % (без шага 2 — 15 % / 46 %).
# Свой ковёр и его двойники не ищутся. Итог уходит прямо на сервер (POST /api/open/xfind_upload по ключу), копия — в Drive; сервер
# и показывает в /m карточкой «на других коврах» (✓ — на тот ковёр ставится подтверждённая рамка с названием образца).
#
# Запуск:
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env CPS_KEY=<ключ из студии: пользователи → «ключ»> (можно и всю ссылку с ?key=…)
#   %run /content/carpevo/cps/tools/colab/xfind_night_colab.py
# Фото кэшируются в Drive (carpet-dna/cps_xfind/raw): следующий запуск качает только новые и изменённые.
import os, sys, io, json, time, zipfile, math, urllib.request, urllib.parse
import numpy as np, cv2

URL = os.environ.get("CPS_URL", "https://cps.jeyran.az").rstrip("/")
KEY = os.environ.get("CPS_KEY", "").strip()
if "key=" in KEY: KEY = urllib.parse.parse_qs(urllib.parse.urlparse(KEY).query).get("key", [""])[0] or KEY.split("key=")[-1]   # вставили всю ссылку — берём ключ
XF_FILE = os.environ.get("XF_FILE", "1RE74rNCMa4Y9HwcsA3PtR9psH7TuPGtL")
NR = int(os.environ.get("XF_NR", "150"))          # сколько ковров шага 1 перепроверять вырезкой
TOPK = int(os.environ.get("XF_TOP", "12"))        # сколько находок на группу отдавать
LONG = 640; D = 128; BETA = 0.5; QW = 0.6; SCALES = (0.6, 0.78, 1.0, 1.28, 1.6)
assert KEY, "задайте ключ: %env CPS_KEY=…"
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f} c]", *a, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if "__file__" in globals() else "/content/carpevo/cps")
import element_embed as EE                         # та же вырезка (_crop) и подготовка (_prep), что на сервере

# ---------- Drive ----------
DRV = "/content/cps_xfind"
try:
    from google.colab import drive
    drive.mount("/content/drive"); DRV = "/content/drive/MyDrive/carpet-dna/cps_xfind"     # кэш фото между запусками
except Exception as e:
    log("Drive не подключён (", e, ") — кэш в /content, фото придётся качать каждый раз")
RAW = os.path.join(DRV, "raw"); os.makedirs(RAW, exist_ok=True)

def get(path, **q):
    q["key"] = KEY
    with urllib.request.urlopen(URL + path + "?" + urllib.parse.urlencode(q), timeout=600) as r: return r.read()

# ---------- карточки и фото ----------
log("карточки…")
Z = zipfile.ZipFile(io.BytesIO(get("/api/open/cards.zip", since=0)))
R = {}
for n in Z.namelist():
    if n.startswith("rugs/") and n.endswith(".json"):
        r = json.loads(Z.read(n)); R[r["id"]] = r
live = [rid for rid, r in R.items() if not (r.get("hidden") or r.get("dup_of") or r.get("split_into"))]
man = json.loads(get("/api/open/manifest", since=0, live=1)).get("rugs") or []
mt = {x["id"]: float((x.get("t") or {}).get("raw") or 0) for x in man if x.get("id")}
live = [rid for rid in live if rid in mt]                                         # live=1: без скрытых и дублей
if os.environ.get("XF_ONLY"): live = [x for x in open(os.environ["XF_ONLY"]).read().split(",") if x in set(live)]   # проба на части каталога
idx_f = os.path.join(RAW, "_index.json")
try: IDX = json.load(open(idx_f))
except Exception: IDX = {}
need = [rid for rid in live if not os.path.exists(os.path.join(RAW, rid + ".jpg")) or IDX.get(rid, -1) < mt.get(rid, 0)]
log(f"ковров {len(live)}, качать фото {len(need)}")
for i in range(0, len(need), 40):
    part = need[i:i + 40]
    try: zb = get("/api/open/files.zip", kind="raw", ids=",".join(part), limit=500)
    except Exception as e: log("пачка не скачалась:", e); continue
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        for n in z.namelist():
            if n.startswith("raw/") and n.endswith(".jpg"):
                rid = n[4:-4]; open(os.path.join(RAW, rid + ".jpg"), "wb").write(z.read(n)); IDX[rid] = mt.get(rid, time.time())
    if i % 400 == 0: log(f"  фото {i + len(part)}/{len(need)}"); json.dump(IDX, open(idx_f, "w"))
json.dump(IDX, open(idx_f, "w"))

def img(rid, side=1400):
    p = os.path.join(RAW, rid + ".jpg"); a = cv2.imread(p)
    if a is None: return None
    k = side / max(a.shape[:2])
    return cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else a

# ---------- сеть ----------
import torch, torch.nn.functional as F
try: import timm
except ImportError: os.system("pip -q install timm"); import timm
from safetensors.torch import load_file
DEV = "cuda" if torch.cuda.is_available() else "cpu"; log("устройство", DEV)
wf = "/content/dinov3_s.safetensors"
if not os.path.exists(wf): open(wf, "wb").write(get("/api/open/wfile/dinov3_s"))
NET = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0, dynamic_img_size=True)
NET.load_state_dict(load_file(wf)); NET = NET.eval().to(DEV).half() if DEV == "cuda" else NET.eval()
NP = NET.num_prefix_tokens; MU = np.array([.485, .456, .406], np.float32); SD = np.array([.229, .224, .225], np.float32)

@torch.no_grad()
def grid(a):
    h, w = a.shape[:2]; k = LONG / max(h, w); W, H = max(16, int(round(w * k / 16)) * 16), max(16, int(round(h * k / 16)) * 16)
    x = ((cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255 - MU) / SD).transpose(2, 0, 1)[None]
    t = torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
    f = NET.forward_features(t.half() if DEV == "cuda" else t)[0, NP:].float()
    return f.reshape(H // 16, W // 16, -1)

@torch.no_grad()
def emb(crops):
    out = []
    for i in range(0, len(crops), 256):
        x = torch.from_numpy(np.stack([EE._prep(c) for c in crops[i:i + 256]])).to(DEV)
        f = NET.forward_features(x.half() if DEV == "cuda" else x).float()
        out.append(F.normalize(torch.cat([f[:, 0], f[:, NP:].mean(1)], 1), dim=1))
    return torch.cat(out) if out else torch.zeros((0, 768), device=DEV)

def crop(a, b): return EE._crop(a, b, 112, 0.15, False, "ellipse")

# ---------- шаг 0: сетки всех ковров ----------
log("сетки признаков…")
G = {}
for i, rid in enumerate(live):
    a = img(rid, 1200)
    if a is None: continue
    G[rid] = grid(a)
    if i % 500 == 0: log(f"  {i}/{len(live)}")
S_ = torch.cat([g.reshape(-1, g.shape[-1])[::11] for g in list(G.values())[:600]])
mu = S_.mean(0); U, s, V = torch.linalg.svd(S_ - mu, full_matrices=False); PM = V[:D].T
G = {k: ((g - mu) @ PM).permute(2, 0, 1).contiguous() for k, g in G.items()}       # (D, gh, gw)
GM = {k: g.mean((1, 2)) for k, g in G.items()}
log(f"сеток {len(G)}")

def wfeat(g, gm, sh, sw):
    """признаки всех окон sh×sw (целое + 4 четверти, минус BETA·среднее ковра), нормированы: (ny, nx, 5D) — как motif_search._feat"""
    C, gh, gw = g.shape; ny, nx = gh - sh + 1, gw - sw + 1
    ht, wl = max(1, sh // 2), max(1, sw // 2); hb, wr = max(1, sh - ht), max(1, sw - wl)
    oy, ox = min(ht, sh - hb), min(wl, sw - wr)
    P = [F.avg_pool2d(g[None], (sh, sw), 1)[0, :, :ny, :nx] - BETA * gm[:, None, None]]
    for hh, ww, dy, dx in ((ht, wl, 0, 0), (ht, wr, 0, ox), (hb, wl, oy, 0), (hb, wr, oy, ox)):
        P.append(QW * (F.avg_pool2d(g[None], (hh, ww), 1)[0, :, dy:dy + ny, dx:dx + nx] - BETA * gm[:, None, None]))
    V_ = torch.cat(P, 0).permute(1, 2, 0)
    return F.normalize(V_, dim=-1)

def qvec(g, gm, b):
    C, gh, gw = g.shape; x, y, w, h = b
    x0 = max(0, min(gw - 1, int(math.floor(x * gw + 0.25)))); y0 = max(0, min(gh - 1, int(math.floor(y * gh + 0.25))))
    x1 = max(x0 + 1, min(gw, int(math.ceil((x + w) * gw - 0.25)))); y1 = max(y0 + 1, min(gh, int(math.ceil((y + h) * gh - 0.25))))
    return wfeat(g[:, y0:y1, x0:x1], gm, y1 - y0, x1 - x0)[0, 0]

def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])); i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)

def windows(g, gm, q, rw, rh, per=3):
    C, gh, gw = g.shape; cand = []
    for k in SCALES:
        sw, sh = max(1, min(gw, int(round(rw * gw * k)))), max(1, min(gh, int(round(rh * gh * k))))
        sc = (wfeat(g, gm, sh, sw) @ q).flatten(); n = min(12, sc.numel())
        v, ix = torch.topk(sc, n); nx = gw - sw + 1
        for vv, ii in zip(v.tolist(), ix.tolist()):
            yy, xx = divmod(ii, nx); cand.append((vv, [xx / gw, yy / gh, sw / gw, sh / gh]))
    cand.sort(key=lambda c: -c[0]); out = []
    for s, b in cand:
        if all(iou(b, o[1]) < 0.3 for o in out): out.append((s, b))
        if len(out) >= per: break
    return out

def jit(b):
    x, y, w, h = b; cx, cy = x + w / 2, y + h / 2; out = []
    for sc, dx, dy in ((1, 0, 0), (1, -.25, 0), (1, .25, 0), (1, 0, -.25), (1, 0, .25), (.8, 0, 0), (1.25, 0, 0)):
        ww, hh = w * sc, h * sc; out.append([max(0.0, cx + dx * w - ww / 2), max(0.0, cy + dy * h - hh / 2), min(1.0, ww), min(1.0, hh)])
    return out

def twins(r):
    s = set()
    for x in r.get("same") or []:
        i = x.get("id") if isinstance(x, dict) else x
        if i and ((r.get("same_ok") or {}).get(i) or {}).get("ok") is not False: s.add(i)
    for i, v in (r.get("same_ok") or {}).items():
        if (v or {}).get("ok"): s.add(i)
    if r.get("dup_of"): s.add(r["dup_of"])
    if r.get("same_group"): s |= {k for k, t in R.items() if t.get("same_group") == r["same_group"]}
    return s

# ---------- запросы: группы подтверждённых рамок ----------
Q = []
for rid in live:
    if rid not in G: continue
    gr = {}
    for m in R[rid].get("marks") or []:
        if m.get("status") == "confirmed" and isinstance(m.get("box"), list) and len(m["box"]) == 4: gr.setdefault(m.get("group") or m["id"], []).append(m)
    for g, ms in gr.items():
        ms.sort(key=lambda m: (bool(m.get("machine")), -m["box"][2] * m["box"][3]))
        Q.append({"rid": rid, "g": g, "boxes": [m["box"] for m in ms[:4]], "n": len(ms)})
if os.environ.get("XF_GROUPS"): Q = Q[:int(os.environ["XF_GROUPS"])]   # проба
log(f"групп {len(Q)}")

items = []; out_f = os.path.join(DRV, "xfind_night.json")
for qi, u in enumerate(Q):
    try:
        g0, gm0 = G[u["rid"]], GM[u["rid"]]
        q = qvec(g0, gm0, u["boxes"][0])                          # одна рамка-образец: среднее по группе размывает (замер 25.11: 41 % → 59 % в первых 5)
        rw = float(np.median([b[2] for b in u["boxes"]])); rh = float(np.median([b[3] for b in u["boxes"]]))
        ex = twins(R[u["rid"]]) | {u["rid"]}
        s1 = []
        for rid, g in G.items():
            if rid in ex: continue
            w = windows(g, GM[rid], q, rw, rh)
            if w: s1.append((w[0][0], rid, w))
        s1.sort(key=lambda t: -t[0])
        a0 = img(u["rid"]); qc = []
        for b in u["boxes"]:
            c = crop(a0, b); qc += [np.rot90(c[:, ::-1] if f else c, r) for f in (0, 1) for r in range(4)]
        QV = emb([np.ascontiguousarray(c) for c in qc])
        res = []
        for s, rid, w in s1[:NR]:
            a = img(rid)
            if a is None: continue
            bs = [j for _, b in w for j in jit(b)]
            sc = (emb([crop(a, b) for b in bs]) @ QV.T).max(1).values.cpu().numpy(); i = int(sc.argmax())
            res.append({"rid": rid, "box": [round(v, 4) for v in bs[i]], "s": round(float(sc[i]), 4), "s1": round(float(s), 4)})
        res.sort(key=lambda t: -t["s"])
        items.append({"q": {"rid": u["rid"], "g": u["g"], "box": u["boxes"][0], "n": u["n"]}, "c": res[:TOPK]})
    except Exception as e:
        log("группа", u["rid"], u["g"], "ошибка:", e)
    if qi % 10 == 0:
        log(f"  групп {qi + 1}/{len(Q)}")
        json.dump({"t": int(time.time()), "partial": True, "items": items}, open(out_f, "w"))

res = {"t": int(time.time()), "ver": 1, "model": "dinov3_s grid640+pca128 → oval112", "nr": NR, "groups": len(Q), "rugs": len(G), "items": items}
json.dump(res, open(out_f, "w"), ensure_ascii=False)
log("итог:", out_f)
try:                                                             # итог — прямо на сервер (по ключу)
    rq = urllib.request.Request(URL + "/api/open/xfind_upload?key=" + urllib.parse.quote(KEY), data=json.dumps(res, ensure_ascii=False).encode("utf-8"),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(rq, timeout=300) as r: log("на сервере:", r.read().decode()[:200])
except Exception as e:
    log("!!! не удалось отправить на сервер:", e, "— итог лежит в", out_f)
sc = [c["s"] for it in items for c in it["c"][:1]]
print("\n===== ИТОГ =====")
print(f"групп {len(items)}, ковров в поиске {len(G)}, время {(time.time() - T0) / 60:.0f} мин")
if sc: print(f"лучшая находка на группу: медиана сходства {np.median(sc):.3f}, ≥ 0,85 — у {sum(v >= 0.85 for v in sc)} групп")
