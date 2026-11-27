# elements_cluster_colab.py — v12.49: «кучи элементов» (пилот).
#
# Идея «снизу вверх»: не искать по образцам, которые человек обвёл, а разобрать выборку ковров на повторяющиеся элементы САМИМ и собрать
# похожие по форме в кучи (кластеры). Человеку остаётся одно решение на кучу («одна форма» / «разные» / «не элементы»), а не на каждую рамку.
#   1. Для каждого ковра: сетка признаков DINOv3-S (как в xfind_night_colab.py); окна 6 размеров; окно — кандидат, если на этом же ковре
#      есть ≥ EC_MINREP непересекающихся окон, похожих на него (≥ EC_TAU), и оно заметно отличается от среднего по ковру (не фон).
#   2. Кандидаты (до EC_PER на ковёр) → овальная вырезка (как element_embed «match») → вектор DINOv3-S, усреднённый по 8 положениям
#      (4 поворота × зеркало): сходство не зависит от поворота и зеркала.
#   3. Кластеризация по всем коврам (средняя связь, косинус, порог EC_THR); куча — ≥ 4 места на ≥ EC_MINRUGS коврах.
#   4. Итог уходит на сервер: POST /api/open/cluster_upload (по ключу) → в /m карточка «кучи». Копия — файл ec_clusters.json (Drive / /content).
#
# Запуск (Colab, T4):
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env CPS_KEY=<ключ из студии: пользователи → «ключ»> (можно всю ссылку с ?key=…)
#   %run /content/carpevo/cps/tools/colab/elements_cluster_colab.py
# Настройки (env): EC_REUSE=1 — не считать заново, взять ec_cands.npz с Drive (только кластеризация, секунды) · EC_MAXC=120 макс. размер кучи · EC_N=400 ковров в пробе · EC_SEED=1 · EC_EXCL=kerimov (префиксы id, которые пропустить — схемы на бумаге) · EC_TAU=0.70 · EC_MINREP=3 ·
#   EC_PER=14 · EC_THR=0.22 (расстояние в кластере; меньше — чище и мельче) · EC_MINRUGS=3 · EC_APPEND=0.
import os, sys, io, json, time, zipfile, math, random, urllib.request, urllib.parse
import numpy as np, cv2

URL = os.environ.get("CPS_URL", "https://studio.carpets.wiki").rstrip("/")
KEY = os.environ.get("CPS_KEY", "").strip()
if "key=" in KEY: KEY = urllib.parse.parse_qs(urllib.parse.urlparse(KEY).query).get("key", [""])[0] or KEY.split("key=")[-1]
N = int(os.environ.get("EC_N", "400")); SEED = int(os.environ.get("EC_SEED", "1")); TAU = float(os.environ.get("EC_TAU", "0.70"))
MINREP = int(os.environ.get("EC_MINREP", "3")); PER = int(os.environ.get("EC_PER", "14")); THR = float(os.environ.get("EC_THR", "0.22"))
MINRUGS = int(os.environ.get("EC_MINRUGS", "3")); EXCL = tuple(x for x in os.environ.get("EC_EXCL", "kerimov").split(",") if x)
APPEND = os.environ.get("EC_APPEND", "0") == "1"; LONG = 640
SIZES = ((3, 3), (4, 4), (6, 6), (8, 8))                                  # v12.51: только квадратные окна (полосы 3×8 давали куски каймы, а не узоры)
assert KEY, "задайте ключ: %env CPS_KEY=…"
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f} c]", *a, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if "__file__" in globals() else "/content/carpevo/cps")
import element_embed as EE

DRV = "/content/cps_xfind"
try:
    from google.colab import drive
    drive.mount("/content/drive"); DRV = "/content/drive/MyDrive/carpet-dna/cps_xfind"           # тот же кэш фото, что у ночного поиска
except Exception as e:
    log("Drive не подключён (", e, ") — кэш в /content")
RAW = os.path.join(DRV, "raw"); os.makedirs(RAW, exist_ok=True)

def get(path, **q):
    q["key"] = KEY
    with urllib.request.urlopen(URL + path + "?" + urllib.parse.urlencode(q), timeout=600) as r: return r.read()

# ---------- выборка ковров ----------
log("карточки…")
Z = zipfile.ZipFile(io.BytesIO(get("/api/open/cards.zip", since=0)))
R = {}
for n in Z.namelist():
    if n.startswith("rugs/") and n.endswith(".json"):
        r = json.loads(Z.read(n)); R[r["id"]] = r
man = json.loads(get("/api/open/manifest", since=0, live=1)).get("rugs") or []
mt = {x["id"]: float((x.get("t") or {}).get("raw") or 0) for x in man if x.get("id")}
pool = [rid for rid, r in R.items() if rid in mt and r.get("frame") == "whole" and not rid.startswith(EXCL)]
random.Random(SEED).shuffle(pool)
live = sorted(pool[:N]) if N > 0 else sorted(pool)
log(f"целых ковров {len(pool)}, берём {len(live)}")
idx_f = os.path.join(RAW, "_index.json")
try: IDX = json.load(open(idx_f))
except Exception: IDX = {}
need = [rid for rid in live if not os.path.exists(os.path.join(RAW, rid + ".jpg")) or IDX.get(rid, -1) < mt.get(rid, 0)]
log(f"качать фото {len(need)}")
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
    a = cv2.imread(os.path.join(RAW, rid + ".jpg"))
    if a is None: return None
    k = side / max(a.shape[:2])
    return cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else a

# ---------- сеть ----------
import torch, torch.nn.functional as F
try: import timm
except ImportError: os.system("pip -q install timm"); import timm
from safetensors.torch import load_file
DEV = "cuda" if torch.cuda.is_available() else "cpu"; log("устройство", DEV)
wf = "/content/dinov3_s.safetensors" if os.path.isdir("/content") else "/tmp/dinov3_s.safetensors"
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

def candidates(a):
    """→ [(box[x,y,w,h] в долях фото, балл, повторов)] — повторяющиеся на этом ковре места, без фона и без пересечений"""
    f = F.normalize(grid(a), dim=-1); gh, gw, C = f.shape
    mu = f.reshape(-1, C).mean(0); x = f.permute(2, 0, 1)[None]; out = []
    for sh, sw in SIZES:
        if sh > gh or sw > gw: continue
        P = F.avg_pool2d(x, (sh, sw), 1)[0]; ny, nx = P.shape[1:]
        V = (P - mu[:, None, None]).reshape(C, -1).T; nrm = V.norm(dim=1); Vn = F.normalize(V, dim=1); S = Vn @ Vn.T
        yy, xx = torch.meshgrid(torch.arange(ny, device=S.device), torch.arange(nx, device=S.device), indexing="ij"); yy, xx = yy.reshape(-1), xx.reshape(-1)
        far = ((yy[:, None] - yy[None]).abs() >= sh) | ((xx[:, None] - xx[None]).abs() >= sw)
        rep0 = ((S >= TAU) & far).sum(1).float()
        sc0 = nrm * rep0.clamp(max=60).sqrt()
        top = torch.argsort(-sc0)[:60]; med = nrm.median()
        for i in top.tolist():
            if nrm[i] < med or rep0[i] < MINREP: continue
            m = ((S[i] >= TAU) & far[i]).nonzero().reshape(-1).tolist(); m.sort(key=lambda j: -float(S[i, j])); sel = []
            for j in m:                                                        # настоящие повторы: между собой тоже не пересекаются
                if all(abs(yy[j] - yy[k]) >= sh or abs(xx[j] - xx[k]) >= sw for k in sel): sel.append(j)
            if len(sel) < MINREP: continue
            out.append(([xx[i].item() / gw, yy[i].item() / gh, sw / gw, sh / gh], float(nrm[i]) * min(len(sel), 10) * (sh * sw) ** 0.25, len(sel) + 1))
    out.sort(key=lambda c: -c[1]); keep = []
    for b, s, n in out:                                                        # NMS между размерами
        if all(_iou(b, k[0]) < 0.3 for k in keep): keep.append((b, s, n))
        if len(keep) >= PER: break
    return keep

def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])); i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)

@torch.no_grad()
def descr(crops):
    """вектор вырезки, усреднённый по 8 положениям (4 поворота × зеркало) → инвариантен к повороту и зеркалу"""
    outs = []
    for c in crops:
        vs = [np.rot90(c if not fl else c[:, ::-1], k) for fl in (0, 1) for k in range(4)]
        x = torch.from_numpy(np.stack([EE._prep(np.ascontiguousarray(v)) for v in vs])).to(DEV)
        f = NET.forward_features(x.half() if DEV == "cuda" else x).float()
        e = F.normalize(torch.cat([f[:, 0], f[:, NP:].mean(1)], 1), dim=1).mean(0)
        outs.append(F.normalize(e, dim=0).cpu().numpy())
    return np.stack(outs) if outs else np.zeros((0, 768), np.float32)

# ---------- шаги 1–2: кандидаты и векторы ----------
CF = os.path.join(DRV if os.path.isdir(DRV) else ".", "ec_cands.npz")
rows = []; vecs = []; skipped = 0; REUSE = os.environ.get("EC_REUSE") == "1" and os.path.exists(CF)
if REUSE:
    z = np.load(CF, allow_pickle=True); rows = json.loads(str(z["rows"])); vecs = list(z["V"]); live = sorted({r["rid"] for r in rows}); log(f"взяты сохранённые кандидаты: {len(rows)}")
for i, rid in enumerate([] if REUSE else live):
    a = img(rid, 1400)
    if a is None: skipped += 1; continue
    try: cs = candidates(a)
    except Exception as e: log("ковёр", rid, "пропущен:", e); skipped += 1; continue
    if not cs: continue
    v = descr([EE._crop(a, b, 112, 0.15, False, "ellipse") for b, s, n in cs])
    for (b, s, n), e in zip(cs, v): rows.append({"rid": rid, "box": [round(float(t), 5) for t in b], "s": round(float(s), 3), "rep": n}); vecs.append(e)
    if i % 25 == 0: log(f"  {i + 1}/{len(live)} ковров, кандидатов {len(rows)}")
V = np.stack(vecs) if vecs else np.zeros((0, 768), np.float32)
if not REUSE and len(rows): np.savez(CF, rows=json.dumps(rows), V=V)
log(f"кандидатов {len(rows)} на {len({r['rid'] for r in rows})} коврах (без фото/с ошибкой: {skipped})")
assert len(rows) >= 20, "слишком мало кандидатов — понизьте EC_TAU (например 0.6) или проверьте ковры"

# ---------- шаг 3: кучи ----------
from sklearn.cluster import AgglomerativeClustering
MAXN = 15000
if len(rows) > MAXN:
    pick = np.random.RandomState(SEED).choice(len(rows), MAXN, replace=False); pick.sort(); rows = [rows[i] for i in pick]; V = V[pick]; log("кандидатов больше", MAXN, "— взята выборка")
log("кластеры…")
def agg(ix, thr): return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=thr).fit_predict(V[ix])
MAXC = int(os.environ.get("EC_MAXC", "120"))                              # куча больше этого — смесь форм: делим строже, пока не станет меньше
lab = np.full(len(rows), -1); nxt = 0; stack = [(np.arange(len(rows)), THR, 0)]
while stack:
    ix, thr, dep = stack.pop(); l = agg(ix, thr) if len(ix) > 1 else np.zeros(len(ix), int)
    for c in np.unique(l):
        sub = ix[l == c]
        if len(sub) > MAXC and dep < 4 and thr > 0.04: stack.append((sub, thr * 0.7, dep + 1))
        else: lab[sub] = nxt; nxt += 1
cl = []
for c in np.unique(lab):
    ix = np.where(lab == c)[0]; rugs = {rows[i]["rid"] for i in ix}
    if len(ix) < 4 or len(rugs) < MINRUGS: continue
    cen = V[ix].mean(0); cen /= np.linalg.norm(cen) + 1e-9; d = V[ix] @ cen; order = ix[np.argsort(-d)]
    reps, seen = [], {}
    for i in order:                                                           # представители: ближе к центру, не больше 2 с одного ковра
        if seen.get(rows[i]["rid"], 0) >= 2: continue
        seen[rows[i]["rid"]] = seen.get(rows[i]["rid"], 0) + 1; reps.append(rows[i])
        if len(reps) >= 16: break
    more = [rows[i] for i in order if rows[i] not in reps][:200]
    cl.append({"id": f"c{int(time.time()) % 100000}_{int(c)}", "n": int(len(ix)), "rugs": len(rugs), "coh": round(float(d.mean()), 3),
               "reps": [{"rid": r["rid"], "box": r["box"], "s": r["s"]} for r in reps], "more": [{"rid": r["rid"], "box": r["box"]} for r in more]})
cl.sort(key=lambda c: (-c["coh"], -c["rugs"]))      # сначала самые плотные (вероятнее одна форма)
log(f"куч {len(cl)} (из {len(np.unique(lab))} кластеров); крупнейшие: {[(c['n'], c['rugs']) for c in cl[:8]]}")
out = {"v": 1, "t": int(time.time()), "append": APPEND, "params": {"n": len(live), "tau": TAU, "minrep": MINREP, "per": PER, "thr": THR, "minrugs": MINRUGS, "cands": len(rows)}, "clusters": cl}
json.dump(out, open(os.path.join(DRV if os.path.isdir(DRV) else ".", "ec_clusters.json"), "w"), ensure_ascii=False)

# ---------- шаг 4: на сервер ----------
if os.environ.get("EC_NOUPLOAD"): log("EC_NOUPLOAD — на сервер не отправляем"); sys.exit(0)
req = urllib.request.Request(URL + "/api/open/cluster_upload?key=" + urllib.parse.quote(KEY), data=json.dumps(out).encode(), headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=600) as r: log("сервер принял:", r.read().decode()[:200])
except Exception as e:
    log("НЕ ОТПРАВИЛОСЬ:", e, "— файл ec_clusters.json сохранён, повторите отправку")
print("\nИТОГ:", json.dumps({"кучи": len(cl), "кандидатов": len(rows), "ковров": len(live), "время_с": int(time.time() - T0)}, ensure_ascii=False))
