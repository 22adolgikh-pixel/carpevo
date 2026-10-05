# cnn_colab.py — свёрточная сеть для клеток схем (v10.15). Запускается в Google Colab (GPU) одной ячейкой, см. README рядом.
# Что делает: берёт данные CPS с GitHub (ветка cps-data), строит «выпрямленные» кропы по вашей итоговой сетке,
# учит сеть с аугментацией, сравнивает её с порогами и с нынешней маленькой моделью (MLP) на ОДНИХ И ТЕХ ЖЕ листах
# (3 фолда по листам: рисунки одного листа не делятся между обучением и проверкой) и сохраняет итоговую сеть.
# Результат: /content/cnn_result/ (summary.json, net_all.pt) и копия в Google Drive carpet-dna/cps_cnn/ (если Drive подключён).
import os, sys, json, glob, time, subprocess, shutil, pickle
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
ITERS = int(os.environ.get("CPS_ITERS", "4000")); LIMIT = int(os.environ.get("CPS_LIMIT", "0")); BS = int(os.environ.get("CPS_BS", "16"))
OUT = os.environ.get("CPS_OUT", "/content/cnn_result"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("CPS_DATA")
if not DATA:
    DATA = "/content/cpsdata/data"
    if not os.path.isdir(DATA + "/work"):
        print("скачиваю данные (ветка cps-data, ~0.5 ГБ)…", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", "-b", "cps-data", "https://github.com/22adolgikh-pixel/carpevo.git", "/content/cpsdata"], check=True)
os.environ["CPS_DATA"] = DATA
sys.path.insert(0, CODE)
import autogrid as A, learn, hard
import torch, torch.nn as nn, torch.nn.functional as F
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("устройство:", dev, "| итераций на фолд:", ITERS, flush=True)
if dev == "cpu" and ITERS > 200: print("ВНИМАНИЕ: нет GPU (Среда выполнения → Сменить тип → T4 GPU), на процессоре будет очень долго", flush=True)
torch.manual_seed(0); np.random.seed(0)
S, PAD = 12, 2

# ---------------------------------------------------------------- данные
def build():
    t0 = time.time(); ds = []
    ex = {e[0]: e for e in learn.load_examples(DATA + "/work", DATA + "/crops")}   # признаки MLP, пороги, маски проверенных кусков
    print("рисунков для MLP:", len(ex), "| %.0f c" % (time.time() - t0), flush=True)
    for fid, e in ex.items():
        w = json.load(open(f"{DATA}/work/{fid}.json")); g = e[3]; mask = e[6]
        gr, m = w["grid"], w["matrix"]
        try:
            q = np.float32(w["quad"]); W, H = A.quad_size([list(p) for p in q])
            x0, y0 = m.get("origin", [0, 0])
            cols = max(int(gr.get("cols") or 0), x0 + int(m["w"])); rows = max(int(gr.get("rows") or 0), y0 + int(m["h"]))   # у ручных сеток cols/rows бывают устаревшими
            Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), q)
            ox, oy, pw, ph = gr.get("ox", 0), gr.get("oy", 0), gr["pw"], gr["ph"]
            T = np.float64([[pw / S, 0, ox - PAD * pw], [0, ph / S, oy - PAD * ph], [0, 0, 1]])
            img = cv2.warpPerspective(g, Hm.astype(np.float64) @ T, ((cols + 2 * PAD) * S, (rows + 2 * PAD) * S),
                                      flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        except Exception as ex_:
            print("пропуск", fid, ex_); continue
        U = e[2]; x0, y0 = m.get("origin", [0, 0]); h, wd = U.shape
        Y = np.zeros((rows, cols), np.int8) if mask is None else np.full((rows, cols), -1, np.int8)
        hh, ww = min(h, rows - y0), min(wd, cols - x0)
        if mask is None: Y[y0:y0 + hh, x0:x0 + ww] = U[:hh, :ww]
        else:
            sub = np.where(mask[:hh, :ww], U[:hh, :ww], -1); Y[y0:y0 + hh, x0:x0 + ww] = sub
        base = learn.baseline_labels(g, e[4], e[5])                              # пороги в координатах матрицы
        p90 = float(max(np.percentile(img, 90), 1))
        ds.append(dict(id=fid, sheet=w.get("sheet") or fid.split("_f")[0], done=mask is None, img=img.astype(np.uint8), p90=p90,
                       Y=Y, org=(x0, y0), mh=h, mw=wd, U=U, base=base, X=e[1], mask=mask))
        if LIMIT and len(ds) >= LIMIT: break
    print("датасет: %d рисунков (принятых %d) | %.0f c" % (len(ds), sum(d["done"] for d in ds), time.time() - t0), flush=True)
    return ds

# ---------------------------------------------------------------- сеть
class Net(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        s.pix = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                              nn.Conv2d(16, c, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                              nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.ReLU(), nn.AvgPool2d(3))   # 12 px → 1 клетка
        s.ctx = nn.ModuleList([nn.Conv2d(2 * c, 2 * c, 3, padding=r, dilation=r) for r in (1, 2, 4, 8, 1)])
        s.bn = nn.ModuleList([nn.BatchNorm2d(2 * c) for _ in s.ctx])
        s.head = nn.Conv2d(2 * c, 3, 1)
    def forward(s, x):
        h = s.pix(x)
        for cv, bn in zip(s.ctx, s.bn): h = h + F.relu(bn(cv(h)))
        return s.head(h)

N = 24      # размер патча в клетках

def patch(d):
    rows, cols = d["Y"].shape; H, W = rows + 2 * PAD, cols + 2 * PAD
    Yp = np.full((H, W), -1, np.int64); Yp[PAD:PAD + rows, PAD:PAD + cols] = d["Y"]
    # патч чаще берём там, где есть метки не-фон (иначе на пустых краях сеть учится «всё фон»)
    ys, xs = np.nonzero(Yp > 0)
    if len(ys) and np.random.rand() < .7:
        k = np.random.randint(len(ys)); i0 = int(np.clip(ys[k] - np.random.randint(N), 0, max(0, H - N))); j0 = int(np.clip(xs[k] - np.random.randint(N), 0, max(0, W - N)))
    else:
        i0 = np.random.randint(0, max(1, H - N + 1)); j0 = np.random.randint(0, max(1, W - N + 1))
    hh, ww = min(N, H - i0), min(N, W - j0)
    x = np.full((N * S, N * S), d["p90"], np.float32); y = np.full((N, N), -1, np.int64)
    x[:hh * S, :ww * S] = d["img"][i0 * S:(i0 + hh) * S, j0 * S:(j0 + ww) * S]; y[:hh, :ww] = Yp[i0:i0 + hh, j0:j0 + ww]
    x = x / d["p90"]
    if np.random.rand() < .5: x, y = x[:, ::-1], y[:, ::-1]
    if np.random.rand() < .5: x, y = x[::-1], y[::-1]
    if np.random.rand() < .5: x, y = x.T, y.T                                          # поворот на 90° (клетки квадратные)
    x = x.copy(); y = y.copy()
    if np.random.rand() < .5: x = x ** np.random.uniform(.7, 1.4)                      # гамма: разная печать/засветка
    x = x * np.random.uniform(.85, 1.15) + np.random.uniform(-.08, .08)               # яркость/контраст
    if np.random.rand() < .3: x = cv2.GaussianBlur(x, (0, 0), np.random.uniform(.5, 1.6))   # размытие скана
    if np.random.rand() < .3: x = x + np.random.normal(0, np.random.uniform(.01, .05), x.shape).astype(np.float32)
    return x.astype(np.float32), y

def predict(net, d):
    net.eval(); x = torch.from_numpy(d["img"].astype(np.float32) / d["p90"])[None, None].to(dev)
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        lo = net(x)[0].float()
    rows, cols = d["Y"].shape; pr = F.softmax(lo, 0)[:, PAD:PAD + rows, PAD:PAD + cols].cpu().numpy()
    x0, y0 = d["org"]; return pr[:, y0:y0 + d["mh"], x0:x0 + d["mw"]].argmax(0).astype(np.uint8)    # в координатах матрицы

def train(tr, iters=ITERS, bs=BS, tag=""):
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=iters); scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    wts = torch.tensor([1.0, 2.0, 2.0], device=dev); t0 = time.time()
    for it in range(iters):
        net.train(); b = [patch(tr[np.random.randint(len(tr))]) for _ in range(bs)]
        x = torch.from_numpy(np.stack([a for a, _ in b]))[:, None].to(dev); y = torch.from_numpy(np.stack([c for _, c in b])).to(dev)
        with torch.autocast(dev, enabled=dev == "cuda"):
            loss = F.cross_entropy(net(x).float(), y, weight=wts, ignore_index=-1)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        if it % max(1, iters // 8) == 0: print(f"  {tag} шаг {it}/{iters} потеря {loss.item():.3f} · {time.time() - t0:.0f} c", flush=True)
    return net

# ---------------------------------------------------------------- оценка
def tiles_stat(truth, base, other):
    """Как в «трудных местах»: куски 10×10, где другой метод расходится с порогами; остаток ошибок порогов, если эти куски исправлены."""
    D = base != other; E = base != truth; h, w = D.shape; n = cov = tot = 0; resid = 0
    keep = np.ones_like(E)
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            if D[y:y + 10, x:x + 10].any(): n += 1; keep[y:y + 10, x:x + 10] = False
    tot = int(E.sum()); resid = int((E & keep).sum())
    return n, tot, resid

def evaluate(ds, folds, K=3):
    acc = [d for d in ds if d["done"]]
    lost = {d["id"] for d in acc if (d["U"] == 2).sum() == 0 and (d["base"] == 2).sum() > .10 * max(1, (d["base"] > 0).sum())}   # потерянное серое в эталоне — не считаем
    sheets = sorted({d["sheet"] for d in ds}); rng = np.random.default_rng(1); rng.shuffle(sheets); fold = {s: i % K for i, s in enumerate(sheets)}
    res = {"cnn": [], "mlp": [], "thr": [], "ink": []}; per = []
    for f in range(K):
        tr = [d for d in ds if fold[d["sheet"]] != f]; te = [d for d in acc if fold[d["sheet"]] == f and d["id"] not in lost]
        print(f"фолд {f + 1}/{K}: учим на {len(tr)}, проверяем на {len(te)}", flush=True)
        net = train(tr, tag=f"фолд {f + 1}")
        X, y = learn._xy([(d["id"], d["X"], d["U"], None, None, None, d["mask"]) for d in tr]); P = learn.fit(X, y, class_w=np.array([1., 2., 1.5]))
        for d in te:
            U = d["U"]; pc = predict(net, d); pm = learn.classify_cells(P, d["X"]); pb = d["base"]
            ink = max(1, int(((pc > 0) | (U > 0)).sum()))
            res["cnn"].append(int((pc != U).sum())); res["mlp"].append(int((pm != U).sum())); res["thr"].append(int((pb != U).sum())); res["ink"].append(int(((U > 0) | (pb > 0)).sum()))
            per.append(dict(id=d["id"], thr=tiles_stat(U, pb, pb), mlp=tiles_stat(U, pb, pm), cnn=tiles_stat(U, pb, pc),
                            cnn_prim=tiles_stat(U, pc, pb), cnn_prim_mlp=tiles_stat(U, pc, pm),
                            cnn_prim_any=tiles_stat(U, pc, np.where(pc != pb, pb, pm)), ink=res["ink"][-1]))
    I = max(1, sum(res["ink"])); out = {k: round(100 * sum(res[k]) / I, 2) for k in ("cnn", "mlp", "thr")}
    out["n_eval"] = len(per)
    for k in ("mlp", "cnn", "cnn_prim", "cnn_prim_mlp", "cnn_prim_any"):
        n = np.array([p[k][0] for p in per]); tot = sum(p[k][1] for p in per); rs = np.array([p[k][2] for p in per])
        out[f"tiles_{k}"] = dict(median_tiles=float(np.median(n)), le10=int((n <= 10).sum()), errors_in_tiles_pct=round(100 * (1 - rs.sum() / max(1, tot)), 1),
                                 residual_gt2_figs_pct=round(100 * float((rs > 2).mean()), 1))
    return out

if __name__ == "__main__":
    t0 = time.time(); ds = build()
    if len(ds) < 10: sys.exit("слишком мало рисунков")
    r = evaluate(ds, None); r["minutes"] = round((time.time() - t0) / 60, 1); r["iters"] = ITERS; r["device"] = dev
    json.dump(r, open(OUT + "/summary.json", "w"), ensure_ascii=False, indent=1)
    print("\n=== ИТОГ (ошибка по закрашенным клеткам, листы не пересекаются) ===")
    print(f"пороги {r['thr']}% | маленькая модель (MLP) {r['mlp']}% | СЕТЬ {r['cnn']}%  (рисунков в проверке: {r['n_eval']})")
    print("(остаток = ошибки итога после исправления спорных кусков; итог = пороги, кроме строк «итог=сеть»)")
    for k, nm in (("tiles_mlp", "итог=пороги, спор с MLP"), ("tiles_cnn", "итог=пороги, спор с сетью"), ("tiles_cnn_prim", "итог=сеть, спор с порогами"),
                  ("tiles_cnn_prim_mlp", "итог=сеть, спор с MLP"), ("tiles_cnn_prim_any", "итог=сеть, спор с порогами ИЛИ MLP")):
        t = r[k]; print(f"{nm}: медиана {t['median_tiles']:.0f} на рисунок, ≤10 кусков у {t['le10']} рис., в кусках {t['errors_in_tiles_pct']}% ошибок итога, остаток >2 клеток у {t['residual_gt2_figs_pct']}% рисунков")
    print("\nобучаю итоговую сеть на всех данных…", flush=True)
    net = train(ds, tag="итог"); torch.save(net.state_dict(), OUT + "/net_all.pt")
    json.dump(r, open(OUT + "/summary.json", "w"), ensure_ascii=False, indent=1)
    try:
        from google.colab import files
        for fn in ("net_all.pt", "summary.json"): files.download(OUT + "/" + fn)      # скачать в браузер (если запрос разрешений — «Разрешить»)
    except Exception as e:
        print("скачивание в браузер не вышло:", e)
    try:
        from google.colab import drive
        drive.mount("/content/drive"); dst = "/content/drive/MyDrive/carpet-dna/cps_cnn"; os.makedirs(dst, exist_ok=True)
        for fn in ("summary.json", "net_all.pt"): shutil.copy(OUT + "/" + fn, dst + "/" + fn)
        print("скопировано на Google Drive:", dst)
    except Exception as e:
        print("Drive не подключён:", e, "— файлы лежат в", OUT)
    print("\nГОТОВО за %.1f мин. Пришлите мне текст блока «ИТОГ» выше (или скажите, что файлы на Drive)." % ((time.time() - t0) / 60))
