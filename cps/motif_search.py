# motif_search.py — v11.4: «где ещё встречается этот узор» — поиск элемента по всем коврам каталога.
#
# Как устроено.
# 1. Индекс. Для каждого фото ковра один раз считается «карта признаков»: фото в оттенках серого (узор важнее цвета —
#    один и тот же элемент ткут разными красками), длинная сторона 518 px, сеть DINOv2 (та же dinov2_vits14, что в Colab)
#    даёт по признаку на каждый квадрат 14×14 px — сетка примерно 37×26. Признаки сжимаются до 64 чисел (PCA, считается
#    один раз по первым фото) и хранятся в data/motif/grid/<id>.npy (≈120 КБ на ковёр). Фото, которые человек поменял,
#    пересчитываются; разворот, разделённый на части, не индексируется (индексируются части).
# 2. Запрос. Рамка элемента → средний признак квадратов внутри рамки минус половина среднего по всему ковру
#    (убирает общий «стиль» фото: свет, ворс, цвет фона). Несколько образцов одного узора (подтверждённые рамки
#    с тем же узором словаря или той же схемой) → их среднее — «портрет» узора; чем больше ✓, тем он точнее.
# 3. Поиск. На каждом ковре перебираются окна той же формы, что рамка, в пяти масштабах (×0.6…×1.6 — один и тот же
#    элемент на маленьком и большом ковре занимает разную долю фото), сходство — косинус. С каждого ковра берутся
#    до 3 лучших окон, ковры сортируются по лучшему.
# 4. Ответы человека (✓ это он / ✗ не он) пишутся в data/motif/labels.jsonl — обучающий набор; ✓ сразу добавляет на
#    тот ковёр подтверждённую рамку с тем же узором и схемой.
#
# Ограничения (честно): повёрнутый на 90° элемент (боковые каймы) DINOv2 считает другим — ищется в том положении, в каком
# обведён; очень мелкие элементы (меньше 1/25 ширины ковра) на сетке 37 квадратов различаются плохо.
#
# Без torch (Mac без сети, автотесты) работает упрощённый извлекатель признаков (CPS_MOTIF_FAKE=1): серые квадраты 14×14
# со случайной проекцией — годится проверить механику, не качество.
import os, json, time, threading, math
import numpy as np, cv2

P = 14
LONG = int(os.environ.get("CPS_MOTIF_SIDE") or 518)
D = 64
BETA = float(os.environ.get("CPS_MOTIF_BETA") or 0.5)
SCALES = (0.6, 0.78, 1.0, 1.28, 1.6)

state = {"running": False, "done": 0, "total": 0, "skipped": 0, "errors": 0, "error": "", "stage": "", "started": 0, "finished": 0,
         "model": "", "last_result": ""}
_cfg = {}
_lock = threading.Lock()          # индекс (meta) и загрузка модели
_model = {"net": None, "torch": None, "pca": None, "R": None}
_stop = {"flag": False}


def fake():
    return bool(os.environ.get("CPS_MOTIF_FAKE"))


def _dir(*p):
    d = os.path.join(_cfg["data"], "motif", *p)
    return d


def configure(data, img_path, rugs_dir):
    _cfg.update(data=data, img_path=img_path, rugs_dir=rugs_dir)
    os.makedirs(_dir("grid"), exist_ok=True)


# ---------------------------------------------------------------- метаданные индекса
def _meta():
    p = _dir("index.json")
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return {}


def _meta_save(m):
    p = _dir("index.json"); t = p + ".tmp"
    with open(t, "w", encoding="utf-8") as f: json.dump(m, f)
    os.replace(t, p)


def available():
    """Можно ли считать признаки: torch установлен (модель скачается при первом расчёте) или включён упрощённый режим."""
    if fake(): return True
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- признаки
def _prep(bgr):
    """серое фото, длинная сторона LONG, стороны кратны 14 (квадрат сети)"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = g.shape[:2]; s = LONG / max(h, w)
    W, H = max(4 * P, int(round(w * s / P)) * P), max(4 * P, int(round(h * s / P)) * P)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def _fake_tokens(g):
    """упрощённые признаки без сети (тесты, Mac без torch): направления контуров в квадрате 14×14 (8 направлений × 4 четверти)"""
    H, W = g.shape; gh, gw = H // P, W // P
    x = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0); gy = cv2.Sobel(x, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy); ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 8
    feats = []
    for k in range(8):
        w = np.clip(1 - np.abs(((ang - k + 4) % 8) - 4), 0, 1) * mag
        w = cv2.GaussianBlur(w, (0, 0), 3.0)
        h2 = P // 2
        cell = w[:gh * P, :gw * P].reshape(gh * 2, h2, gw * 2, h2).mean((1, 3))     # 4 четверти квадрата
        feats.append(cell.reshape(gh, 2, gw, 2).transpose(0, 2, 1, 3).reshape(gh, gw, 4))
    f = np.concatenate(feats, -1)
    f = np.sqrt(f / (f.sum(-1, keepdims=True) + 1e-3 * f.mean() + 1e-6))
    return f.astype(np.float32)


def _load_net():
    if _model["net"] is not None: return _model["net"]
    import torch
    os.environ.setdefault("TORCH_HOME", os.path.join(_cfg["data"], "torch"))     # веса один раз скачиваются в том данных
    try: torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))               # половина ядер: студия остаётся отзывчивой
    except Exception: pass
    state["model"] = "загружаю DINOv2 (первый раз — скачивание ≈90 МБ)…"
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval()
    _model.update(net=net, torch=torch); state["model"] = "DINOv2 ViT-S/14"
    return net


def _raw_tokens(g):
    """[gh, gw, 384] признаки DINOv2"""
    net = _load_net(); torch = _model["torch"]
    x = np.repeat((g.astype(np.float32) / 255.0)[None], 3, 0)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    x = (x - mean) / std
    with torch.no_grad():
        t = net.forward_features(torch.from_numpy(x[None]))["x_norm_patchtokens"][0].numpy()
    gh, gw = g.shape[0] // P, g.shape[1] // P
    return t.reshape(gh, gw, -1)


def _pca():
    if _model["pca"] is None:
        p = _dir("pca.npz")
        if os.path.exists(p):
            z = np.load(p); _model["pca"] = (z["mean"].astype(np.float32), z["comp"].astype(np.float32))
    return _model["pca"]


def _fit_pca(samples):
    X = np.concatenate([s.reshape(-1, s.shape[-1]) for s in samples]).astype(np.float32)
    rng = np.random.default_rng(1)
    if len(X) > 60000: X = X[rng.choice(len(X), 60000, replace=False)]
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    comp = Vt[:D]
    np.savez(_dir("pca.npz"), mean=mu, comp=comp)
    _model["pca"] = (mu, comp)


def _project(t):
    mu, comp = _pca()
    return (t - mu) @ comp.T


def grid_of_image(bgr, raw=False):
    g = _prep(bgr)
    if fake(): return _fake_tokens(g)
    t = _raw_tokens(g)
    return t if raw else _project(t)


# ---------------------------------------------------------------- индекс
_rl = {"t": 0, "v": []}


def _rug_list(fresh=False):
    """(id, путь к фото, mtime) всех ковров, которые нужно индексировать (список кэшируется на 30 с — статус опрашивается часто)"""
    if not fresh and time.time() - _rl["t"] < 30: return _rl["v"]
    out = []; d = _cfg["rugs_dir"]
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"): continue
        rid = fn[:-5]
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f: r = json.load(f)
        except Exception: continue
        if r.get("split_into"): continue                      # разворот, разделённый на части: ищем по частям
        p = _cfg["img_path"](rid)
        if not p: continue
        out.append((rid, p, round(os.path.getmtime(p), 1)))
    _rl.update(t=time.time(), v=out)
    return out


def todo():
    m = _meta(); rs = _rug_list(fresh=True)
    return [x for x in rs if (m.get(x[0]) or {}).get("mt") != x[2] or not os.path.exists(_dir("grid", x[0] + ".npy"))], rs


def status():
    m = _meta(); rs = _rug_list(); ids = {x[0] for x in rs}
    fresh = sum(1 for rid, p, mt in rs if (m.get(rid) or {}).get("mt") == mt)
    return {**state, "indexed": fresh, "total_rugs": len(rs), "stale": len(rs) - fresh, "available": available(),
            "kind": "упрощённый (тест)" if fake() else "DINOv2", "orphans": len(set(m) - ids)}


def _save_grid(rid, grid, mt):
    np.save(_dir("grid", rid + ".npy"), grid.astype(np.float16))
    with _lock:
        m = _meta(); m[rid] = {"mt": mt, "g": [int(grid.shape[0]), int(grid.shape[1])], "at": int(time.time())}; _meta_save(m)


def ensure(rid):
    """признаки одного ковра (посчитать сейчас, если их нет или фото изменилось)"""
    p = _cfg["img_path"](rid)
    if not p: return None
    mt = round(os.path.getmtime(p), 1); m = _meta().get(rid) or {}
    if m.get("mt") == mt and os.path.exists(_dir("grid", rid + ".npy")): return grid(rid)
    a = cv2.imread(p)
    if a is None: return None
    if not fake() and _pca() is None: return None            # индекс ещё ни разу не строился — PCA нет
    G = grid_of_image(a); _save_grid(rid, G, mt)
    return G.astype(np.float32)


def grid(rid):
    """признаки ковра с диска (в памяти не держим: 4000 ковров ≈ 500 МБ; файлы и так в кэше системы)"""
    p = _dir("grid", rid + ".npy")
    if not os.path.exists(p): return None
    try: return np.load(p).astype(np.float32)
    except Exception: return None


def build(limit=None, on_done=None):
    """Посчитать признаки всех новых и изменённых фото. Запускается в отдельном потоке."""
    if state["running"]: return False
    _stop["flag"] = False
    state.update(running=True, done=0, skipped=0, errors=0, error="", stage="список фото", started=int(time.time()), finished=0)

    def run():
        try:
            need, rs = todo()
            if limit: need = need[:limit]
            state.update(total=len(need), stage="признаки фото")
            m = _meta(); ids = {x[0] for x in rs}
            gone = [k for k in m if k not in ids]                   # ковёр удалён или разделён — убрать из индекса
            for k in gone:
                try: os.remove(_dir("grid", k + ".npy"))
                except OSError: pass
            if gone:
                with _lock:
                    m = _meta()
                    for k in gone: m.pop(k, None)
                    _meta_save(m)
            if not fake() and _pca() is None and need:
                state["stage"] = "первые 40 фото: подбор сжатия признаков"; first = []
                for rid, p, mt in need[:40]:
                    a = cv2.imread(p)
                    if a is None: continue
                    first.append((rid, mt, grid_of_image(a, raw=True)))
                _fit_pca([t for _, _, t in first])
                for rid, mt, t in first:
                    _save_grid(rid, _project(t), mt); state["done"] += 1
                need = need[40:]
                state["stage"] = "признаки фото"
            for rid, p, mt in need:
                if _stop["flag"]: break
                try:
                    a = cv2.imread(p)
                    if a is None: state["skipped"] += 1; continue
                    _save_grid(rid, grid_of_image(a), mt); state["done"] += 1
                except Exception as e:
                    state["errors"] += 1; state["error"] = f"{rid}: {e}"[:300]
                time.sleep(0.01)
            res = f"посчитано фото: {state['done']}" + (f", ошибок {state['errors']}" if state["errors"] else "") + (" (остановлено)" if _stop["flag"] else "")
            state["last_result"] = res
            if on_done:
                try: on_done(res)
                except Exception: pass
        except Exception as e:
            state["error"] = str(e)[:300]; state["last_result"] = "ошибка: " + state["error"]
        finally:
            state.update(running=False, stage="", finished=int(time.time()))
    threading.Thread(target=run, daemon=True).start()
    return True


def stop(): _stop["flag"] = True


# ---------------------------------------------------------------- запрос и поиск
def _tok_box(G, box):
    gh, gw = G.shape[:2]; x, y, w, h = box
    x0 = int(math.floor(x * gw + 0.25)); y0 = int(math.floor(y * gh + 0.25))
    x1 = int(math.ceil((x + w) * gw - 0.25)); y1 = int(math.ceil((y + h) * gh - 0.25))
    x0 = max(0, min(gw - 1, x0)); y0 = max(0, min(gh - 1, y0)); x1 = max(x0 + 1, min(gw, x1)); y1 = max(y0 + 1, min(gh, y1))
    return x0, y0, x1, y1


QW = 0.6       # вес четвертей рамки рядом с целой: «что в рамке» + «где внутри рамки»


def _integral(G):
    gh, gw, d = G.shape
    S = np.zeros((gh + 1, gw + 1, d), np.float32); S[1:, 1:] = G.cumsum(0).cumsum(1)
    return S


def _wsum(S, h, w):
    """суммы всех окон h×w: [gh-h+1, gw-w+1, d]"""
    return S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w]


def _feat(S, sh, sw, gmean):
    """признаки всех окон sh×sw: целое окно и 4 четверти, каждое — среднее минус BETA×среднее ковра; [ny, nx, 5d], длина 1"""
    gh, gw = S.shape[0] - 1, S.shape[1] - 1
    ny, nx = gh - sh + 1, gw - sw + 1
    ht, wl = max(1, sh // 2), max(1, sw // 2); hb, wr = max(1, sh - ht), max(1, sw - wl)
    oy, ox = min(ht, sh - hb), min(wl, sw - wr)               # окно в 1 квадрат: «четверти» = само окно
    parts = [_wsum(S, sh, sw)[:ny, :nx] / (sh * sw)]
    for (hh, ww, dy, dx) in ((ht, wl, 0, 0), (ht, wr, 0, ox), (hb, wl, oy, 0), (hb, wr, oy, ox)):
        parts.append(QW * _wsum(S, hh, ww)[dy:dy + ny, dx:dx + nx] / (hh * ww))
    V = np.concatenate([x - (BETA * gmean * (1 if i == 0 else QW)) for i, x in enumerate(parts)], -1)
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-8)


def vec(G, box):
    """признак рамки (целое + четверти); размер рамки в квадратах сетки"""
    x0, y0, x1, y1 = _tok_box(G, box)
    S = _integral(G); gmean = S[-1, -1] / (G.shape[0] * G.shape[1])
    V = _feat(S[y0:y1 + 1, x0:x1 + 1] - S[y0:y1 + 1, x0:x0 + 1] - S[y0:y0 + 1, x0:x1 + 1] + S[y0, x0], y1 - y0, x1 - x0, gmean)
    return V[0, 0], (x1 - x0, y1 - y0)


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]; bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    i = iw * ih; u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0


def search_grid(G, q, qw, qh, per_rug=3):
    """лучшие окна на одном ковре: [(score, [x,y,w,h] в долях)]"""
    gh, gw, d = G.shape
    S = _integral(G); gmean = S[-1, -1] / (gh * gw)
    sizes = set()
    for k in SCALES:
        sizes.add((max(1, min(gw, int(round(qw * k)))), max(1, min(gh, int(round(qh * k))))))
    cand = []
    for sw, sh in sizes:
        sc = _feat(S, sh, sw, gmean) @ q
        flat = sc.ravel(); k = min(12, flat.size)
        for i in np.argpartition(-flat, k - 1)[:k]:
            yy, xx = divmod(int(i), sc.shape[1])
            cand.append((float(flat[i]), [xx / gw, yy / gh, sw / gw, sh / gh]))
    cand.sort(key=lambda c: -c[0]); out = []
    for s, b in cand:
        if all(_iou(b, o[1]) < 0.3 for o in out): out.append((s, b))
        if len(out) >= per_rug: break
    return out


def search(q, qsize, exclude=(), only=None, limit=60, per_rug=3, rel_size=None):
    """q — признак, qsize — (ширина, высота) в квадратах сетки исходного ковра. Ответ: ковры по убыванию сходства."""
    m = _meta(); res = []; ex = set(exclude)
    for rid in m:
        if rid in ex or (only is not None and rid not in only): continue
        G = grid(rid)
        if G is None: continue
        qw, qh = qsize
        if rel_size:                                            # размер окна задан долями фото (запрос по нескольким образцам)
            qw, qh = max(1, rel_size[0] * G.shape[1]), max(1, rel_size[1] * G.shape[0])
        hits = search_grid(G, q, qw, qh, per_rug)
        if hits: res.append({"rid": rid, "score": round(hits[0][0], 4), "boxes": [{"box": [round(v, 4) for v in b], "score": round(s, 4)} for s, b in hits]})
    res.sort(key=lambda r: -r["score"])
    return res[:limit]


# ---------------------------------------------------------------- ответы человека (обучающий набор)
def label_log(rec):
    p = _dir("labels.jsonl")
    with _lock:
        with open(p, "a", encoding="utf-8") as f: f.write(json.dumps({**rec, "t": int(time.time())}, ensure_ascii=False) + "\n")


def labels(key=None):
    p = _dir("labels.jsonl"); out = []
    if not os.path.exists(p): return out
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try: r = json.loads(ln)
            except Exception: continue
            if key is None or r.get("key") == key: out.append(r)
    return out


def label_stats():
    L = labels(); return {"total": len(L), "yes": sum(1 for r in L if r.get("ok")), "no": sum(1 for r in L if not r.get("ok"))}
