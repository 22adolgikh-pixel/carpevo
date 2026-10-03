# learn.py — обучаемый классификатор клеток ЧБ-рисунков на ПРАВКАХ ПОЛЬЗОВАТЕЛЯ.
# Обучающие примеры — готовые (done) ЧБ-рисунки: кроп + сетка (quad/grid) + итоговая матрица 0/1/2.
# Признаки клетки — окно 5×5 нормированной «тёмности» t (как в autogrid.classify) + мин/разброс яркости внутри клетки.
# Модель — маленькая нейросеть (numpy, без sklearn): data/cell_model.json. autogrid.auto_figure использует её, если файл есть.
#
# Обучение (внутри контейнера, где лежат кропы):
#   docker exec deploy-cps-1 python learn.py train          # обучить на всех done-ЧБ, показать качество (кросс-проверка по рисункам)
#   docker exec deploy-cps-1 python learn.py eval           # только качество: модель против порогов
import os, sys, json
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CPS_DATA", os.path.join(HERE, "data"))
MODEL_PATH = os.path.join(DATA, "cell_model.json")
NCLS = 3


# ---------------------------------------------------------------- признаки
def _norm_t(g, C, pitch):
    """t — как в autogrid.classify (фон=1, самый тёмный=0), плюс bg/dark."""
    import autogrid
    v = autogrid.cell_means(g, C, pitch)
    _, info = autogrid.classify(v)
    return info["t"], v, info


def cell_features(g, C, pitch):
    """→ (rows, cols, F). g — серое изображение кропа, C — центры клеток (rows, cols, 2)."""
    t, v, info = _norm_t(g, C, pitch)
    rows, cols = t.shape
    tp = np.pad(t, 2, mode="constant", constant_values=1.0)
    win = np.stack([tp[dy:dy + rows, dx:dx + cols] for dy in range(5) for dx in range(5)], -1)
    # внутри клетки: мин и разброс на полном разрешении (тонкая линия 1px теряется в усреднении)
    span = max(info["bg"] - info["dark"], 25.0)
    r = max(1.0, pitch * 0.35)
    mn = np.full((rows, cols), 255.0, np.float32); acc = np.zeros((rows, cols), np.float32); acc2 = np.zeros_like(acc); n = 0
    for dy in (-r, 0, r):
        for dx in (-r, 0, r):
            s = cv2.remap(g.astype(np.float32), (C[..., 0] + dx).astype(np.float32), (C[..., 1] + dy).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mn = np.minimum(mn, s); acc += s; acc2 += s * s; n += 1
    mean = acc / n; sd = np.sqrt(np.maximum(acc2 / n - mean ** 2, 0))
    extra = np.stack([(mn - info["dark"]) / span, sd / span], -1)
    return np.concatenate([win, extra], -1).astype(np.float32)


# ---------------------------------------------------------------- модель
def _forward(P, X):
    H = np.tanh(X @ P["W1"] + P["b1"]); Z = H @ P["W2"] + P["b2"]
    Z -= Z.max(1, keepdims=True); E = np.exp(Z); return H, E / E.sum(1, keepdims=True)


def fit(X, y, hidden=24, epochs=60, lr=3e-3, seed=0, class_w=None):
    rng = np.random.default_rng(seed); F = X.shape[1]
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    P = {"W1": rng.normal(0, 0.3, (F, hidden)), "b1": np.zeros(hidden), "W2": rng.normal(0, 0.3, (hidden, NCLS)), "b2": np.zeros(NCLS)}
    m = {k: np.zeros_like(v) for k, v in P.items()}; s = {k: np.zeros_like(v) for k, v in P.items()}; step = 0
    Y = np.eye(NCLS)[y]
    w = np.ones(len(y)) if class_w is None else class_w[y]
    for ep in range(epochs):
        idx = rng.permutation(len(y))
        for b in range(0, len(y), 512):
            i = idx[b:b + 512]; H, Pr = _forward(P, Xn[i])
            dZ = (Pr - Y[i]) * w[i, None] / len(i); dH = dZ @ P["W2"].T * (1 - H ** 2)
            g = {"W2": H.T @ dZ, "b2": dZ.sum(0), "W1": Xn[i].T @ dH, "b1": dH.sum(0)}
            step += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]; s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(s[k] / (1 - 0.999 ** step)) + 1e-8)
    P["mu"], P["sd"] = mu, sd
    return P


def predict_proba(P, X):
    return _forward(P, (X - P["mu"]) / P["sd"])[1]


def save_model(P, path=MODEL_PATH, meta=None):
    json.dump({**{k: np.asarray(v).tolist() for k, v in P.items()}, "meta": meta or {}}, open(path, "w"))


_cache = {}
def load_model(path=MODEL_PATH):
    if not os.path.exists(path): return None
    mt = os.path.getmtime(path)
    if _cache.get("p") == path and _cache.get("t") == mt: return _cache["m"]
    d = json.load(open(path)); m = {k: np.array(d[k]) for k in ("W1", "b1", "W2", "b2", "mu", "sd")}
    _cache.update(p=path, t=mt, m=m); return m


def classify_cells(P, feats):
    rows, cols, F = feats.shape
    return predict_proba(P, feats.reshape(-1, F)).argmax(1).reshape(rows, cols).astype(np.uint8)


# ---------------------------------------------------------------- обучающие данные из data/work
def final_centers(work):
    """Центры клеток матрицы в координатах кропа по итоговым quad/grid (то, что видел человек)."""
    import autogrid
    q = np.float32(work["quad"]); g = work["grid"]; m = work["matrix"]
    W, H = autogrid.quad_size([list(p) for p in q])
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), q)
    ox, oy = m.get("origin", [0, 0])
    pts = np.float32([[g.get("ox", 0) + (ox + i + .5) * g["pw"], g.get("oy", 0) + (oy + j + .5) * g["ph"]]
                     for j in range(m["h"]) for i in range(m["w"])]).reshape(-1, 1, 2)
    C = cv2.perspectiveTransform(pts, Hm).reshape(m["h"], m["w"], 2)
    return C, float(min(g["pw"], g["ph"]))


def matrix_labels(work):
    m = work["matrix"]
    return np.array([[0 if c == "." else min(int(c, 36), 2) for c in r] for r in m["rows"]], np.uint8)


SKIPPED = []     # битые json (обрыв записи) — пропускаем, не падаем


def load_examples(work_dir=None, crops_dir=None, limit=None):
    work_dir = work_dir or os.path.join(DATA, "work"); crops_dir = crops_dir or os.path.join(DATA, "crops")
    ex = []
    for fn in sorted(os.listdir(work_dir)):
        if not fn.endswith(".json"): continue
        try: w = json.load(open(os.path.join(work_dir, fn)))
        except Exception: SKIPPED.append(fn); continue
        if not (w.get("done") and w.get("matrix") and w.get("quad") and not w.get("color_mode") and w.get("kind") != "photo"): continue
        p = os.path.join(crops_dir, fn[:-5] + ".png")
        if not os.path.exists(p): continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        try:
            C, pitch = final_centers(w); X = cell_features(g, C, pitch); y = matrix_labels(w)
        except Exception:
            continue
        if X.shape[:2] != y.shape: continue
        ex.append((fn[:-5], X, y, g, C, pitch))
        if limit and len(ex) >= limit: break
    return ex


def baseline_labels(g, C, pitch):
    """Текущие пороги ровно как «Авто» по умолчанию: контур + серая заливка (classify + clean)."""
    import autogrid
    v = autogrid.cell_means(g, C, pitch); M, info = autogrid.classify(v)
    return autogrid.clean(M, info["t"])


def cross_val(ex, k=5, **kw):
    n = len(ex); folds = np.arange(n) % k; res = []
    cw = np.array([1.0, 2.0, 1.5])
    for f in range(k):
        tr = [e for e, fl in zip(ex, folds) if fl != f]; te = [e for e, fl in zip(ex, folds) if fl == f]
        X = np.concatenate([e[1].reshape(-1, e[1].shape[-1]) for e in tr]); y = np.concatenate([e[2].ravel() for e in tr])
        P = fit(X, y, class_w=cw, **kw)
        for e in te:
            pm = classify_cells(P, e[1]); bl = baseline_labels(e[3], e[4], e[5]); y = e[2]
            ink = (y > 0) | (pm > 0) | (bl > 0)
            res.append((e[0], float((pm != y).sum()) / max(1, ink.sum()), float((bl != y).sum()) / max(1, ink.sum())))
    return res


def gated(ex, res, gate=0.6):
    """Как работает в «Авто»: модель только там, где уверенность порогов < gate. → (ошибка порогов, ошибка с моделью, доля рисунков с моделью)"""
    import autogrid
    eb, eg, used = [], [], 0
    byid = {r[0]: r for r in res}
    for e in ex:
        r = byid[e[0]]
        v = autogrid.cell_means(e[3], e[4], e[5]); M, info = autogrid.classify(v); M = autogrid.clean(M, info["t"])
        c, _ = autogrid.confidence({"px": 1, "py": 1, "angle": 0, "quad_resid": 0}, M, info["t"])
        eb.append(r[2]); eg.append(r[1] if c < gate else r[2]); used += c < gate
    return np.mean(eb), np.mean(eg), used / max(1, len(ex))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    ex = load_examples()
    print("примеров (готовые ЧБ с кропом):", len(ex))
    if SKIPPED: print("пропущено битых файлов:", len(SKIPPED), SKIPPED[:5], "— список: /api/broken")
    if len(ex) < 10: print("мало данных"); return
    res = cross_val(ex)
    a = np.array([[r[1], r[2]] for r in res])
    print("ошибка на клетках с чернилами (кросс-проверка по рисункам, меньше — лучше):")
    print("  модель %.2f%%  пороги %.2f%%  | лучше у модели: %d из %d" % (100 * a[:, 0].mean(), 100 * a[:, 1].mean(), int((a[:, 0] < a[:, 1]).sum()), len(a)))
    import autogrid
    eb, eg, share = gated(ex, res, autogrid.MODEL_GATE)
    print("в работе (модель только при низкой уверенности порогов <%.2f, это %.0f%% рисунков): ошибка %.2f%% вместо %.2f%%" % (autogrid.MODEL_GATE, 100 * share, 100 * eg, 100 * eb))
    if cmd == "train":
        if eg >= eb:
            print("с моделью не лучше, чем без неё — НЕ сохраняю (останутся пороги). Нужно больше готовых рисунков."); return
        X = np.concatenate([e[1].reshape(-1, e[1].shape[-1]) for e in ex]); y = np.concatenate([e[2].ravel() for e in ex])
        P = fit(X, y, class_w=np.array([1.0, 2.0, 1.5]))
        save_model(P, meta={"n_figures": len(ex), "cv_model": float(eg), "cv_threshold": float(eb)})
        print("модель сохранена:", MODEL_PATH)


if __name__ == "__main__":
    main()
