# sign_lib.py — v11.20: словарь знаков кайм по всему каталогу. Каждая кайма каждого ковра → раппорт (sign_lab.friezes);
# раппорты всех ковров сравниваются по форме (без цвета: один и тот же знак ткут разными красками; с отражениями) и
# собираются в группы. Группа = знак. Человек называет группу один раз и одной кнопкой размечает её на всех коврах.
#
# Хранение: data/signs/tiles/<rid>__<i>.jpg — раппорт (лёжа), data/signs/index.json — {rid: {"ver", "bands": [...]}}
# (шаг, чёткость, места повторов), data/signs/names.json — имена групп (по любому члену группы).
import os, json, time, threading
import numpy as np, cv2
import sign_lab, shape_hint as SH

VER = 1
state = {"running": False, "done": 0, "total": 0, "found": 0, "error": "", "finished": "", "last": ""}
_cfg = {}
_lock = threading.Lock()
_cl = {"t": 0, "v": None, "sig": ""}
_pv = {}                  # точные сравнения пар (ключи раппортов) — не пересчитываются
TF4 = [(0, 0), (180, 0), (0, 1), (180, 1)]       # раппорт лежит вдоль каймы: поворот на 180° и зеркала — тот же знак


def setup(data):
    d = os.path.join(data, "signs"); os.makedirs(os.path.join(d, "tiles"), exist_ok=True)
    _cfg.update(d=d, idx=os.path.join(d, "index.json"), names=os.path.join(d, "names.json"), desc=os.path.join(d, "desc2.npz"))


def _load(p, default):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return default


def _save(p, v):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(v, f, ensure_ascii=False)
    os.replace(tmp, p)


def index(): return _load(_cfg["idx"], {})
def names(): return _load(_cfg["names"], {})


def analyze(rid, img):
    """один ковёр → раппорты кайм (файлы + запись в индексе)"""
    res = sign_lab.friezes(img)
    out = []
    for i, b in enumerate(res["bands"]):
        t = cv2.rotate(b["tile"], cv2.ROTATE_90_CLOCKWISE)
        k = 96.0 / max(1, t.shape[0]); t = cv2.resize(t, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        cv2.imwrite(os.path.join(_cfg["d"], "tiles", f"{rid}__{i}.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 88])
        per, sharp = structure(t)
        out.append({"i": i, "T": b["T"], "q": b["q"], "inset": b["inset"], "n": len(b["boxes"]), "boxes": b["boxes"], "box_sides": b["box_sides"],
                    "ar": round(t.shape[1] / t.shape[0], 3), "per": per, "sharp": sharp, "px": [int(b["tile"].shape[1]), int(b["tile"].shape[0])]})
    return out


def run(todo, img_reader, heavy, on_done=None):
    """todo — [rid]; img_reader(rid) → BGR или None; heavy(fn, name) — очередь тяжёлых расчётов"""
    if state["running"]: return False
    state.update(running=True, done=0, total=len(todo), found=0, error="")

    def go():
        try:
            for rid in todo:
                try:
                    a = img_reader(rid)
                    bands = heavy(lambda: analyze(rid, a), "знаки кайм") if a is not None else []
                except Exception as e:
                    bands = []; state["error"] = f"{rid}: {e}"[:200]
                with _lock:
                    ix = index(); ix[rid] = {"ver": VER, "at": int(time.time()), "bands": bands}; _save(_cfg["idx"], ix)
                state["done"] += 1; state["found"] += len(bands); state["last"] = rid
                time.sleep(0.02)
            state["finished"] = time.strftime("%Y-%m-%d %H:%M")
            if on_done: on_done(f"просмотрено фото: {state['total']}, раппортов кайм: {state['found']}")
        finally:
            state["running"] = False; _cl["t"] = 0
    threading.Thread(target=go, daemon=True).start()
    return True


def fdesc(t):
    """признак раппорта, не зависящий от того, с какого места каймы он начат: гистограммы направлений линий по клеткам
    4 (поперёк) × 16 (вдоль) → модуль спектра вдоль каймы (сдвиг по кругу меняет только фазу). Цвет не учитывается."""
    a = cv2.GaussianBlur(cv2.resize(t, (128, 32), interpolation=cv2.INTER_AREA), (0, 0), 0.8)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    Jxx = Jyy = Jxy = 0
    for c in range(3):
        gx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0); gy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1)
        Jxx = Jxx + gx * gx; Jyy = Jyy + gy * gy; Jxy = Jxy + gx * gy
    Jxx, Jyy, Jxy = [cv2.GaussianBlur(x, (0, 0), 1.0) for x in (Jxx, Jyy, Jxy)]
    ang = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy); mag = np.sqrt(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2))
    pos = ((ang + np.pi / 2) % np.pi) / np.pi * 8
    cells = np.zeros((4, 16, 8), np.float32)
    for b in range(8):
        w = np.clip(1 - np.abs(((pos - b + 4) % 8) - 4), 0, 1) * mag
        cells[..., b] = cv2.resize(w, (16, 4), interpolation=cv2.INTER_AREA)
    F = np.abs(np.fft.rfft(cells, axis=1))[:, 1:7, :]        # без нулевой гармоники: ровные полосы (без узора вдоль каймы) почти пустые
    v = np.sqrt(F.ravel()); return v / (np.linalg.norm(v) + 1e-8)


def emap(t):
    """контуры раппорта 16 (поперёк) × 64 (вдоль), нулевое среднее, единичная норма — для точной проверки пар"""
    g = cv2.cvtColor(cv2.resize(t, (64, 16), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    e = e - e.mean(); return e / (np.linalg.norm(e) + 1e-6)


def ncc_shift(A, B, dy=4):
    """максимум корреляции A и B по всем круговым сдвигам вдоль каймы и небольшим сдвигам поперёк
    (у разных фото ширина полосы каймы определяется по-разному)"""
    best = -1.0; h = A.shape[0]
    for d in range(-dy, dy + 1):
        a = A[max(0, d):h + min(0, d)]; b = B[max(0, -d):h + min(0, -d)]
        a = a - a.mean(); b = b - b.mean(); n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
        c = np.fft.irfft((np.fft.rfft(a, axis=1) * np.conj(np.fft.rfft(b, axis=1))).sum(0), n=A.shape[1]).max() / n
        best = max(best, float(c))
    return best


def structure(t):
    """насколько в раппорте есть узор ВДОЛЬ каймы (а не просто полосы): доля энергии гармоник 1–6 в контурах; и резкость"""
    a = cv2.resize(t, (128, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    prof = cv2.resize(e, (16, 4), interpolation=cv2.INTER_AREA)
    per = float(np.mean(prof.std(1) / (prof.mean(1) + 1e-3)))     # контуры меняются вдоль каймы (узор), а не ровные полосы
    return round(per, 3), round(float(e.mean()), 2)


def _descs(keys):
    """признак формы раппорта в 4 положениях; кэш по ключу «rid__i»"""
    z = {}
    if os.path.exists(_cfg["desc"]):
        try:
            d = np.load(_cfg["desc"], allow_pickle=False); z = {str(k): v for k, v in zip(d["k"], d["x"])}
        except Exception: z = {}
    ch = False; out = {}
    for k in keys:
        if k not in z:
            t = cv2.imread(os.path.join(_cfg["d"], "tiles", k + ".jpg"))
            if t is None: continue
            z[k] = np.stack([fdesc(SH._tf(t, a, f)) for a, f in TF4]).astype(np.float16); ch = True
        out[k] = z[k].astype(np.float32)
    if ch:
        ks = list(z.keys()); np.savez(_cfg["desc"], k=np.array(ks), x=np.stack([z[k] for k in ks]))
    return out


def good(b, min_q=0.55):
    """раппорт годится в словарь: чёткий повтор, есть узор вдоль каймы, фото не размыто, не крошечный"""
    px = b.get("px") or [99, 99]
    return b.get("q", 0) >= min_q and b.get("per", 1) >= 0.25 and b.get("sharp", 99) >= 35 and px[0] >= 12 and px[1] >= 14


def clusters(meta=None, thr=0.65, min_q=0.55):
    """группы похожих раппортов: [{key, members: [{rid, i, q, n, ar}], rugs, name}] — самые большие первыми"""
    ix = index()
    items = [(rid, b) for rid, v in ix.items() if (not meta or rid in meta) for b in v.get("bands") or [] if good(b, min_q)]
    sig = str(len(items)) + str(sum(v.get("at", 0) for v in ix.values()))
    if _cl["v"] is not None and (_cl["sig"] == sig and time.time() - _cl["t"] < 600 or state["running"] and time.time() - _cl["t"] < 120): return _cl["v"]
    keys = [f"{rid}__{b['i']}" for rid, b in items]
    D = _descs(keys); items = [(rid, b) for (rid, b), k in zip(items, keys) if k in D]; keys = [k for k in keys if k in D]
    if not keys: return []
    X = np.stack([D[k] for k in keys])                      # [n, 4, d]
    ar = np.log(np.array([b.get("ar", 1) for _, b in items]))
    NB = []                                                  # 25 ближайших по признаку — кусками, без матрицы n×n×4
    for c0 in range(0, len(keys), 400):
        Sc = np.einsum("itd,jd->ijt", X[c0:c0 + 400], X[:, 0]).max(-1) - 0.15 * np.minimum(1, np.abs(ar[c0:c0 + 400, None] - ar[None, :]))
        for r_, row in enumerate(Sc):
            row[c0 + r_] = -1; top = np.argpartition(-row, min(25, len(row) - 1))[:25]
            NB.append([(int(j), float(row[j])) for j in top])
    # 2-й шаг: у каждого раппорта 25 ближайших по признаку проверяются точно — корреляция контуров с любым сдвигом вдоль каймы
    E = {}
    def em(i):
        if i not in E:
            t = cv2.imread(os.path.join(_cfg["d"], "tiles", keys[i] + ".jpg"))
            E[i] = [emap(SH._tf(t, a, f)) for a, f in TF4] if t is not None else None
        return E[i]
    V = {}
    for i in range(len(keys)):
        for j, sv in NB[i]:
            if sv < 0.7 or (min(i, j), max(i, j)) in V: continue
            pk = (keys[min(i, j)], keys[max(i, j)])
            if pk not in _pv:
                a, b = em(i), em(int(j)); _pv[pk] = max(ncc_shift(x, b[0]) for x in a) if a and b else 0.0
            V[(min(i, j), max(i, j))] = _pv[pk]
    nb = {i: set() for i in range(len(keys))}
    for (i, j), v in V.items():
        if v >= thr: nb[i].add(j); nb[j].add(i)
    lab = -np.ones(len(keys), int); c = 0
    for i in sorted(range(len(keys)), key=lambda i: -len(nb[i])):      # центр с наибольшим числом подтверждённых соседей
        if lab[i] >= 0: continue
        lab[i] = c
        for j in nb[i]:
            if lab[j] < 0: lab[j] = c
        c += 1
    VV = {}
    for (i, j), v in V.items(): VV.setdefault(i, {})[j] = v; VV.setdefault(j, {})[i] = v
    nm = names(); out = []
    for g in range(c):
        idx = np.where(lab == g)[0]
        ctr = max(idx, key=lambda i: sum(VV.get(i, {}).get(j, 0) for j in idx))
        mem = sorted([{"rid": items[i][0], "i": items[i][1]["i"], "q": items[i][1]["q"], "n": items[i][1]["n"], "key": keys[i]} for i in idx],
                     key=lambda m: (m["key"] != keys[ctr], -m["q"]))
        name = next((nm[m["key"]] for m in mem if m["key"] in nm), None)
        qual = float(np.mean([items[i][1].get("q", 0) * min(1.0, items[i][1].get("per", 0.4) / 0.4) * min(1.0, items[i][1].get("sharp", 60) / 60) for i in idx]))
        out.append({"key": keys[ctr], "members": mem, "rugs": len({m["rid"] for m in mem}), "name": name, "qual": round(qual, 3)})
    out.sort(key=lambda g: (-g["rugs"] * (0.3 + g["qual"]), -len(g["members"])))
    _cl.update(t=time.time(), v=out, sig=sig)
    return out


def set_name(keys, name):
    with _lock:
        nm = names()
        for k in keys:
            if name: nm[k] = name
            else: nm.pop(k, None)
        _save(_cfg["names"], nm)
    _cl["t"] = 0
