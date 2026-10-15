# shape_hint.py — v11.5: «на что это похоже» — подсказка названия элемента по форме.
#
# Человек обвёл элемент, но не знает, как он называется. Сравниваем вырезку с фото:
#  1) со всеми схемами студии (data/out/<id>.png, 1 px = 1 узел) — в 8 положениях (4 поворота × зеркало);
#  2) (если есть индекс узоров, motif_search) — с уже названными подтверждёнными рамками на других коврах.
# Ответ — кандидаты с названием, узором словаря и положением схемы; клик в студии ставит название, схему и поворот.
#
# Признак формы — «направления линий» (как гистограммы HOG): вырезка сводится к 48×48, в каждой клетке сетки 6×6
# считается, сколько контура идёт в каждом из 8 направлений. Направление берётся по тензору структуры сразу по всем
# каналам Lab (и по прозрачности у схемы), поэтому не важно, какими красками соткан элемент и светлый он или тёмный.
# Пропорции рамки тоже учитываются (узкая полоса не похожа на квадратный гёль).
import os, json, time, threading, math
import numpy as np, cv2

N, BINS, SIDE = 6, 8, 48
TF = [(a, f) for f in (0, 1) for a in (0, 90, 180, 270)]
_idx = {"t": 0, "ids": [], "X": None, "meta": {}, "ar": None, "sig": ""}
_lock = threading.Lock()


def _tf(t, a, f):
    if f: t = cv2.flip(t, 1)
    if a == 90: t = cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE)
    elif a == 180: t = cv2.rotate(t, cv2.ROTATE_180)
    elif a == 270: t = cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return t


def _desc(chans):
    """chans — список float32-каналов SIDE×SIDE → вектор N·N·BINS (длина 1)"""
    Jxx = Jyy = Jxy = 0
    for ch in chans:
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0); gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1)
        Jxx = Jxx + gx * gx; Jyy = Jyy + gy * gy; Jxy = Jxy + gx * gy
    Jxx, Jyy, Jxy = [cv2.GaussianBlur(x, (0, 0), 1.0) for x in (Jxx, Jyy, Jxy)]
    ang = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)                 # направление (по модулю π)
    mag = np.sqrt(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2))
    pos = ((ang + np.pi / 2) % np.pi) / np.pi * BINS
    out = np.zeros((N, N, BINS), np.float32); c = SIDE // N
    for b in range(BINS):
        w = np.clip(1 - np.abs(((pos - b + BINS / 2) % BINS) - BINS / 2), 0, 1) * mag
        out[..., b] = cv2.resize(w, (N, N), interpolation=cv2.INTER_AREA) * c * c
    out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-3 * out.sum() / (N * N) + 1e-6)
    out = np.minimum(out, 0.35)
    v = np.sqrt(out.ravel()); return v / (np.linalg.norm(v) + 1e-8)


def desc_photo(bgr):
    """вырезка с фото ковра → признак формы"""
    h, w = bgr.shape[:2]
    a = cv2.resize(bgr, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    return _desc([lab[..., 0], lab[..., 1], lab[..., 2]])


def desc_scheme(rgba, ang=0, fl=0):
    """схема студии (1 px = 1 узел, фон прозрачный) → признак формы в положении (угол, зеркало)"""
    if max(rgba.shape[:2]) > 480:
        q = 480 / max(rgba.shape[:2]); rgba = cv2.resize(rgba, (max(3, int(rgba.shape[1] * q)), max(3, int(rgba.shape[0] * q))), interpolation=cv2.INTER_AREA)
    t = _tf(rgba, ang, fl)
    h, w = t.shape[:2]; k = max(1, int(math.ceil(192 / max(h, w))))
    big = cv2.resize(t, (w * k, h * k), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    big = cv2.GaussianBlur(big, (0, 0), 0.45 * k)             # шерсть: узел на фото не квадратик, а пятно
    a = cv2.resize(big, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    al = a[..., 3:4] / 255.0 if a.shape[2] == 4 else np.ones(a.shape[:2] + (1,), np.float32)
    rgb = np.clip(a[..., :3] * al + 128 * (1 - al), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    return _desc([lab[..., 0], lab[..., 1], lab[..., 2], al[..., 0] * 100])


def _load_rgba(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if a is None: return None
    m = max(a.shape[:2])
    if m > 480:                                                # v11.15: «схема» бывает огромной картинкой — float32 × 8 положений съедали гигабайты
        k = 480 / m; a = cv2.resize(a, (max(3, int(a.shape[1] * k)), max(3, int(a.shape[0] * k))), interpolation=cv2.INTER_AREA)
    if a.ndim == 2: a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGRA)
    if a.shape[2] == 3:                                        # без прозрачности: фон — самый частый цвет
        bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0)
        al = (np.abs(a.astype(np.int16) - bg).sum(-1) > 30).astype(np.uint8) * 255
        a = np.dstack([a, al])
    return a


def build_index(d_out, schemes, cache_path=None):
    """schemes: {fid: {"name", "motif", "done"}} → матрица признаков (каждая схема × 8 положений). Кэш на диске по времени файлов."""
    items = sorted((fid, os.path.getmtime(os.path.join(d_out, fid + ".png"))) for fid in schemes if os.path.exists(os.path.join(d_out, fid + ".png")))
    sig = str(hash(tuple(items)))
    with _lock:
        if _idx["sig"] == sig: _idx["meta"] = schemes; return _idx
        old = {}
        if cache_path and os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                for i, (fid, mt) in enumerate(zip(z["ids"], z["mt"])): old[(str(fid), float(mt))] = (z["X"][i], z["ar"][i])
            except Exception: old = {}
        ids, X, AR, MT = [], [], [], []
        for fid, mt in items:
            if (fid, mt) in old: x, ar = old[(fid, mt)]
            else:
                a = _load_rgba(os.path.join(d_out, fid + ".png"))
                if a is None or a.shape[0] < 3 or a.shape[1] < 3: continue
                x = np.stack([desc_scheme(a, ang, fl) for ang, fl in TF]).astype(np.float16)
                ar = np.float32(a.shape[1] / a.shape[0])
            ids.append(fid); X.append(x); AR.append(ar); MT.append(mt)
        _idx.update(t=time.time(), ids=ids, X=np.stack(X).astype(np.float32) if X else np.zeros((0, len(TF), N * N * BINS), np.float32),
                    ar=np.array(AR, np.float32), meta=schemes, sig=sig)
        if cache_path and ids:
            try: np.savez(cache_path, ids=np.array(ids), mt=np.array(MT, np.float64), X=np.stack(X).astype(np.float16), ar=np.array(AR, np.float32))
            except Exception: pass
        return _idx


def match_schemes(crop_bgr, limit=10):
    """→ [{id, name, motif, score, angle, flip}] — лучшие схемы, по одной на узор (название)"""
    if _idx["X"] is None or not len(_idx["ids"]): return []
    q = desc_photo(crop_bgr); h, w = crop_bgr.shape[:2]; arq = w / max(1, h)
    S = _idx["X"] @ q                                           # [схем, 8 положений]
    ar = _idx["ar"][:, None]; rot = np.array([a in (90, 270) for a, f in TF])[None, :]
    ars = np.where(rot, 1 / ar, ar)
    pen = np.exp(-1.2 * np.abs(np.log(arq / ars)))              # пропорции рамки и схемы
    T = S * (0.75 + 0.25 * pen)
    best = T.argmax(1); sc = T[np.arange(len(best)), best]
    order = np.argsort(-sc); out = []; seen = set()
    for i in order:
        fid = _idx["ids"][i]; m = _idx["meta"].get(fid) or {}
        key = m.get("motif") or (m.get("name") or fid).strip().lower()
        if key in seen: continue
        seen.add(key); a, f = TF[int(best[i])]
        out.append({"id": fid, "name": m.get("name") or "", "motif": m.get("motif") or "", "done": bool(m.get("done")),
                    "score": round(float(sc[i]), 3), "angle": a, "flip": f})
        if len(out) >= limit: break
    return out
