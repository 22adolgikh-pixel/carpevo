# element_embed.py — «отпечатки» элементов узора: сеть, обученная на ответах людей в /m (v11.58).
#
# Обучение — tools/colab/element_search_colab.py (метки — tools/elements/make_labels.py). Сеть превращает вырезку
# элемента (рамка + 15 % поля, 96×96) в вектор из 128 чисел; близость векторов = «тот же элемент». Образец берётся
# во всех 8 положениях (поворот на 90° и зеркало) — у ковра элемент бывает повёрнут.
# Применение в студии:
#   verify — близость предложенных рамок к образцу группы: в карточке «элементы» лучшие идут первыми, сомнительные
#            (ниже порога thr_bad) заранее отмечены ✗;
#   search — по образцу найти на ковре остальные повторы (окна размера образца, шаг 1/3, три масштаба и поворот на 90°).
# Веса — DATA или models/element_embed_v1.pt; если нет — скачиваются с Диска (id в DRIVE). Нет torch или весов —
# available() = False, всё работает как раньше.
import json, os, threading
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CPS_DATA") or os.path.join(HERE, "data")
NAME = "element_embed_v1"
DRIVE = {"pt": "", "json": ""}                       # заполняется после обучения в Colab
VERSION = 1
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
_lock = threading.Lock(); _S = {"net": None, "tried": False, "meta": {}, "err": ""}


def _path():
    for p in (os.path.join(DATA, NAME + ".pt"), os.path.join(HERE, "models", NAME + ".pt")):
        if os.path.exists(p): return p
    return None


def fetch(force=False):
    from rugs import drive_fetch
    dst = os.path.join(DATA, NAME + ".pt")
    if os.path.exists(dst) and not force: return True
    if not DRIVE["pt"]: _S["err"] = "сеть ещё не обучена (нет id на Диске)"; return False
    try:
        drive_fetch(DRIVE["pt"], dst + ".part"); os.replace(dst + ".part", dst)
        if DRIVE["json"]:
            try: drive_fetch(DRIVE["json"], dst[:-3] + ".json")
            except Exception: pass
        _S.update(tried=False, net=None, err=""); return True
    except Exception as e:
        _S["err"] = str(e)[:200]; return False


def _load():
    if _S["tried"]: return _S["net"]
    with _lock:
        if _S["tried"]: return _S["net"]
        _S["tried"] = True
        try:
            import torch
            P = _path()
            if P is None and fetch(): P = _path()
            if P:
                torch.set_num_threads(2)
                _S["net"] = torch.jit.load(P, map_location="cpu").eval()
                try: _S["meta"] = json.load(open(P[:-3] + ".json", encoding="utf-8"))
                except (OSError, ValueError): _S["meta"] = {}
        except Exception as e:
            _S["net"] = None; _S["err"] = _S["err"] or str(e)[:200]
    return _S["net"]


def available(): return _load() is not None


def meta(): _load(); return _S["meta"] or {}


def info():
    if _load() is None: return {"ok": False, "error": _S["err"]}
    m = _S["meta"] or {}
    return {"ok": True, **{k: m.get(k) for k in ("ver", "trained", "n_rugs_train", "n_rugs_test", "thr_bad", "thr_search", "test")}}


def small(img):
    L = int(meta().get("long", 1400)); k = L / max(img.shape[:2])
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img


def _crop(a, box, S, ctx):
    h, w = a.shape[:2]; x, y, bw, bh = box
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * ctx)
    M = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
    return cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def _prep(c): return ((np.ascontiguousarray(c)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def _run(arr):
    import torch
    net = _load(); out = []
    with torch.no_grad():
        for i in range(0, len(arr), 128): out.append(net(torch.from_numpy(np.ascontiguousarray(np.stack(arr[i:i + 128])))).numpy())
    return np.concatenate(out) if out else np.zeros((0, 128), np.float32)


def emb(a, boxes):
    m = meta(); S, ctx = int(m.get("S", 96)), float(m.get("ctx", 0.15))
    return _run([_prep(_crop(a, b, S, ctx)) for b in boxes])


def emb_rot(a, box):
    m = meta(); S, ctx = int(m.get("S", 96)), float(m.get("ctx", 0.15)); c = _crop(a, box, S, ctx)
    return _run([_prep(np.rot90(c[:, ::-1] if f else c, r)) for f in (0, 1) for r in range(4)])


def verify(img, sample_box, boxes):
    """близость рамок к образцу (−1..1, выше — похожее); img — уже уменьшенное small()"""
    if _load() is None or not boxes: return None
    q = emb_rot(img, sample_box); return [float(v) for v in (emb(img, boxes) @ q.T).max(1)]


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def search(img, sample_box, exclude=(), quad=None, thr=None, limit=24):
    """найти на ковре повторы образца → [(близость, рамка)], без наложений и без уже отмеченных мест"""
    if _load() is None: return []
    thr = thr if thr is not None else float(meta().get("thr_search") or 0.8)
    h, w = img.shape[:2]; bw, bh = sample_box[2], sample_box[3]; q = emb_rot(img, sample_box); C = []
    X0, X1, Y0, Y1 = 0.0, 1.0, 0.0, 1.0
    if quad:
        X0, X1 = max(0.0, min(p[0] for p in quad)), min(1.0, max(p[0] for p in quad)); Y0, Y1 = max(0.0, min(p[1] for p in quad)), min(1.0, max(p[1] for p in quad))
    for sc in (0.85, 1.0, 1.18):
        for ww, hh in {(bw * sc, bh * sc), (bh * sc * h / w, bw * sc * w / h)}:
            xs = np.arange(X0, X1 - ww, ww / 3); ys = np.arange(Y0, Y1 - hh, hh / 3)
            bs = [[float(x), float(y), ww, hh] for y in ys for x in xs]
            if not bs or len(bs) > 20000: continue
            s = (emb(img, bs) @ q.T).max(1)
            C += [(float(v), b) for v, b in zip(s, bs) if v >= thr]
    C.sort(key=lambda t: -t[0]); keep = []
    ex = list(exclude) + [sample_box]
    for v, b in C:
        if any(_iou(b, e) >= 0.3 for e in ex) or any(_iou(b, k[1]) >= 0.3 for k in keep): continue
        keep.append((v, [round(x, 5) for x in b]))
        if len(keep) >= limit: break
    return keep
