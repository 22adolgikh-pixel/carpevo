# corner_refine.py — уточнение углов ковра сетью, обученной на правках людей в /m (v11.57).
#
# Обучение — tools/colab/corner_refine_colab.py (метки — tools/corners/make_labels.py из резервной копии студии).
# Сеть смотрит на окно фото вокруг угла и говорит, куда его сдвинуть; три прохода: окно 24 %, 12 %, 6 % длинной стороны.
# Стартовые углы — углы модели по картинке (vlm_quad), иначе find_quad. Все четыре угла приводятся зеркалом к «левому верхнему».
# Веса — DATA или models/corner_refine_v1.pt, сервер сам качает их с Диска (TorchScript; нужен только torch, он в образе есть ради cnn.py) и corner_refine_v1.json (замеры).
# Нет torch или весов — available() = False, карточки углов работают как раньше.
import json, os, threading
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CPS_DATA") or os.path.join(HERE, "data")
# веса лежат на Диске (carpet-dna/models, доступ «по ссылке»); сервер скачивает их сам в DATA при первом обращении
DRIVE = {"pt": "1Rj5r3nPzqGe5qMamkKzyimVkqzj_yW9H", "json": "1-G5e0UXSIz_Ohy2-Z7LwXEnw9BLtl8L1"}
VERSION = 1


def _path():
    for p in (os.path.join(DATA, "corner_refine_v1.pt"), os.path.join(HERE, "models", "corner_refine_v1.pt")):
        if os.path.exists(p): return p
    return None


def fetch(force=False):
    """скачать веса с Диска в DATA (нужен доступ «по ссылке»); после успеха сеть перечитывается"""
    from rugs import drive_fetch
    dst = os.path.join(DATA, "corner_refine_v1.pt")
    if os.path.exists(dst) and not force: return True
    try:
        drive_fetch(DRIVE["pt"], dst + ".part"); os.replace(dst + ".part", dst)
        try: drive_fetch(DRIVE["json"], dst[:-3] + ".json")
        except Exception: pass
        _S["tried"] = False; _S["net"] = None; _S["err"] = ""
        return True
    except Exception as e:
        _S["err"] = str(e)[:200]; return False
FLIP = [(1, 1), (-1, 1), (-1, -1), (1, -1)]      # ЛВ, ПВ, ПН, ЛН
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
_lock = threading.Lock(); _S = {"net": None, "tried": False, "meta": None, "err": ""}


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


def info():
    _load(); m = _S["meta"] or {}
    return {"ok": _S["net"] is not None, **{k: m.get(k) for k in ("ver", "trained", "n_train", "n_test", "test")}} if _S["net"] is not None else {"ok": False, "error": _S["err"]}


def _window(a, cx, cy, half, k, W, mag):
    h, w = a.shape[:2]; hp = half * max(h, w); sx, sy = FLIP[k]
    M = np.array([[sx * 2 * hp / W, 0, cx * w - sx * hp], [0, sy * 2 * hp / W, cy * h - sy * hp]], np.float32)
    return cv2.warpAffine(a, M, (W, W), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=mag)


def refine(img, quad):
    """img — BGR (любой размер), quad — 4 угла в долях кадра (ЛВ, ПВ, ПН, ЛН) → уточнённые 4 угла или None"""
    net = _load()
    if net is None or img is None or not quad or len(quad) != 4: return None
    import torch
    m = _S["meta"] or {}; W = int(m.get("W", 128)); L0 = int(m.get("long", 1024)); mag = tuple(m.get("mag", (255, 0, 255)))
    a = img; k0 = L0 / max(a.shape[:2])
    if k0 < 1: a = cv2.resize(a, None, fx=k0, fy=k0, interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]; L = max(h, w); q = [[float(x), float(y)] for x, y in quad]
    with torch.no_grad():
        for half in m.get("halves", (0.12, 0.06, 0.03)):
            b = np.stack([((_window(a, q[k][0], q[k][1], half, k, W, mag)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for k in range(4)])
            o = net(torch.from_numpy(np.ascontiguousarray(b))).numpy()
            for k in range(4):
                sx, sy = FLIP[k]; d = np.clip(o[k], -1, 1)
                q[k][0] += sx * float(d[0]) * half * L / w; q[k][1] += sy * float(d[1]) * half * L / h
    return [[round(min(1.6, max(-0.6, x)), 5), round(min(1.6, max(-0.6, y)), 5)] for x, y in q]
