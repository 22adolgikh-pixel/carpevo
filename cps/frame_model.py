# frame_model.py — v11.44: «целый ли ковёр на фото» по ответам людей (этап 0 конвейера).
#
# Признаки берутся из того, что уже лежит в карточке ковра: предложение машины (r.auto — уверенность, площадь и форма
# четырёхугольника, как близко углы к краям кадра), размеры фото, источник, слова в названии.
# Обучение — tools/train_frame_model.py (sklearn, градиентный бустинг, неглубокие деревья); на сервере sklearn нет,
# поэтому деревья выгружены в models/frame_v1.json и считаются здесь на чистом Python.
import json, math, os

SRC = ("azerbaijanrugs", "wikimedia", "commons", "azcarpetmuseum", "kerimov", "met", "vam", "fig")
NAMES = ["conf", "area", "minmx", "minmy", "meanmx", "meanmy", "maxangdev", "logasp", "logside", "auto_whole", "auto_cropped"] + \
        ["src_" + k for k in SRC] + ["kw_frag", "has_school", "has_comp"]
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "frame_v1.json")
_M = {"m": None, "t": 0}


def feat(r):
    a = r.get("auto") or {}
    q = a.get("quad") or [[0, 0], [1, 0], [1, 1], [0, 1]]
    w, h = r.get("w") or 1, r.get("h") or 1
    mx = [min(x, 1 - x) for x, _ in q]; my = [min(y, 1 - y) for _, y in q]
    px = [(x * w, y * h) for x, y in q]

    def ang(i):
        p, a_, b = px[i], px[i - 1], px[(i + 1) % 4]
        u = (a_[0] - p[0], a_[1] - p[1]); v = (b[0] - p[0], b[1] - p[1])
        nu, nv = math.hypot(*u), math.hypot(*v)
        c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv + 1e-9)))
        return abs(math.degrees(math.acos(c)) - 90)
    s = (r.get("cat") or {}).get("src", "") or ""
    t = (r.get("title") or "").lower()
    return [a.get("conf", 0) or 0, a.get("area") or 0, min(mx), min(my), sum(mx) / 4, sum(my) / 4, max(ang(i) for i in range(4)),
            math.log(w / h), math.log(min(w, h) + 1), float(a.get("frame") == "whole"), float(a.get("frame") == "cropped")] + \
           [float(s.startswith(k)) for k in SRC] + \
           [float(any(k in t for k in ("fragment", "фрагмент", "hissə", "qalıq"))), float(bool(r.get("school"))), float(bool(r.get("composition")))]


def _load():
    try: mt = os.path.getmtime(PATH)
    except OSError: return None
    if _M["m"] is None or mt != _M["t"]:
        with open(PATH, encoding="utf-8") as f: _M["m"] = json.load(f)
        _M["t"] = mt
    return _M["m"]


def _tree(t, x):
    i = 0
    while t["l"][i] != -1:
        i = t["l"][i] if x[t["f"][i]] <= t["th"][i] else t["r"][i]
    return t["v"][i]


def p_whole(r):
    """вероятность «целый ковёр» 0..1 или None (модели нет / машина ещё не смотрела фото)"""
    m = _load()
    if not m or not r.get("auto"): return None
    x = feat(r)
    z = m["init"] + m["lr"] * sum(_tree(t, x) for t in m["trees"])
    return 1 / (1 + math.exp(-z))


def info():
    m = _load()
    return {k: m.get(k) for k in ("ver", "n", "cv_acc", "trained", "thr")} if m else None
