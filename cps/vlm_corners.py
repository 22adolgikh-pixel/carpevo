# vlm_corners.py — v11.47: углы ковра по фото в два шага, глазами модели (Claude через API).
#
# Проба 10.11.2026 на 177 фото с ответами владельца: медиана ошибки 0,91 % кадра против 2,34 % у find_quad.
# Шаг 1 — грубо: целое фото с линейками в процентах по краям, модель называет 4 угла.
# Шаг 2 — уточнение: вокруг каждого грубого угла окно (16 % длинной стороны), 4 окна склеены в картинку 2×2,
#         у каждого окна свои линейки 0–100, за краем фото — пурпурный фон; модель называет положение угла в окне.
# Углы — в долях кадра 0..1, порядок: левый верхний, правый верхний, правый нижний, левый нижний (как r.layout.quad).
#
# Ключ берётся из переменной окружения ANTHROPIC_API_KEY (или CPS_ANTHROPIC_KEY) — только из окружения сервера, в данные не пишется.
# Без ключа модуль сообщает «нет ключа» и ничего не вызывает. CPS_VLM_MOCK=1 — заглушка без сети (для автотестов).
import base64, json, math, os, re, time, urllib.request, urllib.error
import cv2, numpy as np

MODEL = os.environ.get("CPS_VLM_MODEL", "claude-sonnet-5-5")
URL = "https://api.anthropic.com/v1/messages"
WIN = 0.16            # окно уточнения — доля длинной стороны фото
MAX_MOVE = 0.12       # уточнение не должно уйти дальше этой доли кадра от грубого угла — иначе берём грубый
COARSE_SIDE = 1000
TILE = 480


def key():
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CPS_ANTHROPIC_KEY") or ""


def available():
    return bool(os.environ.get("CPS_VLM_MOCK")) or bool(key())


def _jpg(a, q=88):
    ok, b = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(b.tobytes()).decode()


def _ask(images, prompt, max_tokens=300):
    """один запрос к модели → текст ответа. images — список JPEG в base64"""
    if os.environ.get("CPS_VLM_MOCK"):
        return _mock(prompt)
    body = {"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content":
            [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}} for d in images] +
            [{"type": "text", "text": prompt}]}]}
    rq = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "x-api-key": key(), "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(rq, timeout=90) as r:
                j = json.load(r)
            return "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 2: time.sleep(3 * (attempt + 1)); continue
            raise RuntimeError("API %s: %s" % (e.code, e.read()[:200].decode("utf-8", "replace")))
        except (urllib.error.URLError, TimeoutError):
            if attempt < 2: time.sleep(3); continue
            raise
    raise RuntimeError("API: не ответил")


def _mock(prompt):
    if "ОКНА" in prompt: return json.dumps({"corners": [[50, 50]] * 4})
    return json.dumps({"quad": [[10, 10], [90, 10], [90, 90], [10, 90]]})


def _json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else None


def _rulers(a, x0, y0, w, h, fs=0.4):
    """линейки 0–100 % по сторонам прямоугольника фото (x0,y0,w,h) на холсте a; подписи через 10 %"""
    for i in range(0, 101, 10):
        x = int(x0 + w * i / 100); y = int(y0 + h * i / 100)
        for (p, q) in (((x, y0 - 6), (x, y0)), ((x, y0 + h), (x, y0 + h + 6))): cv2.line(a, p, q, (0, 0, 0), 1)
        for (p, q) in (((x0 - 6, y), (x0, y)), ((x0 + w, y), (x0 + w + 6, y))): cv2.line(a, p, q, (0, 0, 0), 1)
        t = str(i)
        cv2.putText(a, t, (x - 6, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(a, t, (x - 6, y0 + h + 18), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(a, t, (x0 - 28, y + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(a, t, (x0 + w + 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)


def _coarse_img(im):
    h, w = im.shape[:2]; k = COARSE_SIDE / max(h, w)
    if k < 1: im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA); h, w = im.shape[:2]
    pad = 34; c = np.full((h + 2 * pad, w + 2 * pad, 3), 255, np.uint8); c[pad:pad + h, pad:pad + w] = im
    _rulers(c, pad, pad, w, h)
    return c


def _window(im, cx, cy):
    """окно вокруг угла (cx,cy — доли кадра); пурпурный фон за краем фото; линейки 0–100 окна"""
    h, w = im.shape[:2]; s = WIN * max(h, w)                      # сторона окна в пикселях фото
    x0, y0 = cx * w - s / 2, cy * h - s / 2
    k = TILE / s
    M = np.array([[k, 0, -x0 * k], [0, k, -y0 * k]], np.float64)
    t = cv2.warpAffine(im, M, (TILE, TILE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 255))
    pad = 30; c = np.full((TILE + 2 * pad, TILE + 2 * pad, 3), 255, np.uint8); c[pad:pad + TILE, pad:pad + TILE] = t
    _rulers(c, pad, pad, TILE, TILE, 0.38)
    return c, (x0, y0, s)


def _poly_ok(q):
    """выпуклый четырёхугольник нормальной площади"""
    s = 0; sg = 0
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(z) > 1e-9: sg += 1 if z > 0 else -1
        s += a[0] * b[1] - b[0] * a[1]
    return abs(sg) == 4 and abs(s) / 2 > 0.04


COARSE_PROMPT = ("На фото один ковёр. По краям фото линейки в процентах (0–100 по ширине и высоте фото; белое поле с линейками — не фото). "
                 "Назови 4 угла ковра (без бахромы) в процентах ширины и высоты фото в порядке: левый верхний, правый верхний, правый нижний, левый нижний. "
                 "Если ковёр срезан рамкой фото, угол может быть за пределами 0–100 (например −5 или 104). Если ковёр снят под углом, это углы трапеции. "
                 'Ответь только JSON: {"quad": [[x,y],[x,y],[x,y],[x,y]]}')
REFINE_PROMPT = ("ОКНА: четыре увеличенных окна 2×2 вокруг углов ковра (слева направо, сверху вниз: левый верхний, правый верхний, правый нижний, левый нижний). "
                 "У каждого окна свои линейки 0–100 по краям; пурпурный фон — это уже за пределами фото. "
                 "В каждом окне назови точное положение угла ковра (там, где сходятся два края ковра, без бахромы) в процентах окна: x слева направо, y сверху вниз. "
                 'Ответь только JSON: {"corners": [[x,y],[x,y],[x,y],[x,y]]}')


def two_step(im):
    """im — BGR фото (любого размера). → {"quad": [[x,y]*4] доли кадра, "coarse": [...], "model": ..} или None, если модель не дала внятного ответа"""
    h, w = im.shape[:2]
    j = _json(_ask([_jpg(_coarse_img(im))], COARSE_PROMPT))
    q = (j or {}).get("quad")
    if not (isinstance(q, list) and len(q) == 4): return None
    coarse = [[float(p[0]) / 100, float(p[1]) / 100] for p in q]
    if not _poly_ok(coarse): return None
    tiles, wins = [], []
    for cx, cy in coarse:
        t, wn = _window(im, cx, cy); tiles.append(t); wins.append(wn)
    m = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    out = list(coarse)
    try:
        j2 = _json(_ask([_jpg(m)], REFINE_PROMPT))
        cs = (j2 or {}).get("corners")
        if isinstance(cs, list) and len(cs) == 4:
            ref = []
            for (x0, y0, s), c, p in zip(wins, coarse, cs):
                x = (x0 + s * float(p[0]) / 100) / w; y = (y0 + s * float(p[1]) / 100) / h
                ref.append([x, y] if math.hypot(x - c[0], y - c[1]) <= MAX_MOVE else list(c))   # ушло слишком далеко — оставляем грубый
            if _poly_ok(ref): out = ref
    except Exception:
        pass                                                      # уточнение не вышло — остаётся грубый ответ
    return {"quad": [[round(x, 5), round(y, 5)] for x, y in out], "coarse": [[round(x, 5), round(y, 5)] for x, y in coarse], "model": "mock" if os.environ.get("CPS_VLM_MOCK") else MODEL}
