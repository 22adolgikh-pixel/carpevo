# rug_match.py — машинные предложения для разметки ковров (v10.23).
#
# 1) find_repeats: человек обвёл один элемент → ищем такие же на этом же фото. Масштаб тот же (одно фото), но повтор
#    может быть повёрнут на 90/180/270° или отражён, а цвета у повторов на коврах часто меняются местами
#    (тот же гёль — то красный на синем, то синий на красном). Поэтому сравниваем две «картинки» сразу:
#    яркость (форма + тон) и карту контуров (только форма — от смены цветов почти не зависит), берём среднее.
# 2) find_scheme: один элемент со схемой уже найден → знаем, сколько пикселей фото приходится на один узел.
#    Тогда схемы других элементов (из студии, 1 px = 1 узел) можно перевести в масштаб фото и искать их контуры.
#    Это грубее: схема — чистый рисунок, фото — шерсть, тени, износ. Даёт кандидатов для проверки человеком.
import cv2, numpy as np

TF = [(a, f) for f in (0, 1) for a in (0, 90, 180, 270)]   # угол, зеркало (по горизонтали — до поворота)


def _tf(t, a, f):
    if f: t = cv2.flip(t, 1)
    if a == 90: t = cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE)
    elif a == 180: t = cv2.rotate(t, cv2.ROTATE_180)
    elif a == 270: t = cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return t


def _maps(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    # контуры по всем трём каналам Lab: смена «красный ↔ синий» при одинаковой яркости тоже даёт контур
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    e = np.zeros(g.shape, np.float32)
    for c in range(3):
        ch = cv2.GaussianBlur(lab[..., c], (0, 0), 1.0)
        e += cv2.magnitude(cv2.Sobel(ch, cv2.CV_32F, 1, 0), cv2.Sobel(ch, cv2.CV_32F, 0, 1))
    e = cv2.GaussianBlur(e, (0, 0), 1.2)
    return g, e


def _nms(c, iou=0.3, cap=200):
    c = sorted(c, key=lambda t: -t["score"]); keep = []
    for a in c:
        if len(keep) >= cap: break
        ok = True
        for b in keep:
            x0, y0 = max(a["px"][0], b["px"][0]), max(a["px"][1], b["px"][1])
            x1 = min(a["px"][0] + a["px"][2], b["px"][0] + b["px"][2]); y1 = min(a["px"][1] + a["px"][3], b["px"][1] + b["px"][3])
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            if inter / min(a["px"][2] * a["px"][3], b["px"][2] * b["px"][3]) > iou: ok = False; break
        if ok: keep.append(a)
    return keep


def _overlaps(px, boxes, W, H, thr=0.35):
    x, y, w, h = px
    for b in boxes:
        bx, by, bw, bh = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        inter = max(0, min(x + w, bx + bw) - max(x, bx)) * max(0, min(y + h, by + bh) - max(y, by))
        if inter / min(w * h, bw * bh + 1e-6) > thr: return True
    return False


def _ori(bgr):
    """v11.5: направление линий узора (тензор структуры по Lab в «двойном угле»): не зависит ни от цвета, ни от того,
    светлый узор на тёмном или тёмный на светлом, — повтор с переставленными красками сравнивается по форме."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    Jxx = Jyy = Jxy = 0
    for c in range(3):
        ch = cv2.GaussianBlur(lab[..., c], (0, 0), 1.0); gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0); gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1)
        Jxx = Jxx + gx * gx; Jyy = Jyy + gy * gy; Jxy = Jxy + gx * gy
    Jxx, Jyy, Jxy = [cv2.GaussianBlur(x, (0, 0), 1.5) for x in (Jxx, Jyy, Jxy)]
    m = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) + 1e-6; n = np.sqrt(m)
    return (Jxx - Jyy) / m * n, 2 * Jxy / m * n


def _tf_ori(c, s, ang, fl):
    """поворот/зеркало поля направлений: картинку поворачиваем, а «двойной угол» меняет знак (90°/270° — оба, зеркало — sin)"""
    c, s = _tf(c, ang, fl), _tf(s, ang, fl)
    if ang in (90, 270): c, s = -c, -s
    if fl: s = -s
    return c, s


def find_repeats(img, box, exclude=(), limit=60, min_score=None, boxes=None, loose=False):
    """img — BGR, box — [x,y,w,h] в долях. → [{box, score, angle, flip, sure}] без пересечений с exclude (уже размеченные).
    boxes — несколько образцов того же элемента (все рамки группы): ищем по каждому, место получает лучший балл —
    так находятся и повторы в других красках. loose — «вижу ещё»: порог ниже, ответ — следующие по сходству места."""
    H0, W0 = img.shape[:2]
    T = [b for b in (boxes or []) if b] or [box]
    if box not in T: T = [box] + T
    T = T[:8]
    bw = float(np.median([b[2] for b in T])) * W0; bh = float(np.median([b[3] for b in T])) * H0
    # шаблон ≈ 56 px по длинной стороне, но фото — не больше 900 px: иначе мелкий элемент (10–20 px) заставляет перебирать
    # фото в полном размере 24 раза, и это минуты. Мелкий шаблон при этом может стать 8–12 px — для поиска хватает.
    s = min(1.0, 56.0 / max(bw, bh), 900.0 / max(H0, W0))
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    H, W = a.shape[:2]
    G, E = _maps(a); C, S = _ori(a); EN = C * C + S * S
    tw, th = max(6, int(round(bw * s))), max(6, int(round(bh * s)))
    cands = []
    for ti, b in enumerate(T):
        x0, y0 = int(round(b[0] * W)), int(round(b[1] * H))
        x0, y0 = max(0, min(W - tw, x0)), max(0, min(H - th, y0))
        tg, te, tc, ts = (M[y0:y0 + th, x0:x0 + tw] for M in (G, E, C, S))
        if tg.shape[0] < 5 or tg.shape[1] < 5: continue
        for sc in (0.92, 1.0, 1.08):                     # повторы на ковре бывают чуть разного размера (ручная работа)
            for ang, fl in TF:
                g = _tf(tg, ang, fl); e = _tf(te, ang, fl); c, ss = _tf_ori(tc, ts, ang, fl)
                if sc != 1.0:
                    g, e, c, ss = (cv2.resize(m, None, fx=sc, fy=sc) for m in (g, e, c, ss))
                if g.shape[0] >= H or g.shape[1] >= W: continue
                hh, ww = g.shape[:2]
                num = cv2.matchTemplate(C, c, cv2.TM_CCORR) + cv2.matchTemplate(S, ss, cv2.TM_CCORR)
                en = cv2.boxFilter(EN, -1, (ww, hh), normalize=False, anchor=(0, 0))[:H - hh + 1, :W - ww + 1]
                ro = num / np.sqrt(np.maximum(en, 1e-6) * (float((c * c + ss * ss).sum()) + 1e-6))
                r = 0.2 * cv2.matchTemplate(G, g, cv2.TM_CCOEFF_NORMED) + 0.35 * cv2.matchTemplate(E, e, cv2.TM_CCOEFF_NORMED) + 0.45 * ro
                # локальные максимумы
                k = max(3, int(min(hh, ww) * 0.5) | 1)
                mx = cv2.dilate(r, np.ones((k, k), np.uint8))
                ys, xs = np.where((r >= mx - 1e-6) & (r > 0.25))
                if len(ys) > 150:                            # мелкие элементы «находятся» везде — берём только лучшие места
                    top = np.argpartition(-r[ys, xs], 150)[:150]; ys, xs = ys[top], xs[top]
                for yy, xx in zip(ys, xs):
                    cands.append({"px": (xx / s, yy / s, ww / s, hh / s), "score": float(r[yy, xx]), "angle": ang, "flip": fl, "tpl": ti})
    if not cands: return []
    cands = sorted(cands, key=lambda t: -t["score"])[:3000]
    cands = _nms(cands)
    ex = list(exclude) + list(T)
    free = [c for c in cands if not _overlaps(c["px"], ex, W0, H0)]
    # порог: образцы находятся с ~1.0; повторы — 0.45–0.85 (в других красках ниже), случайные места — < 0.4.
    scores = [c["score"] for c in free]
    noise = float(np.median(scores[min(len(scores) - 1, 8):])) if len(scores) > 10 else 0.3
    thr = min_score if min_score is not None else max(0.4, noise + 0.1)
    sure_thr = thr
    if loose: thr = max(0.28, noise + 0.03)          # «вижу ещё»: следующие по сходству места (их проверяет человек)
    out = []
    for c in free:
        if c["score"] < thr: break
        x, y, w, h = c["px"]
        out.append({"box": [round(x / W0, 5), round(y / H0, 5), round(w / W0, 5), round(h / H0, 5)], "score": round(c["score"], 3),
                    "angle": c["angle"], "flip": c["flip"], "above": c["score"] >= sure_thr, "tpl": T[c["tpl"]]})
        if len(out) >= limit: break
    return out


def _scheme_edges(png):
    """схема студии (1 px = 1 узел, фон прозрачный/белый) → карта «узор / фон» (0..1)."""
    if png.ndim == 3 and png.shape[2] == 4:
        m = (png[..., 3] > 0).astype(np.float32)
        col = png[..., :3]
        # несколько цветов узора: границы между цветами тоже контуры
        lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB).astype(np.float32) * m[..., None]
        return m, lab
    g = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY) if png.ndim == 3 else png
    m = (g < 200).astype(np.float32)
    return m, cv2.cvtColor(png if png.ndim == 3 else cv2.cvtColor(png, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB).astype(np.float32)


def find_scheme(img, png, kx, ky, exclude=(), limit=12, region=None):
    """Ищем на фото элемент по схеме студии. kx, ky — пикселей фото на узел (известно по уже найденному элементу со схемой)."""
    H0, W0 = img.shape[:2]
    kh, kw = png.shape[:2]
    pw, ph = kw * kx, kh * ky                         # ожидаемый размер элемента на фото, px
    if pw < 8 or ph < 8 or pw > W0 or ph > H0: return []
    s = min(1.0, 64.0 / max(pw, ph), 900.0 / max(H0, W0))
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    _, E = _maps(a)
    m, lab = _scheme_edges(png)
    tw, th = max(6, int(round(pw * s))), max(6, int(round(ph * s)))
    # контуры схемы: границы маски + границы между цветами, в масштабе фото
    lab = cv2.resize(lab, (tw, th), interpolation=cv2.INTER_NEAREST)
    te = np.zeros((th, tw), np.float32)
    for c in range(3):
        ch = cv2.GaussianBlur(lab[..., c], (0, 0), 1.0)
        te += cv2.magnitude(cv2.Sobel(ch, cv2.CV_32F, 1, 0), cv2.Sobel(ch, cv2.CV_32F, 0, 1))
    te = cv2.GaussianBlur(te, (0, 0), 1.2)
    if region:
        rx0, ry0 = int(region[0] * E.shape[1]), int(region[1] * E.shape[0])
        rx1, ry1 = int((region[0] + region[2]) * E.shape[1]), int((region[1] + region[3]) * E.shape[0])
    else: rx0, ry0, rx1, ry1 = 0, 0, E.shape[1], E.shape[0]
    Er = E[ry0:ry1, rx0:rx1]
    cands = []
    for sc in (0.9, 1.0, 1.1):
        for ang, fl in TF:
            t = _tf(te, ang, fl)
            if sc != 1.0: t = cv2.resize(t, None, fx=sc, fy=sc)
            if t.shape[0] >= Er.shape[0] or t.shape[1] >= Er.shape[1] or t.std() < 1e-3: continue
            r = cv2.matchTemplate(Er, t, cv2.TM_CCOEFF_NORMED)
            hh, ww = t.shape[:2]; k = max(3, int(min(hh, ww) * 0.5) | 1)
            mx = cv2.dilate(r, np.ones((k, k), np.uint8))
            ys, xs = np.where((r >= mx - 1e-6) & (r > 0.2))
            for yy, xx in zip(ys, xs):
                cands.append({"px": ((xx + rx0) / s, (yy + ry0) / s, ww / s, hh / s), "score": float(r[yy, xx]), "angle": ang, "flip": fl})
    if not cands: return []
    cands = _nms(cands)
    scores = [c["score"] for c in cands]
    thr = max(0.3, float(np.percentile(scores, 90)) if len(scores) > 10 else 0.3)
    out = []
    for c in cands:
        if c["score"] < thr: break
        if _overlaps(c["px"], exclude, W0, H0): continue
        x, y, w, h = c["px"]
        out.append({"box": [round(x / W0, 5), round(y / H0, 5), round(w / W0, 5), round(h / H0, 5)], "score": round(c["score"], 3),
                    "angle": c["angle"], "flip": c["flip"]})
        if len(out) >= limit: break
    return out


def find_carpets(img):
    """Сколько ковров на фото (развороты книг: два ковра на двух страницах, или фото + фото). → [[x,y,w,h] в долях]."""
    H0, W0 = img.shape[:2]; s = 600 / max(H0, W0)
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    h, w = a.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 1.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    # «текстурность»: у ковра много мелких деталей, у бумаги, полей и стола — почти нет
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    det = cv2.GaussianBlur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.5)), (0, 0), 4)
    m = max(4, int(min(h, w) * 0.02))
    edge = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3), lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    bg = np.median(edge, 0); d = np.linalg.norm(lab - bg, axis=2)
    fg = ((det > max(3.0, np.percentile(det, 30))) & (d > 12)) | (det > np.percentile(det, 60))
    fg = fg.astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    # строки/столбцы почти без ковра — разрез между страницами/фото
    rows = fg.mean(1); cols = fg.mean(0)
    def runs(p, thr):
        out, st = [], None
        for i, v in enumerate(p):
            if v >= thr and st is None: st = i
            if v < thr and st is not None: out.append((st, i)); st = None
        if st is not None: out.append((st, len(p)))
        return out
    best = None
    for axis, prof, L in (("y", rows, h), ("x", cols, w)):
        rs = [r for r in runs(prof, 0.25) if r[1] - r[0] > L * 0.15]
        if len(rs) >= 2 and (best is None or len(rs) > len(best[1])): best = (axis, rs)
    parts = []
    if best:
        axis, rs = best
        for a0, a1 in rs:
            sub = fg[a0:a1, :] if axis == "y" else fg[:, a0:a1]
            pr = sub.mean(0) if axis == "y" else sub.mean(1)
            on = np.where(pr > 0.25)[0]
            if len(on) < 10: continue
            b0, b1 = on[0], on[-1] + 1
            if axis == "y": parts.append([b0 / w, a0 / h, (b1 - b0) / w, (a1 - a0) / h])
            else: parts.append([a0 / w, b0 / h, (a1 - a0) / w, (b1 - b0) / h])
    if len(parts) < 2: return []
    return [[round(v, 4) for v in p] for p in parts]


def align(a, b):
    """Совместить два фото одного ковра (двойники из разных источников): SIFT-точки + гомография (RANSAC).
    → (H 3×3 в долях: точка (x,y) фото a → фото b, число совпавших точек) или (None, n)."""
    def prep(x):
        s = min(1.0, 1200.0 / max(x.shape[:2]))
        g = cv2.cvtColor(cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else x, cv2.COLOR_BGR2GRAY)
        return cv2.createCLAHE(2.0, (8, 8)).apply(g)
    ga, gb = prep(a), prep(b)
    sift = cv2.SIFT_create(nfeatures=6000)
    ka, da = sift.detectAndCompute(ga, None); kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10: return None, 0
    def root(d): d = d / (d.sum(1, keepdims=True) + 1e-7); return np.sqrt(d).astype(np.float32)
    m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64)).knnMatch(root(da), root(db), k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 12: return None, len(good)
    pa = np.float32([ka[g.queryIdx].pt for g in good]) / [ga.shape[1], ga.shape[0]]
    pb = np.float32([kb[g.trainIdx].pt for g in good]) / [gb.shape[1], gb.shape[0]]
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 0.01)
    n = int(inl.sum()) if inl is not None else 0
    if H is None or n < 12: return None, n
    return H, n


def map_box(H, box):
    """рамка [x,y,w,h] в долях фото a → рамка на фото b (охватывающий прямоугольник) + поворот фото b относительно a (0/90/180/270)."""
    x, y, w, h = box
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    x0, y0 = q.min(0); x1, y1 = q.max(0)
    v = q[1] - q[0]; ang = int(round(np.degrees(np.arctan2(v[1], v[0])) / 90.0)) * 90 % 360
    x0, y0 = max(0.0, float(x0)), max(0.0, float(y0)); x1, y1 = min(1.0, float(x1)), min(1.0, float(y1))
    if x1 - x0 < 0.003 or y1 - y0 < 0.003: return None, 0
    return [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)], ang


# ---------------------------------------------------------------- v11.7: все ожидаемые элементы сразу, масштаб неизвестен
def _scheme_tpl(png, tw, th):
    """схема (1 px = 1 узел) → карты контуров и направлений линий размером tw×th (как у фото)"""
    if png.ndim == 2: png = cv2.cvtColor(png, cv2.COLOR_GRAY2BGRA)
    if png.shape[2] == 3:
        bg = np.median(np.concatenate([png[0], png[-1], png[:, 0], png[:, -1]]), 0)
        al = (np.abs(png.astype(np.int16) - bg).sum(-1) > 30).astype(np.uint8) * 255
        png = np.dstack([png, al])
    big = cv2.resize(png, (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    al = big[..., 3:4] / 255.0
    rgb = np.clip(big[..., :3] * al + 128 * (1 - al), 0, 255).astype(np.uint8)
    rgb = cv2.GaussianBlur(rgb, (0, 0), max(0.6, 0.35 * tw / max(1, png.shape[1])))
    _, E = _maps(rgb); C, S = _ori(rgb)
    # прозрачность — тоже граница (силуэт элемента на фоне ковра)
    a = cv2.GaussianBlur(al[..., 0] * 100, (0, 0), 1.0)
    E = E + cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)), (0, 0), 1.2)
    return E, C, S


def _score_maps(PE, PC, PS, PEN, E, C, S):
    """карта сходства: контуры (TM_CCOEFF_NORMED) + направления линий (нормированная корреляция поля направлений)"""
    hh, ww = E.shape
    if E.std() < 1e-3: return None
    re_ = cv2.matchTemplate(PE, E, cv2.TM_CCOEFF_NORMED)
    num = cv2.matchTemplate(PC, C, cv2.TM_CCORR) + cv2.matchTemplate(PS, S, cv2.TM_CCORR)
    en = cv2.boxFilter(PEN, -1, (ww, hh), normalize=False, anchor=(0, 0))[:PE.shape[0] - hh + 1, :PE.shape[1] - ww + 1]
    ro = num / np.sqrt(np.maximum(en, 1e-6) * (float((C * C + S * S).sum()) + 1e-6))
    return 0.45 * re_ + 0.55 * ro


def find_schemes_auto(img, schemes, knots=None, exclude=(), per=4, work=900, progress=None, min_z=4.0):
    """img — фото ковра (BGR); schemes — {id: png схемы}; knots — сколько узлов по ширине фото (если известно по уже найденному
    элементу со схемой), иначе подбирается сразу по всем схемам: у одного ковра один масштаб, и он тот, при котором элементы
    находятся увереннее всего (по крупным схемам: мелкие, до 12 узлов, «находятся» где угодно).
    Уверенность — не абсолютный балл (на шерстяном фото он низкий), а насколько лучшее место выделяется среди всех мест
    этого же фото: z = (балл − среднее) / разброс по 200 лучшим местам. → (узлов по ширине, {id: [{box, score, z, angle, flip}]})"""
    H0, W0 = img.shape[:2]

    def prep(side):
        s = min(1.0, side / max(H0, W0)); a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        _, PE = _maps(a); PC, PS = _ori(a)
        return s, a.shape[1], a.shape[0], PE, PC, PS, PC * PC + PS * PS

    def run(M, K, ids, jit=(1.0,), ky=(1.0,), tfs=TF, cap=200):
        s, W, H, PE, PC, PS, PEN = M; k = W / K; out = {}
        for sid in ids:
            png = schemes[sid]; kh, kw = png.shape[:2]; best = []
            for j in jit:
                for q in ky:
                    tw, th = int(round(kw * k * j)), int(round(kh * k * j * q))
                    if tw < 8 or th < 8 or tw >= W * 0.9 or th >= H * 0.9: continue
                    E0, C0, S0 = _scheme_tpl(png, tw, th)
                    for ang, fl in tfs:
                        E = _tf(E0, ang, fl); C, S = _tf_ori(C0, S0, ang, fl)
                        if E.shape[0] >= H or E.shape[1] >= W: continue
                        r = _score_maps(PE, PC, PS, PEN, E, C, S)
                        if r is None: continue
                        hh, ww = E.shape; kk = max(3, int(min(hh, ww) * 0.5) | 1)
                        mx = cv2.dilate(r, np.ones((kk, kk), np.uint8))
                        ys, xs = np.where(r >= mx - 1e-6)
                        if len(ys) > cap:
                            t = np.argpartition(-r[ys, xs], cap)[:cap]; ys, xs = ys[t], xs[t]
                        for yy, xx in zip(ys, xs):
                            best.append({"px": (xx / s, yy / s, ww / s, hh / s), "score": float(r[yy, xx]), "angle": ang, "flip": fl})
            c = _nms(best, iou=0.3, cap=200)
            if len(c) >= 8:
                v = np.array([x["score"] for x in c[1:]]); mu, sd = float(v.mean()), float(v.std()) + 1e-4
            else: mu, sd = 0.2, 0.1
            for x in c: x["z"] = (x["score"] - mu) / sd
            out[sid] = c
        return out
    ids = list(schemes)
    if not ids: return knots, {}
    big = [i for i in ids if max(schemes[i].shape[:2]) >= 12] or ids
    if not knots:
        M = prep(560)
        med_w = float(np.median([schemes[i].shape[1] for i in big]))
        cands = sorted({max(30, int(round(med_w * f))) for f in (4, 5.5, 7, 9, 12, 16, 22)})     # элемент ≈ 1/4 … 1/22 ширины фото
        sc = []
        def val(r):
            b = sorted((v[0]["z"] if v else 0 for v in r.values()), reverse=True)
            return float(np.mean(b[:max(1, (len(b) + 1) // 2)]))
        for n, K in enumerate(cands):
            sc.append((val(run(M, K, big, tfs=[(a, 0) for a in (0, 90, 180, 270)], cap=120)), K))
            if progress: progress(n + 1, len(cands) + 3)
        best, knots = max(sc)
        for K in (int(knots * 0.87), int(knots * 1.15)):
            v = val(run(M, K, big, tfs=[(a, 0) for a in (0, 90, 180, 270)], cap=120))
            if v > best: best, knots = v, K
    res = run(prep(work), knots, ids, jit=(0.92, 1.0, 1.08), ky=(1.0, 1.18))
    if progress: progress(1, 1)
    out = {}
    for sid, c in res.items():
        keep = []
        if c:
            top = c[0]
            for x in c:
                # первое место — если выделяется (z ≥ min_z; у настоящего элемента обычно 8–11, у случайного места 3–5; сервер предлагает от 6); повторы того же элемента — почти так же хорошие
                if x["z"] < min_z or x["score"] < top["score"] * 0.88: break
                if _overlaps(x["px"], exclude, W0, H0): continue
                X, Y, w, h = x["px"]
                keep.append({"box": [round(X / W0, 5), round(Y / H0, 5), round(w / W0, 5), round(h / H0, 5)], "score": round(x["score"], 3),
                             "z": round(x["z"], 2), "angle": x["angle"], "flip": x["flip"]})
                if len(keep) >= per: break
        out[sid] = keep
    return knots, out
