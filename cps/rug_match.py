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


def _nms(c, iou=0.3):
    c = sorted(c, key=lambda t: -t["score"]); keep = []
    for a in c:
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


def find_repeats(img, box, exclude=(), limit=60, min_score=None):
    """img — BGR, box — [x,y,w,h] в долях. → [{box, score, angle, flip}] без пересечений с exclude (уже размеченные)."""
    H0, W0 = img.shape[:2]
    bw, bh = box[2] * W0, box[3] * H0
    s = min(1.0, 56.0 / max(bw, bh))                 # шаблон ≈ 56 px по длинной стороне: быстро и достаточно деталей
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    H, W = a.shape[:2]
    G, E = _maps(a)
    x0, y0 = int(round(box[0] * W)), int(round(box[1] * H)); tw, th = max(6, int(round(box[2] * W))), max(6, int(round(box[3] * H)))
    tg, te = G[y0:y0 + th, x0:x0 + tw], E[y0:y0 + th, x0:x0 + tw]
    if tg.shape[0] < 6 or tg.shape[1] < 6: return []
    cands = []
    for sc in (0.92, 1.0, 1.08):                     # повторы на ковре бывают чуть разного размера (ручная работа)
        for ang, fl in TF:
            g = _tf(tg, ang, fl); e = _tf(te, ang, fl)
            if sc != 1.0:
                g = cv2.resize(g, None, fx=sc, fy=sc); e = cv2.resize(e, None, fx=sc, fy=sc)
            if g.shape[0] >= H or g.shape[1] >= W: continue
            r = 0.45 * cv2.matchTemplate(G, g, cv2.TM_CCOEFF_NORMED) + 0.55 * cv2.matchTemplate(E, e, cv2.TM_CCOEFF_NORMED)
            hh, ww = g.shape[:2]
            # локальные максимумы
            k = max(3, int(min(hh, ww) * 0.5) | 1)
            mx = cv2.dilate(r, np.ones((k, k), np.uint8))
            ys, xs = np.where((r >= mx - 1e-6) & (r > 0.25))
            for yy, xx in zip(ys, xs):
                cands.append({"px": (xx / s, yy / s, ww / s, hh / s), "score": float(r[yy, xx]), "angle": ang, "flip": fl})
    if not cands: return []
    cands = _nms(cands)
    # порог: сам исходный элемент находится с ~1.0; повторы обычно 0.5–0.8, случайные места — < 0.4.
    scores = sorted((c["score"] for c in cands), reverse=True)
    noise = float(np.median(scores[min(len(scores) - 1, 5):])) if len(scores) > 6 else 0.3
    thr = min_score if min_score is not None else max(0.42, noise + 0.12)
    out = []
    ex = list(exclude) + [box]
    for c in cands:
        if c["score"] < thr: break
        if _overlaps(c["px"], ex, W0, H0): continue
        x, y, w, h = c["px"]
        out.append({"box": [round(x / W0, 5), round(y / H0, 5), round(w / W0, 5), round(h / H0, 5)], "score": round(c["score"], 3),
                    "angle": c["angle"], "flip": c["flip"]})
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
    s = min(1.0, 64.0 / max(pw, ph))
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
