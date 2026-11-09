# rug_rect.py — 4 угла ковра на фото и выпрямление (v11.30).
#
# Фото ковра почти всегда снято с перспективой: ковёр — не прямоугольник, а четырёхугольник. Пока он не выпрямлен,
# у одного и того же элемента на разных краях разный размер и наклон, а каймы не параллельны краям кадра.
# 1) find_quad: фон (цвет по краям кадра) отделяем от ковра, берём крупнейшую область, грубые углы — крайние точки
#    по диагоналям, затем каждую сторону уточняем прямой по средней части контура (бахрома и сколы углов не мешают);
#    углы — пересечения соседних прямых. Это предложение: человек двигает углы.
# 2) rectify: гомография четырёхугольник → прямоугольник; пропорции — по перспективе углов (aspect), иначе по средним сторонам.
import cv2, numpy as np


def _fg(a):
    h, w = a.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 2), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = max(4, int(min(h, w) * 0.02))
    edge = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3), lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    bg = np.median(edge, 0); spread = np.percentile(np.linalg.norm(edge - bg, axis=1), 80)
    fg = (np.linalg.norm(lab - bg, axis=2) > max(18, spread * 2.5)).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.012)) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))           # тонкие нити бахромы отпадают
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    M = (lb == i).astype(np.uint8)
    cs, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    M2 = np.zeros_like(M); cv2.drawContours(M2, cs, -1, 1, -1)                      # дыры внутри (светлое поле) — тоже ковёр
    return M2


def _line(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _cross(l1, l2):
    (p, d), (q, e) = l1, l2
    A = np.array([[d[0], -e[0]], [d[1], -e[1]]]);
    if abs(np.linalg.det(A)) < 1e-6: return None
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def find_quad(img):
    """→ {quad: [[x,y]×4] в долях кадра (ЛВ, ПВ, ПН, ЛН), conf 0..1}"""
    H0, W0 = img.shape[:2]; s = min(1.0, 900 / max(H0, W0))
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    h, w = a.shape[:2]
    full = {"quad": [[0.01, 0.01], [0.99, 0.01], [0.99, 0.99], [0.01, 0.99]], "conf": 0.0}
    M = _fg(a)
    if M is None or M.mean() < 0.12: return full
    cs, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    sx, dx = c[:, 0] + c[:, 1], c[:, 0] - c[:, 1]
    idx = [int(np.argmin(sx)), int(np.argmax(dx)), int(np.argmax(sx)), int(np.argmin(dx))]   # ЛВ, ПВ, ПН, ЛН
    q = c[idx].copy()
    # уточнение: каждая сторона — прямая по средним 60% контура между соседними грубыми углами
    n = len(c); lines = []
    for k in range(4):
        i0, i1 = idx[k], idx[(k + 1) % 4]
        seg = c[i0:i1 + 1] if i0 <= i1 else np.concatenate([c[i0:], c[:i1 + 1]])
        if len(seg) < 20: lines = None; break
        m = len(seg); seg = seg[int(m * 0.2):int(m * 0.8)]
        lines.append(_line(seg))
    if lines:
        q2 = []
        for k in range(4):
            p = _cross(lines[(k - 1) % 4], lines[k])
            if p is None or not (-0.05 * w <= p[0] <= 1.05 * w and -0.05 * h <= p[1] <= 1.05 * h) or np.hypot(*(p - q[k])) > 0.15 * max(w, h): p = q[k]
            q2.append(p)
        q = np.array(q2)
    q[:, 0] = np.clip(q[:, 0], 0, w - 1); q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    P = np.zeros_like(M); cv2.fillPoly(P, [q.astype(np.int32)], 1)
    inter = float((P & M).sum()); union = float((P | M).sum()) or 1.0
    area = cv2.contourArea(q.astype(np.float32)) / (w * h)
    conf = inter / union if area > 0.1 else 0.0
    return {"quad": [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in q], "conf": round(conf, 3)}


def aspect(quad_px, img_wh):
    """Отношение ширины ковра к высоте по 4 углам на снимке (Zhang, He: whiteboard rectification, 2007):
    камера — обычная (пиксели квадратные, центр снимка — главная точка), фокус оцениваем по самим углам.
    Если перспективы почти нет или фокус не оценивается — средние длины противоположных сторон."""
    q = np.asarray(quad_px, np.float64); d = lambda i, j: float(np.hypot(*(q[i] - q[j])))
    naive = ((d(0, 1) + d(3, 2)) / 2) / max(1e-6, (d(0, 3) + d(1, 2)) / 2)
    m1, m2, m3, m4 = [np.array([*q[i], 1.0]) for i in (0, 1, 3, 2)]          # ЛВ, ПВ, ЛН, ПН
    try:
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
        n2 = k2 * m2 - m1; n3 = k3 * m3 - m1
        u0, v0 = img_wh[0] / 2.0, img_wh[1] / 2.0
        if abs(n2[0]) + abs(n2[1]) < 1e-9 or abs(n3[0]) + abs(n3[1]) < 1e-9: return naive
        f = None
        if abs(n2[2] * n3[2]) > 1e-12 * (abs(n2[0]) + abs(n2[1])) * (abs(n3[0]) + abs(n3[1])):
            f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
                   + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
            if np.isfinite(f2) and f2 > 0 and 0.3 * max(img_wh) < np.sqrt(f2) < 20 * max(img_wh): f = np.sqrt(f2)
        if f is None:
            # одна точка схода на бесконечности (ковёр наклонён только «вперёд» или только «вбок») — фокус по углам не найти;
            # берём обычный для съёмки: ≈ длинная сторона кадра (у кадрированных музейных фото ошибка в пропорциях несколько %)
            if abs(n2[2]) < 1e-9 and abs(n3[2]) < 1e-9: return naive          # перспективы нет — средние стороны точны
            f = float(max(img_wh))
        Ai = np.linalg.inv(np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]))
        B = Ai.T @ Ai
        r = np.sqrt((n2 @ B @ n2) / (n3 @ B @ n3))
        return float(r) if np.isfinite(r) and 0.1 < r < 10 else naive
    except Exception:
        return naive


def rect_size(quad_px, side, img_wh=None):
    r = aspect(quad_px, img_wh) if img_wh else None
    if r is None:
        q = np.asarray(quad_px, np.float64); d = lambda i, j: float(np.hypot(*(q[i] - q[j])))
        r = ((d(0, 1) + d(3, 2)) / 2) / max(1e-6, (d(0, 3) + d(1, 2)) / 2)
    W, H = (side, side / r) if r >= 1 else (side * r, side)
    return max(8, int(round(W))), max(8, int(round(H)))


def rectify(img, quad, side=1600):
    """quad — доли кадра (ЛВ, ПВ, ПН, ЛН) → выпрямленный ковёр (большая сторона = side)"""
    h, w = img.shape[:2]
    if max(h, w) > 2 * side:                                   # сильное уменьшение: сначала сжать (warp сам не усредняет пиксели)
        k = 2 * side / max(h, w); img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA); h, w = img.shape[:2]
    src = np.array([[x * w, y * h] for x, y in quad], np.float32)
    W, H = rect_size(src, side, (w, h))
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    T = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))   # за краем кадра — тёмно-серое (части ковра нет на фото)


def triage(img):
    """v11.34: этап 0–1 конвейера — что на фото и где углы. → {frame, quad, conf, touch:[л,в,п,н]}
    frame: whole — ковёр целиком, по краям кадра фон; cropped — край ковра выходит за кадр; unsure — ковёр от фона не отделяется.
    touch — доля каждой стороны кадра, занятая ковром (0..1)."""
    H0, W0 = img.shape[:2]; s = min(1.0, 900 / max(H0, W0))
    a = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    h, w = a.shape[:2]
    M = _fg(a)
    q = find_quad(img)
    if M is None or M.mean() < 0.12:
        return {"frame": "unsure", "quad": q["quad"], "conf": q["conf"], "touch": [1, 1, 1, 1]}
    m = max(2, int(min(h, w) * 0.01))
    touch = [float(M[:, :m].mean()), float(M[:m, :].mean()), float(M[:, -m:].mean()), float(M[-m:, :].mean())]
    quad_out = any(x < 0.005 or x > 0.995 or y < 0.005 or y > 0.995 for x, y in q["quad"])
    qa = float(cv2.contourArea(np.array(q["quad"], np.float32) * np.float32([w, h]))) / (w * h)
    if q["conf"] < 0.6 or qa < 0.1: frame = "unsure"
    elif max(touch) > 0.45 or quad_out: frame = "cropped"
    else: frame = "whole"
    return {"frame": frame, "quad": q["quad"], "conf": q["conf"], "area": round(qa, 3), "touch": [round(t, 3) for t in touch]}
