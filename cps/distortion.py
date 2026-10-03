"""Искажения страницы: честная мера «неровности» сетки.
Что выяснено на 150 готовых рисунках: уточнять углы сетки по фазе линий бумаги в окнах — шумнее, чем сама автосетка
(медиана сдвига узлов 0.00 → 0.15 клетки), а «ошибки» автосетки относительно ручной сетки в большинстве случаев — это неточность
ручной сетки: по печатным линиям и по краям туши автосетка стоит не хуже (в 36 из 150 — заметно лучше, ручная лучше — в 0).
Поэтому сетку НЕ двигаем, а измеряем, насколько она одинаково ложится на всю страницу: рисунок выпрямляется по сетке,
делится на окна, в каждом — комплексный коэффициент гребёнки линий бумаги на периоде клетки. Если решётка равномерна,
фазы окон совпадают (когерентность ≈ 1); изгиб листа, трапеция, накопленная ошибка шага разводят фазы.
dist — разброс фаз по окнам в долях клетки (0 = идеально), по осям отдельно; warp — «изгиб» = линейный тренд фазы (дрейф шага)."""
import numpy as np
import cv2
import autogrid as A

S = 10
NW = 4
MIN_STRENGTH = 0.25


def _rectify(g, q, cols, rows):
    H = cv2.getPerspectiveTransform(np.array(q, np.float32), np.array([[0, 0], [cols * S, 0], [cols * S, rows * S], [0, rows * S]], np.float32))
    return cv2.warpPerspective(g, H, (cols * S, rows * S), flags=cv2.INTER_AREA, borderValue=255)


def _coef(prof, x0):
    z = prof - prof.mean(); tot = np.abs(z).sum()
    if tot < 1e-6: return 0j, 0.0
    xs = np.arange(len(prof)) + x0
    c = (z * np.exp(-2j * np.pi * xs / S)).sum()
    return c / tot, float(abs(c) / tot)


def measure(g, corners, cols, rows):
    """Окна с их фазовыми сдвигами: список (i, j, dx, dy, сила), dx/dy — в клетках (±0.5)."""
    img = _rectify(g, corners, cols, rows)
    hp = A._line_response(img); keep = A._paper_mask(img).astype(np.float32)
    Hh, Ww = hp.shape; out = []
    for j in range(NW):
        for i in range(NW):
            x0, x1 = i * Ww // NW, (i + 1) * Ww // NW; y0, y1 = j * Hh // NW, (j + 1) * Hh // NW
            if x1 - x0 < S * 4 or y1 - y0 < S * 4: continue
            sub, k = hp[y0:y1, x0:x1], keep[y0:y1, x0:x1]
            if k.mean() < 0.25: continue
            cp = (sub * k).sum(0) / np.maximum(k.sum(0), 1); rp = (sub * k).sum(1) / np.maximum(k.sum(1), 1)
            cp[k.sum(0) < 0.15 * k.shape[0]] = np.median(cp); rp[k.sum(1) < 0.15 * k.shape[1]] = np.median(rp)
            cx, sx = _coef(cp, x0); cy, sy = _coef(rp, y0)
            out.append((i, j, -np.angle(cx) / (2 * np.pi), -np.angle(cy) / (2 * np.pi), min(sx, sy)))
    return out


def assess(g, G):
    """{dist, dist_x, dist_y, drift_x, drift_y, strength, windows}: dist — разброс фаз (клетки), drift — линейный дрейф по странице
    (клеток от края до края). None, если окон с надёжной линейкой меньше 6 (чистая бумага без сетки / сплошная тушь)."""
    pts = [p for p in measure(g, G["corners"], G["cols"], G["rows"]) if p[4] >= MIN_STRENGTH]
    if len(pts) < 6: return None
    i = np.array([p[0] for p in pts], float); j = np.array([p[1] for p in pts], float)
    dx = np.array([p[2] for p in pts]); dy = np.array([p[3] for p in pts]); w = np.array([p[4] for p in pts])
    def circ(v):                                              # круговой разброс фазы в клетках
        z = (w * np.exp(2j * np.pi * v)).sum() / w.sum(); R = min(abs(z), 0.999)
        return float(np.sqrt(-2 * np.log(R)) / (2 * np.pi))
    mx, my = np.angle((w * np.exp(2j * np.pi * dx)).sum()) / (2 * np.pi), np.angle((w * np.exp(2j * np.pi * dy)).sum()) / (2 * np.pi)
    ex = (dx - mx + .5) % 1 - .5; ey = (dy - my + .5) % 1 - .5      # отклонение от общей фазы
    def slope(c, e):
        if np.ptp(c) < 1: return 0.0
        return float(np.polyfit(c, e, 1, w=w)[0] * (NW - 1))
    return {"dist_x": circ(dx), "dist_y": circ(dy), "dist": max(circ(dx), circ(dy)),
            "drift_x": max(abs(slope(i, ex)), abs(slope(j, ex))), "drift_y": max(abs(slope(i, ey)), abs(slope(j, ey))),
            "strength": float(w.mean()), "windows": len(pts)}
