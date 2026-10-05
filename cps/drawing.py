# drawing.py — зарисовки пером (без сетки) → матрица узлов. v10.17
# Зарисовка устроена так: сплошная чёрная заливка, тонкий контур, а области «цвета» обозначены штриховкой
# (точки, завитки, чёрточки) разной густоты. Пороги по средней яркости клетки тут не работают: редкие завитки
# почти не темнят клетку (область уходит в фон), а края заливки дают серую кайму.
# Поэтому классифицируем не клетки, а ОБЛАСТИ:
#   1) сплошная заливка = то, что переживает «размыкание» ядром ~⅓ клетки;
#   2) остальные чернила делятся на длинные линии (контуры — границы областей) и мелкие штрихи (текстура);
#   3) области между заливкой и контурами получают цвет по густоте штрихов внутри: пусто → фон,
#      редкие штрихи → цвет 2, густые → цвет 3 (если нужен) — вся область целиком, без «крапа»;
#   4) клетка берёт метку, которой в ней больше всего (заливка и контур — с приоритетом, если их заметная доля).
import cv2, numpy as np


def label_map(g, cell_px, levels=2, dark=128):
    """g — серое изображение (uint8). → карта меток того же размера: 0 фон, 1 заливка/контур, 2.. текстура."""
    bw = (g < dark).astype(np.uint8)
    k = max(3, int(round(cell_px * 0.35)) | 1)
    solid = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    strokes = bw & (1 - cv2.dilate(solid, np.ones((3, 3), np.uint8)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(strokes, 8)
    big = max(cell_px * 2.5, 25)
    is_line = np.zeros(n, bool)
    for j in range(1, n):
        if max(st[j, 2], st[j, 3]) >= big: is_line[j] = True        # длинное — контур
    lines = is_line[lab].astype(np.uint8)
    marks = ((lab > 0) & ~is_line[lab]).astype(np.uint8)          # мелкие штрихи — текстура
    kw = max(3, int(round(cell_px * 0.3)) | 1)                    # контуры в зарисовках с разрывами — «замазать» щели
    walls = cv2.dilate(solid | lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw)))
    free = (1 - walls).astype(np.uint8)
    rn, rlab, rst, _ = cv2.connectedComponentsWithStats(free, 4)
    # штриховка = двумерная россыпь штрихов: «замыкание» склеивает её в пятно, «размыкание» убирает одиночные линии
    k1 = max(3, int(round(cell_px * 0.6)) | 1); k2 = max(3, int(round(cell_px * 0.8)) | 1)
    E = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    txm = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, E(k1))
    txm = cv2.morphologyEx(txm, cv2.MORPH_OPEN, E(k2)) & (1 - solid)
    win = max(5, int(round(cell_px * 1.5)))
    sd = cv2.blur(strokes.astype(np.float32), (win, win))
    tex = np.zeros(g.shape, np.uint8)
    tex[txm > 0] = 3 if levels >= 3 else 2                      # редкая штриховка — светлее (палитра CPS: 3 = #dedede)
    tex[(txm > 0) & (sd > 0.3)] = 2                             # густая — темнее (2 = #b8b8b8)
    out = tex.copy()
    out[(lines > 0) & (txm == 0)] = 1                            # одиночный контур (не внутри штриховки) — тёмная линия
    out[solid > 0] = 1
    if rn > 1:                                                   # замкнутые области — ровной заливкой по своей густоте
        cnt = np.bincount(rlab.ravel(), weights=strokes.ravel().astype(np.float64), minlength=rn)
        dens = cnt / np.maximum(1, rst[:, 4]).astype(np.float64)
        Hh, Ww = g.shape
        cls = np.zeros(rn, np.uint8)
        for r in range(1, rn):
            x, y, w, h = rst[r, :4]
            edge = x == 0 or y == 0 or x + w >= Ww or y + h >= Hh
            d = dens[r]
            if edge or d < 0.012: continue                         # у края кадра — фон; совсем пусто — фон
            cls[r] = 2 if (d > 0.25 or levels < 3) else 3
        reg = cls[rlab]; reg[rlab == 0] = 0
        m = (solid == 0) & (reg > 0)
        out[m] = reg[m]
    return out


def to_matrix(g, grid, levels=2):
    """grid: {pw, ph, ox, oy, cols, rows} в пикселях g. → (rows×cols uint8, карта меток)."""
    pw, ph = float(grid["pw"]), float(grid["ph"]); ox, oy = float(grid.get("ox", 0)), float(grid.get("oy", 0))
    cols, rows = int(grid["cols"]), int(grid["rows"])
    L = label_map(g, min(pw, ph), levels)
    M = np.zeros((rows, cols), np.uint8)
    for j in range(rows):
        y0, y1 = int(round(oy + j * ph)), int(round(oy + (j + 1) * ph))
        for i in range(cols):
            x0, x1 = int(round(ox + i * pw)), int(round(ox + (i + 1) * pw))
            c = L[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            if c.size == 0: continue
            h = np.bincount(c.ravel(), minlength=4)
            if h[1] >= 0.4 * c.size: M[j, i] = 1                   # тёмное заметно — контур/заливка
            else:
                h[1] = 0; M[j, i] = int(h.argmax())
    return M, L
