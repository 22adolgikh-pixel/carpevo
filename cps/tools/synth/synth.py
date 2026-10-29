# synth.py — синтетические ковры из схем студии (v0): ковёр в узлах → «шерсть» → фото + точная разметка.
#
# Зачем: настоящих размеченных ковров 67 (элемент — на 1–3 коврах), детектор по образцам не обобщается.
# Синтетика даёт тысячи ковров, где про каждый узел известно: зона (кайма / поле / медальон / угол) и экземпляр элемента (схема).
# На ней потом учится обычный детектор / сегментатор, а реальные ✓ рамки — для дообучения и проверки.
#
# Ковёр строится по «грамматике»: внешняя малая кайма (раппорт) → главная кайма (раппорт по 4 сторонам, углы) → внутренняя
# малая кайма → поле: решётка (ряды со сдвигом) или медальоны по оси + углы поля; всё симметрично по двум осям, как у настоящих.
# Цвета — палитра красителей (dyes.json): тело схемы (1, 2, …) → красители; повторы одного элемента иногда в разных цветах.
#
# Рендер: узел → прямоугольник s×(s·k) px, разброс цвета по узлу, полосы абраша, ворс (шум + размытие), свет, перспектива,
# фон, обрезка кадра, JPEG. Маски перерисовываются тем же преобразованием (без размытия).
#
# Выход (папка OUT): img/NNNNN.jpg, zone/NNNNN.png (индекс зоны), inst/NNNNN.png (номер экземпляра, 16 бит),
#   ann/NNNNN.json {zones: {...}, inst: [{id, scheme, zone, box (доли кадра), rot, flip}], layout, seed}.
# Запуск: python3 synth.py --data <папка data из cps-data> --out <папка> --n 50 [--seed 1] [--knots]
import os, sys, json, glob, math, random, argparse
import numpy as np, cv2

ZONES = {0: 'фон', 1: 'внешняя малая кайма', 2: 'главная кайма', 3: 'внутренняя малая кайма', 4: 'поле', 5: 'медальон', 6: 'угол поля', 7: 'элемент поля'}


# ---------------- схемы ----------------
def rows_to_arr(rows):
    h = len(rows); w = max(len(r) for r in rows); a = np.zeros((h, w), np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch.isdigit(): a[y, x] = int(ch)
    return a


def load_schemes(data):
    els, strips = [], []
    for f in sorted(glob.glob(os.path.join(data, 'work', '*.json'))):
        try: w = json.load(open(f, encoding='utf-8'))
        except Exception: continue
        mx = w.get('matrix')
        if not mx or not w.get('done') or w.get('kind') in ('photo', 'junk'): continue
        meta = w.get('meta') or {}
        if meta.get('section') == 'Изображение людей': continue
        a = rows_to_arr(mx['rows'])
        if a.max() == 0: continue
        rap = w.get('rapport') or {}
        u = (rap.get('unit') or {})
        if rap.get('found') and u.get('rows') and rap.get('axis') == 'h' and 4 <= u.get('h', 0) <= 40 and u.get('w', 0) >= 4:
            strips.append(dict(id=w['id'], a=rows_to_arr(u['rows']), type=meta.get('type', '')))
            continue
        h, ww = a.shape
        if 9 <= h <= 140 and 9 <= ww <= 140 and max(h, ww) / min(h, ww) <= 2.6 and (a > 0).mean() > 0.12:
            els.append(dict(id=w['id'], a=a, type=meta.get('type', '')))
    return els, strips


def load_dyes(data):
    try: d = json.load(open(os.path.join(data, 'dyes.json'), encoding='utf-8'))['dyes']
    except Exception: d = []
    cols = [tuple(int(x['hex'][i:i + 2], 16) for i in (1, 3, 5)) for x in d if x.get('hex')]
    return cols or [(184, 64, 47), (122, 42, 42), (216, 119, 58), (30, 40, 80), (230, 220, 190), (40, 30, 30), (60, 100, 70), (200, 160, 60)]


# ---------------- ковёр в узлах ----------------
class Rug:
    """C — индекс цвета узла (0..), Z — зона, I — экземпляр (0 = нет)."""
    def __init__(s, H, W):
        s.C = np.zeros((H, W), np.int16); s.Z = np.zeros((H, W), np.uint8); s.I = np.zeros((H, W), np.int32); s.inst = []

    def put(s, a, y, x, cmap, zone, scheme, rot=0, flip=False, inst=True):
        b = np.rot90(a, rot // 90)
        if flip: b = b[:, ::-1]
        h, w = b.shape; H, W = s.C.shape
        y0, x0 = max(0, y), max(0, x); y1, x1 = min(H, y + h), min(W, x + w)
        if y1 <= y0 or x1 <= x0: return None
        sub = b[y0 - y:y1 - y, x0 - x:x1 - x]; m = sub > 0
        s.C[y0:y1, x0:x1][m] = np.asarray(cmap)[sub[m]]
        if inst:
            k = len(s.inst) + 1
            s.I[y0:y1, x0:x1][m] = k
            s.inst.append(dict(id=k, scheme=scheme, zone=zone, knots=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)], rot=rot, flip=bool(flip)))
            return k


def tile_strip(R, strip, y0, x0, length, horizontal, cmap, zone, flip_v=False):
    """полоса раппорта вдоль стороны; вертикальные стороны — поворот на 90°"""
    u = strip['a']
    if flip_v: u = u[::-1]
    if not horizontal: u = np.rot90(u, 3)
    h, w = u.shape; n = 0
    if horizontal:
        x = x0
        while x < x0 + length:
            seg = u[:, :min(w, x0 + length - x)]
            R.put(seg, y0, x, cmap, zone, strip['id'], inst=seg.shape[1] == w); x += w; n += 1
    else:
        y = y0
        while y < y0 + length:
            seg = u[:min(h, y0 + length - y), :]
            R.put(seg, y, x0, cmap, zone, strip['id'], inst=seg.shape[0] == h); y += h; n += 1
    return n


def compose(rng, els, strips, ncol):
    # --- размеры ---
    W = rng.randint(130, 230); H = int(W * rng.uniform(1.25, 1.9))
    R = Rug(H, W)
    # палитра ковра: 0 — поле, далее цвета кайм и элементов
    base = ncol[:]; rng.shuffle(base)
    field_c, border_c, guard_c = 0, 1, 2           # индексы в палитре ковра
    pal = base[:8]
    R.C[:] = field_c; R.Z[:] = 4
    def cm(*idx): return [0] + list(idx) + [idx[-1]] * 8       # 0 схемы = прозрачно (не ставится)

    # --- каймы: внешняя малая, главная, внутренняя малая ---
    def ring(d, t, zone, color):
        R.Z[d:H - d, d:W - d][:t, :] = zone; R.Z[d:H - d, d:W - d][-t:, :] = zone
        R.Z[d:H - d, d:W - d][:, :t] = zone; R.Z[d:H - d, d:W - d][:, -t:] = zone
        R.C[R.Z == zone] = color

    small = [s for s in strips if s['a'].shape[0] <= 12] or strips
    big = [s for s in strips if s['a'].shape[0] >= 12] or strips
    d = rng.randint(1, 3)
    R.C[:d, :] = 3; R.C[-d:, :] = 3; R.C[:, :d] = 3; R.C[:, -d:] = 3; R.Z[:d, :] = 1; R.Z[-d:, :] = 1; R.Z[:, :d] = 1; R.Z[:, -d:] = 1
    layout = []
    for zone, pool in ((1, small), (2, big), (3, small)):
        st = rng.choice(pool); t = st['a'].shape[0]
        if zone == 2 and t < 0.07 * W: t2 = t
        else: t2 = t
        if d + 2 * t2 >= min(H, W) // 2 - 20: break
        bg = rng.choice([border_c, guard_c, 3]) if zone != 2 else border_c
        ring(d, t2, zone, bg)
        ink = [rng.randrange(len(pal)) for _ in range(4)]
        cmap = cm(*ink)
        # верх и низ (низ — зеркально), левая и правая — повёрнутая полоса
        tile_strip(R, st, d, d + t2, W - 2 * (d + t2), True, cmap, zone)
        tile_strip(R, st, H - d - t2, d + t2, W - 2 * (d + t2), True, cmap, zone, flip_v=True)
        tile_strip(R, st, d + t2, d, H - 2 * (d + t2), False, cmap, zone)
        tile_strip(R, dict(st, a=st['a'][::-1]), d + t2, W - d - t2, H - 2 * (d + t2), False, cmap, zone)
        if zone == 2:
            ce = [e for e in els if max(e['a'].shape) <= t2] or [None]
            ce = rng.choice(ce)
            if ce is not None:
                b = ce['a']; ink2 = cm(*[rng.randrange(len(pal)) for _ in range(3)])
                for cy, cx in ((d, d), (d, W - d - t2), (H - d - t2, d), (H - d - t2, W - d - t2)):
                    R.put(b, cy + (t2 - b.shape[0]) // 2, cx + (t2 - b.shape[1]) // 2, ink2, zone, ce['id'])
        layout.append(dict(zone=zone, scheme=st['id'], inset=d, width=t2)); d += t2 + rng.randint(0, 2)
    fy0, fx0, fy1, fx1 = d, d, H - d, W - d
    R.Z[fy0:fy1, fx0:fx1] = 4; R.C[fy0:fy1, fx0:fx1] = field_c
    fh, fw = fy1 - fy0, fx1 - fx0

    kind = rng.choice(['lattice', 'medallion', 'lattice'])
    if kind == 'medallion':
        n = rng.choice([1, 2, 3, 3]); big_els = [e for e in els if min(e['a'].shape) >= 25] or els
        med = rng.choice(big_els); a = med['a']; sc = 1
        while a.shape[0] * (sc + 1) * n < fh * 0.8 and a.shape[1] * (sc + 1) < fw * 0.6 and sc < 3: sc += 1
        a = np.kron(a, np.ones((sc, sc), np.uint8)) if sc > 1 else a
        if a.shape[1] > fw - 6 or a.shape[0] * n > fh - 6: a = med['a']
        gap = (fh - n * a.shape[0]) / (n + 1)
        for i in range(n):
            y = int(fy0 + gap * (i + 1) + a.shape[0] * i); x = fx0 + (fw - a.shape[1]) // 2
            ink = cm(*[rng.randrange(len(pal)) for _ in range(4)])
            k = R.put(a, y, x, ink, 5, med['id'])
            if k: R.Z[R.I == k] = 5
        # углы поля: один элемент, четыре раза с отражениями
        sp = rng.choice(els); b = sp['a']; ink = cm(*[rng.randrange(len(pal)) for _ in range(4)])
        if b.shape[0] < fh / 3 and b.shape[1] < fw / 3:
            for fy, fx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                bb = b[::-1] if fy else b; bb = bb[:, ::-1] if fx else bb
                y = fy0 + 2 if not fy else fy1 - 2 - b.shape[0]; x = fx0 + 2 if not fx else fx1 - 2 - b.shape[1]
                k = R.put(bb, y, x, ink, 6, sp['id'], flip=False)
                if k: R.inst[-1].update(rot=180 if fy and fx else 0, flip=bool(fy) != bool(fx)); R.Z[R.I == k] = 6
        fill_free(R, rng, els, pal, cm, fy0, fx0, fy1, fx1, n=rng.randint(25, 70))
    else:
        e = rng.choice(els); a = e['a']; h, w = a.shape
        gy, gx = int(h * rng.uniform(1.05, 1.35)), int(w * rng.uniform(1.05, 1.4))
        ny, nx = max(1, fh // gy), max(1, fw // gx)
        oy, ox = fy0 + (fh - ny * gy) // 2, fx0 + (fw - nx * gx) // 2
        brick = rng.random() < 0.5
        inks = [cm(*[rng.randrange(len(pal)) for _ in range(4)]) for _ in range(rng.choice([1, 2, 3, 4]))]
        for iy in range(ny):
            sh = gx // 2 if brick and iy % 2 else 0
            for ix in range(nx + (1 if sh else 0)):
                y = oy + iy * gy + (gy - h) // 2; x = ox + ix * gx + (gx - w) // 2 + sh - (gx if sh else 0) // 2 * 0
                if x + w > fx1 or x < fx0: continue
                fl = rng.random() < 0.2
                R.put(a, y, x, inks[(iy + ix) % len(inks)], 7, e['id'], flip=fl)
        layout.append(dict(field='lattice', scheme=e['id'], step=[gx, gy], brick=brick))
        fill_free(R, rng, els, pal, cm, fy0, fx0, fy1, fx1, n=rng.randint(10, 40), small_only=True)
    layout.append(dict(field=kind, rect=[fx0, fy0, fw, fh], size=[W, H]))
    return R, pal, layout


def fill_free(R, rng, els, pal, cm, fy0, fx0, fy1, fx1, n=10, small_only=False):
    """мелкие элементы-заполнители в свободных местах поля (как у настоящих ковров: розетки, гребни, s-знаки)"""
    sm = [e for e in els if max(e['a'].shape) <= 30] or els
    for _ in range(n * 12):
        if n <= 0: break
        e = rng.choice(sm); a = e['a']; h, w = a.shape
        if h >= fy1 - fy0 - 4 or w >= fx1 - fx0 - 4: continue
        y = rng.randint(fy0 + 1, fy1 - h - 1); x = rng.randint(fx0 + 1, fx1 - w - 1)
        if R.I[y:y + h, x:x + w].any(): continue
        # симметричная пара по вертикальной оси (ковры симметричны)
        x2 = (fx0 + fx1) - x - w
        ink = cm(*[rng.randrange(len(pal)) for _ in range(3)])
        R.put(a, y, x, ink, 7, e['id'])
        if abs(x2 - x) > w and not R.I[y:y + h, x2:x2 + w].any(): R.put(a[:, ::-1], y, x2, ink, 7, e['id'], flip=True)
        n -= 1


# ---------------- рендер «под шерсть» ----------------
def render(R, pal, rng, out_side=1100):
    H, W = R.C.shape; P = np.array(pal + [(25, 22, 20)] * 4, np.float32); nr = np.random.default_rng(rng.randrange(1 << 30))
    # цвет узла с разбросом + абраш (горизонтальные полосы оттенка, как от разных партий пряжи)
    col = P[np.clip(R.C, 0, len(P) - 1)]
    col *= nr.uniform(0.92, 1.08, (H, W, 1)).astype(np.float32)
    ab = np.ones((H, 1, 3), np.float32)
    y = 0
    while y < H:
        t = rng.randint(4, 30)
        if rng.random() < 0.25: ab[y:y + t] *= np.float32(rng.uniform(0.9, 1.06)) * np.array([rng.uniform(0.97, 1.03) for _ in range(3)], np.float32)
        y += t
    col *= ab
    # узел → s×(s·k) пикселей (узлы чуть вытянуты по вертикали или горизонтали)
    s = rng.choice([3, 4, 5, 6]); k = rng.uniform(0.75, 1.3); sy = max(2, int(round(s * k)))
    img = cv2.resize(col, (W * s, H * sy), interpolation=cv2.INTER_NEAREST)
    Z = cv2.resize(R.Z, (W * s, H * sy), interpolation=cv2.INTER_NEAREST)
    I = cv2.resize(R.I.astype(np.int32).astype(np.float32), (W * s, H * sy), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    # ворс: шум по пикселям, размытие (ворс сглаживает ступеньки), лёгкая текстура узлов
    img += rng.uniform(4, 14) * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(img.shape[:2] + (1,)).astype(np.float32)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img *= (0.94 + 0.06 * np.cos(2 * np.pi * xx / s)[..., None] * np.cos(2 * np.pi * yy / sy)[..., None]).astype(np.float32)
    b = rng.uniform(0.5, 1.6) * s / 3; img = cv2.GaussianBlur(img, (0, 0), b)
    # износ: светлые потёртые пятна
    for _ in range(rng.randint(0, 4)):
        cx, cy, r = rng.randrange(img.shape[1]), rng.randrange(img.shape[0]), rng.randint(20, 160)
        m = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))[..., None].astype(np.float32)
        img = img * (1 - 0.35 * m) + 0.35 * m * np.float32([150, 135, 115])
    # свет: градиент + общая яркость / баланс белого
    g = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.15), img.shape[1], dtype=np.float32)[None, :, None]
    img = img * g * np.float32([rng.uniform(0.85, 1.12) for _ in range(3)])
    # перспектива и фон
    h, w = img.shape[:2]; pad = int(max(h, w) * rng.uniform(0.03, 0.25))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = lambda: rng.uniform(-0.06, 0.06) * max(w, h)
    dst = np.float32([[pad + jit(), pad + jit()], [pad + w + jit(), pad + jit()], [pad + w + jit(), pad + h + jit()], [pad + jit(), pad + h + jit()]])
    ang = rng.uniform(-4, 4); M0 = cv2.getRotationMatrix2D((w / 2 + pad, h / 2 + pad), ang, 1)
    dst = cv2.transform(dst[None], M0)[0]
    OW, OH = int(dst[:, 0].max() + pad), int(dst[:, 1].max() + pad)
    Mp = cv2.getPerspectiveTransform(src, dst)
    bgc = np.float32(rng.choice([(235, 232, 225), (245, 245, 245), (60, 55, 50), (150, 130, 110), (200, 190, 175)]))
    bg = np.ones((OH, OW, 3), np.float32) * bgc + rng.uniform(2, 10) * np.random.default_rng(rng.randrange(1 << 30)).standard_normal((OH, OW, 1)).astype(np.float32)
    warped = cv2.warpPerspective(img, Mp, (OW, OH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), Mp, (OW, OH), flags=cv2.INTER_LINEAR)[..., None]
    img = warped * mask + bg * (1 - mask)
    Zw = cv2.warpPerspective(Z, Mp, (OW, OH), flags=cv2.INTER_NEAREST, borderValue=0)
    Iw = cv2.warpPerspective(I.astype(np.float32), Mp, (OW, OH), flags=cv2.INTER_NEAREST, borderValue=0).astype(np.int32)
    # кадр: иногда ковёр целиком, иногда фрагмент (деталь)
    if rng.random() < 0.25:
        cw, ch = int(OW * rng.uniform(0.35, 0.7)), int(OH * rng.uniform(0.35, 0.7))
        x0, y0 = rng.randrange(OW - cw), rng.randrange(OH - ch)
        img, Zw, Iw = img[y0:y0 + ch, x0:x0 + cw], Zw[y0:y0 + ch, x0:x0 + cw], Iw[y0:y0 + ch, x0:x0 + cw]
    sc = out_side / max(img.shape[:2])
    nsz = (int(img.shape[1] * sc), int(img.shape[0] * sc))
    img = cv2.resize(img, nsz, interpolation=cv2.INTER_AREA)
    Zw = cv2.resize(Zw, nsz, interpolation=cv2.INTER_NEAREST); Iw = cv2.resize(Iw.astype(np.float32), nsz, interpolation=cv2.INTER_NEAREST).astype(np.int32)
    img = np.clip(img, 0, 255).astype(np.uint8)[..., ::-1]   # RGB → BGR для cv2
    q = rng.randint(55, 92); ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), Zw, Iw


def boxes(R, Iw):
    out = []; Hh, Ww = Iw.shape
    for it in R.inst:
        ys, xs = np.nonzero(Iw == it['id'])
        if len(xs) < 20: continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        out.append(dict(it, box=[round(x0 / Ww, 4), round(y0 / Hh, 4), round((x1 - x0) / Ww, 4), round((y1 - y0) / Hh, 4)], px=int(len(xs))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=20); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--side', type=int, default=1100)
    ap.add_argument('--knots', action='store_true', help='сохранять ещё ковёр в узлах (1 px = 1 узел)')
    A = ap.parse_args()
    els, strips = load_schemes(A.data); dyes = load_dyes(A.data)
    print(f'схем-элементов {len(els)}, полос кайм {len(strips)}, красителей {len(dyes)}', flush=True)
    for d in ('img', 'zone', 'inst', 'ann') + (('knots',) if A.knots else ()): os.makedirs(os.path.join(A.out, d), exist_ok=True)
    for i in range(A.n):
        seed = A.seed * 100000 + i; rng = random.Random(seed)
        R, pal, layout = compose(rng, els, strips, dyes)
        img, Z, I = render(R, pal, rng, A.side)
        nm = f'{seed:08d}'
        cv2.imwrite(os.path.join(A.out, 'img', nm + '.jpg'), img)
        cv2.imwrite(os.path.join(A.out, 'zone', nm + '.png'), Z)
        cv2.imwrite(os.path.join(A.out, 'inst', nm + '.png'), np.clip(I, 0, 65535).astype(np.uint16))
        if A.knots:
            P = np.array(pal + [(25, 22, 20)] * 4, np.uint8); cv2.imwrite(os.path.join(A.out, 'knots', nm + '.png'), P[np.clip(R.C, 0, len(P) - 1)][..., ::-1])
        json.dump(dict(seed=seed, size=[img.shape[1], img.shape[0]], zones=ZONES, layout=layout, inst=boxes(R, I)),
                  open(os.path.join(A.out, 'ann', nm + '.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        if (i + 1) % 10 == 0: print(i + 1, flush=True)


if __name__ == '__main__':
    main()
