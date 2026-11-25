#!/usr/bin/env python3
"""Автотесты CPS: поднимают студию на копии кода с тестовыми данными и проходят экраны в браузере (Playwright).

Запуск (из корня репозитория):   python3 cps/tests/run_tests.py
  --keep      не удалять временную папку (посмотреть данные после тестов)
  --only rug  запустить только тесты, в названии которых есть «rug»
Нужно: pip install fastapi uvicorn python-multipart opencv-python-headless numpy pymupdf playwright && python3 -m playwright install chromium
Скриншоты каждого шага — cps/tests/out/*.png (в git не попадают). Код выхода 0 — всё прошло, 1 — есть падения.

Правило: каждое изменение интерфейса — сначала (или вместе с ним) тест сюда. Горячие клавиши проверяются здесь же —
если клавиша перестала работать, тест упадёт с её названием.
"""
import re
import os, sys, json, time, shutil, socket, subprocess, tempfile, traceback, urllib.request, urllib.parse, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); CPS = os.path.dirname(HERE); OUT = os.path.join(HERE, 'out')

ap = argparse.ArgumentParser(); ap.add_argument('--keep', action='store_true'); ap.add_argument('--only', default='')
ARGS = ap.parse_args()


# ---------------- тестовые данные ----------------
def make_rugs(data):
    import cv2, numpy as np
    os.makedirs(f'{data}/rugs', exist_ok=True); os.makedirs(f'{data}/rug_img', exist_ok=True)

    def carpet(w, h, seed):
        rng = np.random.default_rng(seed); a = np.zeros((h, w, 3), np.uint8); a[:] = (40, 40, 150)
        b = int(min(w, h) * .12); a[b:h - b, b:w - b] = (120, 60, 20)
        for y in range(b + 20, h - b - 20, 60):
            for x in range(b + 20, w - b - 20, 60):
                c = tuple(int(v) for v in rng.integers(60, 255, 3)); cv2.drawMarker(a, (x, y), c, cv2.MARKER_DIAMOND, 40, 4)
        return a

    def save(rid, img, **kw):
        cv2.imwrite(f'{data}/rug_img/{rid}.jpg', img)
        r = {"id": rid, "title": "", "source": "тест", "museum": "", "inv": "", "url": "", "license": "", "place": "", "date": "", "school": "",
             "composition": "", "technique": "", "size": "", "notes": "", "status": "new", "w": img.shape[1], "h": img.shape[0], "marks": [],
             "created": int(time.time()), "updated": int(time.time())}
        r.update(kw); json.dump(r, open(f'{data}/rugs/{rid}.json', 'w'), ensure_ascii=False)

    page = np.full((700, 1100, 3), 235, np.uint8); page[50:650, 60:480] = carpet(420, 600, 1); page[50:650, 620:1040] = carpet(420, 600, 2)
    pf = [{"label": "Рис. 12", "caption": "Ковёр «Гымыл»", "composition": "", "school": "guba-shirvan", "date": "XIX в.", "technique": "ворсовый", "text": "", "title": "Керимов, т. II, Рис. 12. Гымыл"},
          {"label": "Рис. 13", "caption": "Ковёр «Шахназарли»", "composition": "", "school": "guba-shirvan", "date": "1850", "technique": "ворсовый", "text": "", "title": "Керимов, т. II, Рис. 13. Шахназарли"}]
    save("kerimov_vol2_p040_1", page, title="Керимов, т. II (стр. 40)", school="guba-shirvan", technique="ворсовый", license="книга",
         cat={"src": "kerimov_vol2", "page_figs": pf}, multi=[[0.05, 0.07, 0.39, 0.86], [0.56, 0.07, 0.39, 0.86]])
    k = cv2.resize(carpet(420, 600, 3), (280, 400)); m = cv2.resize(carpet(420, 600, 3), (840, 1200))
    save("kerimov_vol3_p100_1", k, title="Керимов, т. III, Челеберд", cat={"src": "kerimov_vol3"}, same_group="g1", same=[{"id": "met_123", "score": 80}], school="karabakh")
    save("met_123", m, title="Carpet, Karabagh", museum="Метрополитен", cat={"src": "met"}, same_group="g1", same=[{"id": "kerimov_vol3_p100_1", "score": 80}])
    def frieze_rug():                # v11.19: ковёр с каймой из повторяющегося узора (раппорт 40 px)
        a = np.full((760, 560, 3), 235, np.uint8); a[30:730, 30:530] = (40, 40, 150)
        t = np.full((40, 60, 3), (60, 140, 200), np.uint8); cv2.rectangle(t, (6, 6), (28, 34), (240, 240, 230), -1); cv2.line(t, (34, 4), (56, 36), (20, 90, 40), 5)
        for y in range(30, 730 - 40, 40): a[y:y + 40, 30:90] = t; a[y:y + 40, 470:530] = cv2.flip(t, 1)
        tt = cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE)
        for x in range(90, 470 - 40, 40): a[30:90, x:x + 40] = tt; a[670:730, x:x + 40] = cv2.flip(tt, 0)
        a[90:670, 90:470] = carpet(380, 580, 77)[:580, :380]
        return a
    save("frieze_1", frieze_rug(), title="Rug with border", cat={"src": "azerbaijanrugs"})
    # v11.30: тот же ковёр с каймой, снятый в перспективе на сером фоне — для «4 углов»
    fr = frieze_rug()[30:730, 30:530]; Wp, Hp = 900, 800
    QT = np.float32([[239, 167], [624, 144], [628, 597], [325, 584]])   # проекция ковра 500×700 обычной камерой (фокус 900, наклон 28°, поворот 12°)
    T = cv2.getPerspectiveTransform(np.float32([[0, 0], [500, 0], [500, 700], [0, 700]]), QT)
    ph = cv2.warpPerspective(fr, T, (Wp, Hp), dst=np.full((Hp, Wp, 3), (205, 208, 212), np.uint8), borderMode=cv2.BORDER_TRANSPARENT)
    save("persp_1", ph, title="Rug photographed at an angle", cat={"src": "azerbaijanrugs"})
    tl = np.full((1500, 700, 3), (210, 210, 214), np.uint8); tl[60:1440, 60:640] = cv2.resize(frieze_rug(), (580, 1380))   # v11.38: длинная дорожка 1:2 — в плитке видны все 4 угла
    save("tall_1", tl, title="Long runner", cat={"src": "azerbaijanrugs"})
    pg_ = np.full((1100, 800, 3), 238, np.uint8); rugimg = carpet(520, 700, 93); pg_[200:900, 140:660] = rugimg   # v11.22: лист книги и вырезанный с него ковёр
    save("kerimov_vol9_p050_1", pg_, title="Керимов, т. IX (стр. 50)", cat={"src": "kerimov_vol9"})
    save("kerimov_vol9_p050_2", cv2.resize(rugimg, (390, 525), interpolation=cv2.INTER_AREA), title="Керимов, т. IX (стр. 50), ковёр", cat={"src": "kerimov_vol9"},
         marks=[{"id": "mp1", "box": [0.2, 0.2, 0.2, 0.2], "zone": "field", "motif": "", "motif_text": "лист", "scheme": "", "group": "", "certainty": "sure", "status": "confirmed", "note": "", "angle": 0, "flip": 0}])
    big = carpet(600, 840, 91)            # v11.21: один и тот же снимок в двух размерах (ресайз Wikimedia)
    save("cm_shot_o", big, title="Shot original", cat={"src": "commons"})
    save("cm_shot_s", cv2.resize(big, (300, 420), interpolation=cv2.INTER_AREA), title="Shot small", cat={"src": "wikimedia"},
         marks=[{"id": "ms1", "box": [0.1, 0.1, 0.2, 0.2], "zone": "field", "motif": "", "motif_text": "тест", "scheme": "", "group": "", "certainty": "sure", "status": "confirmed", "note": "", "angle": 0, "flip": 0}])
    f2 = frieze_rug(); f2[90:670, 90:470] = (30, 110, 60); f2 = cv2.resize(f2, None, fx=1.2, fy=1.2)
    save("frieze_2", f2, title="Another rug, same border", cat={"src": "azerbaijanrugs"})
    for i in (1, 2):                 # v11.18: фото с людьми (лица найдены машиной заранее)
        save(f"ppl_{i}", carpet(300, 420, 40 + i), title=f"Rug photo {i}", cat={"src": "azerbaijanrugs"}, people={"n": i, "ver": 1, "faces": []})
    def star(a, cx, cy, r, col):     # v11.4: один и тот же «узор» на трёх коврах — для поиска по всем коврам
        pts = [(int(cx + (r if i % 2 == 0 else r * .45) * np.cos(i * np.pi / 8)), int(cy + (r if i % 2 == 0 else r * .45) * np.sin(i * np.pi / 8))) for i in range(16)]
        cv2.fillPoly(a, [np.array(pts)], col); cv2.circle(a, (cx, cy), int(r * .25), (250, 250, 250), -1)
    STARS = {0: (150, 200, (30, 200, 230)), 3: (230, 140, (200, 90, 40)), 5: (110, 300, (90, 220, 90))}
    for i in range(6):
        a = carpet(300 + 10 * i, 420, 10 + i)
        if i in STARS: star(a, *STARS[i][:2], 34, STARS[i][2])
        save(f"vam_{i}", a, title=f"V&A rug {i}", museum="V&A", cat={"src": "vam"})
    a = carpet(320, 420, 31); star(a, 160, 220, 34, (60, 200, 200)); save("vam_star", a, title="Star rug extra", museum="V&A", cat={"src": "vam"})   # v11.24: эталоны
    # v11.5: кайма с одним элементом в разных красках (как у «Гаджигайыба»: красный на светлом / светлый на чёрном)
    a = np.full((620, 300, 3), (40, 30, 120), np.uint8)
    pal = [((40, 40, 160), (230, 225, 210)), ((20, 20, 20), (230, 225, 210)), ((230, 225, 210), (40, 40, 160)), ((40, 40, 160), (20, 20, 20))]
    for i in range(6):
        bg, fg = pal[i % 4]; y0 = 20 + i * 100
        cv2.rectangle(a, (20, y0), (100, y0 + 80), bg, -1); cv2.circle(a, (60, y0 + 40), 9, fg, -1)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.ellipse(a, (60 + dx * 22, y0 + 40 + dy * 22), (14, 9), 45 * dx * dy, 0, 300, fg, 4)
        cv2.rectangle(a, (180, y0), (260, y0 + 80), (110, 60, 20), -1); cv2.line(a, (190, y0 + 10), (250, y0 + 70), (200, 200, 200), 5)
    save("border_rep", a, title="Hajigayib rug, Guba school, XIX c", cat={"src": "commons"})
    # v11.5: тип ковра по названию и одинаковый файл Wikimedia в двух размерах
    save("cm_pir", carpet(320, 420, 31), title="Pirəbədil carpet, Quba group of Azerbaijani carpets", url="https://commons.wikimedia.org/wiki/File:Pir%C9%99b%C9%99dil_carpet.jpg", cat={"src": "commons"})
    save("am_bag", carpet(400, 260, 41), title="Məfrəş (Мафраш), Ширван, Азербайджан", museum="Музей ковра", cat={"src": "azcarpetmuseum"})   # v11.6: изделие
    small = cv2.resize(carpet(320, 420, 31), (160, 210))
    save("wm_pir_small", small, title="Pirəbədil carpet.jpg", url="https://commons.wikimedia.org/wiki/File:Pirəbədil_carpet.jpg", cat={"src": "wikimedia"},
         marks=[{"id": "mSm", "box": [0.3, 0.3, 0.2, 0.2], "zone": "field", "status": "confirmed", "motif_text": "с маленькой копии"}])


def make_color_fig(data):
    """Цветной рисунок, принятый до v11.1: красители подписаны, но цвета — как на скане (связи с палитрой ещё нет)."""
    os.makedirs(f'{data}/work', exist_ok=True)
    w = {"id": "tc_f01", "sheet": "test", "table": 900, "fig": 1, "quad": [[20, 20], [140, 20], [140, 100], [20, 100]],
         "grid": {"pw": 20, "ph": 20, "ox": 0, "oy": 0, "cols": 6, "rows": 4, "square": True},
         "classify": {"tones": 4, "inner": 45, "mode": "color"},
         "matrix": {"w": 6, "h": 4, "origin": [0, 0], "rows": ["123321", "234432", "234432", "123321"]},
         "palette": ["#f2eee2", "#83423b", "#7f3336", "#2b2a2c", "#c9a23a"],
         "palette_names": ["ağ", "zoğalı", "zoğalı 2", "qara", "sarı"],
         "palette_dyes": ["ag", "zogali", "zogali", "qara", "sari"], "palette_touched": False, "color_mode": True,
         "done": True, "meta": {"name_az": "test"}, "auto": {"mode": "color", "reviewed": True}}
    json.dump(w, open(f'{data}/work/tc_f01.json', 'w'), ensure_ascii=False)
    w2 = dict(w, id="tc_f02", done=False, palette_dyes=["ag", "qirmizi", "zogali", "qara", "sari"])   # не принят: красители — только предложения
    json.dump(w2, open(f'{data}/work/tc_f02.json', 'w'), ensure_ascii=False)


def make_auto_sheet(data):
    """Лист, который нарезала машина (ночной прогон), — должен попасть в «Нужно решение»."""
    json.dump({"path": "test/auto_page_001.jpg", "table": 901, "boxes": [{"x": 10, "y": 10, "w": 100, "h": 100, "id": "t901_f01"}],
               "figures": ["t901_f01"], "auto_segmented": True}, open(f'{data}/sheets/test_auto_page_001_jpg.json', 'w'))


def make_crops(data):
    """Сканы страниц в git не лежат — для каждого рисунка с сеткой рисуем серую «вырезку» нужного размера, чтобы редактор открывался."""
    import cv2, numpy as np
    os.makedirs(f'{data}/crops', exist_ok=True)
    for fn in os.listdir(f'{data}/work'):
        if not fn.endswith('.json'): continue
        fid = fn[:-5]; p = f'{data}/crops/{fid}.png'
        if os.path.exists(p): continue
        try: w = json.load(open(f'{data}/work/{fn}'))
        except Exception: continue
        q = w.get('quad') or [[0, 0], [300, 0], [300, 300], [0, 300]]
        W = int(max(x for x, y in q)) + 40; H = int(max(y for x, y in q)) + 40
        a = np.full((H, W, 3), 230, np.uint8); a[::7, :] = 200; a[:, ::7] = 200
        cv2.imwrite(p, a)


def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=30))


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=60))


# ---------------- тесты ----------------
TESTS = []
def test(fn): TESTS.append(fn); return fn


def rg_tools_open(pg):
    """v11.92: инструменты каталога ковров — в выдвижной панели; открыть, если закрыта"""
    if not pg.is_visible('#rgTools'): pg.click('#rgToolsBtn'); pg.wait_for_timeout(500)


def rgtool(pg, tool_id):
    rg_tools_open(pg)
    pg.evaluate("id=>{const b=document.getElementById(id),d=b&&b.closest('details');if(d)d.open=true;}", tool_id)   # v12.04: устаревшие — в свёрнутом блоке
    pg.click('#' + tool_id)


def errors_ok(ctx, where):
    errs = [e for e in ctx['errors'] if 'favicon' not in e]
    ctx['errors'].clear()
    assert not errs, f'ошибки JS ({where}): ' + ' | '.join(errs[:3])


@test
def t01_load(pg, ctx):
    pg.goto(ctx['url']); pg.wait_for_timeout(2500)
    v = pg.text_content('#appVer') or ''
    assert v.startswith('v'), 'нет номера версии рядом с логотипом'
    assert pg.evaluate("getComputedStyle(document.querySelector('header')).getPropertyValue('--bg') !== undefined")
    # кнопка пользователя — в одной строке с логотипом (не отдельной линией)
    hb = pg.evaluate("(()=>{const h=document.querySelector('header').getBoundingClientRect();return h.height})()")
    assert hb < 70, f'шапка слишком высокая ({hb:.0f} px) — что-то переносится на вторую строку'
    errors_ok(ctx, 'загрузка'); pg.screenshot(path=f'{OUT}/01_load.png')


@test
def t02_tabs(pg, ctx):
    for t in ('tab-pipe', 'tab-queue', 'tab-rugs'):
        pg.click('#' + t); pg.wait_for_timeout(900)
        errors_ok(ctx, 'вкладка ' + t)
    pg.click('#tab-queue'); pg.wait_for_timeout(600); pg.click('#catalog [data-vs="sheet"]'); pg.wait_for_timeout(900)
    assert pg.evaluate("S.mode") == 'sheet', 'вкладка «Листы» внутри «Узоров» не открывается'
    errors_ok(ctx, 'листы')
    pg.screenshot(path=f'{OUT}/02_tabs.png')


@test
def t03_pattern_hotkeys(pg, ctx):
    pg.click('#tab-queue'); pg.wait_for_timeout(600)
    fid = pg.evaluate("S.figures.find(f=>f.has_matrix||f.done)?.id || S.figures[0]?.id")
    assert fid, 'нет рисунков в тестовых данных'
    pg.evaluate(f"openFigure('{fid}')"); pg.wait_for_timeout(1500)
    mode = pg.evaluate("S.mode")
    assert mode in ('pix', 'grid'), f'редактор узора не открылся (режим {mode})'
    pg.keyboard.press('Tab'); pg.wait_for_timeout(400)
    m2 = pg.evaluate("S.mode"); assert m2 != mode and m2 in ('pix', 'grid'), 'Tab не переключает Геометрия ↔ Пиксели'
    if pg.evaluate("S.mode") != 'pix': pg.keyboard.press('Tab'); pg.wait_for_timeout(400)
    if pg.evaluate("!!(S.fig&&S.fig.M)"):
        for key, tool in (('e', 'eraser'), ('g', 'fill'), ('f', 'fill'), ('l', 'line'), ('r', 'rect'), ('c', 'copy'), ('p', 'pen'), ('b', 'pen')):
            pg.keyboard.press(key); pg.wait_for_timeout(120)
            assert pg.evaluate("S.fig.tool") == tool, f'клавиша {key.upper()} не включает инструмент {tool}'
        if pg.evaluate("S.fig.palette.length") >= 2:
            pg.keyboard.press('2'); pg.wait_for_timeout(120)
            assert pg.evaluate("S.fig.cur") == 1, 'клавиша 2 не выбирает цвет 2'
            pg.keyboard.press('1'); pg.wait_for_timeout(120); pg.keyboard.press('x'); pg.wait_for_timeout(120)
            assert pg.evaluate("S.fig.cur") == 1, 'клавиша X не возвращает предыдущий цвет'
        u0 = pg.evaluate("S.fig.under>0")
        pg.keyboard.press('u'); pg.wait_for_timeout(120)
        assert pg.evaluate("S.fig.under>0") != u0, 'клавиша U не включает/выключает подложку'
        pg.keyboard.press('u'); pg.wait_for_timeout(120)
    pg.keyboard.press(']'); pg.wait_for_timeout(1200)
    assert pg.evaluate("S.fig.id") != fid, 'клавиша ] не переходит к следующему рисунку'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(800)
    assert pg.evaluate("S.mode") not in ('pix', 'grid'), 'Esc не закрывает редактор узора'
    errors_ok(ctx, 'редактор узора'); pg.screenshot(path=f'{OUT}/03_pattern.png')


@test
def t04b_rug_quad(pg, ctx):
    """v11.30: 4 угла ковра → выпрямление → каймы на выпрямленном; контрольный набор"""
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    pg.evaluate("rugOpen('persp_1')"); pg.wait_for_timeout(1500)
    pg.keyboard.press('z'); pg.wait_for_timeout(400)
    assert pg.is_visible('#rgQAuto'), 'в «Поле и каймы» нет кнопки «найти 4 угла»'
    pg.click('#rgQAuto'); pg.wait_for_timeout(2500)
    q = pg.evaluate("R.rug.layout&&R.rug.layout.quad")
    assert q and len(q) == 4, 'углы не найдены'
    truth = [[x / 900, y / 800] for x, y in [[239, 167], [624, 144], [628, 597], [325, 584]]]
    err = max(max(abs(a[0] - b[0]) * 900, abs(a[1] - b[1]) * 800) for a, b in zip(q, truth))
    assert err < 15, f'углы найдены неточно: ошибка {err:.0f} px'
    assert pg.evaluate("document.querySelectorAll('#rgSvg circle[data-q]').length") == 4, 'на фото нет 4 точек-углов'
    pg.wait_for_function("(()=>{const i=document.getElementById('rgRectImg');return i&&i.complete&&i.naturalWidth>0})()", timeout=8000)
    rw, rh = pg.evaluate("[document.getElementById('rgRectImg').naturalWidth,document.getElementById('rgRectImg').naturalHeight]")
    assert abs(rw / rh - 500 / 700) < 0.06, f'пропорции выпрямленного ковра неверны: {rw}×{rh}'
    # тянем угол мышью — статус «поправлены»
    c = pg.query_selector('#rgSvg circle[data-q="0"]').bounding_box(); x, y = c['x'] + c['width'] / 2, c['y'] + c['height'] / 2
    pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 12, y + 8, steps=4); pg.mouse.up(); pg.wait_for_timeout(1200)
    assert pg.evaluate("R.rug.layout.quad_status") == 'edited', 'перетаскивание угла не меняет статус'
    q2 = pg.evaluate("R.rug.layout.quad[0]"); assert abs(q2[0] - q[0][0]) > 0.003, 'угол не сдвинулся'
    # v11.32: угол можно вынести за край фото — точка и линии видны поверх тёмного поля
    c = pg.query_selector('#rgSvg circle[data-q="3"]').bounding_box(); x, y = c['x'] + c['width'] / 2, c['y'] + c['height'] / 2
    ib = pg.query_selector('#rgImg').bounding_box()
    pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(ib['x'] - 25, y, steps=5); pg.mouse.up(); pg.wait_for_timeout(1200)
    assert pg.evaluate("R.rug.layout.quad[3][0]") < 0, 'угол нельзя вынести за край фото'
    c2 = pg.query_selector('#rgSvg circle[data-q="3"]').bounding_box()
    assert c2 and c2['width'] > 0 and c2['x'] + c2['width'] / 2 < ib['x'], 'точка угла за краем фото не видна'
    assert pg.evaluate("getComputedStyle(document.getElementById('rgSvg')).overflow") == 'visible', 'линии за краем фото обрезаются'
    pg.mouse.move(c2['x'] + c2['width'] / 2, c2['y'] + c2['height'] / 2); pg.mouse.down(); pg.mouse.move(x, y, steps=5); pg.mouse.up(); pg.wait_for_timeout(1200)
    pg.click('#rgQOk'); pg.wait_for_timeout(300); assert pg.evaluate("R.rug.layout.quad_status") == 'confirmed', '«✓ углы верны» не подтверждает'
    pg.evaluate("R.layOld=true;document.querySelectorAll('details.tlold').forEach(d=>d.open=true)")   # v12.04: каймы шагами — в свёрнутом «устарело»
    pg.click('#rgLayAuto'); pg.wait_for_timeout(2500)
    nb = pg.evaluate("R.rug.layout.bounds.length"); assert nb >= 1, 'на выпрямленном ковре не найдено ни одной границы каймы'
    bx = pg.evaluate("R.rug.layout.bounds.map(b=>b.ix)"); by = pg.evaluate("R.rug.layout.bounds.map(b=>b.iy)")
    assert any(abs(v - 0.12) < 0.02 for v in bx) and any(abs(v - 60 / 700) < 0.02 for v in by), f'граница кайма/поле не найдена: ix={bx}, iy={by} (ждали ≈0.12 и ≈0.086)'
    assert pg.evaluate("layZoneAt(R.rug.layout,0.5,0.5)") == 'field', 'центр ковра не попадает в поле'
    assert pg.evaluate("layZoneAt(R.rug.layout,0.03,0.03)") is None, 'фон за углом ковра считается ковром'
    # v11.33: два шага — на шаге 1 видна и берётся мышью только линия поля (последняя), на шаге 2 — все; «＋ слой» режет каймы
    pg.evaluate("laySetStep(1)"); pg.wait_for_timeout(200)
    assert pg.evaluate("document.querySelectorAll('#rgSvg path[fill-rule=evenodd]').length") >= 1, 'на шаге 1 нет заливки «все каймы»'
    assert pg.is_visible('#rgStep2'), 'нет переключателя шагов'
    pg.evaluate("laySetStep(2)"); pg.wait_for_timeout(200)
    nb0 = pg.evaluate("R.rug.layout.bounds.length"); pg.click('#rgLayAdd'); pg.wait_for_timeout(300)
    assert pg.evaluate("R.rug.layout.bounds.length") == nb0 + 1, '«＋ слой» не добавляет линию'
    assert pg.evaluate("R.rug.layout.bounds.every((b,i,a)=>i==0||a[i-1].ix<=b.ix)"), 'линии после добавления слоя не по порядку'
    pg.evaluate("R.layBound=R.rug.layout.bounds.length-2;layRemove(R.rug.layout)"); pg.wait_for_timeout(200)
    # v11.31: рамка элемента поверх линии каймы не мешает выбрать и удалить линию (кликом, Delete и кнопкой)
    def line_pt(i):
        return pg.evaluate(f"""(()=>{{const L=R.rug.layout,b=L.bounds[{i}],[x,y]=layToImg(L,b.ix,0.5),V=document.getElementById('rgView').getBoundingClientRect();
          return [V.left+R.ox+x*R.iw*R.z, V.top+R.oy+y*R.ih*R.z];}})()""")
    pg.evaluate("""(()=>{const L=R.rug.layout,b=L.bounds[0],[x,y]=layToImg(L,b.ix,0.5);
      R.rug.marks.push({id:'mz1',box:[x-0.03,y-0.05,0.06,0.1],zone:'border',motif:'',motif_text:'кайма',scheme:'',group:'',certainty:'sure',status:'proposed',note:'',angle:0,flip:0});rugDraw();})()""")
    n0 = pg.evaluate("R.rug.layout.bounds.length"); x, y = line_pt(0)
    pg.mouse.click(x, y); pg.wait_for_timeout(300)
    assert pg.evaluate("R.layBound") == 0, 'клик по линии каймы под рамкой элемента не выбирает линию'
    assert pg.evaluate("R.sel") is None, 'в режиме зон клик выбрал рамку элемента вместо линии'
    pg.keyboard.press('Delete'); pg.wait_for_timeout(300)
    assert pg.evaluate("R.rug.layout.bounds.length") == n0 - 1, 'Delete не удаляет выбранную линию каймы'
    assert pg.evaluate("R.rug.layout.rings.length") == n0, 'после удаления линии число полос не сходится'
    if n0 - 1 >= 1:
        x, y = line_pt(0); pg.mouse.click(x, y); pg.wait_for_timeout(300)
        assert pg.is_visible('#rgLayRm'), 'нет кнопки «✕ выбранную линию»'
        pg.click('#rgLayRm'); pg.wait_for_timeout(300)
        assert pg.evaluate("R.rug.layout.bounds.length") == n0 - 2, 'кнопка не удаляет линию'
    pg.evaluate("R.rug.marks=R.rug.marks.filter(m=>m.id!=='mz1');rugSaveSoon()")
    pg.evaluate("R.layOld=true;document.querySelectorAll('details.tlold').forEach(d=>d.open=true)")   # v12.04: каймы шагами — в свёрнутом «устарело»
    pg.click('#rgLayAuto'); pg.wait_for_timeout(2500)
    # перетаскивание линии: типы полос едут вместе с линиями
    rg0 = pg.evaluate("R.rug.layout.rings.slice()"); x, y = line_pt(0)
    pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 15, y, steps=4); pg.mouse.up(); pg.wait_for_timeout(400)
    assert pg.evaluate("R.rug.layout.status") == 'edited', 'линия каймы не перетаскивается'
    pg.check('#rgCtl'); pg.wait_for_timeout(1500)
    r = json.load(urllib.request.urlopen(ctx['url'] + 'api/rugs/persp_1'))
    assert r.get('control') is True, 'отметка «контрольный набор» не сохранилась'
    assert (r.get('layout') or {}).get('quad_status') == 'confirmed' and len(r['layout']['quad']) == 4, 'углы не сохранились на сервере'
    errors_ok(ctx, '4 угла'); pg.screenshot(path=f'{OUT}/04b_rug_quad.png')
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)


@test
def t04_rug_grid_and_editor(pg, ctx):
    pg.click('#tab-rugs'); pg.wait_for_timeout(1500)
    n = pg.evaluate("document.querySelectorAll('.rgcard').length"); assert n >= 3, f'в плитке ковров {n} карточек'
    pg.screenshot(path=f'{OUT}/04_rug_grid.png')
    pg.click('.rgcard[data-id="vam_2"]'); pg.wait_for_timeout(1500)
    assert pg.evaluate("R.view") == 'edit', 'ковёр не открылся в редакторе'
    bb = pg.query_selector('#rgView').bounding_box(); x, y = bb['x'] + bb['width'] * .45, bb['y'] + bb['height'] * .4
    before = pg.evaluate("R.rug.marks.length")
    pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 50, y + 50, steps=5); pg.mouse.up(); pg.wait_for_timeout(2500)
    assert pg.evaluate("R.rug.marks.length") > before, 'рамка мышью не создаётся'
    for key, tool in (('h', 'hand'), ('b', 'box'), ('z', 'zones')):
        pg.keyboard.press(key); pg.wait_for_timeout(200)
        assert pg.evaluate("R.tool") == tool, f'клавиша {key.upper()} не включает «{tool}»'
    pg.keyboard.press('z'); pg.wait_for_timeout(200); assert pg.evaluate("R.tool") == 'box', 'повторное Z не выходит из режима зон'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200); assert pg.evaluate("R.sel") is None, 'Esc не снимает выделение'
    pg.keyboard.press('Tab'); pg.wait_for_timeout(200); assert pg.evaluate("!!R.sel"), 'Tab не выбирает элемент'
    sid = pg.evaluate("R.sel.id"); pg.keyboard.press('c'); pg.wait_for_timeout(200)
    assert pg.evaluate(f"R.rug.marks.find(m=>m.id==='{sid}').status") == 'confirmed', 'C не подтверждает элемент'
    sid = pg.evaluate("R.sel.id"); pg.keyboard.press('x'); pg.wait_for_timeout(200)
    assert pg.evaluate(f"R.rug.marks.find(m=>m.id==='{sid}').status") == 'rejected', 'X не отклоняет элемент'
    pg.keyboard.press('2'); pg.wait_for_timeout(200); assert pg.evaluate("R.sel.zone") == 'medallion', 'цифра 2 не ставит зону «медальон»'
    k0 = pg.evaluate("R.rug.marks.length"); pg.keyboard.press('Delete'); pg.wait_for_timeout(300)
    assert pg.evaluate("R.rug.marks.length") == k0 - 1, 'Delete не удаляет рамку'
    pg.wait_for_timeout(1200)
    errors_ok(ctx, 'редактор ковра'); pg.screenshot(path=f'{OUT}/05_rug_editor.png')


@test
def t05_rug_passport_popup(pg, ctx):
    pg.evaluate("rugOpen('kerimov_vol3_p100_1')"); pg.wait_for_timeout(1500)
    pg.click('#rgPassBtn'); pg.wait_for_timeout(500)
    assert pg.is_visible('#modal') and pg.is_visible('#modal [data-f="title"]'), 'паспорт ковра не открылся во всплывающем окне'
    pg.screenshot(path=f'{OUT}/06_rug_passport.png')
    # v12.31: ID и кнопки «скачать» в паспорте
    assert (pg.text_content('#modal #rgIdTxt') or '').strip() == 'kerimov_vol3_p100_1', 'в паспорте не показан ID ковра'
    assert pg.is_visible('#modal #rgIdCopy'), 'в паспорте нет кнопки «копировать» ID'
    for sel, frag in (('#rgDlRaw', '/api/open/file/raw/kerimov_vol3_p100_1.jpg'), ('#rgDlRect', '/api/open/file/rect/kerimov_vol3_p100_1.jpg'), ('#rgDlCard', '/api/open/card/kerimov_vol3_p100_1.json')):
        h = pg.get_attribute('#modal ' + sel, 'href') or ''
        assert h.endswith(frag), f'кнопка скачивания {sel} ведёт не туда: «{h}»'
        assert pg.get_attribute('#modal ' + sel, 'download'), f'у кнопки {sel} нет атрибута download'
    assert get(ctx['url'] + 'api/open/card/kerimov_vol3_p100_1.json')['id'] == 'kerimov_vol3_p100_1', 'карточка по ссылке из паспорта не отдаётся'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
    assert not pg.is_visible('#modal'), 'Esc не закрывает паспорт'
    pg.keyboard.press('i'); pg.wait_for_timeout(400)
    assert pg.is_visible('#modal [data-f="title"]'), 'клавиша I не открывает паспорт'
    pg.fill('#modal [data-f="title"]', 'Тестовое название'); pg.wait_for_timeout(1500)
    pg.click('#mClose'); pg.wait_for_timeout(300)
    assert not pg.is_visible('#modal'), 'кнопка «Закрыть» не закрывает паспорт'
    assert get(ctx['url'] + 'api/rugs/kerimov_vol3_p100_1')['title'] == 'Тестовое название', 'правка в паспорте не сохранилась'
    errors_ok(ctx, 'паспорт ковра')


@test
def t06_rug_statusbar(pg, ctx):
    pg.evaluate("rugOpen('kerimov_vol3_p100_1')"); pg.wait_for_timeout(1200)
    import re
    txt = pg.text_content('#sbar') or ''
    assert re.search(r'\d+\s*/\s*\d+', txt), f'в строке состояния нет позиции ковра (‹ N / M ›): «{txt[:80]}»'
    pg.keyboard.press('ArrowRight'); pg.wait_for_timeout(1200)
    assert pg.evaluate("R.rug.id") != 'kerimov_vol3_p100_1', '→ не переходит к следующему ковру'
    pg.keyboard.press('ArrowLeft'); pg.wait_for_timeout(1200)
    assert pg.evaluate("R.rug.id") == 'kerimov_vol3_p100_1', '← не возвращает к предыдущему ковру'
    pg.keyboard.press(']'); pg.wait_for_timeout(1200); assert pg.evaluate("R.rug.id") != 'kerimov_vol3_p100_1', '] не переходит к следующему ковру'
    pg.keyboard.press('['); pg.wait_for_timeout(1200); assert pg.evaluate("R.rug.id") == 'kerimov_vol3_p100_1', '[ не возвращает к предыдущему ковру'
    pg.screenshot(path=f'{OUT}/07_rug_statusbar.png')
    errors_ok(ctx, 'строка состояния ковра')


@test
def t07_rug_split_keeps_passport(pg, ctx):
    u = ctx['url']
    j = post(u + 'api/rugs/kerimov_vol2_p040_1/split', {'parts': [[0.05, 0.07, 0.39, 0.86], [0.56, 0.07, 0.39, 0.86]]})
    assert len(j['made']) == 2, 'разделение не дало 2 части'
    a = get(u + 'api/rugs/' + j['made'][0]); b = get(u + 'api/rugs/' + j['made'][1])
    assert a['school'] == 'guba-shirvan' and a['license'] == 'книга', 'часть не получила паспорт разворота'
    assert 'Гымыл' in a['title'] and 'Шахназарли' in b['title'], 'части не получили подписи своих рисунков по порядку'
    post(u + 'api/rugs/' + a['id'], {'marks': [{'id': 'mT', 'box': [0.4, 0.4, 0.2, 0.1], 'zone': 'field', 'status': 'proposed'}]})
    j2 = post(u + 'api/rugs/kerimov_vol2_p040_1/split', {'parts': [[0.02, 0.07, 0.42, 0.86], [0.56, 0.07, 0.39, 0.86]], 'ids': j['made']})
    a2 = get(u + 'api/rugs/' + j2['made'][0])
    assert a2['id'] == a['id'] and len(a2['marks']) == 1, 'правка разделения потеряла рамки части'
    post(u + 'api/rugs/kerimov_vol2_p040_1/unsplit', {})


@test
def t08_api(pg, ctx):
    u = ctx['url']
    for p in ('api/rugs', 'api/dyes', 'api/figures', 'api/me'):
        get(u + p)


@test
def t09_dyes_palette_api(pg, ctx):
    u = ctx['url']
    for _ in range(30):                       # переход на ссылки на красители идёт в фоне при старте
        w = get(u + 'api/work/tc_f01')
        if w.get('palette_v') == 1: break
        time.sleep(0.5)
    L = {d['id']: d for d in get(u + 'api/dyes')['dyes']}
    assert w.get('palette_link') == [True] * 5, f'принятый рисунок не связан с палитрой: {w.get("palette_link")}'
    assert w['palette_scan'][1] == '#83423b', 'цвет со скана не сохранился как замер'
    assert w['palette'][3] == L['qara']['hex'], 'связанный цвет не взят из библиотеки'
    assert w['palette'][1] != w['palette'][2] and L['zogali']['hex'] in w['palette'][1:3], 'два оттенка одного красителя слились или не из библиотеки'
    w2 = get(u + 'api/work/tc_f02')
    assert w2['palette_link'] == [False] * 5 and w2['palette'][1] == '#83423b', 'непринятый рисунок перекрашен без подтверждения'
    j = post(u + 'api/dyes', {'id': 'qara', 'hex': '#101060'})
    assert j['recolor'] == 'started', f'перекраска не запустилась: {j}'
    for _ in range(40):
        if not get(u + 'api/dyes')['recolor']['running'] and get(u + 'api/work/tc_f01')['palette'][3] == '#101060': break
        time.sleep(0.5)
    assert get(u + 'api/work/tc_f01')['palette'][3] == '#101060', 'смена цвета красителя не дошла до рисунка'
    assert get(u + 'api/work/tc_f02')['palette'][3] == '#2b2a2c', 'перекрашен рисунок, где краситель не подтверждён'
    import cv2
    img = cv2.imread(os.path.join(ctx['app'], 'data', 'out', 'tc_f01.png'), cv2.IMREAD_UNCHANGED)
    assert img is not None and tuple(int(v) for v in img[0, 2][:3]) == (0x60, 0x10, 0x10), 'выгрузка PNG не перерисована новым цветом'
    n = post(u + 'api/dyes', {'az': 'tünd göy', 'ru': 'тёмно-синий', 'hex': '#22335a'})
    assert n['dye']['id'] == 'tund_goy', 'новый краситель не добавился'
    # тот же расчёт оттенков в браузере и на сервере
    pal = {'palette': ['#fff', '#83423b', '#7f3336', '#97463a'], 'palette_dyes': [None, 'zogali', 'zogali', 'zogali'],
           'palette_link': [False, True, True, True], 'palette_scan': ['#ffffff', '#83423b', '#7f3336', '#97463a'], 'palette_names': ['', '', '', '']}
    srv = post(u + 'api/dyes/resolve', pal)
    pg.goto(u); pg.wait_for_timeout(2500)
    js = pg.evaluate("""p=>{const F={palette:p.palette.slice(),palDyes:p.palette_dyes.slice(),palLink:p.palette_link.slice(),palScan:p.palette_scan.slice(),palNames:p.palette_names.slice()};resolvePal(F);return F.palette;}""", pal)
    assert js == srv['palette'], f'оттенки в редакторе и на сервере считаются по-разному: {js} / {srv["palette"]}'


@test
def t10_dyes_screen_and_picker(pg, ctx):
    pg.click('#tab-dyes'); pg.wait_for_timeout(1500)
    assert pg.is_visible('#dyesScr'), 'экран «Красители» не открылся'
    n = pg.evaluate("document.querySelectorAll('.dycard[data-id]').length"); assert n >= 16, f'на экране красителей {n} карточек'
    pg.click('.dycard[data-id="qara"]'); pg.wait_for_timeout(1200)
    assert pg.is_visible('.dyfig[data-fig="tc_f01"]'), 'у красителя не показаны рисунки, где он используется'
    pg.screenshot(path=f'{OUT}/08_dyes.png')
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300); assert not pg.is_visible('#dyesScr'), 'Esc не закрывает экран красителей'
    pg.click('#tab-dyes'); pg.wait_for_timeout(1200)
    pg.click('.dycard[data-id="qara"]'); pg.wait_for_timeout(1200)
    pg.click('.dyfig[data-fig="tc_f01"]'); pg.wait_for_timeout(2000)
    assert pg.evaluate("S.fig&&S.fig.id") == 'tc_f01', 'клик по рисунку на экране красителей не открывает его'
    if pg.evaluate("S.mode") != 'pix': pg.keyboard.press('Tab'); pg.wait_for_timeout(600)
    pg.evaluate("openDyePicker(4)"); pg.wait_for_timeout(400)
    pg.click('#modal .dy[data-id="krem"]'); pg.wait_for_timeout(300)
    k = pg.evaluate("S.dyes.find(d=>d.id==='krem').hex")
    assert pg.evaluate("S.fig.palette[4]") == k and pg.evaluate("S.fig.palLink[4]"), 'выбор красителя меняет только название, а не цвет'
    pg.evaluate("openDyePicker(4)"); pg.wait_for_timeout(400); pg.check('#dyKeep'); pg.click('#modal .dy[data-id="sari"]'); pg.wait_for_timeout(300)
    assert pg.evaluate("S.fig.palette[4]") == '#c9a23a' and not pg.evaluate("S.fig.palLink[4]"), '«не связывать» не вернуло цвет со скана'
    pg.keyboard.press('Control+z'); pg.wait_for_timeout(300)
    assert pg.evaluate("S.fig.palette[4]") == k, 'отмена (Ctrl+Z) не возвращает выбор красителя'
    pg.screenshot(path=f'{OUT}/09_palette.png')
    pg.keyboard.press('Escape'); pg.wait_for_timeout(800)
    errors_ok(ctx, 'красители')


@test
def t11_events_api(pg, ctx):
    u = ctx['url']
    j = get(u + 'api/events')
    for k in ('decisions', 'system', 'people', 'automations', 'need', 'can', 'unseen'): assert k in j, f'в /api/events нет «{k}»'
    D = {d['key']: d for d in j['decisions']}
    assert 'train' in D and D['train']['level'] == 'need', f'нет пункта «нужно дообучить» (готовых ч/б в тестовых данных > 100): {list(D)}'
    assert 'sheets' in D and D['sheets']['sheets'][0]['path'] == 'test/auto_page_001.jpg', 'лист, нарезанный машиной, не попал в «Нужно решение»'
    assert {a['id'] for a in j['automations']} >= {'night', 'train', 'weekly'}, 'нет автоматических задач'
    post(u + 'api/events/dismiss', {'key': 'train', 'sig': D['train']['sig']})
    assert next(d for d in get(u + 'api/events')['decisions'] if d['key'] == 'train')['hidden'], '«скрыть» не скрывает пункт'
    post(u + 'api/events/dismiss', {'key': 'train', 'sig': None})
    post(u + 'api/sheet_checked', {'paths': ['test/auto_page_001.jpg']})
    assert 'sheets' not in {d['key'] for d in get(u + 'api/events')['decisions']}, '«рамки верны» не убирает лист из проверки'
    a = post(u + 'api/automation', {'id': 'night', 'auto': False, 'hour': 4})
    n = next(x for x in a['automations'] if x['id'] == 'night'); assert not n['auto'] and n['hour'] == 4 and '04:00' in n['rule'], 'настройка задачи не сохранилась'
    post(u + 'api/automation', {'id': 'night', 'auto': True, 'hour': 3})
    r = get(u + 'api/events/report?days=7')
    assert r['done_total'] >= 100 and r['text'].startswith('Неделя'), 'отчёт за неделю не собирается'
    c = get(u + 'api/events/count'); assert c['need'] >= 1, 'счётчик колокольчика не видит «нужно решение»'


@test
def t12_events_screen(pg, ctx):
    pg.goto(ctx['url']); pg.wait_for_timeout(3500)
    assert pg.is_visible('#bellBtn') and pg.is_visible('#bellN'), 'нет колокольчика со счётчиком рядом с пользователем'
    hb = pg.evaluate("document.querySelector('header').getBoundingClientRect().height"); assert hb < 70, 'шапка с колокольчиком переносится'
    pg.click('#bellBtn'); pg.wait_for_timeout(1500)
    assert pg.is_visible('#evScr'), 'центр событий не открылся'
    assert pg.evaluate("document.querySelectorAll('#evList .evit').length") >= 1, 'в «Нужно решение» пусто'
    assert pg.evaluate("document.querySelectorAll('#evAuto .evau').length") >= 3, 'нет автоматических задач справа'
    assert pg.is_visible('#evRep .nums'), 'нет отчёта за неделю'
    pg.screenshot(path=f'{OUT}/10_events.png')
    pg.click('#evSeg [data-t="sys"]'); pg.wait_for_timeout(300); pg.click('#evSeg [data-t="ppl"]'); pg.wait_for_timeout(300)
    pg.click('#evSeg [data-t="need"]'); pg.wait_for_timeout(300)
    pg.uncheck('.evau[data-a="train"] [data-f="enabled"]'); pg.wait_for_timeout(800)
    assert not next(a for a in get(ctx['url'] + 'api/events')['automations'] if a['id'] == 'train')['enabled'], 'галочка задачи не сохраняется'
    pg.check('.evau[data-a="train"] [data-f="enabled"]'); pg.wait_for_timeout(800)
    pg.keyboard.press('2'); pg.wait_for_timeout(300)
    assert pg.is_visible('#evScr'), 'клавиши редакторов срабатывают поверх центра событий'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300); assert not pg.is_visible('#evScr'), 'Esc не закрывает центр событий'
    pg.click('#bellBtn'); pg.wait_for_timeout(1200)
    if pg.query_selector('#evList [data-act="accept"]'):          # кнопка «открыть» ведёт на нужный экран
        pg.click('#evList [data-act="accept"]'); pg.wait_for_timeout(1500)
        assert not pg.is_visible('#evScr') and pg.evaluate("S.mode") == 'queue', 'кнопка пункта не открывает «Рисунки»'
    errors_ok(ctx, 'центр событий')


@test
def t13_rail_crumbs_keys(pg, ctx):
    pg.goto(ctx['url']); pg.wait_for_timeout(2500)
    ids = pg.evaluate("[...document.querySelectorAll('#rail button')].map(b=>b.id)")
    for t in ('tab-pipe', 'tab-queue', 'tab-rugs', 'tab-dyes', 'railKeys'): assert t in ids, f'в колонке разделов нет {t}'
    assert not pg.is_visible('#tab-sheet'), '«Листы» должны быть внутри «Узоров», а не отдельным разделом'
    for t, word in (('tab-pipe', 'Обзор'), ('tab-queue', 'Узоры'), ('tab-rugs', 'Ковры'), ('tab-dyes', 'Красители')):
        pg.click('#' + t); pg.wait_for_timeout(900)
        assert (pg.text_content('#crumbs') or '').startswith(word), f'«где вы сейчас» не показывает {word}: {pg.text_content("#crumbs")}'
        assert pg.evaluate(f"document.getElementById('{t}').classList.contains('on')"), f'раздел {word} не подсвечен в колонке'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
    pg.click('#tab-pipe'); pg.wait_for_timeout(1500)
    # v11.92: один дашборд на все разделы — в «Узорах» своей сводки нет
    for h in ('Узоры', 'Ковры', 'Красители'): assert h in (pg.inner_text('#catalog .dash') or ''), f'в общем «Обзоре» нет раздела «{h}»'
    assert pg.is_visible('#dbNeed'), 'в «Обзоре» нет блока «Ждёт вашего решения»'
    pg.wait_for_selector('#dbRugs b', timeout=8000); pg.wait_for_selector('#dbDyes b', timeout=8000)
    assert not pg.is_visible('#tab-edit'), 'кнопка «К рисунку» не должна показываться в «Обзоре»'
    pg.keyboard.press('Shift+Slash'); pg.wait_for_timeout(400)
    assert pg.is_visible('#modal .kbgrid'), '«?» не открывает шпаргалку клавиш'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300); assert not pg.is_visible('#modal'), 'Esc не закрывает шпаргалку'
    pg.click('#railKeys'); pg.wait_for_timeout(300); assert pg.is_visible('#modal .kbgrid'), 'кнопка «Клавиши» не открывает шпаргалку'
    pg.click('#mClose'); pg.wait_for_timeout(200)
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
    pg.keyboard.press('2'); pg.wait_for_timeout(600); assert pg.evaluate("S.mode") == 'queue', 'клавиша 2 не открывает «Узоры»'
    pg.keyboard.press('1'); pg.wait_for_timeout(600); assert pg.evaluate("S.mode") == 'sheet', 'клавиша 1 не открывает «Листы»'
    assert (pg.text_content('#crumbs') or '').startswith('Узоры') and pg.evaluate("document.getElementById('tab-queue').classList.contains('on')"), 'листы не отмечены как часть «Узоров»'
    assert pg.is_visible('#left [data-vs="sheet"]'), 'на листах нет переключателя Листы · Очередь · Все рисунки'
    errors_ok(ctx, 'колонка разделов')


@test
def t14_patterns_tree(pg, ctx):
    pg.click('#tab-queue'); pg.wait_for_timeout(800)
    pg.evaluate("setMode('cat')"); pg.wait_for_timeout(600)
    assert pg.is_visible('#left .ptree'), 'у «Узоров» слева нет дерева источников'
    assert not pg.query_selector('#catalog .kpirow'), 'у «Узоров» не должно быть своей сводки: она одна, в «Обзоре»'
    assert not pg.query_selector('#catalog [data-vs="pipe"]'), 'в «Узорах» не должно быть вкладки «Сводка»'
    assert pg.is_visible('#catalog [data-upl="pat"]'), 'в «Узорах» нет кнопки «Загрузить»'
    pg.click('#catalog [data-upl="pat"]'); pg.wait_for_timeout(300)
    mt = pg.inner_text('#popMenu'); assert all(w in mt for w in ('Сканы листов', 'Готовые вырезанные рисунки', 'Легенда', 'Зарисовки')), f'меню «Загрузить» неполное: {mt}'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200); assert not pg.query_selector('#popMenu'), 'Esc не закрывает меню «Загрузить»'
    src = pg.evaluate("[...document.querySelectorAll('#left .pn[data-src]')].map(b=>b.dataset.src).filter(Boolean)")
    assert src, 'в дереве нет источников'
    k = src[0]; n_all = pg.evaluate("document.querySelectorAll('#catalog .card').length")
    pg.click(f'#left .pn[data-src="{k}"]:not(.sub)'); pg.wait_for_timeout(500)
    ids = pg.evaluate("[...document.querySelectorAll('#catalog .card')].map(c=>c.dataset.id)")
    assert ids and all(pg.evaluate(f"figSrc(S.figures.find(f=>f.id==='{i}'))") == k for i in ids[:30]), 'фильтр по источнику пропускает чужие рисунки'
    assert k.replace('_', ' ')[:6].lower() in (pg.text_content('#crumbs') or '').lower() or pg.evaluate("srcTitle(S.fsrc)") in (pg.text_content('#crumbs') or ''), 'источник не показан в «где вы сейчас»'
    tabs = pg.evaluate(f"[...document.querySelectorAll('#left .pn.sub[data-src=\"{k}\"]')].map(b=>b.dataset.tab)")
    if tabs:
        pg.click(f'#left .pn.sub[data-tab="{tabs[0]}"]'); pg.wait_for_timeout(500)
        assert all(str(pg.evaluate(f"S.figures.find(f=>f.id==='{i}').table")) == tabs[0] for i in pg.evaluate("[...document.querySelectorAll('#catalog .card')].map(c=>c.dataset.id)")[:20]), 'фильтр по таблице не работает'
    pg.click('#left .pn[data-src=""]'); pg.wait_for_timeout(500)
    assert pg.evaluate("document.querySelectorAll('#catalog .card').length") == n_all, '«Все источники» не снимает фильтр'
    pg.screenshot(path=f'{OUT}/11_patterns_tree.png')
    errors_ok(ctx, 'дерево источников')


@test
def t14b_patterns_filters_tools(pg, ctx):
    """v11.93: «Узоры» устроены как «Ковры»: фильтры слева, чипы, Вид, Инструменты, выбор в очереди."""
    pg.evaluate("setMode('cat')"); pg.wait_for_timeout(600)
    for sel in ('#left details.fg', '#patChips', '#patViewBtn', '#patToolsBtn', '#patSort', '#catQ'):
        assert pg.query_selector(sel), f'в «Всех рисунках» нет {sel}'
    assert not pg.is_visible('#patTools'), 'панель инструментов открыта по умолчанию'
    n_all = pg.evaluate("document.querySelectorAll('#catalog .card').length")
    pg.click('#left [data-ft="bw"]'); pg.wait_for_timeout(400)
    assert pg.is_visible('#patChips .rgfch'), 'после выбора типа нет чипа активного фильтра'
    assert pg.evaluate("S.catType") == 'bw'
    pg.click('#patChips .rgfch button'); pg.wait_for_timeout(400)
    assert pg.evaluate("document.querySelectorAll('#catalog .card').length") == n_all and not pg.query_selector('#patChips .rgfch'), 'крестик на чипе не снимает фильтр'
    pg.click('#left [data-fs="rev"]'); pg.wait_for_timeout(300); pg.click('#patChipsAll'); pg.wait_for_timeout(300)
    assert not pg.evaluate("S.catRev"), '«сбросить всё» не снимает фильтры'
    pg.click('#patViewBtn'); pg.wait_for_timeout(300)
    pg.click('#popMenu [data-sz="l"]'); pg.wait_for_timeout(200)
    assert pg.evaluate("document.getElementById('catalog').style.getPropertyValue('--pcw')") == '250px', '«Вид» не меняет размер карточек'
    pg.keyboard.press('Escape'); pg.click('#patViewBtn'); pg.wait_for_timeout(200); pg.click('#popMenu [data-sz="m"]'); pg.keyboard.press('Escape')
    pg.click('#patToolsBtn'); pg.wait_for_timeout(500)
    assert pg.is_visible('#patTools'), 'панель инструментов не открылась'
    for i in ('qFast', 'qHard', 'pwGo', 'pwRedo', 'mdTrain', 'rkBack', 'lgDl'):
        assert pg.query_selector(f'#patTools #{i}'), f'в панели нет инструмента {i}'
    assert pg.is_visible('#patToolsBtn') and pg.is_visible('[data-upl="pat"]'), 'панель закрыла кнопки сверху'
    pg.click('#patToolsX'); pg.wait_for_timeout(300); assert not pg.is_visible('#patTools')
    pg.click('#patToolsBtn'); pg.wait_for_timeout(300)
    pg.click('[data-vs="queue"]'); pg.wait_for_timeout(700)
    assert pg.is_visible('#patTools'), 'панель инструментов пропала при переходе в очередь'
    assert pg.evaluate("document.getElementById('qFast').disabled") == (pg.evaluate("patCounts().q") == 0), 'быстрая проверка: доступность не совпадает с очередью'
    n = pg.evaluate("document.querySelectorAll('#catalog .card').length")
    if n:
        pg.click('#qThr', force=True) if False else None
        pg.evaluate("document.querySelector('#catalog [data-sel]').click()"); pg.wait_for_timeout(300)
        assert pg.is_visible('#qAccept'), 'после выбора рисунка нет панели «принять выбранные»'
        pg.click('#qPickNone'); pg.wait_for_timeout(300); assert not pg.query_selector('#qAccept'), '«снять выбор» не убирает панель'
    pg.click('#patToolsX'); pg.evaluate("setMode('sheet')"); pg.wait_for_timeout(400)
    assert not pg.is_visible('#patTools'), 'панель инструментов осталась на листах'
    pg.evaluate("setMode('cat')"); pg.wait_for_timeout(300)
    pg.screenshot(path=f'{OUT}/11b_patterns_filters.png')
    errors_ok(ctx, 'фильтры и инструменты узоров')


@test
def t15_rugs_overview(pg, ctx):
    pg.click('#tab-rugs'); pg.wait_for_timeout(1500)
    if pg.evaluate("R.view") == 'edit': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    assert pg.is_visible('#rgTree .ptree') and pg.is_visible('#rgPeek') and pg.is_visible('#rgChips'), 'у «Ковров» нет фильтров / карточки справа / строки активных фильтров'
    assert not pg.is_visible('#rgKpi'), 'инструменты каталога должны быть в отдельной панели, а не над плиткой'
    for b in ('rgTriBtn', 'rgTypesBtn', 'rgEtBtn', 'rgSignsBtn', 'rgDupChk', 'rgSync', 'rgSyncBooks'): assert not pg.is_visible('#' + b), f'кнопка {b} снова над плиткой'
    rg_tools_open(pg); assert pg.is_visible('#rgKpi') and pg.is_visible('#rgTypesBtn'), 'в панели «Инструменты» нет инструментов'
    pg.click('#rgToolsX'); pg.wait_for_timeout(200); assert not pg.is_visible('#rgTools'), 'панель «Инструменты» не закрывается'
    pg.click('#rgUpBtn'); pg.wait_for_timeout(300); mt = pg.inner_text('#popMenu')
    assert all(w in mt for w in ('Фото ковров', 'Архив', 'Каталог музея', 'Отпечатки', 'Обновить каталог', 'Ковры из книг')), f'меню «Загрузить» (ковры) неполное: {mt}'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    # размер фото + строка активных фильтров
    n0 = pg.evaluate("document.querySelectorAll('.rgcard').length"); pg.select_option('#rgFSize', '1500'); pg.wait_for_timeout(400)
    n1 = pg.evaluate("document.querySelectorAll('.rgcard').length"); assert n1 < n0, f'фильтр «размер фото от 1500 px» ничего не отсёк: {n0} → {n1}'
    assert 'Фото: от 1500 px' in pg.inner_text('#rgChips'), 'активный фильтр не показан чипом'
    pg.click('#rgChips [data-clr="size"]'); pg.wait_for_timeout(400); assert pg.evaluate("document.querySelectorAll('.rgcard').length") == n0, 'чип не снимает фильтр'
    pg.click('#rgViewBtn'); pg.wait_for_timeout(200); pg.click('#popMenu [data-sz="s"]'); pg.wait_for_timeout(300)
    assert pg.evaluate("getComputedStyle(document.getElementById('rugs')).getPropertyValue('--cw').trim()") == '140px', '«Вид: мелко» не меняет размер плиток'
    pg.click('#popMenu [data-sz="m"]'); pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    n_all = pg.evaluate("document.querySelectorAll('.rgcard').length")
    pg.click('#rgTree .pn[data-k="museum"]'); pg.wait_for_timeout(500)
    ids = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")
    assert ids and all(i.startswith(('vam_', 'met_', 'cm_', 'border_rep')) for i in ids), f'фильтр «музеи» пропускает другие: {ids}'
    pg.click('#rgTree .pn[data-k="museum"]'); pg.wait_for_timeout(300)   # раскрыть подисточники
    if pg.query_selector('#rgTree .pn.sub[data-s="vam"]'):
        pg.click('#rgTree .pn.sub[data-s="vam"]'); pg.wait_for_timeout(400)
        assert all(i.startswith('vam_') for i in pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")), 'фильтр по одному музею не работает'
    pg.click('#rgTree .pn[data-k=""]'); pg.wait_for_timeout(400)
    pg.check('#rgTree [data-sch="karabakh"]'); pg.wait_for_timeout(400)
    ids = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")
    assert ids == ['kerimov_vol3_p100_1'] or set(ids) <= {'kerimov_vol3_p100_1', 'met_123'}, f'фильтр по школе: {ids}'
    pg.uncheck('#rgTree [data-sch="karabakh"]'); pg.wait_for_timeout(300)
    pg.select_option('#rgFShow', 'big'); pg.wait_for_timeout(400)
    assert pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)") == ['kerimov_vol3_p100_1'], 'чип «есть фото крупнее» не работает'
    pg.select_option('#rgFShow', ''); pg.wait_for_timeout(400)
    assert pg.evaluate("document.querySelectorAll('.rgcard').length") == n_all, 'фильтры не снимаются'
    pg.hover('.rgcard[data-id="kerimov_vol3_p100_1"]'); pg.wait_for_timeout(500)
    assert 'Метрополитен' in (pg.text_content('#rgPeek') or ''), 'карточка справа не показывает двойника'
    pg.screenshot(path=f'{OUT}/12_rugs_overview.png')
    pg.click('#rgPeekOpen'); pg.wait_for_timeout(1200)
    assert pg.evaluate("R.view") == 'edit' and pg.evaluate("R.rug.id") == 'kerimov_vol3_p100_1', '«Открыть в редакторе» не открывает ковёр'
    pg.keyboard.press('Escape'); errors_ok(ctx, 'обзор ковров')


@test
def t16_motif_search(pg, ctx):
    """v11.4: «где ещё встречается этот узор» — индекс, поиск по рамке, ✓ добавляет рамку на найденный ковёр, клавиша G"""
    u = ctx['url']
    st = post(u + 'api/motif/index', {})
    for _ in range(60):
        st = get(u + 'api/motif/status')
        if not st['running'] and st['indexed']: break
        time.sleep(0.5)
    assert st['indexed'] >= 6 and not st['stale'], f'индекс не построился: {st}'
    W = 300; box = [(150 - 40) / W, (200 - 40) / 420, 80 / W, 80 / 420]
    post(u + 'api/rugs/vam_0', {'marks': [{'id': 'mStar', 'box': box, 'zone': 'field', 'status': 'confirmed', 'motif_text': 'звезда'}]})
    j = post(u + 'api/motif/search', {'rug': 'vam_0', 'mark': 'mStar'})
    top = [r['rid'] for r in j['results'][:3]]
    assert 'vam_3' in top and 'vam_5' in top, f'поиск не нашёл тот же узор на других коврах: {top}'
    best = j['results'][0]
    assert 'vam_0' not in [r['rid'] for r in j['results']], 'в находках — сам ковёр запроса'
    # интерфейс: выбрать рамку, G — окно поиска, ✓ на первой находке
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    pg.evaluate("rugOpen('vam_0')"); pg.wait_for_timeout(1500)
    pg.evaluate("rugSelect(R.rug.marks.find(m=>m.id==='mStar'))"); pg.wait_for_timeout(300)
    assert pg.is_visible('#rgMotSearch'), 'у выбранной рамки нет кнопки «найти на других коврах»'
    pg.keyboard.press('g'); pg.wait_for_timeout(2500)
    assert pg.is_visible('#modal .msm') and pg.evaluate("document.querySelectorAll('.mscard').length") >= 2, 'клавиша G не открывает окно находок'
    pg.screenshot(path=f'{OUT}/13_motif_search.png')
    rid = pg.evaluate("MSS.res.results[0].rid")
    pg.click('.mscard[data-i="0"] [data-ok]'); pg.wait_for_timeout(1200)
    t = get(u + 'api/rugs/' + rid)
    mk = [m for m in t['marks'] if m.get('machine') == 'motif']
    assert mk and mk[-1]['status'] == 'confirmed' and mk[-1]['motif_text'] == 'звезда', '✓ не добавил подтверждённую рамку на найденный ковёр'
    assert pg.evaluate("document.querySelector('.mscard[data-i=\"0\"]').classList.contains('yes')"), 'карточка не отмечена как ✓'
    pg.click('.mscard[data-i="0"] [data-no]'); pg.wait_for_timeout(1000)
    assert not [m for m in get(u + 'api/rugs/' + rid)['marks'] if m.get('machine') == 'motif'], '✗ после ✓ не убрал добавленную рамку'
    pg.click('.mscard[data-i="1"] .msim'); pg.wait_for_timeout(2000)
    assert pg.is_hidden('#modal') and pg.evaluate("R.rug.id") == j['results'][1]['rid'], 'клик по находке не открывает тот ковёр'
    assert pg.evaluate("!!R.hl"), 'найденное место не подсвечено'
    pg.screenshot(path=f'{OUT}/14_motif_open.png')
    assert get(u + 'api/motif/status')['labels']['total'] >= 2, 'ответы ✓/✗ не записаны в обучающий набор'
    errors_ok(ctx, 'поиск узора')


@test
def t17_feedback_v115(pg, ctx):
    """v11.5: повторы в других красках и «вижу ещё», тип ковра по названию, уменьшенная копия Wikimedia скрыта, подсказка по форме, лента двойников не прыгает"""
    u = ctx['url']
    # 1) повторы: образец — элемент в первой клетке каймы; остальные 5 — в других красках
    box = [20 / 300, 20 / 620, 80 / 300, 80 / 620]
    post(u + 'api/rugs/border_rep', {'marks': [{'id': 'mB', 'box': box, 'zone': 'border', 'status': 'confirmed', 'motif_text': 'qarmaq'}]})
    j = post(u + 'api/rugs/border_rep/repeats', {'box': box, 'mark': 'mB'})
    left = [c for c in j['cands'] if c['box'][0] < 0.4]
    assert len(left) >= 5, f'повторы в других красках: найдено {len(left)} из 5 — {[c["score"] for c in j["cands"]]}'
    assert not [c for c in j['cands'] if c['box'][0] > 0.5 and c['score'] > 0.6], 'полосы справа приняты за тот же элемент'
    jl = post(u + 'api/rugs/border_rep/repeats', {'box': box, 'mark': 'mB', 'loose': True})
    assert len(jl['cands']) >= len(j['cands']), '«вижу ещё» даёт меньше мест, чем обычный поиск'
    # 1б) v11.11: долгий расчёт отдаётся номером задачи и забирается опросом; фоновая очередь не мешает интерактивной
    import time as _t
    j = post(u + 'api/rugs/border_rep/repeats', {'box': box, 'mark': 'mB'})
    assert 'cands' in j, 'повторы не вернулись'
    h = get(u + 'api/heavy'); assert 'ui_wait' in h and 'wait' in h, 'нет полос очереди в /api/heavy'
    # 2) тип ковра по названию + принять; копия Wikimedia скрыта, её рамка перенесена в оригинал
    for _ in range(20):
        L = {r['id']: r for r in get(u + 'api/rugs')['rugs']}
        if 'wm_pir_small' not in L: break
        time.sleep(0.5)
    assert 'wm_pir_small' not in L, 'уменьшенная копия того же файла Wikimedia не скрыта'
    assert any(m.get('motif_text') == 'с маленькой копии' for m in get(u + 'api/rugs/cm_pir')['marks']), 'рамка уменьшенной копии не перенесена в оригинал'
    assert L['cm_pir'].get('comp_sug') == 'pirebedil', f'тип по названию не предложен: {L["cm_pir"].get("comp_sug")}'
    assert L['border_rep'].get('comp_sug') == 'haciqayib', f'«Hajigayib» не узнан как hacıqayıb: {L["border_rep"].get("comp_sug")}'
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    if pg.evaluate("R.view") == 'edit': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    rg_tools_open(pg); assert pg.is_visible('#rgKpiComp [data-cacc="title"]'), 'в панели «Инструменты» нет «принять по названию»'
    pg.on('dialog', lambda d: d.accept())
    pg.click('#rgKpiComp [data-cacc="title"]'); pg.wait_for_timeout(1500)
    assert get(u + 'api/rugs/cm_pir')['composition'] == 'pirebedil', '«принять по названию» не поставило тип'
    # 3) подсказка по форме: звезда на vam_3 → схемы студии из тестовых данных (качество формы проверено отдельно на 28 схемах)
    j = post(u + 'api/rugs/vam_3/shape_hint', {'box': [(230 - 40) / 330, (140 - 40) / 420, 80 / 330, 80 / 420]})
    assert len(j['schemes']) >= 3 and j['n_schemes'] >= 3, f'подсказка по форме пустая: {j}'
    pg.evaluate("rugOpen('vam_3')"); pg.wait_for_timeout(1500)
    pg.evaluate("(()=>{const m={id:'mNew',box:[190/330,100/420,80/330,80/420],zone:'field',status:'proposed',motif:'',motif_text:'',scheme:'',group:'',angle:0,flip:0};R.rug.marks.push(m);rugSelect(m);})()".replace('190/330,100/420', '(230-40)/330,(140-40)/420'))
    pg.wait_for_timeout(2500)
    assert pg.is_visible('#rgShape [data-hs="0"]'), 'у новой рамки без названия нет подсказок по форме'
    pg.screenshot(path=f'{OUT}/15_shape_hint.png')
    sid = j['schemes'][0]['id']
    pg.click('#rgShape [data-hs="0"]'); pg.wait_for_timeout(400)
    assert pg.evaluate("R.sel.scheme") == sid and pg.evaluate("!!(R.sel.motif||R.sel.motif_text)"), 'клик по подсказке не поставил схему и название'
    pg.keyboard.press('Delete'); pg.wait_for_timeout(1200)
    # 4) лента двойников: порядок не меняется при переключении
    pg.evaluate("rugOpen('kerimov_vol3_p100_1')"); pg.wait_for_timeout(1500)
    o1 = pg.evaluate("[...document.querySelectorAll('#rgStrip .rgtw')].map(e=>e.dataset.open)")
    pg.evaluate("rugOpen('met_123')"); pg.wait_for_timeout(1500)
    o2 = pg.evaluate("[...document.querySelectorAll('#rgStrip .rgtw')].map(e=>e.dataset.open)")
    assert o1 == o2 and len(o1) == 2, f'порядок миниатюр двойников меняется: {o1} → {o2}'
    assert pg.evaluate("document.querySelector('#rgStrip .rgtw.cur').dataset.open") == 'met_123', 'открытое фото не подсвечено'
    errors_ok(ctx, 'v11.5')


@test
def t18_feedback_v116(pg, ctx):
    """v11.6: изделия отдельно, скрыть фото, Tab — явный фокус, пробел прячет рамки, автоповторы только точные, вариант элемента, K — разделить, проверка склеек"""
    u = ctx['url']
    L = {r['id']: r for r in get(u + 'api/rugs')['rugs']}
    assert L['am_bag']['obj'] == 'bag', f'мафраш не узнан как сумка: {L["am_bag"]["obj"]}'
    assert L['vam_1']['obj'] == 'carpet'
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    if pg.evaluate("R.view") == 'edit': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    ids = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")
    assert 'am_bag' not in ids, 'сумка показана среди ковров'
    pg.click('#rgTree [data-kind="items"]'); pg.wait_for_timeout(400)
    _it = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)"); assert _it == ['am_bag'], f'фильтр «Изделия» не работает: {_it}'
    pg.click('#rgTree [data-kind="carpets"]'); pg.wait_for_timeout(300)
    # скрыть
    pg.evaluate("rugOpen('vam_4')"); pg.wait_for_timeout(1400)
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200); pg.click('#rgTbHide'); pg.wait_for_timeout(300); pg.click('#modal [data-hr="0"]'); pg.wait_for_timeout(1500)
    assert get(u + 'api/rugs/vam_4').get('hidden'), 'фото не скрылось'
    assert 'vam_4' not in pg.evaluate("R.list.filter(r=>rugKindOf(r)==='carpets').map(r=>r.id)"), 'скрытое фото осталось среди ковров'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200); pg.click('#rgTbHide'); pg.wait_for_timeout(1200)
    assert not get(u + 'api/rugs/vam_4').get('hidden'), '«вернуть» не вернуло фото'
    # автоповторы после новой рамки — только «точно такие же»
    pg.evaluate("rugOpen('border_rep')"); pg.wait_for_timeout(1500)
    pg.evaluate("R.rug.marks=[];rugSaveSoon();"); pg.wait_for_timeout(900)
    pg.evaluate("(()=>{const m={id:'mA',box:[20/300,120/620,80/300,80/620],zone:'border',status:'proposed',motif:'',motif_text:'',scheme:'',group:'',angle:0,flip:0};R.rug.marks.push(m);rugSelect(m);rugFindRepeats(m,true);})()")
    pg.wait_for_timeout(3500)
    mk = pg.evaluate("R.rug.marks.filter(m=>m.machine).map(m=>m.certainty)")
    assert mk and all(c == 'sure' for c in mk), f'после новой рамки добавлены неуверенные повторы: {mk}'
    # v11.12: кнопка «🔍 найти повторы» в панели — до конца, с нажатием мышью
    n0 = pg.evaluate("R.rug.marks.length")
    pg.evaluate("R.rug.marks.forEach(m=>{if(m.id==='mA')m.status='confirmed'});rugSaveSoon();rugDraw();rugSide();"); pg.wait_for_timeout(900)
    pg.evaluate("rugSelect(R.rug.marks.find(m=>m.id==='mA'))"); pg.wait_for_timeout(300)
    pg.click('#rgFindRep'); pg.wait_for_timeout(6000)
    assert pg.evaluate("!!R.sel"), 'клик по кнопке панели снял выбор рамки'
    n1 = pg.evaluate("R.rug.marks.length"); st = pg.evaluate("document.getElementById('stat').title")
    assert n1 > n0, f'кнопка «найти повторы» ничего не добавила (было {n0}, стало {n1}); строка состояния: {st}'
    # Tab: ореол, подпись, пинг; пробел — рамки скрыты
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200); pg.keyboard.press('Tab'); pg.wait_for_timeout(300)
    assert pg.evaluate("!!document.querySelector('#rgSvg .rgping')"), 'Tab не подсвечивает выбранный элемент'
    pg.keyboard.down(' '); pg.wait_for_timeout(200)
    assert pg.evaluate("getComputedStyle(document.getElementById('rgSvg')).opacity") == '0', 'пробел не прячет рамки'
    pg.keyboard.up(' '); pg.wait_for_timeout(200)
    assert pg.evaluate("getComputedStyle(document.getElementById('rgSvg')).opacity") == '1', 'рамки не вернулись после пробела'
    # вариант элемента
    pg.click('#rgSide [data-var="detail"]'); pg.wait_for_timeout(1500)
    sid = pg.evaluate("R.sel.id")
    assert [m for m in get(u + 'api/rugs/border_rep')['marks'] if m['id'] == sid][0].get('variant') == 'detail', 'вариант элемента не сохранился'
    pg.screenshot(path=f'{OUT}/16_focus_variant.png')
    # K — разделение
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200); pg.keyboard.press('k'); pg.wait_for_timeout(1500)
    assert pg.evaluate("R.tool") == 'split', 'K не включает разделение фото'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
    # проверка склеек: Керимов т. III и Метрополитен — один ковёр в разном размере
    post(u + 'api/rugs_dupcheck', {})
    for _ in range(60):
        j = get(u + 'api/rugs_dupcheck')
        if not j['state']['running']: break
        time.sleep(0.5)
    pr = [p for p in j['pairs'] if {p['a'], p['b']} == {'kerimov_vol3_p100_1', 'met_123'}]
    assert pr and pr[0]['chk'] and pr[0]['chk']['v'] == 'same', f'машинная проверка не узнала один и тот же ковёр: {pr}'
    pg.evaluate("rugBack()"); pg.wait_for_timeout(500)
    rgtool(pg, 'rgDupChk'); pg.wait_for_timeout(1200)
    pg.click('#dcBar [data-df="same"]'); pg.wait_for_timeout(300)
    assert pg.evaluate("document.querySelectorAll('.dcrow').length") >= 1, 'в окне проверки нет пары «точно тот же»'
    pg.screenshot(path=f'{OUT}/17_dupcheck.png')
    pg.click('.dcrow [data-dno]'); pg.wait_for_timeout(800)
    assert (get(u + 'api/rugs/met_123').get('same_ok') or {}).get('kerimov_vol3_p100_1', {}).get('ok') is False, '✗ в окне проверки не записался'
    pg.click('#dcBar [data-df="done"]'); pg.wait_for_timeout(300)
    pg.click('.dcrow [data-dok]'); pg.wait_for_timeout(800)
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
    errors_ok(ctx, 'v11.6')


@test
def t19_type_worksheet(pg, ctx):
    """v11.7: рабочий лист типа — типы с элементами Керимова, фото типа, «найти элементы» находит схемы на ковре без известного масштаба"""
    import cv2, numpy as np
    u = ctx['url']; D = ctx['app'] + '/data'
    T = get(u + 'api/types')['types']
    t = next((t for t in T if not t['region'] and t['el'] >= 2 and t['id'] != 'pirebedil'), None)
    assert t, f'нет типов с элементами Керимова: {T[:3]}'
    d = get(u + 'api/types/' + t['id'])
    els = [e for e in d['elements'] if os.path.exists(f"{D}/out/{e['id']}.png")]
    big = sorted(els, key=lambda e: -max(cv2.imread(f"{D}/out/{e['id']}.png", cv2.IMREAD_UNCHANGED).shape[:2]))[:2]
    assert len(big) == 2, 'у типа меньше двух схем с картинкой'
    rng = np.random.default_rng(5); k = 4
    img = np.zeros((1500, 1100, 3), np.uint8); img[:] = (60, 40, 120); img = img + rng.integers(0, 20, img.shape).astype(np.uint8)
    truth = {}
    for i, e in enumerate(big):
        p = cv2.imread(f"{D}/out/{e['id']}.png", cv2.IMREAD_UNCHANGED)
        if p.shape[2] == 3: p = np.dstack([p, np.full(p.shape[:2], 255, np.uint8)])
        p = p.copy(); al = p[..., 3] > 0; p[..., :3][al] = rng.integers(30, 230, 3)
        b = cv2.resize(p, (p.shape[1] * k, int(p.shape[0] * k * 1.1)), interpolation=cv2.INTER_NEAREST)
        x, y = 80 + i * 520, 120 + i * 650; h, w = b.shape[:2]; a = b[..., 3:4] / 255.
        img[y:y + h, x:x + w] = (b[..., :3] * a + img[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)
        truth[e['id']] = [x / 1100, y / 1500, w / 1100, h / 1500]
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    cv2.imwrite(f'{D}/rug_img/ty_test.jpg', img)
    json.dump({"id": "ty_test", "title": "Тест типа", "source": "тест", "museum": "", "inv": "", "url": "", "license": "", "place": "", "date": "", "school": "",
               "composition": "", "technique": "", "size": "", "notes": "", "status": "new", "w": 1100, "h": 1500, "marks": [], "cat": {"src": "upload"},
               "created": int(time.time()), "updated": int(time.time())}, open(f'{D}/rugs/ty_test.json', 'w'), ensure_ascii=False)
    post(u + 'api/rugs_set_comp', {'composition': t['id'], 'ids': ['ty_test']})
    post(u + 'api/rugs/ty_test/find_expected', {'only': [e['id'] for e in els]})
    for _ in range(240):
        st = get(u + 'api/rugs/ty_test/find_expected')
        if not st.get('running'): break
        time.sleep(1)
    assert not st.get('error'), st
    mk = [m for m in get(u + 'api/rugs/ty_test')['marks'] if m.get('machine') == 'kerimov']
    def iou(a, b):
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])); i = ix * iy
        return i / (a[2] * a[3] + b[2] * b[3] - i)
    for sid, tb in truth.items():
        assert any(m['scheme'] == sid and iou(m['box'], tb) > 0.5 for m in mk), f'элемент {sid} не найден: {[(m["scheme"], m["box"], m["score"]) for m in mk]}'
    # интерфейс
    pg.click('#tab-rugs'); pg.wait_for_timeout(1000)
    if pg.evaluate("R.view") == 'edit': pg.evaluate("rugBack()"); pg.wait_for_timeout(500)
    rgtool(pg, 'rgTypesBtn'); pg.wait_for_timeout(1500)
    pg.click(f'#tyList [data-ty="{t["id"]}"]'); pg.wait_for_timeout(1200)
    assert pg.is_visible('#tySheet .tyel') and pg.is_visible('#tySheet .tyc[data-p="ty_test"]'), 'рабочий лист не показал элементы и фото типа'
    assert pg.evaluate(f"document.querySelector('#tySheet .tyel[data-el=\"{big[0]['id']}\"]').classList.contains('pr')"), 'элемент, найденный машиной, не помечен «ждёт проверки»'
    pg.screenshot(path=f'{OUT}/18_type_sheet.png')
    pg.click('#tySheet .tyc[data-p="ty_test"] .im'); pg.wait_for_timeout(1500)
    assert pg.evaluate("R.view") == 'edit' and pg.evaluate("R.rug.id") == 'ty_test', 'фото из листа не открылось'
    pg.evaluate("rugBack()"); pg.wait_for_timeout(1200)
    assert pg.evaluate("R.view") == 'types', '«← все ковры» из редактора не вернул к рабочему листу'
    errors_ok(ctx, 'рабочий лист типа')


@test
def t20_twin_verdicts_v118(pg, ctx):
    """v11.8: в ленте двойников ✓ тот же / ≈ похожи / ✗ другой; очередь тяжёлых расчётов; список ковров из кэша"""
    u = ctx['url']
    assert 'run' in get(u + 'api/heavy'), 'нет /api/heavy'
    h = get(u + 'api/health'); assert h['rss_mb'] > 0 and 'mem_tail' in h, f'/api/health: {h}'
    post(u + 'api/rugs/kerimov_vol3_p100_1/same', {'other': 'met_123', 'ok': None})   # t18 уже ответил на эту пару
    pg.click('#tab-rugs'); pg.wait_for_timeout(800)
    pg.evaluate("R.twinLater&&R.twinLater.clear();rugsLoadList()"); pg.wait_for_timeout(600)
    pg.evaluate("rugOpen('kerimov_vol3_p100_1')"); pg.wait_for_timeout(1500)
    tw = '#rgStrip .rgtw[data-open="met_123"]'
    # v11.15: вопрос «это тот же ковёр?» — поверх фото, один раз; в ленте только значок
    assert not pg.query_selector(tw + ' [data-vd]'), 'кнопки ✓ ≈ ✗ остались в ленте'
    assert not pg.query_selector('#rgTwinAsk'), 'вопрос про двойников всплыл сам — мешает размечать (v11.23)'
    assert pg.is_visible('#rgTwinQ'), 'в ленте нет кнопки «? двойники: … не проверено»'
    pg.click('#rgTwinQ'); pg.wait_for_timeout(300)
    assert pg.is_visible('#rgTwinAsk [data-vd="sim"]'), 'кнопка в ленте не открывает вопрос «это тот же ковёр?»'
    pg.click('#rgTwinAsk [data-vd="sim"]'); pg.wait_for_timeout(1200)
    ok = (get(u + 'api/rugs/kerimov_vol3_p100_1').get('same_ok') or {}).get('met_123') or {}
    assert ok.get('similar') and ok.get('ok') is False, f'«≈ похожи» не записалось: {ok}'
    assert (pg.text_content(tw + ' .st') or '').strip() == '≈', 'значок «≈» не появился на миниатюре'
    assert not pg.query_selector('#rgTwinAsk'), 'вопрос не исчез после ответа'
    assert pg.evaluate("R.rug.id") == 'kerimov_vol3_p100_1', 'ответ открыл другой ковёр'
    post(u + 'api/rugs/kerimov_vol3_p100_1/same', {'other': 'met_123', 'ok': None})
    pg.evaluate("rugOpen('kerimov_vol3_p100_1')"); pg.wait_for_timeout(1500)
    pg.click('#rgTwinQ'); pg.wait_for_timeout(300)
    pg.click('#rgTwinAsk [data-vd="later"]'); pg.wait_for_timeout(300); assert not pg.query_selector('#rgTwinAsk'), '«позже» не убрало вопрос'
    pg.evaluate("rugOpen('met_123')"); pg.wait_for_timeout(1200); pg.evaluate("rugOpen('kerimov_vol3_p100_1')"); pg.wait_for_timeout(1500)
    assert not pg.query_selector('#rgTwinAsk'), 'после «позже» вопрос снова появился в этом же запуске'
    pg.evaluate("R.twinLater.clear();rugTwinAsk()"); pg.wait_for_timeout(200)
    pg.click('#rgTwinAsk [data-vd="ok"]'); pg.wait_for_timeout(1200)
    assert ((get(u + 'api/rugs/met_123').get('same_ok') or {}).get('kerimov_vol3_p100_1') or {}).get('ok') is True, '✓ не записалось у второго ковра'
    t0 = time.time(); get(u + 'api/rugs'); get(u + 'api/rugs'); assert time.time() - t0 < 5
    # «только каталог»: json вместо zip — паспорт обновляется, фото не трогаются
    cp = ctx['app'] + '/cat_only.json'
    json.dump({"built": "test", "groups": [], "items": [{"id": "vam_1", "src": "vam", "title": "V&A rug 1", "museum": "V&A", "inv": "T.123-1920", "img": "img/vam_1.jpg"}]},
              open(cp, 'w'), ensure_ascii=False)
    post(u + 'api/rugs_sync', {'bundle_id': cp})
    for _ in range(40):
        st = get(u + 'api/rugs_sync')
        if not st.get('running'): break
        time.sleep(0.5)
    assert not st.get('error'), st
    assert get(u + 'api/rugs/vam_1')['inv'] == 'T.123-1920', 'каталог без фото не обновил паспорт'
    errors_ok(ctx, 'v11.8')

@test
def t21_rug_editor_zones(pg, ctx):
    """v11.9: слева «Узоры» (вкладки), над фото панель инструментов, справа только выбранная рамка; пояснения — в «i»."""
    pg.goto(ctx['url']); pg.wait_for_timeout(2000)
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200); pg.evaluate("rugOpen('vam_1')"); pg.wait_for_timeout(1500)
    tb = pg.query_selector('.rgtb').bounding_box(); v = pg.query_selector('#rgView').bounding_box()
    assert tb['x'] - v['x'] < 30 and tb['y'] - v['y'] < 30 and tb['height'] > tb['width'], 'панель инструментов не слева сверху столбиком (как у узоров)'
    assert pg.evaluate("document.getElementById('rgFindRep').disabled"), 'кнопки элемента активны без выбранной рамки'
    assert pg.evaluate("document.querySelectorAll('#rgPassBtn,#rgPassBtn2').length") == 1, 'кнопка «Паспорт» на экране не одна'
    title = pg.evaluate("R.rug.title")
    for sel in ('#rgLeft', '#rgSide', '#sbar'):
        assert title not in (pg.text_content(sel) or ''), f'название ковра повторяется в {sel}'
    # v11.19: всё про фото целиком — справа, когда рамка не выбрана
    for sel in ('#rgSide #rgTZones', '#rgSide #rgTbHide', '#rgSide #rgColBy', '#rgSide #rgDim'): assert pg.is_visible(sel), f'справа нет {sel}'
    assert pg.is_visible('header #rgPassBtn'), 'кнопки «Паспорт ковра» нет в шапке у названия'
    # пояснения спрятаны за «i»
    assert pg.evaluate("[...document.querySelectorAll('#rgSide .hint')].every(h=>h.offsetParent===null)"), 'пояснения видны без «i»'
    pg.click('#rgSide .ib-i'); pg.wait_for_timeout(150); assert pg.evaluate("[...document.querySelectorAll('#rgSide .hint')].some(h=>h.offsetParent!==null)"), '«i» не раскрывает пояснение'
    # новая рамка → слева «Похожие», справа рамка, кнопки элемента активны
    x, y = v['x'] + v['width'] * .45, v['y'] + v['height'] * .45
    pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 50, y + 50, steps=5); pg.mouse.up(); pg.wait_for_timeout(2500)
    assert pg.evaluate("R.lib") == 'shape' and pg.is_visible('#rgShape'), 'у новой рамки без названия слева не открылись «Похожие»'
    assert not pg.evaluate("document.getElementById('rgFindRep').disabled"), 'кнопки элемента не включились'
    # «Все схемы»: клик назначает схему выбранной рамке
    pg.click('#rgLeft [data-lib="all"]'); pg.wait_for_timeout(400)
    if pg.query_selector('#rgLibAll .rgsch'):
        f = pg.evaluate("document.querySelector('#rgLibAll .rgsch').dataset.f"); pg.click('#rgLibAll .rgsch'); pg.wait_for_timeout(400)
        assert pg.evaluate("R.sel.scheme") == f, 'клик по схеме в «Все схемы» не назначает её рамке'
    pg.click('#rgLeft [data-lib="ker"]'); pg.wait_for_timeout(800); assert pg.is_visible('#rgKerType'), 'во вкладке «Керимов» нет выбора типа ковра'
    pg.click('#rgLeft [data-lib="rug"]'); pg.wait_for_timeout(300); assert pg.evaluate("document.querySelectorAll('#rgLeft .rgallit').length") >= 1, 'во вкладке «На ковре» нет элементов'
    pg.keyboard.press('r'); pg.wait_for_timeout(200); assert pg.evaluate("!!R.repeatOf"), 'R (повтор) перестал работать'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    pg.screenshot(path=f'{OUT}/13_rug_zones.png')
    errors_ok(ctx, 'зоны редактора ковра')


@test
def t22_rug_mark_panel_v1115(pg, ctx):
    """v11.15: справа фото и схема рядом, зона списком, «Верно / Неверно → следующая» только у названной рамки; поиск узора слева"""
    u = ctx['url']; pg.goto(u); pg.wait_for_timeout(2000)
    pg.click('#tab-rugs'); pg.wait_for_timeout(1000); pg.evaluate("rugOpen('vam_3')"); pg.wait_for_timeout(1500)
    v = pg.query_selector('#rgView').bounding_box(); x, y = v['x'] + v['width'] * .45, v['y'] + v['height'] * .5
    pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 50, y + 50, steps=5); pg.mouse.up(); pg.wait_for_timeout(2500)
    assert pg.query_selector('#rgOk') and pg.query_selector('#rgNo'), 'у рамки без названия нет «Верно / Неверно» (с v11.67 проверять можно и без названия)'
    assert pg.is_visible('#rgSide .rghead2 .pair'), 'нет пары «на фото / схема»'
    assert pg.is_visible('#rgZone') and not pg.query_selector('#rgSide [data-z]'), 'зона не выпадающим списком'
    assert not pg.query_selector('#rgSide #rgMotQ') and pg.is_visible('#rgLeft #rgMotQ'), 'поиск узора не слева'
    assert not pg.query_selector('#rgSide #rgSchList'), 'список схем остался справа'
    pg.select_option('#rgZone', 'medallion'); pg.wait_for_timeout(300); assert pg.evaluate("R.sel.zone") == 'medallion', 'зона из списка не ставится'
    pg.keyboard.press('3'); pg.wait_for_timeout(200); assert pg.evaluate("R.sel.zone") == 'spandrel', 'клавиша 3 не ставит зону'
    pg.click('#rgLeft [data-lib="all"]'); pg.wait_for_timeout(400)
    if pg.query_selector('#rgLibAll .rgsch'):
        pg.click('#rgLibAll .rgsch'); pg.wait_for_timeout(500)
        assert pg.is_visible('#rgOk') and pg.is_visible('#rgSide .rghead2 figure img[src^="/out/"]'), 'после выбора схемы нет схемы рядом с фото или кнопок проверки'
        n0 = pg.evaluate("R.sel.id"); pg.evaluate("R.rug.marks.push({...R.sel,id:'mX',box:[0.1,0.1,0.05,0.05],status:'proposed'})")
        pg.click('#rgOk'); pg.wait_for_timeout(600)
        assert pg.evaluate(f"R.rug.marks.find(m=>m.id==='{n0}').status") == 'confirmed' and pg.evaluate("R.sel.id") != n0, '«Верно → следующая» не отметило и не перешло'
    pg.fill('#rgMotQ', 'dəvə'); pg.wait_for_timeout(900)
    errors_ok(ctx, 'панель рамки v11.15')


# ---------------- запуск ----------------
@test
def t23_tab_group_sweep(pg, ctx):
    """v11.16: Tab — только внутри группы выбранного элемента и сначала непроверенные; клик мимо рамки выбор не снимает; Shift+протяжка убирает машинные рамки"""
    pg.goto(ctx['url']); pg.wait_for_timeout(2000)
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200); pg.evaluate("rugOpen('border_rep')"); pg.wait_for_timeout(1500)
    pg.evaluate("""(()=>{const mk=(id,x,y,g,st,mach)=>({id,box:[x,y,0.08,0.05],zone:'field',status:st,motif:'',motif_text:'',scheme:'',group:g,angle:0,flip:0,machine:mach,certainty:'sure'});
      R.rug.marks=[mk('a1',.1,.1,'a1','confirmed',0),mk('a2',.3,.1,'a1','proposed',1),mk('a3',.5,.1,'a1','proposed',1),mk('b1',.1,.5,'b1','proposed',0),mk('b2',.3,.5,'b1','proposed',1)];
      rugSelect(R.rug.marks[0]);rugDraw();rugSide();})()""")
    seen = []
    for _ in range(4):
        pg.keyboard.press('Tab'); pg.wait_for_timeout(150); seen.append(pg.evaluate("R.sel.id"))
    assert seen == ['a2', 'a3', 'a1', 'a2'], f'Tab ушёл за пределы группы или начал не с непроверенных: {seen}'
    pg.keyboard.press('Shift+Tab'); pg.wait_for_timeout(150)
    assert pg.evaluate("R.sel.id") == 'a1'
    # ✓ — переход к следующей непроверенной той же группы
    pg.evaluate("rugSelect(R.rug.marks.find(m=>m.id==='a2'))"); pg.keyboard.press('c'); pg.wait_for_timeout(200)
    assert pg.evaluate("R.sel.id") == 'a3', 'после «верно» не выбрана следующая непроверенная'
    # клик мимо рамки выбор не снимает
    b = pg.query_selector('#rgView').bounding_box()
    pg.mouse.click(b['x'] + b['width'] * 0.9, b['y'] + b['height'] * 0.9); pg.wait_for_timeout(200)
    assert pg.evaluate("!!R.sel"), 'клик по пустому месту снял выбор'
    # Shift+протяжка вокруг b2 убирает машинную рамку b2 и не трогает ручную b1
    pg.evaluate("R.z=1;R.ox=0;R.oy=0;rugApply()"); pg.wait_for_timeout(200)
    p0 = pg.evaluate("(()=>{const r=document.getElementById('rgView').getBoundingClientRect(),i=document.getElementById('rgSvg').getBoundingClientRect();return [i.left,i.top,i.width,i.height]})()")
    x0, y0, w, h = p0
    pg.keyboard.down('Shift'); pg.mouse.move(x0 + w * .2, y0 + h * .40); pg.mouse.down(); pg.mouse.move(x0 + w * .45, y0 + h * .65, steps=4); pg.mouse.up(); pg.keyboard.up('Shift'); pg.wait_for_timeout(300)
    ids = pg.evaluate("R.rug.marks.map(m=>m.id)")
    assert 'b2' not in ids and 'b1' in ids, f'Shift+протяжка: {ids}'
    errors_ok(ctx, "t23")


@test
def t24_people_schemes_twins_v1118(pg, ctx):
    """v11.18: фото с людьми — фильтр и «скрыть разом» (с исключением); «Все схемы» — счётчик; склейка по цветовой шкале не считается «тем же ковром»"""
    import cv2, numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import rug_match
    rng = np.random.default_rng(5)
    def rug(seed):
        r = np.random.default_rng(seed); a = np.full((900, 700, 3), (40, 60, 150), np.uint8)
        for _ in range(250):
            c = tuple(int(x) for x in r.integers(0, 255, 3)); x, y = int(r.integers(0, 700)), int(r.integers(0, 820))
            cv2.fillPoly(a, [np.int32([[x, y], [x + int(r.integers(10, 60)), y + int(r.integers(-30, 30))], [x + int(r.integers(-20, 40)), y + int(r.integers(10, 60))]])], c)
        return cv2.GaussianBlur(a, (3, 3), 0)
    chart = np.zeros((80, 700, 3), np.uint8)
    for i in range(24): chart[10:60, i * 28 + 4:i * 28 + 26] = rng.integers(0, 255, 3)
    A, B = rug(1), rug(2); A[820:] = chart; B[820:] = chart
    n = rug_match.align(A, B)[1]
    assert n < 20, f'разные ковры с одинаковой цветовой шкалой: совпало {n} точек (должно быть «разные»)'
    D = cv2.resize(A[100:500, 100:450], None, fx=2, fy=2)
    assert rug_match.align(D, A)[1] >= 60, 'крупный план того же ковра больше не узнаётся'
    u = ctx['url']
    pg.goto(u); pg.wait_for_timeout(1800); pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    if pg.evaluate("R.view") != 'grid': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    pg.select_option('#rgFCheck', 'ppl'); pg.wait_for_timeout(500)
    ids = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")
    assert sorted(ids) == ['ppl_1', 'ppl_2'], f'фильтр «люди на фото»: {ids} · ' + str(pg.evaluate("R.list.filter(r=>r.id.startsWith('ppl')).map(r=>[r.id,r.people,r.obj,r.hidden,rugKindOf(r)])")) + ' fk=' + str(pg.evaluate("R.fkind"))
    pg.click('[data-keep="ppl_2"]'); pg.wait_for_timeout(300)
    pg.evaluate('window.confirm=()=>true'); pg.click('#rgPplHide'); pg.wait_for_timeout(1500)
    assert get(u + 'api/rugs/ppl_1').get('hidden'), 'фото с людьми не скрылось'
    assert not get(u + 'api/rugs/ppl_2').get('hidden'), 'оставленное фото скрылось'
    pg.select_option('#rgFCheck', ''); pg.wait_for_timeout(300)
    # «Все схемы» в редакторе: счётчик всех схем, не обрезано до 150
    pg.evaluate("rugOpen('vam_1')"); pg.wait_for_timeout(1500)
    pg.click('#rgLibTabs [data-lib="all"]'); pg.wait_for_timeout(500)
    tx = pg.evaluate("document.getElementById('rgLibAll').innerText")
    assert 'схем:' in tx, 'нет счётчика схем во «Все схемы»'
    st = post(u + 'api/rugs_people', {'all': True})
    for _ in range(60):
        st = get(u + 'api/rugs_people')
        if not st['running']: break
        time.sleep(1)
    assert not st['running'] and not st['error'] and st['done'] == st['total'] > 0, f'проверка фото на людей: {st}'
    errors_ok(ctx, 't24')


@test
def t25_friezes_v1119(pg, ctx):
    """v11.19: каймы как фризы — машина находит раппорт без базы, «разметить повторы» ставит группу рамок"""
    u = ctx['url']
    j = post(u + 'api/rugs/frieze_1/friezes', {})
    for _ in range(40):
        if 'pending' not in j: break
        time.sleep(1); j = get(u + 'api/jobs/' + j['pending'])
        j = j.get('res', j) if isinstance(j, dict) and j.get('done') else j
    assert j.get('bands'), f'раппорт каймы не найден: {str(j)[:300]}'
    b = max(j['bands'], key=lambda b: len(b['boxes']))
    assert len(b['boxes']) >= 12 and b['tile'].startswith('data:image'), f'мало повторов каймы: {len(b["boxes"])}'
    pg.goto(u); pg.wait_for_timeout(1800); pg.click('#tab-rugs'); pg.wait_for_timeout(1000); pg.evaluate("rugOpen('frieze_1')"); pg.wait_for_timeout(1500)
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    pg.click('#rgFriez'); pg.wait_for_selector('#modal [data-fz]', timeout=30000)
    n0 = pg.evaluate("R.rug.marks.length"); pg.click('#modal [data-fz]'); pg.wait_for_timeout(800)
    n1 = pg.evaluate("R.rug.marks.length"); g = pg.evaluate("new Set(R.rug.marks.map(m=>m.group)).size")
    assert n1 - n0 >= 8 and g == 1, f'«разметить повторы»: было {n0}, стало {n1}, групп {g}'
    errors_ok(ctx, 't25')


@test
def t26_sign_library_v1120(pg, ctx):
    """v11.20: словарь знаков кайм — прогон по всем фото, группа из двух ковров, назвать, разметить на всех"""
    u = ctx['url']
    post(u + 'api/signs/run', {})
    for _ in range(90):
        d = get(u + 'api/signs?min_rugs=1')
        if not d['state']['running'] and d['rugs_done']: break
        time.sleep(1)
    assert not d['state']['error'], d['state']
    g = next((g for g in d['clusters'] if {'frieze_1', 'frieze_2'} <= {m['rid'] for m in g['members']}), None)
    assert g, 'кайма двух ковров не попала в одну группу: ' + str([[m['rid'] for m in c['members']] for c in d['clusters']][:8])
    post(u + 'api/signs/name', {'keys': [m['key'] for m in g['members']], 'name': {'motif': '', 'motif_text': 'тест-знак', 'az': 'тест-знак'}})
    j = post(u + 'api/signs/mark', {'keys': [m['key'] for m in g['members']], 'name': {'motif': '', 'motif_text': 'тест-знак'}})
    assert j['rugs'] >= 2 and j['marks'] >= 16, j
    r = get(u + 'api/rugs/frieze_2'); assert any(m.get('motif_text') == 'тест-знак' for m in r['marks']), 'имя знака не перешло на рамки'
    pg.goto(u); pg.wait_for_timeout(1800); pg.click('#tab-rugs'); pg.wait_for_timeout(1000)
    if pg.evaluate("R.view") != 'grid': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    rgtool(pg, 'rgSignsBtn'); pg.wait_for_selector('#rgSignsV .sgc', timeout=20000)
    assert 'тест-знак' in (pg.text_content('#rgSignsV') or ''), 'названный знак не виден в словаре'
    pg.click('#sgBack'); pg.wait_for_timeout(400); assert pg.evaluate("R.view") == 'grid'
    errors_ok(ctx, 't26')


@test
def t27_multiselect_phash_v1121(pg, ctx):
    """v11.21: один снимок в двух размерах склеивается (рамки переходят в оригинал); множественный выбор в плитке — «один ковёр», «скрыть»"""
    u = ctx['url']
    post(u + 'api/rugs_phash', {})
    for _ in range(60):
        if not get(u + 'api/rugs_phash')['running']: break
        time.sleep(1)
    s = get(u + 'api/rugs/cm_shot_s'); o = get(u + 'api/rugs/cm_shot_o')
    assert s.get('dup_of') == 'cm_shot_o', f'ресайз не склеен с оригиналом: {s.get("dup_of")}'
    assert any(m.get('motif_text') == 'тест' for m in o['marks']), 'рамки копии не перешли в оригинал'
    assert not get(u + 'api/rugs/vam_1').get('dup_of'), 'разные ковры склеены как один снимок'
    pg.goto(u); pg.wait_for_timeout(1800); pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    if pg.evaluate("R.view") != 'grid': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    pg.click('#rgTree .pn[data-k="museum"]'); pg.wait_for_timeout(500)
    ids = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")
    a, b = [i for i in ids if i.startswith('vam_')][:2]
    pg.click(f'.rgcard[data-id="{a}"]', modifiers=['Meta']); pg.wait_for_timeout(200)
    assert pg.evaluate("R.view") == 'grid' and pg.is_visible('#rgMsel'), 'Cmd+клик не выбирает плитку'
    pg.click(f'.rgcard[data-id="{b}"]'); pg.wait_for_timeout(200)      # пока что-то выбрано — клик тоже выбирает
    assert pg.evaluate("R.msel.size") == 2
    pg.click('#msOne'); pg.wait_for_timeout(1500)
    assert (get(u + f'api/rugs/{a}').get('same_ok') or {}).get(b, {}).get('ok'), '«один ковёр» не связал фото'
    ids2 = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")
    assert len(ids2) == len(ids) - 1, f'после «один ковёр» плиток не стало меньше: {len(ids)} → {len(ids2)}'
    pg.click(f'.rgcard[data-id="{ids2[-1]}"] .rgck'); pg.wait_for_timeout(200)
    pg.evaluate('window.confirm=()=>true'); pg.click('#msHide'); pg.wait_for_timeout(1500)
    assert get(u + f'api/rugs/{ids2[-1]}').get('hidden'), 'выбранное фото не скрылось'
    pg.keyboard.press('Escape'); errors_ok(ctx, 't27')


@test
def t28_pages_grades_fragments_v1122(pg, ctx):
    """v11.22: лист книги, с которого вырезан ковёр, склеивается с вырезкой (рамки переводятся); грейды; фрагменты пачкой"""
    u = ctx['url']
    post(u + 'api/rugs_pages_dedupe', {})
    for _ in range(60):
        a, b = get(u + 'api/rugs/kerimov_vol9_p050_1'), get(u + 'api/rugs/kerimov_vol9_p050_2')
        if a.get('dup_of') or b.get('dup_of'): break
        time.sleep(1)
    assert a.get('dup_of') == 'kerimov_vol9_p050_2' and not b.get('dup_of'), f'лист не склеен с вырезанным ковром: {a.get("dup_of")} / {b.get("dup_of")}'
    L = get(u + 'api/rugs')
    assert 'elx' in L and all('q' in r and 'fragment' in r for r in L['rugs']), 'нет данных для грейда'
    pg.goto(u); pg.wait_for_timeout(1800); pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    if pg.evaluate("R.view") != 'grid': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    assert pg.evaluate("document.querySelectorAll('.rgcard .rggr').length") == pg.evaluate("document.querySelectorAll('.rgcard').length"), 'не у всех плиток есть грейд'
    n = sum(pg.evaluate("R.gradeCnt")[g] for g in 'ABCD')
    pg.click('#rgTree [data-grade="D"]'); pg.wait_for_timeout(400)
    assert pg.evaluate("[...document.querySelectorAll('.rgcard .rggr')].every(b=>b.textContent==='D')"), 'фильтр грейда пропускает другие'
    pg.click('#rgTree [data-grade="D"]'); pg.wait_for_timeout(300)
    ids = pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)")
    pg.click(f'.rgcard[data-id="{ids[0]}"] .rgck'); pg.wait_for_timeout(200); pg.click('#msFrag'); pg.wait_for_timeout(1500)
    assert get(u + f'api/rugs/{ids[0]}').get('fragment') is True, 'отметка «фрагмент» пачкой не сохранилась'
    pg.select_option('#rgFShow', 'frag'); pg.wait_for_timeout(400)
    assert ids[0] in pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)"), 'фильтр «фрагменты» не показывает отмеченное'
    pg.select_option('#rgFShow', ''); errors_ok(ctx, 't28')


@test
def t29_etalon_v1124(pg, ctx):
    """v11.24: эталоны — детектор учится на ✓ рамках элемента и находит его на коврах того же типа; проверка находок пачкой"""
    u = ctx['url']
    st = post(u + 'api/motif/index', {})
    for _ in range(80):
        st = get(u + 'api/motif/status')
        if not st['running'] and st['indexed'] and not st['stale']: break
        time.sleep(0.5)
    def box(cx, cy, W, H=420): return [(cx - 40) / W, (cy - 40) / H, 80 / W, 80 / H]
    mk = lambda i, b: {'id': 'et' + i, 'box': b, 'zone': 'field', 'status': 'confirmed', 'motif_text': 'звезда-эталон'}
    for rid, b in (('vam_0', box(150, 200, 300)), ('vam_3', box(230, 140, 330)), ('vam_star', box(160, 220, 320))):
        r = get(u + f'api/rugs/{rid}'); post(u + f'api/rugs/{rid}', {'composition': 'pirebedil', 'marks': [m for m in r['marks'] if not m['id'].startswith('et')] + [mk(rid, b)]})
    for rid in ('vam_5', 'vam_1'): post(u + f'api/rugs/{rid}', {'composition': 'pirebedil', 'marks': [], 'hidden': ''})   # звезду на vam_5 мог отметить t16, скрыть — t27
    E = get(u + 'api/etalon'); e = next((x for x in E['elements'] if x['key'] == 't:звезда-эталон'), None)
    assert e and e['npos'] == 3 and e['types'].get('pirebedil') == 3, f'элемент не собран из ✓ рамок: {e}'
    post(u + 'api/etalon/run', {'keys': ['t:звезда-эталон']})
    for _ in range(60):
        s = get(u + 'api/etalon')['state']
        if not s['running']: break
        time.sleep(0.5)
    assert not s['error'], s
    F = get(u + 'api/etalon/finds?key=' + urllib.parse.quote('t:звезда-эталон'))['finds']
    hit = [f for f in F if f['rid'] == 'vam_5']
    sb = box(110, 300, 350)
    assert any(max(0, min(f['box'][0] + f['box'][2], sb[0] + sb[2]) - max(f['box'][0], sb[0])) > 0 and max(0, min(f['box'][1] + f['box'][3], sb[1] + sb[3]) - max(f['box'][1], sb[1])) > 0 for f in hit), \
        f'детектор не нашёл звезду на vam_5: {[(f["rid"], f["box"], f["score"]) for f in F][:8]}'
    j = post(u + 'api/marks_status', {'items': [{'rid': hit[0]['rid'], 'mid': hit[0]['mid'], 'status': 'confirmed'}]})
    assert j['n'] == 1 and next(m for m in get(u + 'api/rugs/vam_5')['marks'] if m['id'] == hit[0]['mid'])['status'] == 'confirmed'
    # экран: тип → элементы и ковры-эталоны; проверка находок клавишами
    pg.goto(u); pg.wait_for_timeout(1800); pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    if pg.evaluate("R.view") != 'grid': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    rgtool(pg, 'rgEtBtn'); pg.wait_for_selector('#etTypes [data-et="pirebedil"]', timeout=15000); pg.click('#etTypes [data-et="pirebedil"]'); pg.wait_for_timeout(400)
    assert 'звезда-эталон' in (pg.text_content('#etMain .etels') or ''), 'элемент не показан у типа'
    assert pg.evaluate("document.querySelectorAll('#etMain .etrug').length") >= 4, 'нет ковров-эталонов этого типа'
    left = sum(1 for f in F if f['mid'] != hit[0]['mid'])
    if left:
        pg.click('#etFinds'); pg.wait_for_selector('#modal .efc', timeout=10000)
        pg.keyboard.press('x'); pg.wait_for_timeout(800)
        assert pg.evaluate("document.querySelectorAll('#modal .efc.no').length") == 1, 'X не отклонил находку'
        pg.keyboard.press('Escape'); pg.wait_for_timeout(500)
    pg.click('#etMain [data-etdone]'); pg.wait_for_timeout(800)
    assert pg.evaluate("document.querySelectorAll('#etMain [data-etdone].on').length") >= 1, '«отметить эталоном» не сработало'
    pg.click('#etBack'); pg.wait_for_timeout(300); errors_ok(ctx, 't29')


@test
def t30_sam_contour(pg, ctx):
    """v11.28: контур элемента (SAM) — рамка хранит poly, инструмент «контур» есть в студии; без torch сервер честно отвечает 400"""
    u = ctx['url']
    r = get(u + 'api/rugs/vam_2')
    poly = [[0.1, 0.1], [0.3, 0.12], [0.28, 0.3], [0.12, 0.28]]
    keep = [m for m in r['marks'] if m['id'] != 'samt']
    post(u + 'api/rugs/vam_2', {'marks': keep + [{'id': 'samt', 'box': [0.1, 0.1, 0.2, 0.2], 'zone': 'field', 'status': 'proposed', 'motif_text': 'контур', 'poly': poly}]})
    m = next(m for m in get(u + 'api/rugs/vam_2')['marks'] if m['id'] == 'samt')
    assert m.get('poly') == poly, f'контур не сохранился: {m}'
    try: post(u + 'api/rugs/vam_2/segment', {'points': [[0.2, 0.2, 1]]}); ok = True
    except Exception as e: ok = '400' in str(e)
    assert ok, 'segment без SAM должен отвечать 400'
    pg.goto(u); pg.wait_for_timeout(1500); pg.click('#tab-rugs'); pg.wait_for_timeout(1000)
    pg.evaluate("rugOpen('vam_2')"); pg.wait_for_selector('#rgTMagic', timeout=10000); pg.wait_for_timeout(800)
    assert pg.evaluate("document.querySelectorAll('#rgSvg polygon').length") == 0, 'контур невыбранной рамки рисуется (каша)'
    pg.evaluate("rugSelect(R.rug.marks.find(m=>m.id==='samt'))"); pg.wait_for_timeout(200)
    assert pg.evaluate("document.querySelectorAll('#rgSvg polygon').length") == 1, 'контур выбранной рамки не нарисован'
    pg.keyboard.press('m'); pg.wait_for_timeout(200)
    assert pg.evaluate("R.tool") == 'magic' and pg.evaluate("$('rgTMagic').classList.contains('on')"), 'M не включил инструмент «контур»'
    pg.keyboard.press('b'); pg.wait_for_timeout(200)
    # сохранение из студии не теряет контур
    pg.evaluate("R.sel=R.rug.marks.find(m=>m.id==='samt');rugSaveNow()"); pg.wait_for_timeout(800)
    m = next(m for m in get(u + 'api/rugs/vam_2')['marks'] if m['id'] == 'samt')
    assert m.get('poly') == poly, 'студия потеряла контур при сохранении'
    post(u + 'api/rugs/vam_2', {'marks': keep})


@test
def t31_etalon_shapes_v1129(pg, ctx):
    """v11.29: образцы одного названия разной формы — делятся на формы; чужую форму можно переименовать вместе с повторами"""
    u = ctx['url']
    sq = lambda x, y: [x, y, 0.12, 0.12]
    bar = lambda x, y: [x, y, 0.4, 0.04]
    saved = {}
    for i, rid in enumerate(('vam_0', 'vam_1', 'vam_3')):
        r = get(u + f'api/rugs/{rid}'); saved[rid] = r['marks']
        ms = [m for m in r['marks'] if not m['id'].startswith('fs')]
        ms += [{'id': f'fsq{i}', 'box': sq(0.1, 0.1), 'zone': 'field', 'status': 'confirmed', 'motif_text': 'форма-тест'},
               {'id': f'fsb{i}', 'box': bar(0.3, 0.6), 'zone': 'field', 'status': 'confirmed', 'motif_text': 'форма-тест', 'group': f'fsb{i}'},
               {'id': f'fsr{i}', 'box': bar(0.3, 0.8), 'zone': 'field', 'status': 'proposed', 'motif_text': 'форма-тест', 'group': f'fsb{i}'}]
        post(u + f'api/rugs/{rid}', {'marks': ms})
    e = next(x for x in get(u + 'api/etalon')['elements'] if x['key'] == 't:форма-тест')
    assert len(e['shapes']) == 2 and sorted(len(s) for s in e['shapes']) == [3, 3], f'формы не разделены: {[len(s) for s in e["shapes"]]}'
    bars = next(s for s in e['shapes'] if s[0]['box'][2] > 0.3)
    j = post(u + 'api/etalon/relabel', {'items': [{'rid': x['rid'], 'mid': x['mid']} for x in bars], 'name': 'планка-тест'})
    assert j['n'] == 6, f'переименованы не все рамки формы с повторами: {j}'
    E = {x['key']: x for x in get(u + 'api/etalon')['elements']}
    assert len(E['t:форма-тест']['shapes']) == 1 and E['t:планка-тест']['npos'] == 3
    for rid, ms in saved.items(): post(u + f'api/rugs/{rid}', {'marks': ms})



@test
def t32_triage_batch(pg, ctx):
    """v11.34: этап 0–1 — вид кадра и углы пачкой; клик по неверным, остальное принимается (последним: меняет разметку фикстур)"""
    import urllib.request, json as _j
    def post(path, body):
        rq = urllib.request.Request(ctx['url'] + path, data=_j.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        return _j.load(urllib.request.urlopen(rq))
    j = post('api/rugs_triage', {'ids': ['persp_1']})['res']['persp_1']
    assert j['frame'] == 'whole', f'ковёр на сером фоне должен быть «целый», а не {j["frame"]}'
    assert j['conf'] > 0.8 and len(j['quad']) == 4, 'углы не найдены уверенно'
    L = _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs'))['rugs']
    assert L and all(x.get('use') in ('work', 'maybe', 'skip') for x in L), 'у ковров нет отбора work/maybe/skip'
    assert any(x['use'] == 'skip' and x.get('use_why') for x in L) or len(L) < 3, 'отложенные без причины'
    kp = [x for x in L if x['id'].startswith('kerimov') and not x.get('multi') and x['use'] != 'skip']
    assert all(x['use'] == 'work' for x in kp), 'ковры из Керимова не должны отбрасываться по размеру и рамке'
    q = _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs_triage?limit=5'))
    assert 'ids' in q and 'stats' in q, 'нет очереди этапа 0–1'
    pg.click('#tab-rugs'); pg.wait_for_timeout(800)
    rgtool(pg, 'rgTriBtn'); pg.wait_for_timeout(4000)
    n = pg.evaluate("document.querySelectorAll('#rgTri .trt').length")
    if n:
        assert pg.evaluate("document.querySelectorAll('#rgTri .trt polygon').length") >= 1, 'на плитках нет контура ковра'
        ids = pg.evaluate("[...document.querySelectorAll('#rgTri .trt')].map(e=>e.dataset.i)")
        moved = None
        if n >= 2:
            hb = pg.locator('#rgTri .trt:nth-child(2) .hd').first.bounding_box()
            if hb:
                cx, cy = hb['x'] + hb['width'] / 2, hb['y'] + hb['height'] / 2
                pg.mouse.move(cx, cy); pg.mouse.down(); pg.mouse.move(cx + 22, cy + 18, steps=6); pg.mouse.up(); pg.wait_for_timeout(200)
                moved = pg.evaluate("document.querySelector('#rgTri .trt:nth-child(2)').classList.contains('ed')")
                pg.mouse.move(cx + 22, cy + 18); pg.mouse.down(); pg.mouse.move(cx + 26, cy + 20, steps=3)
                assert pg.evaluate("getComputedStyle(document.getElementById('triLoupe')).display") == 'block', 'при перетаскивании угла нет лупы'
                pg.screenshot(path=f'{OUT}/32_triage_loupe.png'); pg.mouse.up(); pg.wait_for_timeout(150)
                assert pg.evaluate("getComputedStyle(document.getElementById('triLoupe')).display") == 'none', 'лупа не прячется'
                assert moved, 'перетаскивание угла в плитке не сработало'
        pg.evaluate("document.querySelector('#rgTri .trt').click()"); pg.wait_for_timeout(200)
        first_bad = pg.evaluate("document.querySelector('#rgTri .trt').classList.contains('bad')")
        pg.screenshot(path=f'{OUT}/32_triage.png')
        if 'tall_1' in ids:          # все 4 угла длинной дорожки видны в пределах плитки
            vis = pg.evaluate("""(()=>{const e=[...document.querySelectorAll('#rgTri .trt')].find(x=>x.dataset.i==='tall_1'),r=e.getBoundingClientRect();
              return [...e.querySelectorAll('.hd')].map(h=>{const b=h.getBoundingClientRect(),y=b.top+b.height/2;return y>=r.top&&y<=r.bottom;});})()""")
            assert vis and all(vis), f'в плитке длинного ковра видны не все углы: {vis}' 
        pg.click('#triOk'); pg.wait_for_timeout(2500)
        r0 = _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs/' + ids[0]))
        if first_bad: assert r0.get('tri') == 'review', 'неверный ковёр не помечен «проверить вручную»'
        if moved:
            r1 = _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs/' + ids[1]))
            assert r1.get('tri') == 'hand' and r1['layout']['quad_status'] == 'accepted', 'поправленные углы не сохранились'
        acc = [i for i in ids[1:] if ((_j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs/' + i)).get('layout') or {}).get('quad_status') == 'accepted')]
        assert acc or n == 1, 'принятые ковры не получили углы'
    errors_ok(ctx, 'проверка пачкой')
    pg.evaluate("(document.getElementById('rgTri')||{remove(){}}).remove()")

@test
def t33_micro_phone(pg, ctx):
    """v11.39: телефонный экран /m — карточки «целый ли ковёр» и «углы верны», правка угла панорамой, контрольные карточки"""
    import urllib.request, urllib.error, json as _j
    def post(path, body):
        rq = urllib.request.Request(ctx['url'] + path, data=_j.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        return _j.load(urllib.request.urlopen(rq))
    post('api/rugs_triage', {'ids': ['tall_1', 'frieze_1', 'cm_shot_o']})
    for i in ('tall_1', 'frieze_1', 'cm_shot_o'): post('api/micro/answer', {'type': 'frame', 'rid': i, 'a': 'whole'})   # углы спрашиваем только у признанных целыми
    nx = _j.load(urllib.request.urlopen(ctx['url'] + 'api/micro/next?n=6'))
    assert nx['tasks'], f'нет карточек: {nx}'
    mc = pg.context.browser.new_context(viewport={'width': 390, 'height': 844}, has_touch=True); m = mc.new_page()
    errs = []; m.on('pageerror', lambda e: errs.append(str(e)))
    m.goto(ctx['url'] + 'm?k=quad,frame'); m.wait_for_selector('.q', timeout=15000); m.wait_for_timeout(800)   # v11.52: остальные виды карточек — в t36
    m.screenshot(path=f'{OUT}/33_micro_first.png')
    hv = m.inner_text('#me'); assert re.search(r'v1[1-9]\.', hv), f'в шапке /m нет номера версии: {hv!r}'      # v11.46
    assert m.query_selector('#ib') and not m.is_visible('#tip'), 'подсказка должна быть спрятана под «i»'
    m.click('#ib'); assert m.is_visible('#tip'), '«i» не раскрывает подсказку'; m.click('#ib')
    cdp = mc.new_cdp_session(m); sb = m.query_selector('#stage').bounding_box(); cx, cy = sb['x'] + sb['width'] / 2, sb['y'] + sb['height'] / 2
    cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': cx - 30, 'y': cy, 'id': 1}, {'x': cx + 30, 'y': cy, 'id': 2}]})
    for st_ in range(1, 7): cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [{'x': cx - 30 - st_ * 12, 'y': cy, 'id': 1}, {'x': cx + 30 + st_ * 12, 'y': cy, 'id': 2}]})
    cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []}); m.wait_for_timeout(200)
    zk = m.evaluate("window.__zoom&&window.__zoom()"); m.screenshot(path=f'{OUT}/33_micro_zoom.png')
    assert zk and zk > 2, f'щипок двумя пальцами не увеличил фото: {zk}'
    m.click('.zb'); m.wait_for_timeout(150); assert m.evaluate("window.__zoom()") == 1, 'сброс масштаба не работает'
    kinds = []; fixed = False
    for _ in range(10):
        if m.query_selector('.end'): break
        q = m.inner_text('.q')
        if 'целый' in q:
            kinds.append('frame'); m.click('#a1')
        else:
            kinds.append('quad'); m.wait_for_timeout(600)
            bg = m.evaluate("getComputedStyle(document.querySelector('#cz .cz')).backgroundImage")
            assert 'rugmid' in bg, 'в карточке углов нет крупного вида угла'
            if not fixed:
                m.check('#fd')                                   # отметка «край повреждён» переживает правку угла
                m.click('#cz .cz'); m.wait_for_selector('#fx')                                   # v11.46: кнопки «поправить» нет — правка по тапу на угол; m.wait_for_timeout(300)
                b = m.query_selector('#fx').bounding_box(); cx, cy = b['x'] + b['width'] / 2, b['y'] + b['height'] / 2
                m.mouse.move(cx, cy); m.mouse.down(); m.mouse.move(cx - 30, cy - 20, steps=5); m.mouse.up()
                m.screenshot(path=f'{OUT}/33_micro_fix.png'); m.click('#ok'); fixed = True
            else: m.click('#a1')
        m.wait_for_timeout(700)
    m.screenshot(path=f'{OUT}/33_micro_end.png')
    assert 'quad' in kinds, f'не было карточек углов: {kinds}'
    assert m.query_selector('.end'), 'пачка не закончилась экраном итога'
    assert not errs, f'ошибки на телефонном экране: {errs}'
    fm = _j.load(urllib.request.urlopen(ctx['url'] + 'api/micro/stats')).get('frame_model')
    assert fm and fm.get('n', 0) >= 40, f'модель «целый ли ковёр» не загрузилась: {fm}'
    pws = [x['pw'] for x in _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs'))['rugs'] if x.get('seen')]
    assert pws and all(p is not None and 0 <= p <= 1 for p in pws), f'нет вероятности «целый» у просмотренных: {pws[:5]}'
    st = _j.load(urllib.request.urlopen(ctx['url'] + 'api/micro/stats'))['users']
    assert sum(u['total'] for u in st.values()) >= len(kinds), f'ответы не записаны: {st}'
    ch = [i for i in ('tall_1', 'frieze_1', 'cm_shot_o') if (_j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs/' + i)).get('layout') or {}).get('quad_status') == 'confirmed']
    assert ch, 'ответ админа об углах не применился сразу'
    ed = [x['id'] for x in _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs'))['rugs'] if 'damaged' in (x.get('edge') or [])]
    assert ed, 'отметка «край повреждён» не сохранилась'
    # v11.45: три значения кадра (целый / обрезан / деталь) и галочка «снят в перспективе»
    post('api/micro/answer', {'type': 'frame', 'rid': 'persp_1', 'a': 'detail', 'flags': {'skew': True}})
    d = _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs/persp_1'))
    assert d.get('frame') == 'detail', f'ответ «только деталь» не сохранился: {d.get("frame")}'
    pr = [x for x in _j.load(urllib.request.urlopen(ctx['url'] + 'api/rugs'))['rugs'] if x['id'] == 'persp_1'][0]
    assert 'skew' in (pr.get('edge') or []), f'галочка «снят в перспективе» не сохранилась: {pr.get("edge")}'
    assert pr['use'] == 'skip' and 'деталь' in pr['use_why'], f'деталь ковра должна уходить из рабочей очереди: {pr["use"], pr["use_why"]}'
    assert 'damaged' not in (pr.get('edge') or []), 'карточка «что на фото» не должна трогать галочки другой карточки'
    bad = urllib.request.Request(ctx['url'] + 'api/micro/answer', data=_j.dumps({'type': 'frame', 'rid': 'persp_1', 'a': 'fragment'}).encode(), headers={'Content-Type': 'application/json'})
    try: urllib.request.urlopen(bad); assert False, 'снятый ответ «fragment» всё ещё принимается'
    except urllib.error.HTTPError as e: assert e.code == 400, f'ожидали 400 на снятый ответ, получили {e.code}'
    mc.close()


@test
def t34_vlm_corners(pg, ctx):
    """v11.47: углы моделью в два шага — фоновый расчёт, подмена углов на карточке, «да» принимает именно их (модель — заглушка CPS_VLM_MOCK)"""
    import urllib.request, json as _j
    def post(path, body):
        rq = urllib.request.Request(ctx['url'] + path, data=_j.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        return _j.load(urllib.request.urlopen(rq))
    def get(path): return _j.load(urllib.request.urlopen(ctx['url'] + path))
    st = get('api/vlm/status'); assert st['key'] and 'queued' in st, f'нет статуса модели: {st}'
    ids = ['frieze_2', 'met_123', 'kerimov_vol9_p050_1']
    post('api/rugs_triage', {'ids': ids})
    for i in ids: post('api/micro/answer', {'type': 'frame', 'rid': i, 'a': 'whole'})
    get('api/micro/next?n=1')                                    # первая карточка показана, остальные ставятся в очередь расчёта
    for _ in range(60):
        st = get('api/vlm/status')
        if st['done_today'] >= 1 and st['queued'] == 0: break
        time.sleep(0.5)
    assert st['done_today'] >= 1, f'модель углов ничего не посчитала: {st}'
    assert st['refine']['ok'] >= 1 and st['refine']['fail'] == 0, f'уточнение углов не сработало: {st}'   # v11.50
    tk = [t for t in get('api/micro/next?n=8')['tasks'] if t['type'] == 'quad' and t.get('qsrc') == 'vlm']
    assert tk, 'в очереди нет карточки с углами модели'
    q = tk[0]['quad']; assert abs(q[0][0] - 0.1) < 1e-3 and abs(q[2][0] - 0.9) < 1e-3, f'углы модели не те: {q}'
    r = post('api/micro/answer', {'type': 'quad', 'rid': tk[0]['rid'], 'a': 'ok', 's': 'vlm'})
    L = get('api/rugs/' + tk[0]['rid']).get('layout') or {}
    assert r.get('applied') == 'quad' and abs(L['quad'][0][0] - 0.1) < 1e-3, f'«да» не приняло углы модели: {r} {L.get("quad")}'
    ml = [_j.loads(x) for x in open(os.path.join(ctx['app'], 'data', 'micro.jsonl'), encoding='utf-8') if x.strip()]
    assert any(x.get('s') == 'vlm' for x in ml), 'в журнале нет пометки, что углы были от модели'


@test
def t35_micro_hide(pg, ctx):
    """v11.49: ковры из micro_hide.json не попадают в очередь «что на фото», данные ковра не меняются"""
    import urllib.request, json as _j
    hid = _j.load(open(os.path.join(ctx['app'], 'micro_hide.json'), encoding='utf-8'))['ids']
    assert len(hid) > 100, f'список скрытых подозрительно мал: {len(hid)}'
    seen = set()
    for _ in range(4):
        j = _j.load(urllib.request.urlopen(ctx['url'] + 'api/micro/next?n=12'))
        seen |= {t['rid'] for t in j['tasks'] if t['type'] == 'frame'}
    assert not (seen & set(hid)), f'скрытые ковры попали в очередь: {sorted(seen & set(hid))[:5]}'
    assert j['left'].get('hidden') == len(hid), f'в left нет числа скрытых: {j["left"]}'


@test
def t36_micro_more_cards(pg, ctx):
    """v11.54: карточки /m «элементы» (✓/✗ рамок машины), «имена» (схема из закрытого списка типа), «тип» (да/нет), «двойники» (пара фото);
    выбор видов карточек, «не знаю» не возвращает карточку, отклонённый тип больше не предлагается"""
    import cv2
    u = ctx['url']; D = ctx['app'] + '/data'
    T = get(u + 'api/types')['types']
    ty = next((t for t in T if not t['region'] and t['el'] >= 2 and len([e for e in get(u + 'api/types/' + t['id'])['elements'] if os.path.exists(f"{D}/out/{e['id']}.png")]) >= 2), None)
    assert ty, 'нет типа с двумя схемами с картинкой'
    els = [e for e in get(u + 'api/types/' + ty['id'])['elements'] if os.path.exists(f"{D}/out/{e['id']}.png")]
    src = cv2.imread(f'{D}/rug_img/frieze_1.jpg'); now = int(time.time())
    def rug(rid, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', src)
        r = {"id": rid, "title": rid, "source": "тест", "museum": "", "inv": "", "url": "", "license": "", "place": "", "date": "", "school": "", "composition": "",
             "technique": "", "size": "", "notes": "", "status": "new", "w": src.shape[1], "h": src.shape[0], "marks": [], "cat": {"src": "upload"}, "created": now, "updated": now}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    def mk(i, box, st, g, **kw): return {"id": i, "box": box, "zone": "border", "motif": "", "motif_text": "", "scheme": "", "group": g, "status": st, **kw}
    rug('mc_marks', composition=ty['id'], marks=[
        mk('a1', [.06, .10, .10, .05], 'confirmed', 'gA', motif_text='уже назван'), mk('a2', [.06, .15, .10, .05], 'proposed', 'gA', machine=1), mk('a3', [.06, .20, .10, .05], 'proposed', 'gA', machine=1), mk('a5', [.06, .25, .10, .05], 'proposed', 'gA', machine=1),
        mk('a4', [.30, .40, .10, .05], 'rejected', 'gA', machine=1), mk('b1', [.40, .30, .12, .08], 'confirmed', 'gB'), mk('b2', [.55, .30, .12, .08], 'confirmed', 'gB')])
    rug('mc_comp', title='Pirəbədil carpet, Quba'); rug('mc_comp2', title='Pirebedil rug')
    rug('mc_s1', same=[{"id": "mc_s2", "score": 30}]); rug('mc_s2', same=[{"id": "mc_s1", "score": 30}])
    rug('mc_s3', same=[{"id": "mc_s4", "score": 25}]); rug('mc_s4')
    K = 'marks,name,comp,same'
    nx = get(u + 'api/micro/next?n=12&k=' + K); kinds = [t['type'] for t in nx['tasks']]
    assert set(kinds) == set(K.split(',')), f'в очереди не все виды карточек: {kinds} {nx["left"]}'
    assert nx['left']['marks'] >= 1 and nx['left']['comp'] >= 2 and nx['left']['same'] >= 2 and nx['left']['name'] >= 1, nx['left']
    tm = next(t for t in nx['tasks'] if t['type'] == 'marks' and t['rid'] == 'mc_marks')
    assert tm['sample'] == [.06, .10, .10, .05] and {'a2', 'a3'} <= {i['id'] for i in tm['items']}, f'карточка элементов: {tm}'
    tn = next(t for t in nx['tasks'] if t['type'] == 'name' and t['rid'] == 'mc_marks')
    assert tn['g'] == 'gB' and len(tn['opts']) >= 2 and tn['opts'][0]['img'].startswith('/out/'), f'карточка имён: {tn}'
    tc = next(t for t in nx['tasks'] if t['type'] == 'comp' and t['rid'] == 'mc_comp'); assert tc['cid'] == 'pirebedil', tc
    assert get(u + 'api/micro/next?n=6&k=quad,frame')['kinds'] == ['quad', 'frame'], 'выбор видов карточек не работает'
    # «не знаю» и «нет»: карточка не возвращается, отклонённый тип больше не предлагается
    post(u + 'api/micro/answer', {'type': 'same', 'rid': 'mc_s3', 'other': 'mc_s4', 'a': 'skip'})
    r = post(u + 'api/micro/answer', {'type': 'comp', 'rid': 'mc_comp2', 'cid': 'pirebedil', 'a': 'no'}); assert r.get('applied') == 'comp_no', r
    n2 = get(u + 'api/micro/next?n=12&k=' + K)['tasks']
    assert not [t for t in n2 if (t['type'] == 'same' and t['rid'] == 'mc_s3') or (t['type'] == 'comp' and t['rid'] == 'mc_comp2')], 'отвеченные карточки вернулись в очередь'
    assert get(u + 'api/rugs/mc_comp2').get('comp_no') == ['pirebedil'] and not get(u + 'api/rugs/mc_comp2').get('composition')
    # телефон: проходим по одной карточке каждого вида
    mc = pg.context.browser.new_context(viewport={'width': 390, 'height': 844}, has_touch=True); m = mc.new_page()
    errs = []; m.on('pageerror', lambda e: errs.append(str(e)))
    m.goto(u + 'm?k=' + K); m.wait_for_selector('.q', timeout=15000)
    assert m.eval_on_selector_all('#chips button.on', 'e=>e.length') == 4, 'кнопки видов карточек не отражают выбор'
    seen = set()
    for _ in range(14):
        m.wait_for_timeout(500)
        if m.query_selector('.end'):
            if len(seen) == 4 or not m.query_selector('.end button'): break
            m.click('.end button'); m.wait_for_selector('.q, .end', timeout=15000); continue
        q = m.inner_text('.q')
        kind = 'marks' if ('элемент' in q or 'Тап' in q) and 'Какой' not in q else 'name' if 'Какой' in q else 'comp' if 'типа' in q else 'same'
        rid = m.evaluate("S.tasks[S.i].rid")
        if kind not in seen:
            m.wait_for_function("[...document.querySelectorAll('#main img')].every(i=>i.complete&&i.naturalWidth>0)", timeout=15000)
            m.screenshot(path=f'{OUT}/36_micro_{kind}.png'); seen.add(kind)
        if kind == 'marks':
            if rid == 'mc_marks':
                m.click('#gr .tl[data-id="a3"]'); assert m.query_selector('#gr .tl.x'), 'тап по плитке не ставит ✗'
                m.click('#gr .tl[data-id="a5"]'); m.click('#gr .tl[data-id="a5"]')                       # v11.75: второй тап — ≈ «тот же, но иначе»
                assert m.query_selector('#gr .tl.v[data-id="a5"]') and not m.query_selector('#gr .tl.x[data-id="a5"]'), 'второй тап не ставит ≈'
                m.screenshot(path=f'{OUT}/36_micro_marks_var.png')
            m.click('#a1')
        elif kind == 'name':
            assert m.is_disabled('#a1'), '✓ должна ждать выбора схемы'
            m.click('#gr .tl'); m.click('#a1')
        else: m.click('#a1')
    assert seen == set(K.split(',')), f'на телефоне показаны не все виды карточек: {seen}'
    assert not errs, f'ошибки на телефонном экране: {errs}'
    mc.close(); time.sleep(0.5)
    R = get(u + 'api/rugs/mc_marks'); st = {x['id']: x for x in R['marks']}
    assert st['a2']['status'] == 'confirmed' and st['a3']['status'] == 'rejected' and st['a4']['status'] == 'rejected', f'ответ по рамкам не применился: {[(i, x["status"]) for i, x in st.items()]}'
    assert st['a5']['status'] == 'confirmed' and st['a5'].get('variant') == 'detail' and not st['a2'].get('variant'), f'«≈ тот же, но иначе» не применился: {st["a5"]}'
    assert st['b1']['scheme'] == tn['opts'][0]['id'] and st['b2']['scheme'] == st['b1']['scheme'] and not st['a1']['scheme'], 'имя не поставлено на всю группу (или задело чужую)'
    assert get(u + 'api/rugs/mc_comp')['composition'] == 'pirebedil', 'тип ковра не принят'
    a, b = get(u + 'api/rugs/mc_s1'), get(u + 'api/rugs/mc_s2')
    assert (a.get('same_ok') or {}).get('mc_s2', {}).get('ok') and (b.get('same_ok') or {}).get('mc_s1', {}).get('ok'), '«тот же ковёр» не записан у обоих'
    lg = [json.loads(x) for x in open(f'{D}/micro.jsonl', encoding='utf-8') if x.strip()]
    assert {'marks', 'name', 'comp', 'same'} <= {x.get('type') for x in lg}, 'ответы новых карточек не попали в журнал'
    # v11.56: «дубли» — пары почти одинаковых названий из словаря; ответ лежит отдельно от ковров
    q = get(u + 'api/micro/next?n=3&k=cdup'); assert q['tasks'] and q['tasks'][0]['type'] == 'cdup' and q['left']['cdup'] >= 1, 'нет карточек «дубли»'
    t0 = q['tasks'][0]; n0 = q['left']['cdup']
    assert post(u + 'api/micro/answer', {'type': 'cdup', 'a': 'same', 'pid': t0['pid']}).get('applied') == 'cdup:same', 'ответ «одно и то же» не принят'
    q2 = get(u + 'api/micro/next?n=3&k=cdup'); assert q2['left']['cdup'] == n0 - 1 and all(x['pid'] != t0['pid'] for x in q2['tasks']), 'решённая пара вернулась в очередь'
    assert next(x for x in get(u + 'api/concept_dups')['pairs'] if x['id'] == t0['pid'])['verdict'] == 'same', 'решение не видно в /api/concept_dups'
    mc = pg.context.browser.new_context(viewport={'width': 390, 'height': 844}, has_touch=True); m = mc.new_page(); errs2 = []
    m.on('pageerror', lambda e: errs2.append(str(e))); m.goto(u + 'm?k=cdup'); m.wait_for_selector('.cds .cd'); m.screenshot(path=f'{OUT}/36_micro_cdup.png')
    assert m.locator('.cds .cd').count() == 2 and not errs2, f'карточка «дубли» не показана: {errs2}'
    m.click('#a3'); time.sleep(0.4); mc.close()


@test
def t37_micro_split_spreads(pg, ctx):
    """v11.54: «несколько ковров» в карточке кадра убирает фото из очередей и ищет рамки; карточка «развороты»: ✓ режет на отдельные ковры, «один ковёр» снимает пометку;
    фоновый проход сам находит развороты у фото, которых ещё не смотрели"""
    import cv2, numpy as np
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    one = cv2.imread(f'{D}/rug_img/frieze_1.jpg'); gap = np.full((one.shape[0], 120, 3), 235, np.uint8)
    two = np.hstack([one, gap, one])
    def rug(rid, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', two)
        r = {"id": rid, "title": rid, "source": "тест", "museum": "", "inv": "", "url": "", "license": "", "place": "", "date": "", "school": "", "composition": "",
             "technique": "", "size": "", "notes": "", "status": "new", "w": two.shape[1], "h": two.shape[0], "marks": [], "cat": {"src": "upload"}, "created": now, "updated": now}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    P = [[0.03, 0.04, 0.44, 0.92], [0.53, 0.04, 0.44, 0.92]]
    rug('sp_a', multi=P); rug('sp_b', multi=P); rug('sp_c')
    rug('sp_d', auto={'frame': 'cropped', 'quad': [[.1, .1], [.9, .1], [.9, .9], [.1, .9]], 'conf': .5, 'area': .6, 'ver': 1, 't': now}, multi=[])   # для карточки кадра
    # 1. «несколько ковров» из карточки кадра: фото уходит из рабочих, рамки находятся сами
    r = post(u + 'api/micro/answer', {'type': 'frame', 'rid': 'sp_c', 'a': 'multi'}); assert r.get('applied') == 'multi', r
    c = get(u + 'api/rugs/sp_c'); assert c.get('frame') == 'multi' and len(c.get('multi') or []) == 2, f'рамки ковров не найдены: {c.get("frame")} {c.get("multi")}'
    row = [x for x in get(u + 'api/rugs')['rugs'] if x['id'] == 'sp_c'][0]; assert row['use'] == 'skip' and 'разворот' in row['use_why'] or 'несколько' in row['use_why'], row['use_why']
    # 2. карточки «развороты»
    nx = get(u + 'api/micro/next?n=12&k=split'); tk = {t['rid']: t for t in nx['tasks'] if t['type'] == 'split'}
    assert {'sp_a', 'sp_b', 'sp_c'} <= set(tk) and len(tk['sp_a']['parts']) == 2 and nx['left']['split'] >= 3, f'очередь разворотов: {list(tk)} {nx["left"]}'
    assert 'sp_c' not in [t['rid'] for t in get(u + 'api/micro/next?n=12&k=frame,marks,name,comp,same')['tasks']], 'разворот остался в других очередях'
    r = post(u + 'api/micro/answer', {'type': 'split', 'rid': 'sp_a', 'a': 'split'}); assert r.get('applied') == 'split', r
    a = get(u + 'api/rugs/sp_a'); assert len(a.get('split_into') or []) == 2, f'разворот не разрезан: {a.get("split_into")}'
    assert all(get(u + 'api/rugs/' + i).get('split_from') == 'sp_a' for i in a['split_into']), 'части без ссылки на разворот'
    r = post(u + 'api/micro/answer', {'type': 'split', 'rid': 'sp_b', 'a': 'one'}); assert r.get('applied') == 'one', r
    assert get(u + 'api/rugs/sp_b').get('multi') == [], 'пометка разворота не снята'
    post(u + 'api/micro/answer', {'type': 'split', 'rid': 'sp_c', 'a': 'skip'})
    assert 'sp_c' not in [t['rid'] for t in get(u + 'api/micro/next?n=12&k=split')['tasks']], '«не знаю» вернул карточку'
    # 3. телефон
    mc = pg.context.browser.new_context(viewport={'width': 390, 'height': 844}, has_touch=True); m = mc.new_page()
    errs = []; m.on('pageerror', lambda e: errs.append(str(e)))
    m.goto(u + 'm?k=frame'); m.wait_for_selector('#a6', timeout=15000)
    assert 'несколько ковров' in m.inner_text('#a6'), 'в карточке кадра нет кнопки «несколько ковров»'
    m.screenshot(path=f'{OUT}/37_micro_frame_multi.png')
    rug('sp_e', multi=P); m.goto(u + 'm?k=split'); m.wait_for_selector('#pg, svg rect', timeout=15000)
    m.wait_for_function("[...document.querySelectorAll('#main img')].every(i=>i.complete&&i.naturalWidth>0)", timeout=15000)
    assert m.eval_on_selector_all('svg rect', 'e=>e.length') >= 2, 'на карточке нет рамок разрезания'
    m.screenshot(path=f'{OUT}/37_micro_split.png'); mc.close()
    assert not errs, errs
    # 4. фоновый проход сам находит развороты и не трогает уже решённые
    rug('sp_f')
    st = post(u + 'api/rugs_catalog', {'force': False})
    for _ in range(120):
        if not get(u + 'api/rugs_catalog').get('running'): break
        time.sleep(0.5)
    f = get(u + 'api/rugs/sp_f'); assert len(f.get('multi') or []) == 2, f'проход не нашёл разворот: {f.get("multi")}'
    assert get(u + 'api/rugs/sp_b').get('multi') == [], 'проход перезаписал решение человека'


@test
def t38_rect_view_v1170(pg, ctx):
    """v11.70: выпрямленный вид (T): картинка /rugrect/, рамка, нарисованная на ровном ковре, ложится на фото по 4 углам; непроверенные — бледно/скрыть (N)"""
    pg.goto(ctx['url']); pg.wait_for_timeout(2000)
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200); pg.evaluate("rugOpen('border_rep')"); pg.wait_for_timeout(1500)
    pg.evaluate("""(async()=>{R.rug.marks=[];R.rug.layout={quad:[[.06,.05],[.95,.09],[.92,.95],[.08,.91]],quad_status:'confirmed',bounds:[],rings:[],status:'auto',carpet:[.05,.05,.9,.9]};rugSaveSoon();await rugSaveNow();await rugRectify();})()""")
    pg.wait_for_timeout(800); pg.keyboard.press('t'); pg.wait_for_timeout(1500)
    assert '/rugrect/' in pg.evaluate("document.getElementById('rgImg').src"), 'T не включил выпрямленный ковёр'
    assert pg.evaluate("rugRectOn()"), 'выпрямленный режим не включён'
    i = pg.evaluate("(()=>{const r=document.getElementById('rgSvg').getBoundingClientRect();return [r.left,r.top,r.width,r.height]})()")
    x0, y0, w, h = i
    pg.mouse.move(x0 + w * .40, y0 + h * .40); pg.mouse.down(); pg.mouse.move(x0 + w * .55, y0 + h * .52, steps=5); pg.mouse.up(); pg.wait_for_timeout(800)
    pg.screenshot(path=f'{OUT}/rect_view.png')
    b = pg.evaluate("R.sel&&R.sel.box")
    assert b, 'рамка не создана в выпрямленном виде'
    # центр новой рамки на фото = центр нарисованного прямоугольника, пересчитанный по углам
    c = pg.evaluate("layToImg(R.rug.layout,.475,.46)")
    assert abs(b[0] + b[2] / 2 - c[0]) < .02 and abs(b[1] + b[3] / 2 - c[1]) < .02, f'рамка легла не туда: {b} центр должен быть {c}'
    d = pg.evaluate("rugPx(R.sel.box)")
    assert abs(d[0] / pg.evaluate("R.iw") - .40) < .03 and abs(d[1] / pg.evaluate("R.ih") - .40) < .03, f'на экране рамка не там, где нарисована: {d}'
    # непроверенные чужие: скрыть
    pg.evaluate("R.rug.marks.push({...R.sel,id:'zz',group:'zz',box:[.2,.2,.05,.05],status:'proposed',machine:1});rugSelect(R.rug.marks[0]);")
    pg.evaluate("rugPropSet('hide')"); n_hide = pg.evaluate("document.querySelectorAll('#rgSvg [data-m=\"zz\"]').length")
    pg.evaluate("rugPropSet('show')"); n_show = pg.evaluate("document.querySelectorAll('#rgSvg [data-m=\"zz\"]').length")
    assert n_hide == 0 and n_show == 1, f'скрытие непроверенных не работает: {n_hide} {n_show}'
    pg.evaluate("rugPropSet('faint')")
    pg.keyboard.press('t'); pg.wait_for_timeout(1200)
    assert '/rugimg/' in pg.evaluate("document.getElementById('rgImg').src"), 'T не вернул обычное фото'
    errors_ok(ctx, 'выпрямленный вид v11.70')


@test
def t39_rect_working_v1172(pg, ctx):
    """v11.72: выпрямленный ковёр — рабочая картинка: по подтверждённым углам редактор открывает /rugrect/ сам, миниатюра каталога — выпрямленная
    (?raw=1 — исходная), поиск повторов идёт по выпрямленному и отвечает в долях фото, «похожие ковры» по выпрямленным"""
    import urllib.request as U, json as _j, cv2, numpy as np
    pg.goto(ctx['url']); pg.wait_for_timeout(2000); pg.evaluate("localStorage.removeItem('cpsRect')")
    pg.click('#tab-rugs'); pg.wait_for_timeout(1200)
    Q = "[[.06,.05],[.95,.09],[.92,.95],[.08,.91]]"
    for rid in ('frieze_1', 'tall_1', 'persp_1'):
        pg.evaluate(f"(async()=>{{await rugOpen('{rid}');R.rug.layout={{quad:{Q},quad_status:'confirmed',bounds:[],rings:[],status:'auto',carpet:[.05,.05,.9,.9]}};rugSaveSoon();await rugSaveNow();}})()")
        pg.wait_for_timeout(900)
    pg.evaluate("rugOpen('frieze_1')"); pg.wait_for_timeout(1800)
    assert '/rugrect/' in pg.evaluate("document.getElementById('rgImg').src"), 'редактор не открыл выпрямленный ковёр по подтверждённым углам'
    pg.keyboard.press('t'); pg.wait_for_timeout(900)
    assert pg.evaluate("localStorage.getItem('cpsRect')") == '0', 'выбор «обычное фото» не запомнился'
    pg.evaluate("rugOpen('tall_1')"); pg.wait_for_timeout(1500)
    assert '/rugimg/' in pg.evaluate("document.getElementById('rgImg').src"), 'после T редактор всё равно открывает выпрямленный'
    pg.keyboard.press('t'); pg.wait_for_timeout(600)
    # миниатюры
    rd = lambda u: cv2.imdecode(np.frombuffer(U.urlopen(ctx['url'] + u).read(), np.uint8), 1)
    a, b = rd('rugthumb/persp_1.jpg'), rd('rugthumb/persp_1.jpg?raw=1')
    assert a is not None and b is not None and a.shape != b.shape or not np.array_equal(a, b), 'миниатюра не выпрямленная'
    # повторы на выпрямленном: рамка-образец находит хоть что-то, ответ — в долях исходного фото
    req = U.Request(ctx['url'] + 'api/rugs/frieze_1/repeats', data=_j.dumps({"box": [.3, .3, .1, .1]}).encode(), headers={'Content-Type': 'application/json'})
    j = _j.load(U.urlopen(req))
    j = j.get('result', j)
    assert 'cands' in j, f'поиск повторов упал: {str(j)[:200]}'
    assert all(0 <= c['box'][0] <= 1 and 0 <= c['box'][1] <= 1 for c in j['cands']), 'кандидаты не в долях фото'
    # похожие
    s = _j.load(U.urlopen(ctx['url'] + 'api/rugs/frieze_1/rect_similar?n=48'))
    ids = [x['id'] for x in s['items']]
    dbg = {i: (lambda r: [(r.get('layout') or {}).get('quad_status'), r.get('hidden'), r.get('dup_of'), r.get('split_into')])(_j.load(U.urlopen(ctx['url'] + 'api/rugs/' + i))) for i in ('tall_1', 'persp_1')}
    assert s['pool'] >= 2 and {'tall_1', 'persp_1'} <= set(ids), f'похожие по выпрямленным: {dbg} {ids}'
    pg.evaluate("rugOpen('frieze_1')"); pg.wait_for_timeout(1500)
    pg.click('#rgSim'); pg.wait_for_timeout(1500)
    assert pg.evaluate("document.querySelectorAll('#modal [data-simok]').length") >= 1, 'окно похожих ковров пустое'
    pg.screenshot(path=f'{OUT}/rect_similar.png')
    pg.click('#modal [data-simok="persp_1"]'); pg.wait_for_timeout(1000)
    assert pg.evaluate("((R.rug.same_ok||{})['persp_1']||{}).ok") is True, '«тот же» из окна похожих не записался'
    pg.click('#mClose')
    errors_ok(ctx, 'выпрямленный ковёр — рабочая картинка v11.72')


@test
def t40_rect_dups_m_v1173(pg, ctx):
    """v11.73: новые двойники по выпрямленным: копия ковра другого размера находится фоновым поиском, попадает в /m вид «rsame»
    выпрямленными (rect=true, /rectmid/), ответ «тот же» дважды (вес ≥2) ставит двойников обоим коврам"""
    import urllib.request as U, json as _j, cv2, numpy as np, shutil
    D = os.path.join(ctx['app'], 'data')
    src = cv2.imread(f'{D}/rug_img/frieze_1.jpg'); cp = cv2.resize(src, None, fx=.6, fy=.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{D}/rug_img/frieze_copy.jpg', cp)
    r = _j.load(open(f'{D}/rugs/frieze_1.json', encoding='utf-8'))
    if (r.get('layout') or {}).get('quad_status') != 'confirmed':                       # не зависим от порядка тестов (t39 ставит те же углы)
        r['layout'] = {'quad': [[.06, .05], [.95, .09], [.92, .95], [.08, .91]], 'quad_status': 'confirmed', 'bounds': [], 'rings': [], 'status': 'auto', 'carpet': [.05, .05, .9, .9]}
        _j.dump(r, open(f'{D}/rugs/frieze_1.json', 'w', encoding='utf-8'), ensure_ascii=False)
    r2 = {**r, 'id': 'frieze_copy', 'title': 'Frieze copy', 'w': cp.shape[1], 'h': cp.shape[0], 'marks': [], 'same': [], 'same_ok': {}, 'micro': {}}
    _j.dump(r2, open(f'{D}/rugs/frieze_copy.json', 'w', encoding='utf-8'), ensure_ascii=False)
    get = lambda path: _j.load(U.urlopen(ctx['url'] + path))
    def post(path, body):
        return _j.load(U.urlopen(U.Request(ctx['url'] + path, data=_j.dumps(body).encode(), headers={'Content-Type': 'application/json'})))
    assert (r.get('layout') or {}).get('quad_status') == 'confirmed', 'углы frieze_1 должны быть подтверждены (тест t39)'
    time.sleep(1.5)
    img = U.urlopen(ctx['url'] + 'rectmid/frieze_1.jpg').read(); assert img[:2] == b'\xff\xd8', '/rectmid не отдал картинку'
    post('api/rugs_rect_all', {}); post('api/rugs_rect_dups', {})
    for _ in range(120):
        st = get('api/rugs_rect_dups')
        if not st['running'] and st['pairs']: break
        time.sleep(0.5)
    assert not st['err'], f"поиск упал: {st['err']}"
    P = _j.load(open(f'{D}/rect_pairs.json')); k = 'frieze_1~frieze_copy'
    assert k in P and P[k]['n'] >= 60, f'копия не найдена по выпрямленным: {P.get(k)} всего пар {len(P)}'
    j = get('api/micro/next?n=12&k=rsame')
    ts = [t for t in j['tasks'] if {t['rid'], t['other']} == {'frieze_1', 'frieze_copy'}]
    assert ts and ts[0]['rect'] and ts[0]['inl'] >= 60 and ts[0]['type'] == 'same', f"пара не попала в очередь /m: {j['tasks'][:2]} left={j['left']}"
    assert j['left']['rsame'] >= 1
    # две телефонные карточки: в тесте один пользователь с весом 2 (админ без пароля) — решение применяется сразу
    t0 = ts[0]; post('api/micro/answer', {'type': 'same', 'rid': t0['rid'], 'other': t0['other'], 'a': 'same'})
    d = get('api/rugs/' + t0['rid']); assert (d.get('same_ok') or {}).get(t0['other'], {}).get('ok') is True, f"«тот же» не записался: {d.get('same_ok')}"
    j = get('api/micro/next?n=12&k=rsame'); assert not [t for t in j['tasks'] if {t['rid'], t['other']} == {'frieze_1', 'frieze_copy'}], 'отвеченная пара вернулась в очередь'
    pg.goto(ctx['url'] + 'm?k=rsame'); pg.wait_for_timeout(1500)
    assert pg.evaluate("[...document.querySelectorAll('#chips button')].some(b=>b.dataset.k==='rsame')"), 'нет кнопки «новые дубли» в /m'
    errors_ok(ctx, '/m новые дубли v11.73')


@test
def t41_mesh_field_v1181(pg, ctx):
    """v11.81: края к рамке (сетка) и граница поля: у ковра с подтверждёнными углами строятся кандидаты в границу «поле / кайма»,
    в /m вид «поле» показывает линии; выбор линии пишется в layout.field, по нему строится выровненный ковёр (/rugmesh/<id>.jpg?f=1)"""
    import urllib.request as U, json as _j, cv2, numpy as np
    D = os.path.join(ctx['app'], 'data')
    img = np.full((900, 700, 3), (60, 60, 60), np.uint8); cv2.rectangle(img, (100, 100), (600, 800), (30, 30, 160), -1)
    pts = [[180 + int(10 * np.sin(y / 60)), y] for y in range(180, 721)] + [[520 + int(10 * np.sin(y / 60)), y] for y in range(720, 179, -1)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], (170, 200, 220)); cv2.imwrite(f'{D}/rug_img/fieldrug.jpg', img)
    r = _j.load(open(f'{D}/rugs/frieze_1.json', encoding='utf-8'))
    r2 = {**r, 'id': 'fieldrug', 'title': 'Field rug', 'w': 700, 'h': 900, 'marks': [], 'same': [], 'same_ok': {}, 'micro': {}, 'frame': 'whole', 'frame_by': 'micro',
          'layout': {'quad': [[1 / 7, 1 / 9], [6 / 7, 1 / 9], [6 / 7, 8 / 9], [1 / 7, 8 / 9]], 'quad_status': 'confirmed', 'bounds': [], 'rings': [], 'status': 'auto', 'carpet': [.14, .11, .72, .78]}}
    for k in ('dup_of', 'hidden', 'split_into', 'split_from', 'multi', 'lib_img'): r2.pop(k, None)
    _j.dump(r2, open(f'{D}/rugs/fieldrug.json', 'w', encoding='utf-8'), ensure_ascii=False)
    get = lambda path: _j.load(U.urlopen(ctx['url'] + path))
    def post(path, body):
        return _j.load(U.urlopen(U.Request(ctx['url'] + path, data=_j.dumps(body).encode(), headers={'Content-Type': 'application/json'})))
    time.sleep(1.5)
    m = get('api/rugs/fieldrug/mesh')
    assert m['cands'], f'кандидатов поля нет: {m}'
    c0 = min(m['cands'], key=lambda c: abs(c['depth_pct'] - 16))
    assert abs(c0['depth_pct'] - 16) < 3, f"граница поля не там (ожидалось ≈16 %): {[c['depth_pct'] for c in m['cands']]}"
    a = U.urlopen(ctx['url'] + 'rugmesh/fieldrug.jpg').read(); assert a[:2] == b'\xff\xd8', '/rugmesh не отдал картинку'
    j = get('api/micro/next?n=12&k=field')
    ts = [t for t in j['tasks'] if t['rid'] == 'fieldrug']
    assert ts and ts[0]['type'] == 'field' and ts[0]['cands'], f"нет карточки «поле»: {[t['rid'] for t in j['tasks']]} left={j['left']}"
    pg.goto(ctx['url'] + 'm?k=field'); pg.wait_for_timeout(1500)
    for _ in range(12):
        if pg.evaluate("S.tasks[S.i]&&S.tasks[S.i].rid") == 'fieldrug': break
        pg.evaluate("S.i++;show()"); pg.wait_for_timeout(200)
    assert pg.query_selector('#f0') and pg.query_selector('#ph svg polyline'), 'карточка «поле» не нарисовала линии и кнопки'
    pg.screenshot(path=os.path.join(OUT, 't41_field.png'))
    k = m['cands'].index(c0); pg.click('#f%d' % k); pg.wait_for_timeout(1200)
    d = get('api/rugs/fieldrug'); F = (d.get('layout') or {}).get('field') or {}
    assert F.get('curves') and abs(F['depth_pct'] - c0['depth_pct']) < 0.01, f'выбор линии не записался: {F}'
    a = U.urlopen(ctx['url'] + 'rugmesh/fieldrug.jpg?f=1').read(); assert a[:2] == b'\xff\xd8', 'выровненный по полю ковёр не построен'
    j = get('api/micro/next?n=12&k=field'); assert not [t for t in j['tasks'] if t['rid'] == 'fieldrug'], 'отвеченный ковёр вернулся в очередь'
    # v11.84: части ковра (поле и ленты каймы) и перевод точки части в доли фото
    pm = get('api/rugs/fieldrug/parts'); assert set(pm['boxes']) == {'field', 'top', 'bot', 'left', 'right'}, pm
    for k in pm['boxes']:
        a = U.urlopen(ctx['url'] + 'rugpart/fieldrug/%s.jpg' % k).read(); assert a[:2] == b'\xff\xd8', f'часть {k} не отдана'
    fw, fh = pm['size']['field']; mp = get('api/rugs/fieldrug/part_map?part=field&b=0,0,%d,%d' % (fw, fh))
    bx = mp['box']; exp = [180 / 700, 180 / 900, 340 / 700, 540 / 900]
    assert all(abs(a - b) < 0.03 for a, b in zip(bx, exp)), f'поле в долях фото не там: {bx} ожидалось {exp}'
    assert pm['size']['left'][0] > pm['size']['left'][1] * 3, f"левая лента не повёрнута вдоль: {pm['size']['left']}"
    assert b'rugpart/fieldrug/field.jpg' in U.urlopen(ctx['url'] + 'rugparts/fieldrug').read(), 'страница частей'
    # v11.84: «нет нужной» возвращает ковёр — линию можно поставить самому; четыре стороны отдельно
    r3 = {**r2, 'id': 'fieldrug2', 'title': 'Field rug 2'}; cv2.imwrite(f'{D}/rug_img/fieldrug2.jpg', img)
    _j.dump(r3, open(f'{D}/rugs/fieldrug2.json', 'w', encoding='utf-8'), ensure_ascii=False); time.sleep(1.5); get('api/rugs/fieldrug2/mesh')
    post('api/micro/answer', {'type': 'field', 'rid': 'fieldrug2', 'a': 'none'})
    assert [t for t in get('api/micro/next?n=12&k=field')['tasks'] if t['rid'] == 'fieldrug2'], '«нет нужной» не вернул ковёр в очередь'
    pg.goto(ctx['url'] + 'm?k=field'); pg.wait_for_timeout(1500)
    for _ in range(12):
        if pg.evaluate("S.tasks[S.i]&&S.tasks[S.i].rid") == 'fieldrug2': break
        pg.evaluate("S.i++;show()"); pg.wait_for_timeout(200)
    pg.click('#fn'); pg.wait_for_selector('#ed', timeout=3000); pg.wait_for_timeout(600)
    ei = pg.query_selector('#ei').bounding_box(); h = pg.query_selector('#ed .eh[data-k="top"]').bounding_box()
    pg.mouse.move(h['x'] + h['width'] / 2, h['y'] + h['height'] / 2); pg.mouse.down(); pg.mouse.move(h['x'] + h['width'] / 2, ei['y'] + ei['height'] * 0.30, steps=5); pg.mouse.up()
    pg.screenshot(path=os.path.join(OUT, 't41_field_edit.png')); pg.click('#eo'); pg.wait_for_timeout(1200)
    F2 = (get('api/rugs/fieldrug2').get('layout') or {}).get('field') or {}
    assert F2.get('src') == 'hand' and abs(F2['curves']['top'][0] / F2['h'] - 0.30) < 0.03 and abs(F2['curves']['bot'][0] / F2['h'] - 0.30) > 0.05, f'линия вручную не записалась: {F2.get("src")} {F2.get("curves", {}).get("top", [0])[0]}'
    assert get('api/rugs/fieldrug2/parts')['size']['top'][1] > 200, 'части по ручной линии не построены'
    errors_ok(ctx, '/m поле v11.81')


@test
def t42_marks_near_fix_v1182(pg, ctx):
    """v11.82: карточка «элементы»: рамка, промахнувшаяся мимо элемента, перед показом сдвигается к нему (по образцу группы, с поворотом);
    ответ «⌖ рядом» сам находит элемент рядом и возвращает рамку на проверку; долгое нажатие на плитку — правка рамки пальцем, ✓ пишет новую рамку"""
    import cv2, numpy as np
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(60, 120, (800, 800, 3)).astype(np.uint8), (0, 0), 3)
    mot = np.zeros((60, 90, 3), np.uint8); mot[:] = (30, 30, 150); cv2.rectangle(mot, (10, 10), (50, 50), (220, 220, 240), -1); cv2.circle(mot, (70, 30), 12, (40, 200, 240), -1)
    def put(x, y, m): img[y:y + m.shape[0], x:x + m.shape[1]] = m
    put(100, 100, mot); put(500, 200, mot); put(300, 550, np.ascontiguousarray(np.rot90(mot))); put(560, 600, mot); put(150, 400, mot)
    cv2.imwrite(f'{D}/rug_img/mk_ref.jpg', img)
    f = lambda x, y, w, h: [x / 800, y / 800, w / 800, h / 800]
    def mk(i, box, st, **kw): return {"id": i, "box": box, "zone": "field", "motif": "", "motif_text": "", "scheme": "", "group": "gR", "status": st, **kw}
    r = {"id": "mk_ref", "title": "mk_ref", "source": "тест", "museum": "", "inv": "", "url": "", "license": "", "place": "", "date": "", "school": "", "composition": "",
         "technique": "", "size": "", "notes": "", "status": "new", "w": 800, "h": 800, "cat": {"src": "upload"}, "created": now, "updated": now,
         "marks": [mk('r0', f(100, 100, 90, 60), 'confirmed'),
                   mk('r1', f(536, 215, 90, 60), 'proposed', machine=1),          # промах на 40 % — должна сама встать на элемент (500,200)
                   mk('r2', f(318, 520, 60, 90), 'proposed', machine=1),          # повёрнутый элемент (300,550), промах
                   mk('r3', f(600, 600, 90, 60), 'proposed', machine=1, rf=1),    # для «⌖ рядом»: элемент (560,600); подгонка до показа уже была
                   mk('r4', f(150, 400, 90, 60), 'proposed', machine=1)]}         # для правки пальцем
    json.dump(r, open(f'{D}/rugs/mk_ref.json', 'w'), ensure_ascii=False)
    nx = get(u + 'api/micro/next?n=12&k=marks')
    tm = next((t for t in nx['tasks'] if t['rid'] == 'mk_ref'), None); assert tm, f"нет карточки: {[t['rid'] for t in nx['tasks']]}"
    iou = lambda p, s: (lambda ix, iy: ix * iy / (p[2] * p[3] + s[2] * s[3] - ix * iy))(max(0, min(p[0] + p[2], s[0] + s[2]) - max(p[0], s[0])), max(0, min(p[1] + p[3], s[1] + s[3]) - max(p[1], s[1])))
    B = {i['id']: i['box'] for i in tm['items']}
    assert iou(B['r1'], f(500, 200, 90, 60)) > 0.8, f"рамка r1 не подогнана: {B['r1']}"
    assert iou(B['r2'], f(300, 550, 60, 90)) > 0.8, f"рамка r2 (повёрнутый элемент) не подогнана: {B['r2']}"
    d = get(u + 'api/rugs/mk_ref'); m1 = next(m for m in d['marks'] if m['id'] == 'r1'); assert m1.get('box0') and m1.get('rf') == 1, f'подгонка не записана: {m1}'
    # «⌖ рядом» через API: r3 — элемент на 40 px левее
    post(u + 'api/micro/answer', {'type': 'marks', 'rid': 'mk_ref', 'g': 'gR', 'a': 'done', 'ok': [], 'bad': [], 'var': [], 'near': ['r3'], 'fix': {}})
    d = get(u + 'api/rugs/mk_ref'); m3 = next(m for m in d['marks'] if m['id'] == 'r3')
    assert m3['status'] == 'proposed' and iou(m3['box'], f(560, 600, 90, 60)) > 0.7 and m3.get('near_by'), f'«рядом» не нашёл элемент: {m3}'
    # телефон: долгое нажатие на плитку r4 → редактор → сдвиг → готово → ✓
    mc = pg.context.browser.new_context(viewport={'width': 390, 'height': 844}, has_touch=True); m = mc.new_page()
    errs = []; m.on('pageerror', lambda e: errs.append(str(e)))
    m.goto(u + 'm?k=marks'); m.wait_for_selector('.q', timeout=15000)
    for _ in range(12):
        if m.evaluate("S.tasks[S.i]&&S.tasks[S.i].rid") == 'mk_ref': break
        m.evaluate("S.i++;show()"); m.wait_for_timeout(200)
    tile = m.query_selector('#gr .tl[data-id="r4"]'); assert tile, 'нет плитки r4'
    bb = tile.bounding_box(); m.mouse.move(bb['x'] + bb['width'] / 2, bb['y'] + bb['height'] / 2); m.mouse.down(); m.wait_for_timeout(700); m.mouse.up()
    m.wait_for_selector('#ed', timeout=3000); m.wait_for_timeout(500)
    eb = m.query_selector('#eb').bounding_box(); ei = m.query_selector('#ei').bounding_box()
    m.mouse.move(eb['x'] + eb['width'] / 2, eb['y'] + eb['height'] / 2); m.mouse.down(); m.mouse.move(eb['x'] + eb['width'] / 2 + ei['width'] * 0.1, eb['y'] + eb['height'] / 2, steps=5); m.mouse.up()
    m.screenshot(path=os.path.join(OUT, 't42_edit.png')); m.click('#eo'); m.wait_for_timeout(300)
    assert 'f' in (m.get_attribute('#gr .tl[data-id="r4"]', 'class') or '').split(), 'после правки плитка не отмечена ✎'
    m.click('#a1'); m.wait_for_timeout(1200)
    d = get(u + 'api/rugs/mk_ref'); m4 = next(x for x in d['marks'] if x['id'] == 'r4')
    assert m4.get('fixed_by') and m4['status'] == 'confirmed' and m4['box'][0] > f(150, 400, 90, 60)[0] + 0.02, f'правка рамки не записалась: {m4}'
    assert not errs, f'ошибки JS: {errs[:2]}'
    mc.close()


@test
def t43_rug_names_v1185(pg, ctx):
    """v11.85: название ковра = тип (композиция) + музей/книга; подпись источника остаётся в title. Q/G-двойники словаря различаются
    по написанию; «X group» — не тип; школа ставится по типу; пара двойников с разными подтверждёнными типами не идёт в /m."""
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np
    img = np.full((300, 200, 3), 120, np.uint8)
    def mk(rid, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', img)
        r = {"id": rid, "title": "", "source": "тест", "museum": "", "inv": "", "school": "", "composition": "", "status": "new", "w": 200, "h": 300,
             "marks": [], "created": now, "updated": now, "cat": {"src": "azcarpetmuseum"}}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    mk("nm_q", title="Ковёр «Qollu Çiçi» Губа")
    mk("nm_g", title="Göllü çiçi xalçası")
    mk("nm_grp", title="Azerbaijani carpet \"Unknown\" (Barda group) of Karabakh")
    mk("nm_ok", title="Pirəbədil carpet, Quba", composition="pirebedil", museum="Национальный музей азербайджанского ковра", inv="1234")
    mk("nm_book", title="Керимов, т. I, табл. 202, фото 1 (стр. скана 324)", composition="zeyve", cat={"src": "kerimov_vol1"})
    mk("nm_w1", title="twin a", composition="pirebedil", same=[{"id": "nm_w2", "score": 40}])
    mk("nm_w2", title="twin b", composition="zeyve", same=[{"id": "nm_w1", "score": 40}])
    mk("nm_w3", title="twin c", composition="pirebedil", same=[{"id": "nm_w4", "score": 40}])
    mk("nm_w4", title="twin d", composition="pirebedil", same=[{"id": "nm_w3", "score": 40}])
    by = {r["id"]: r for r in get(u + 'api/rugs')["rugs"]}
    q, g, grp, ok, bk = by["nm_q"], by["nm_g"], by["nm_grp"], by["nm_ok"], by["nm_book"]
    assert q.get("comp_sug") == "qollu-chichi" and q["nm_q"] == 1 and q["nm"].lower() == "qollu çiçi", q
    assert g.get("comp_sug") == "golluchichi", g
    assert not grp.get("comp_sug"), ("«Barda group» — группа, не тип", grp.get("comp_sug"))
    assert ok["nm"] == "Pirəbədil" and ok["nm_q"] == 0 and ok["nm_sub"] == "Национальный музей азербайджанского ковра, инв. 1234", ok
    assert ok["title"] == "Pirəbədil carpet, Quba"                       # подпись источника не тронута
    assert bk["nm_sub"] == "Керимов, т. I, табл. 202", bk
    assert ok.get("sch_by_type") == "guba-shirvan"
    n = post(u + 'api/rugs_school_from_type', {})["n"]; assert n >= 2, n
    r = get(u + 'api/rugs/nm_ok'); assert r["school"] == "guba-shirvan" and "по типу" in r.get("school_by", ""), r.get("school")
    assert r["_nm"]["nm"] == "Pirəbədil" and r["_composition"]["school"] == "guba-shirvan"
    # тип, поставленный в паспорте, заполняет пустую школу
    r = post(u + 'api/rugs/nm_g', {"composition": "berde", "school": ""}); assert r["school"] == "karabakh", r["school"]
    # /m: пара двойников с разными подтверждёнными типами не спрашивается
    j = get(u + 'api/micro/next?k=same&n=50')
    pairs = [{t["rid"], t["other"]} for t in j.get("tasks", []) if t.get("type") == "same"]
    assert {"nm_w1", "nm_w2"} not in pairs, pairs
    assert {"nm_w3", "nm_w4"} in pairs, ("контроль: пара одного типа спрашивается", pairs)
    # студия: в плитке название — тип, подпись — во всплывающей подсказке
    pg.goto(u); pg.wait_for_timeout(1500); pg.click('#tab-rugs'); pg.wait_for_timeout(2500)
    pg.evaluate("()=>{R.fuse='';R.q='pirəbədil carpet';rugList();}"); pg.wait_for_timeout(600)
    t = pg.inner_text('.rgcard[data-id="nm_ok"] .t'); assert t.startswith("Pirəbədil") and "инв. 1234" in t, t
    errors_ok(ctx, 'названия ковров')


@test
def t44_twins_fast_v1186(pg, ctx):
    """v11.86: двойники — кандидаты по размеру и музейному номеру, совмещение фото, порог по ответам людей, авторешение, цепочка,
    карточка /m «двойники ковра» (отмеченные — тот же, остальные — другой)"""
    import cv2, numpy as np, sys as _s
    _s.path.insert(0, CPS); import twins as TW
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    def rug_img(seed):
        rng = np.random.default_rng(seed); a = cv2.GaussianBlur(rng.integers(0, 255, (60, 40, 3)).astype(np.uint8), (0, 0), 0.6)
        a = cv2.resize(a, (400, 600), interpolation=cv2.INTER_NEAREST)
        for _ in range(40):
            c = tuple(int(v) for v in rng.integers(0, 255, 3)); x, y = int(rng.integers(20, 380)), int(rng.integers(20, 580))
            cv2.drawMarker(a, (x, y), c, int(rng.integers(0, 6)), int(rng.integers(15, 50)), 3)
        b = np.full((700, 520, 3), 230, np.uint8); b[50:650, 60:460] = a; return b
    def warp(a, k=0.06, gain=1.1):
        h, w = a.shape[:2]; src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = np.float32([[w * k, h * k * .5], [w * (1 - k * .3), 0], [w, h * (1 - k)], [w * k * .4, h]])
        return np.clip(cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(200, 200, 200)) * gain, 0, 255).astype(np.uint8)
    def mk(rid, img, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', img)
        r = {"id": rid, "title": rid, "source": "тест", "museum": "", "inv": "", "school": "", "composition": "", "status": "new", "w": img.shape[1], "h": img.shape[0],
             "marks": [], "created": now, "updated": now, "cat": {"src": kw.pop("src", "azcarpetmuseum")}}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    # авторешение должно сработать на паре, которая не контрольная
    a1, a2 = next((f"tw_a{i}", f"tw_b{i}") for i in range(100) if not TW.is_ctl(TW.key(f"tw_a{i}", f"tw_b{i}")))
    P1 = rug_img(11)
    mk(a1, P1, size="150×230 см", src="azcarpetmuseum"); mk(a2, warp(P1), size="151×229 см", src="wikimedia")
    mk("tw_c", cv2.resize(warp(P1, 0.03, 0.95), None, fx=0.8, fy=0.8), same=[{"id": a2, "score": 0}], same_ok={a2: {"ok": True, "by": "dolgikh", "at": now}})
    js = json.load(open(f'{D}/rugs/{a2}.json')); js["same"] = [{"id": "tw_c", "score": 0}]; js["same_ok"] = {"tw_c": {"ok": True, "by": "dolgikh", "at": now}}
    json.dump(js, open(f'{D}/rugs/{a2}.json', 'w'), ensure_ascii=False)
    mk("tw_x", rug_img(99), size="150×231 см", src="book:intangible_2849")      # тот же размер, другой ковёр
    # ответы людей для порога: 2 «тот же» (фото одного ковра), 1 «разные»
    for i, sd in enumerate((21, 22)):
        P = rug_img(sd); mk(f"tw_h{i}a", P, same=[{"id": f"tw_h{i}b"}], same_ok={f"tw_h{i}b": {"ok": True, "by": "dolgikh", "at": now}})
        mk(f"tw_h{i}b", warp(P, 0.05), same=[{"id": f"tw_h{i}a"}], same_ok={f"tw_h{i}a": {"ok": True, "by": "dolgikh", "at": now}})
    mk("tw_hda", rug_img(31), same=[{"id": "tw_hdb"}], same_ok={"tw_hdb": {"ok": False, "by": "dolgikh", "at": now}})
    mk("tw_hdb", rug_img(32), same=[{"id": "tw_hda"}], same_ok={"tw_hda": {"ok": False, "by": "dolgikh", "at": now}})
    # один музей и номер — кандидат, хотя фото разные
    mk("tw_m1", rug_img(41), museum="Национальный музей ковра", inv="№ 777"); mk("tw_m2", rug_img(42), museum="Национальный музей ковра", inv="777", src="book:x")
    post(u + 'api/twins_run', {})
    for _ in range(240):
        j = get(u + 'api/twins')
        if not j["state"]["running"] and j["last"].get("t"): break
        time.sleep(1)
    else: raise AssertionError(("проход двойников не закончился", j["state"]))
    L = j["last"]; assert L["thr"], ("порог не найден", L.get("suspects"), L)
    assert j["by_why"]["size"] >= 1 and j["by_why"]["inv"] >= 1 and j["by_why"]["calib"] >= 3, j["by_why"]
    ra = get(u + f'api/rugs/{a1}'); ok = (ra.get("same_ok") or {}).get(a2)
    assert ok and ok["ok"] and ok["by"].startswith("машина"), ("авторешение", ok, L)
    assert ((ra.get("same_ok") or {}).get("tw_c") or {}).get("by", "").startswith("по цепочке"), ("цепочка a1 = a2 = c", ra.get("same_ok"))
    assert "tw_x" not in (ra.get("same_ok") or {}), "другой ковёр того же размера не должен решиться сам"
    # /m: карточка «двойники ковра» с парой по музейному номеру
    t = [t for t in get(u + 'api/micro/next?k=tw&n=12')["tasks"] if t["type"] == "tw" and {t["rid"]} | {x["id"] for x in t["items"]} >= {"tw_m1", "tw_m2"}]
    assert t, "нет карточки с парой по музейному номеру"
    t = t[0]; r = post(u + 'api/micro/answer', {"type": "tw", "rid": t["rid"], "a": "done", "ids": [x["id"] for x in t["items"]], "same": []})
    assert r.get("applied"), r
    assert (get(u + 'api/rugs/tw_m1').get("same_ok") or {}).get("tw_m2", {}).get("ok") is False
    # v11.88: ≈ очень похож, но другой ковёр → связь similar; школа похожего — подсказкой
    mk("tw_s1", rug_img(61), museum="Музей Б", inv="55"); mk("tw_s2", rug_img(62), museum="Музей Б", inv="55", src="book:z", school="karabakh")
    t0 = get(u + 'api/twins')["last"]["t"]; post(u + 'api/twins_run', {})
    for _ in range(120):
        j = get(u + 'api/twins')
        if not j["state"]["running"] and j["last"].get("t", 0) > t0: break
        time.sleep(1)
    t = [t for t in get(u + 'api/micro/next?k=tw&n=12')["tasks"] if t["type"] == "tw" and {t["rid"]} | {x["id"] for x in t["items"]} >= {"tw_s1", "tw_s2"}]
    assert t, "нет карточки tw_s1/tw_s2"
    t = t[0]; o = "tw_s2" if t["rid"] == "tw_s1" else "tw_s1"
    post(u + 'api/micro/answer', {"type": "tw", "rid": t["rid"], "a": "done", "ids": [x["id"] for x in t["items"]], "same": [], "similar": [o]})
    v = (get(u + 'api/rugs/tw_s1').get("same_ok") or {}).get("tw_s2") or {}
    assert v.get("similar") and v.get("ok") is False, v
    sg = get(u + 'api/rugs/tw_s1').get("school_suggest") or {}
    assert sg.get("school") == "karabakh" and "похож" in sg.get("why", ""), sg
    # телефон: карточка показывается, отметка плитки меняет кнопку
    mk("tw_n1", rug_img(51), museum="Музей", inv="1"); mk("tw_n2", warp(rug_img(51)), museum="Музей", inv="1", src="book:y")
    p2 = pg
    p2.goto(u + 'm?k=tw'); p2.wait_for_timeout(2500)
    assert p2.locator('.grid.tw .tl').count() >= 1, p2.inner_text('main')[:300]
    p2.locator('.grid.tw .tl').first.click(); p2.wait_for_timeout(200)
    assert 'готово' in p2.inner_text('#a1'), p2.inner_text('#a1')
    p2.locator('.grid.tw .tl').first.click(); p2.wait_for_timeout(200)          # второй тап — ≈ похож
    assert p2.locator('.grid.tw .tl.v').count() == 1 and 'похож' in p2.inner_text('#a1'), p2.inner_text('#a1')
    p2.screenshot(path=os.path.join(OUT, 't44_m_tw_sim.png'))
    p2.locator('.grid.tw .tl').first.click(); p2.wait_for_timeout(200)          # третий — снять
    assert p2.locator('.grid.tw .tl.v, .grid.tw .tl.sel').count() == 0
    p2.locator('.grid.tw .tl').first.click(); p2.wait_for_timeout(200)
    p2.screenshot(path=os.path.join(OUT, 't44_m_tw.png'))
    wc = pg.context.browser.new_context(viewport={'width': 1440, 'height': 800}); pw = wc.new_page(); pw.goto(u + 'm?k=tw'); pw.wait_for_timeout(2500)   # v11.89: широкий экран — ковёр слева, кандидаты справа
    bl = pw.locator('.twl-l').bounding_box(); bg = pw.locator('.grid.tw').bounding_box()
    assert bl and bg and bg['x'] > bl['x'] + bl['width'] - 5 and bg['width'] > bl['width'] * 1.6, (bl, bg)
    pw.screenshot(path=os.path.join(OUT, 't44_m_tw_wide.png'))
    pw.click('#hz'); pw.wait_for_timeout(600); assert pw.locator('.grid.tw.hz').count() == 1
    pw.screenshot(path=os.path.join(OUT, 't44_m_tw_wide_hz.png')); wc.close()
    pg.goto(u); pg.wait_for_timeout(1500); pg.click('#tab-rugs'); pg.wait_for_timeout(3000)
    rg_tools_open(pg); assert 'двойники' in pg.inner_text('#rgKpi'), pg.inner_text('#rgKpi')
    errors_ok(ctx, 'двойники')


@test
def t45_schools_7_v1190(pg, ctx):
    """v11.90: 7 школ — главный уровень, 4 типа Керимова — выше. Губа-Ширван / Гянджа-Газах уточняются по подписи, типу, двойнику;
    «Kuba Shirvan» — неясно; «Chelaberd Kazak» — Карабах; фильтр по группе включает её школы; Губа и Губа-Ширван не расходятся."""
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np
    img = np.full((300, 200, 3), 120, np.uint8)
    def mk(rid, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', img)
        r = {"id": rid, "title": "", "source": "тест", "museum": "", "inv": "", "school": "", "composition": "", "status": "new", "w": 200, "h": 300,
             "marks": [], "created": now, "updated": now, "cat": {"src": "azerbaijanrugs"}}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    mk("sc_kuba", title="kuba medallion rug 12", school="guba-shirvan")
    mk("sc_ks", title="kuba shirvan medallion 3", school="guba-shirvan")
    mk("sc_baku", title="Qollu chichi carpet, Baku group of Azerbaijani carpets", school="guba-shirvan")
    mk("sc_kaz", title="Borchaly rug", school="ganja-gazakh")
    mk("sc_tw1", title="unknown", school="guba-shirvan", same_ok={"sc_kuba": {"ok": True, "by": "/m: t", "at": now}})
    mk("sc_kar", title="Karabakh rug, Guba-Shirvan per dealer", school="karabakh")
    st = post(u + 'api/school_refine', {})
    by = {k: get(u + 'api/rugs/' + k) for k in ("sc_kuba", "sc_ks", "sc_baku", "sc_kaz", "sc_tw1", "sc_kar")}
    assert by["sc_kuba"]["school"] == "guba" and by["sc_kuba"]["school_prev"] == "guba-shirvan", by["sc_kuba"]["school"]
    assert by["sc_ks"]["school"] == "guba-shirvan", ("«Kuba Shirvan» — неясно, не уточнять", by["sc_ks"]["school"])
    assert by["sc_baku"]["school"] == "baku", by["sc_baku"]["school"]
    assert by["sc_kaz"]["school"] == "gazakh", by["sc_kaz"]["school"]
    assert by["sc_tw1"]["school"] == "guba" and "двойник" in by["sc_tw1"]["school_by"], by["sc_tw1"]
    assert by["sc_kar"]["school"] == "karabakh"                                   # другая группа не трогается
    assert st.get("text", 0) >= 3 and st.get("twin", 0) >= 1, st
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import rugs as RG
    assert RG.school_guess("Chelaberd Kazak")[0] == "karabakh" and RG.school_fine("Chelaberd Kazak")[0] == "karabakh"
    assert RG.school_compat("guba", "guba-shirvan") and not RG.school_compat("guba", "shirvan") and not RG.school_compat("karabakh", "guba-shirvan")
    pg.goto(u); pg.wait_for_timeout(1500); pg.click('#tab-rugs'); pg.wait_for_timeout(2500)
    pg.evaluate("()=>{R.fuse='';R.fschs=new Set(['guba-shirvan']);rugList();}"); pg.wait_for_timeout(600)
    ids = pg.evaluate("()=>[...document.querySelectorAll('.rgcard')].map(e=>e.dataset.id)")
    assert "sc_kuba" in ids and "sc_baku" in ids and "sc_ks" in ids and "sc_kar" not in ids, ids[:20]
    pg.evaluate("()=>{R.fschs=new Set(['guba-shirvan!']);rugList();}"); pg.wait_for_timeout(600)
    ids = pg.evaluate("()=>[...document.querySelectorAll('.rgcard')].map(e=>e.dataset.id)")
    assert "sc_ks" in ids and "sc_kuba" not in ids, ids[:20]
    pg.evaluate("()=>{R.fschs=new Set();rugList();}"); pg.wait_for_timeout(400)
    pg.evaluate("rugOpen('sc_kuba')"); pg.wait_for_timeout(1500); pg.click('#rgPassBtn'); pg.wait_for_timeout(600)
    html = pg.inner_html('#modal')
    assert 'optgroup label="Губа-Ширван (тип Керимова)"' in html and 'было: Губа-Ширван' in html, 'паспорт: школы группами, видно прежнее значение'
    pg.screenshot(path=os.path.join(OUT, 't45_schools.png'))
    errors_ok(ctx, 'школы 7')


@test
def t46_frame_auto_v1191(pg, ctx):
    """v11.91: «целый ковёр» — очевидные решает машина (вероятность ≥ auto_thr): в карточке «кадр» их нет, кроме контрольных (1 из 10),
    они идут дальше в «углы»; ответ человека пишет вероятность машины (pw) для подсчёта контрольных."""
    u = ctx['url']; D = ctx['app'] + '/data'; MP = ctx['app'] + '/models/frame_v1.json'; now = int(time.time())
    old = open(MP).read(); M = json.loads(old)
    leaf = lambda v: {"l": [-1], "r": [-1], "f": [-2], "th": [-2.0], "v": [v]}
    split = {"l": [1, -1, -1], "r": [2, -1, -1], "f": [0, -2, -2], "th": [0.5, -2.0, -2.0], "v": [0, -50.0, 50.0]}   # conf > 0.5 → целый
    json.dump({**M, "auto_thr": 0.95, "trained": "2000-01-01 00:00", "init": 0.0, "lr": 1.0, "trees": [split]}, open(MP, 'w'))
    try:
        import cv2, numpy as np, hashlib
        img = np.full((900, 600, 3), 120, np.uint8)
        ids = []
        for i in range(40):
            rid = f"fa_{i}"; ids.append(rid)
            cv2.imwrite(f'{D}/rug_img/{rid}.jpg', img)
            r = {"id": rid, "title": "rug", "source": "тест", "school": "", "composition": "", "status": "new", "w": 600, "h": 900, "marks": [],
                 "created": now, "updated": now, "cat": {"src": "azerbaijanrugs"},
                 "auto": {"frame": "whole", "quad": [[.05, .05], [.95, .05], [.95, .95], [.05, .95]], "conf": 0.9 if i < 30 else 0.1, "area": .8, "ver": 99, "t": now}}
            json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
        ctl = {x for x in ids[:30] if int(hashlib.md5(("frctl:" + x).encode()).hexdigest(), 16) % 10 == 0}
        time.sleep(31)                                                           # кэш времени модели — 30 с
        j = get(u + 'api/micro/next?k=frame&n=12')
        got = {t["rid"] for t in j["tasks"] if t["type"] == "frame" and t["rid"].startswith("fa_")}
        auto = set(ids[:30]) - ctl
        assert not (got & auto), ("очевидные «целый» не должны идти в кадр", sorted(got & auto))
        assert got & set(ids[30:]), ("сомнительные — в кадр", sorted(got))
        assert j["left"]["frame_auto"] >= len(auto) and j["left"]["frame_auto_on"], j["left"]
        assert all(t.get("ctl") for t in j["tasks"] if t.get("rid") in ctl)
        q = get(u + 'api/micro/next?k=quad&n=12')
        assert q["left"]["quad"] >= len(auto), ("решённые машиной идут в «углы»", q["left"])
        st = get(u + 'api/micro/stats')["frame_model"]["auto"]; assert st["thr"] == 0.95 and st["on"], st
        if ctl:
            c = sorted(ctl)[0]; post(u + 'api/micro/answer', {"type": "frame", "rid": c, "a": "whole"})
            rec = [json.loads(l) for l in open(f'{D}/micro.jsonl', encoding='utf-8') if c in l][-1]; assert rec.get("pw", 0) >= 0.95, rec
        pg.goto(u + 'm?k=frame'); pg.wait_for_timeout(2500)
        assert 'машина решила сама' in pg.inner_html('main'), pg.inner_html('main')[:500]
    finally:
        open(MP, 'w').write(old)
    errors_ok(ctx, 'кадр авто')


@test
def t47_corner_upload_v1194(pg, ctx):
    """v11.94: «Загрузить» → «Веса сети углов»: без одного из файлов — понятная ошибка; плохие веса не подключаются и не ломают сервер."""
    import urllib.request
    u = ctx['url']
    def up(files):
        bd = 'xxBOUNDxx'; body = b''
        for name, fn, data in files:
            body += f'--{bd}\r\nContent-Disposition: form-data; name="{name}"; filename="{fn}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode() + data + b'\r\n'
        body += f'--{bd}--\r\n'.encode()
        rq = urllib.request.Request(u + 'api/corner_refine/upload', data=body, headers={'Content-Type': 'multipart/form-data; boundary=' + bd}, method='POST')
        try:
            with urllib.request.urlopen(rq) as r: return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
    c, j = up([('meta', 'corner_refine_v4.json', b'{"ver": 4}')])
    assert c == 400 and 'оба' in j['error'], (c, j)
    c, j = up([('pt', 'corner_refine_v4.pt', b'not a net'), ('meta', 'corner_refine_v4.json', b'{"ver": 4, "W": 128}')])
    assert c == 400 and j.get('error'), (c, j)
    c, j = up([('pt', 'corner_refine_v3.pt', b'x'), ('meta', 'corner_refine_v3.json', b'{"ver": 3}')])
    assert c == 400, (c, j)
    assert not os.path.exists(ctx['app'] + '/data/corner_refine_v4.pt')
    pg.goto(u); pg.wait_for_timeout(1500); pg.click('#tab-rugs'); pg.wait_for_timeout(1500)
    assert pg.locator('#rgCR').count() == 1 and 'Веса сети углов' in pg.content()
    errors_ok(ctx, 'веса углов')


@test
def t48_one_rug_v1196(pg, ctx):
    """v11.96: один ковёр = снимки, связанные «тот же». В ленте — снимки ковра; обложка; «не этот ковёр» (связи → «другой», взятые поля снимаются);
    «в другой ковёр» (переносит связь); плитка каталога показывает обложку и число снимков."""
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np
    def mk(rid, col, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', np.full((300, 200, 3), col, np.uint8))
        r = {"id": rid, "title": rid, "source": "тест", "museum": "", "inv": "", "school": "", "composition": "", "status": "new", "w": 200, "h": 300,
             "marks": [], "created": now, "updated": now, "cat": {"src": "azerbaijanrugs"}, "frame": "whole", "frame_by": "hand"}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    ok = lambda o: {"ok": True, "by": "тест", "at": now}
    mk("og_a", 60, same=[{"id": "og_b"}, {"id": "og_c"}], same_ok={"og_b": ok(0), "og_c": ok(0)}, school="karabakh")
    mk("og_b", 90, same=[{"id": "og_a"}], same_ok={"og_a": ok(0)}, school="karabakh", from_twin={"school": "og_a"})
    mk("og_c", 120, same=[{"id": "og_a"}], same_ok={"og_a": ok(0)})
    mk("og_x", 150, same=[{"id": "og_c"}], same_ok={"og_c": {"ok": False, "similar": True, "by": "тест", "at": now}})
    j = get(u + 'api/rugs/og_b'); assert {x["id"] for x in j["group"]} == {"og_a", "og_b", "og_c"}, j["group"]
    j = post(u + 'api/rugs/og_c/group', {"act": "cover"}); assert any(x["cover"] and x["id"] == "og_c" for x in j["group"])
    j = post(u + 'api/rugs/og_b/group', {"act": "out"})
    assert [x["id"] for x in j["group"]] == ["og_b"], j["group"]
    assert j["same_ok"]["og_a"]["ok"] is False and j["school"] == "" and not j.get("from_twin"), (j["same_ok"], j["school"], j.get("from_twin"))
    assert {x["id"] for x in get(u + 'api/rugs/og_a')["group"]} == {"og_a", "og_c"}
    def bad(path, b):
        try: post(u + path, b); return False
        except urllib.error.HTTPError as e: return e.code == 400
    assert bad('api/rugs/og_b/group', {"act": "out"})
    j = post(u + 'api/rugs/og_c/group', {"act": "move", "to": "og_x"})
    assert {x["id"] for x in j["group"]} == {"og_c", "og_x"} and j["same_ok"]["og_a"]["ok"] is False and not j.get("cover"), (j["group"], j["same_ok"])
    assert bad('api/rugs/og_c/group', {"act": "move", "to": "og_x"})
    # интерфейс: лента ковра и кнопки; перенос через окно
    post(u + 'api/rugs/og_c/group', {"act": "move", "to": "og_a"})
    pg.goto(u); pg.wait_for_timeout(1500); pg.click('#tab-rugs'); pg.wait_for_timeout(2500)
    pg.evaluate("rugOpen('og_a')"); pg.wait_for_timeout(1500)
    st = pg.inner_text('#rgStrip'); assert 'этот ковёр · 2 снимка' in st and '★ обложка' in st and 'не этот ковёр' in st, st
    pg.click('#rgGMove'); pg.wait_for_timeout(400); pg.fill('#mvQ', 'og_x'); pg.wait_for_timeout(300)
    pg.click('.rgmv[data-to="og_x"]'); pg.click('#mvGo'); pg.wait_for_timeout(1800)
    assert {x["id"] for x in get(u + 'api/rugs/og_a')["group"]} == {"og_a", "og_x"}                 # переносится снимок, og_c остаётся сам по себе
    assert [x["id"] for x in get(u + 'api/rugs/og_c')["group"]] == ["og_c"]
    pg.screenshot(path=os.path.join(OUT, 't48_one_rug.png'))
    errors_ok(ctx, 'один ковёр')


@test
def t49_versions_v1197(pg, ctx):
    """v11.97: версии школы / типа / техники по снимкам ковра; главное — по весу источника; спор; правка человека — прежнее в историю;
    техники и «Карабах» уходят из «типа ковра» (один раз) и не предлагаются как тип."""
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np, sys
    def mk(rid, src, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', np.full((300, 200, 3), 100, np.uint8))
        r = {"id": rid, "title": rid, "source": "тест", "museum": "", "inv": "", "school": "", "composition": "", "status": "new", "w": 200, "h": 300,
             "marks": [], "created": now, "updated": now, "cat": {"src": src}}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    ok = {"ok": True, "by": "тест", "at": now}
    mk("vr_k", "kerimov_vol2", school="karabakh", same=[{"id": "vr_d"}, {"id": "vr_m"}], same_ok={"vr_d": ok, "vr_m": ok})
    mk("vr_d", "azerbaijanrugs", school="guba", same=[{"id": "vr_k"}], same_ok={"vr_k": ok})
    mk("vr_m", "met", school="guba-shirvan", same=[{"id": "vr_k"}], same_ok={"vr_k": ok}, composition="sumax")
    v = get(u + 'api/rugs/vr_d')["versions"]["school"]
    assert v["main"] == "karabakh" and v["rank"] == 1 and not v["dispute"] and v["differ"], v
    post(u + 'api/rugs/vr_d', {"school": "shirvan"})                            # человек решил
    j = get(u + 'api/rugs/vr_d'); v = j["versions"]["school"]
    assert v["main"] == "shirvan" and v["rank"] == 0, v
    assert j["hist"]["school"][-1]["v"] == "guba" and any(x["how"] == "old" and x["v"] == "guba" for x in v["items"]), (j.get("hist"), v)
    post(u + 'api/rugs/vr_k', {"school": "ganja"})                              # второй человек с другой школой — спор
    v = get(u + 'api/rugs/vr_k')["versions"]["school"]; assert v["dispute"] and v["main"] is None, v
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import rugs as RG
    assert "sumax" in RG.COMP_MOVE and "talish" not in RG.COMP_MOVE
    pg.goto(u); pg.wait_for_timeout(1500); pg.click('#tab-rugs'); pg.wait_for_timeout(2000)
    pg.evaluate("rugOpen('vr_k')"); pg.wait_for_timeout(1500); pg.click('#rgPassBtn'); pg.wait_for_timeout(600)
    t = pg.inner_text('#modal'); assert 'Версии по источникам' in t and 'спор' in t, t[:600]
    pg.screenshot(path=os.path.join(OUT, 't49_versions.png'))
    errors_ok(ctx, 'версии')


@test
def t50_open_palette_v1198(pg, ctx):
    """v11.98: палитра ковра полоской 1024×1 (цвета по убыванию площади), список ковров, архив; ключ для программ — только GET /api/open."""
    import zipfile, io as _io
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np
    img = np.zeros((600, 400, 3), np.uint8); img[:] = (200, 120, 40)            # голубой (BGR) фон ≈ 75 %
    img[150:450, 100:300] = (30, 30, 180)                                        # красный медальон ≈ 25 %…
    img[280:320, 180:220] = (40, 200, 230)                                       # …с жёлтой серединой ≈ 0,7 %
    cv2.imwrite(f'{D}/rug_img/pal_a.jpg', img)
    json.dump({"id": "pal_a", "title": "тест палитры", "source": "тест", "school": "karabakh", "composition": "", "status": "new", "w": 400, "h": 600,
               "marks": [], "created": now, "updated": now, "cat": {"src": "kerimov_vol2"}, "frame": "whole", "frame_by": "hand",
               "layout": {"quad": [[0, 0], [1, 0], [1, 1], [0, 1]], "quad_status": "confirmed"}}, open(f'{D}/rugs/pal_a.json', 'w'), ensure_ascii=False)
    j = get(u + 'api/open/palette/pal_a.json')
    hx = [c["hex"] for c in j["colors"]]; sh = [c["share"] for c in j["colors"]]
    assert len(hx) >= 3 and sh == sorted(sh, reverse=True) and sum(c["px"] for c in j["colors"]) == 1024, j
    r0, g0, b0 = j["colors"][0]["rgb"]; assert b0 > 150 and r0 < 80 and 0.7 < sh[0] < 0.8, j["colors"][0]
    assert j["how"] == "углы подтверждены"
    b = urllib.request.urlopen(u + 'api/open/palette/pal_a.png').read()
    a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR); assert a.shape == (1, 1024, 3), a.shape
    a = cv2.imdecode(np.frombuffer(urllib.request.urlopen(u + 'api/open/palette/pal_a.png?w=512&h=8&blend=6').read(), np.uint8), cv2.IMREAD_COLOR)
    assert a.shape == (8, 512, 3)
    L = get(u + 'api/open/rugs?school=karabakh&quad=1'); assert any(x["id"] == "pal_a" for x in L["rugs"]), L
    z = zipfile.ZipFile(_io.BytesIO(urllib.request.urlopen(u + 'api/open/palettes.zip?school=karabakh&quad=1').read()))
    assert "pal_a.png" in z.namelist() and "pal_a" in z.read("index.csv").decode("utf-8-sig")
    assert 'Палитра ковра полоской' in urllib.request.urlopen(u + 'api/open').read().decode()
    # ключ: работает только для GET /api/open (запрос «из интернета» — с X-Forwarded-For)
    try: post(u + 'api/register', {"name": "maks", "password": "maks-pass-123"})
    except urllib.error.HTTPError: pass
    k = post(u + 'api/users/key', {"name": "maks"})["key"]
    def rq(path, key=None, method="GET"):
        h = {"X-Forwarded-For": "8.8.8.8"}
        if key: h["X-CPS-Key"] = key
        r = urllib.request.Request(u + path, headers=h, method=method, data=b"{}" if method == "POST" else None)
        try: return urllib.request.urlopen(r).status
        except urllib.error.HTTPError as e: return e.code
    assert rq('api/open/palette/pal_a.json', k) == 200
    assert rq('api/open/palette/pal_a.json?key=' + k) == 200
    assert rq('api/open/palette/pal_a.json') == 401 and rq('api/open/palette/pal_a.json', 'x' * 32) == 401
    assert rq('api/rugs', k) == 401 and rq('api/rugs/pal_a', k, "POST") == 401
    post(u + 'api/users/key', {"name": "maks", "off": True}); assert rq('api/open/palette/pal_a.json', k) == 401


@test
def t51_school_card_v1199(pg, ctx):
    """v11.99: /m «школа»: спор источников одного веса, «Губа-Ширван» без уточнения, школа не указана; одна карточка на ковёр (группа);
    ответ редактора — главное значение (человек), прежнее — в истории; «оставить» и «не знаю» снимают вопрос."""
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np
    def mk(rid, src, **kw):
        cv2.imwrite(f'{D}/rug_img/{rid}.jpg', np.full((300, 200, 3), (60, 90, 160), np.uint8))
        r = {"id": rid, "title": rid, "source": "тест", "museum": "", "inv": "", "school": "", "composition": "", "status": "new", "w": 200, "h": 300,
             "marks": [], "created": now, "updated": now, "cat": {"src": src}, "frame": "whole", "frame_by": "hand"}
        r.update(kw); json.dump(r, open(f'{D}/rugs/{rid}.json', 'w'), ensure_ascii=False)
    ok = {"ok": True, "by": "тест", "at": now}
    mk("sk_a", "kerimov_vol2", school="karabakh", same=[{"id": "sk_b"}], same_ok={"sk_b": ok})        # спор двух азербайджанских
    mk("sk_b", "azcarpetmuseum", school="guba", same=[{"id": "sk_a"}], same_ok={"sk_a": ok})
    mk("sk_c", "azerbaijanrugs", school="guba-shirvan", composition="pirebedil")                      # не уточнено (тип известен — раньше)
    mk("sk_d", "met", school="", composition="talish")                                               # не указана (тип известен — раньше остальных без школы)
    mk("sk_e", "met", school="guba-shirvan", school_by="dolgikh")                                   # человек оставил группу — тоже спрашиваем (не уточнено)
    def collect():                                                                # очередь отдаётся по 12: чужие карточки пропускаем, свои собираем
        T, order = {}, []
        for _ in range(40):
            ts = [t for t in get(u + 'api/micro/next?k=sch&n=12')["tasks"] if t["type"] == "sch"]
            other = [t for t in ts if not t["rid"].startswith("sk_")]
            for t in ts:
                if t["rid"].startswith("sk_") and t["rid"] not in T: T[t["rid"]] = t; order.append(t["rid"])
            if not other: break
            for t in other: post(u + 'api/micro/answer', {"type": "sch", "rid": t["rid"], "a": "skip"})
        return T, order
    T, order = collect()
    assert ("sk_a" in T) != ("sk_b" in T), ("один ковёр — одна карточка", sorted(T))
    d = T.get("sk_a") or T["sk_b"]; assert d["kind"] == "dispute" and [o["v"] for o in d["opts"]][:2] == ["guba", "karabakh"], d
    assert T["sk_c"]["kind"] == "coarse" and [o["v"] for o in T["sk_c"]["opts"]] == ["guba", "shirvan", "baku"] and T["sk_c"]["keep"].startswith("Губа-Ширван")
    assert T["sk_d"]["kind"] == "none" and len(T["sk_d"]["opts"]) == 8
    assert order[0] in ("sk_a", "sk_b"), order
    rid = d["rid"]
    assert post(u + 'api/micro/answer', {"type": "sch", "rid": rid, "a": "karabakh"})["applied"] == "school"
    r = get(u + 'api/rugs/' + rid); v = r["versions"]["school"]
    assert r["school"] == "karabakh" and r["school_by"].startswith("/m:") and v["main"] == "karabakh" and v["rank"] == 0, (r["school"], v)
    assert post(u + 'api/micro/answer', {"type": "sch", "rid": "sk_c", "a": "keep"})["applied"] == "keep"
    assert post(u + 'api/micro/answer', {"type": "sch", "rid": "sk_d", "a": "unk"})["applied"] == "unk"
    post(u + 'api/micro/answer', {"type": "sch", "rid": "sk_e", "a": "skip"})
    left, _ = collect(); assert not left, sorted(left)
    # интерфейс: карточка показывает источники и кнопки школ
    mk("sk_f", "azerbaijanrugs", school="ganja-gazakh")
    post(u + 'api/micro/answer', {"type": "sch", "rid": "sk_e", "a": "skip"})     # любой ответ сбрасывает кэш очереди (2 мин)
    pg.goto(u + 'm?k=sch'); pg.wait_for_timeout(2500)
    t = pg.inner_text('main'); assert 'что говорят источники' in t and 'не знаю' in t, t[:500]
    cur = pg.evaluate("S.tasks[S.i]"); v = cur["opts"][0]["v"]
    pg.screenshot(path=os.path.join(OUT, 't51_school_card.png'))
    pg.click(f'.schb [data-s="{v}"]'); pg.wait_for_timeout(1200)
    assert get(u + 'api/rugs/' + cur["rid"])["school"] == v, (cur["rid"], v)
    errors_ok(ctx, 'школа /m')


@test
def t52_frame_edge_v1200(pg, ctx):
    """v12.00: признаки края фото считаются на сервере для всех ковров (r.auto_edge), модель кадра берёт их, если обучена с ними;
    в паре элементов — ответ «тот же, другое заполнение» (тот же + пометка var)."""
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np, sys as _s
    img = np.full((300, 200, 3), 235, np.uint8); img[30:270, 20:180] = np.random.default_rng(1).integers(0, 255, (240, 160, 3))
    cv2.imwrite(f'{D}/rug_img/fe_a.jpg', img)
    json.dump({"id": "fe_a", "title": "fe_a", "source": "тест", "w": 200, "h": 300, "marks": [], "created": now, "updated": now, "status": "new",
               "auto": {"frame": "whole", "quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], "conf": 0.9, "area": 0.6}}, open(f'{D}/rugs/fe_a.json', 'w'))
    post(u + 'api/frame_edge_run', {})
    for _ in range(60):
        st = get(u + 'api/frame_edge')
        if not st["running"] and json.load(open(f'{D}/rugs/fe_a.json')).get("auto_edge"): break
        time.sleep(1)
    e = json.load(open(f'{D}/rugs/fe_a.json')).get("auto_edge")
    _s.path.insert(0, ctx['app']); import frame_edge as FE, frame_model as FM
    assert e and e["v"] == FE.VER and len(e["f"]) == len(FE.NAMES), e
    assert e["f"][FE.NAMES.index("e_bg_mean")] > 0.9, ("вокруг ковра фон", e["f"])
    r = json.load(open(f'{D}/rugs/fe_a.json'))
    assert len(FM.feat_for(r, FM.NAMES)) == len(FM.NAMES) and len(FM.feat_for(r, FM.NAMES + FE.NAMES)) == len(FM.NAMES) + len(FE.NAMES)
    assert FM.feat_for({"auto": r["auto"]}, FM.NAMES + FE.NAMES)[-1] == -1.0, "нет признаков края — пропуск (-1)"
    k = "xa|g1~xb|g1"
    j = post(u + 'api/micro/answer', {"type": "xsame", "rid": "xa", "key": k, "a": "same", "var": 1})
    assert j.get("ok"), j
    x = get(u + 'api/xsame'); assert k in x["same"] and k in x["same_var"], x


@test
def t53_open_work_v1202(pg, ctx):
    """v12.02: рабочие данные по ключу: manifest, карточка, исходное фото, выпрямленное, пачка zip; запись по ключу закрыта."""
    import zipfile, io as _io
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np
    img = np.full((500, 400, 3), 230, np.uint8); img[40:460, 60:340] = (40, 40, 160)
    cv2.imwrite(f'{D}/rug_img/ow_a.jpg', img)
    json.dump({"id": "ow_a", "title": "ow_a", "source": "тест", "w": 400, "h": 500, "marks": [], "created": now, "updated": now, "status": "new",
               "frame": "whole", "edge": {"wavy": ["t"]},
               "layout": {"quad": [[0.15, 0.08], [0.85, 0.08], [0.85, 0.92], [0.15, 0.92]], "quad_status": "confirmed"}}, open(f'{D}/rugs/ow_a.json', 'w'))
    try: post(u + 'api/register', {"name": "owk", "password": "owk-pass-123"})
    except urllib.error.HTTPError: pass
    k = post(u + 'api/users/key', {"name": "owk"})["key"]
    def rq(path, key=k):
        h = {"X-Forwarded-For": "8.8.8.8"}
        if key: h["X-CPS-Key"] = key
        try:
            r = urllib.request.urlopen(urllib.request.Request(u + path, headers=h)); return r.status, r.read()
        except urllib.error.HTTPError as e: return e.code, b""
    st, b = rq('api/open/manifest?since=%d' % (now - 5)); assert st == 200, st
    m = {x["id"]: x for x in json.loads(b)["rugs"]}
    assert "ow_a" in m and m["ow_a"]["quad_status"] == "confirmed" and m["ow_a"]["edge"] == {"wavy": ["t"]} and m["ow_a"]["t"]["raw"] > 0, m.get("ow_a")
    st, b = rq('api/open/card/ow_a.json'); assert st == 200 and json.loads(b)["id"] == "ow_a"
    st, b = rq('api/open/file/raw/ow_a.jpg'); a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR); assert st == 200 and a.shape == (500, 400, 3)
    st, b = rq('api/open/file/rect/ow_a.jpg'); a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    assert st == 200 and a is not None and a[a.shape[0] // 2, a.shape[1] // 2, 2] > 120, "выпрямленный: середина — красный ковёр"
    st, b = rq('api/open/files.zip?kind=raw,rect&ids=ow_a'); z = zipfile.ZipFile(_io.BytesIO(b))
    assert st == 200 and {"raw/ow_a.jpg", "rect/ow_a.jpg", "index.json"} <= set(z.namelist()), z.namelist()
    st, b = rq('api/open/cards.zip?ids=ow_a'); assert st == 200 and "rugs/ow_a.json" in zipfile.ZipFile(_io.BytesIO(b)).namelist()
    assert rq('api/open/manifest', None)[0] == 401 and rq('api/rugs/ow_a')[0] == 401


@test
def t54_vlm_frame_v1203(pg, ctx):
    """v12.03: «что на фото» моделью по картинке: уверенный «целый» решает машина (кроме контрольных); прежний ответ человека,
    с которым модель уверенно не согласна, возвращается на перепроверку с подсказкой."""
    import hashlib
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    import cv2, numpy as np
    img = np.full((500, 400, 3), 20, np.uint8); img[40:460, 60:340] = (40, 40, 160)
    ids = []
    for n in range(40):                                                        # некотрольный ковёр: md5 % 10 != 0
        i = 'vf_a%d' % n
        if int(hashlib.md5(("frctl:" + i).encode()).hexdigest(), 16) % 10: ids.append(i); break
    A = ids[0]; B = 'vf_b'
    for i in (A, B): cv2.imwrite(f'{D}/rug_img/{i}.jpg', img)
    base = {"source": "тест", "w": 400, "h": 500, "marks": [], "created": now, "updated": now, "status": "new",
            "auto": {"frame": "whole", "quad": [[0.15, 0.08], [0.85, 0.08], [0.85, 0.92], [0.15, 0.92]], "conf": 0.9, "area": 0.6}}
    json.dump({**base, "id": A, "title": A}, open(f'{D}/rugs/{A}.json', 'w'))
    json.dump({**base, "id": B, "title": B, "frame": "cropped", "frame_by": "micro", "micro": {"frame": {"zz": {"a": "cropped", "w": 2, "t": now - 100}}}}, open(f'{D}/rugs/{B}.json', 'w'))
    q = post(u + 'api/vlm/frame_all', {}); assert q["queued"] >= 2, q
    for _ in range(60):
        if all((get(u + 'api/rugs/' + i).get('auto') or {}).get('vlm_frame') for i in (A, B)): break
        time.sleep(0.5)
    else: raise AssertionError('модель «кадр» не ответила: %s' % get(u + 'api/vlm/frame'))
    st = get(u + 'api/vlm/frame'); assert st["vs_people"]["n"] >= 1 and "cropped→whole" in st["vs_people"]["pairs"], st
    T = get(u + 'api/micro/next?n=50&k=frame')
    fr = {t['rid']: t for t in T['tasks'] if t['type'] == 'frame'}
    assert A not in fr, 'уверенный «целый» по модели — машина решает сама'
    assert B in fr and fr[B].get('rc') == 1 and fr[B]['vf'] == 'whole' and fr[B]['was'] == 'cropped', fr.get(B)
    post(u + 'api/micro/answer', {'type': 'frame', 'rid': B, 'a': 'whole'})
    r = get(u + 'api/rugs/' + B); assert r['frame'] == 'whole', r.get('frame')
    T = get(u + 'api/micro/next?n=50&k=frame'); assert B not in [t['rid'] for t in T['tasks']], 'после ответа перепроверка уходит'
    for i in (A, B): os.remove(f'{D}/rugs/{i}.json')


@test
def t55_elem_name_v1204(pg, ctx):
    """v12.04: название элемента по картинке среди всех: банк из названных рамок других ковров и схем; замер без своего ковра."""
    import sys, tempfile, cv2, numpy as np
    sys.path.insert(0, ctx['app']); import elem_name as EN
    cols = {'deve': (0, 0, 200), 'qeychi': (0, 200, 0), 'alma': (200, 0, 0)}
    class FE:                                                                   # «сеть»: вектор = средний цвет рамки
        @staticmethod
        def version(m='match'): return 1
        @staticmethod
        def small(a): return a
        @staticmethod
        def _v(a, b):
            h, w = a.shape[:2]; x, y, bw, bh = b; c = a[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)].reshape(-1, 3).mean(0) + 1
            return c / np.linalg.norm(c)
        @classmethod
        def emb(cls, a, boxes, m='match'): return np.stack([cls._v(a, b) for b in boxes])
        @classmethod
        def emb_rot(cls, a, b, m='match'): return np.stack([cls._v(a, b)] * 8)
    T = tempfile.mkdtemp(); rugs = []
    for n in range(3):
        a = np.full((200, 200, 3), 128, np.uint8); a[10:60, 10:60] = cols['deve']; a[100:150, 100:150] = cols['qeychi']
        cv2.imwrite(f'{T}/r{n}.jpg', a)
        rugs.append({'id': f'r{n}', 'composition': 'x', 'marks': [
            {'id': 'a', 'status': 'confirmed', 'box': [0.05, 0.05, 0.25, 0.25], 'motif': 'deve'},
            {'id': 'b', 'status': 'confirmed', 'box': [0.5, 0.5, 0.25, 0.25], 'motif': 'qeychi'}]})
    os.makedirs(f'{T}/out'); s = np.full((40, 40, 3), cols['alma'], np.uint8); cv2.imwrite(f'{T}/out/t1_f01.png', s)
    smeta = {'t1_f01': {'name': 'Alma', 'motif': 'alma'}}
    B = EN.build(FE, rugs, smeta, f'{T}/out', lambda rid: f'{T}/{rid}.jpg', lambda p, side: cv2.imread(p), log=lambda *a: None)
    assert len(B['U']) == 6 and len(B['S']) == 1, (len(B['U']), len(B['S']))
    ev = EN.evaluate(B, closed=lambda u: ['alma'], shape=None)
    assert ev['n'] == 6 and ev['rugs']['top1'] == 1.0 and ev['closed']['in_list'] == 0.0, ev
    q = FE.emb_rot(np.full((10, 10, 3), cols['alma'], np.uint8), [0, 0, 1, 1])
    sg = EN.suggest(B, q8=q, k=3); assert sg[0][0] == 'alma' and sg[0][2] == 'scheme', sg
    op = EN.options(B, sg, lambda l: l.upper()); assert op[0]['id'] == 'mo:alma' and op[0]['sc'] == 't1_f01' and op[0]['img'], op
    # ответ карточки «имена» вариантом из подсказки по картинке: узор словаря и просто название
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    json.dump({"id": "en_r", "title": "en_r", "source": "тест", "w": 200, "h": 200, "created": now, "updated": now, "status": "new",
               "marks": [{"id": "m1", "status": "confirmed", "box": [0.1, 0.1, 0.2, 0.2]}, {"id": "m2", "group": "m1", "status": "confirmed", "box": [0.5, 0.5, 0.2, 0.2]},
                         {"id": "m3", "status": "confirmed", "box": [0.7, 0.1, 0.2, 0.2]}]}, open(f'{D}/rugs/en_r.json', 'w'))
    post(u + 'api/micro/answer', {'type': 'name', 'rid': 'en_r', 'g': 'm1', 'a': 'mo:deve'})
    post(u + 'api/micro/answer', {'type': 'name', 'rid': 'en_r', 'g': 'm3', 'a': 'tx:qoşa qarmaq'})
    mk = {m['id']: m for m in get(u + 'api/rugs/en_r')['marks']}
    assert mk['m1'].get('motif') == 'deve' and mk['m2'].get('motif') == 'deve' and mk['m3'].get('motif_text') == 'qoşa qarmaq', mk
    os.remove(f'{D}/rugs/en_r.json')


@test
def t56_comp_nn_v1205(pg, ctx):
    """v12.05: тип ковра по соседям по картинке: ковёр не находит сам себя и своих двойников; рисунок Gans-Ruedin голосует."""
    import sys, numpy as np
    sys.path.insert(0, ctx['app']); import rug_nn as RN
    class FE:
        @staticmethod
        def meta(m='match'): return {"S": 8}
        @staticmethod
        def _prep(c): return c.astype(np.float32).mean((0, 1))
        @staticmethod
        def _run(arr, m): return np.stack(arr)
    def img(c): return np.full((20, 10, 3), c, np.uint8)
    rugs = [{'id': 'a1', 'composition': 'pirebedil'}, {'id': 'a2', 'composition': 'pirebedil'}, {'id': 'a3', 'composition': 'pirebedil', 'same': [{'id': 'a4'}]},
            {'id': 'a4', 'composition': 'kasimushagi', 'same': [{'id': 'a3'}]}, {'id': 'b1', 'composition': 'kasimushagi'}, {'id': 'x', 'composition': ''}]
    col = {'a1': (0, 0, 200), 'a2': (0, 10, 200), 'a3': (0, 5, 200), 'a4': (0, 5, 201), 'b1': (200, 0, 0), 'x': (0, 3, 199)}
    B = RN.build(FE, rugs, lambda r: img(col[r['id']]), gans=[('gr1', img((200, 5, 0)), 'kasimushagi')], log=lambda *a: None)
    i = B['ids'].index('a4'); sg = RN.suggest(B, i, k=2, use_gans=False)
    assert sg and sg[0][0] == 'pirebedil', ('двойник a3 не должен подсказывать сам себя: ', sg)
    ev = RN.evaluate(B, k=2); assert ev['n'] == 5 and 'rugs' in ev, ev
    q = RN.vec(FE, img((199, 4, 0))); g = RN.suggest(B, q=q, k=1, skip=set()); assert any(c == 'kasimushagi' for c, *_ in g), g
    # v12.07: английские торговые названия → тип словаря (подписи Gans-Ruedin, дилеры)
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    for i, t in (('tr_a', "'DRAGON' RUG, KARABAKH OR SHIRVAN"), ('tr_b', 'SEICHUR SOUMAK')):
        json.dump({"id": i, "title": t, "source": "тест", "w": 10, "h": 10, "marks": [], "created": now, "updated": now, "status": "new"}, open(f'{D}/rugs/{i}.json', 'w'))
    L = {r['id']: r for r in get(u + 'api/rugs')['rugs'] if r['id'] in ('tr_a', 'tr_b')}
    assert L['tr_a'].get('comp_sug') == 'ejdahali' and L['tr_b'].get('comp_sug') == 'zeyve', {k: v.get('comp_sug') for k, v in L.items()}
    for i in ('tr_a', 'tr_b'): os.remove(f'{D}/rugs/{i}.json')


@test
def t57_text_facts_v1206(pg, ctx):
    """v12.06: факты из подписи: значение берётся только с дословной цитатой; сводка показывает новое и расхождения с карточкой."""
    u = ctx['url']; D = ctx['app'] + '/data'; now = int(time.time())
    json.dump({"id": "tf_a", "title": "Əjdahalı", "source": "тест", "w": 10, "h": 10, "marks": [], "created": now, "updated": now, "status": "new",
               "school": "guba", "cat": {"src": "book:test", "text": "Əjdahalı. Xalça. Qarabağ qrupu. Şuşa. XIX əsr. Yun."}}, open(f'{D}/rugs/tf_a.json', 'w'))
    q = post(u + 'api/text_facts/all', {}); assert q['queued'] >= 1, q
    for _ in range(40):
        F = (get(u + 'api/rugs/tf_a').get('auto') or {}).get('text_facts')
        if F: break
        time.sleep(0.5)
    else: raise AssertionError('факты не извлечены')
    assert F.get('group') == 'Qarabağ' and F['quotes']['group'] == 'Qarabağ' and F.get('date') == 'XIX əsr', F
    try: post(u + 'api/register', {"name": "tfk", "password": "tfk-pass-123"})
    except urllib.error.HTTPError: pass
    k = post(u + 'api/users/key', {"name": "tfk"})["key"]
    st = json.loads(urllib.request.urlopen(urllib.request.Request(u + 'api/open/text_facts', headers={"X-CPS-Key": k, "X-Forwarded-For": "8.8.8.8"})).read())
    assert st['fields']['school']['conflict'] >= 1 and any(c['id'] == 'tf_a' and c['field'] == 'school' for c in st['conflicts']), st
    assert st['fields']['date']['new'] >= 1, st['fields']['date']
    os.remove(f'{D}/rugs/tf_a.json')


@test
def m_edge_ids_dont_clash(pg, ctx):
    """v12.35: у карточки «край» были id (ed, ew, eh…), совпавшие с CSS редактора рамок: кнопка «готово» растягивалась на весь экран и закрывала страницу.
    Проверяем: ни один id и класс карточки не упомянут в стилях страницы."""
    html = open(os.path.join(ctx['app'], 'm.html'), encoding='utf-8').read()
    css = html[html.index('<style>'):html.index('</style>')]
    blk = html[html.index('function showEdge(t){'):html.index('function showXsame(t){')]
    ids = set(re.findall(r'id="([^"]+)"', blk)); cls = set(re.findall(r'class="([a-z]+)"', blk)) - {'yes', 'no', 'sm', 'unk', 'mut', 'btns'}
    bad = [i for i in ids if re.search(r'#' + re.escape(i) + r'(?![\w-])', css)] + [c for c in cls if re.search(r'\.' + re.escape(c) + r'(?![\w-])', css)]
    assert not bad, f'id/классы карточки «край» есть в стилях страницы: {bad}'
    assert all(i.startswith('kx_') for i in ids), f'id карточки «край» должны начинаться с kx_: {ids}'


@test
def t58_repeats_v2_v1231(pg, ctx):
    """v12.31: быстрый поиск повторов (v2) находит не меньше v1, работает при сотнях повторов (v1 поднимал порог выше 1) и быстрее
    при многих образцах; замер /api/open/rep_eval отвечает и держит прежний способ, пока замера нет."""
    import sys, numpy as np, cv2; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import rug_match as RM
    rng = np.random.default_rng(3)
    def rug(el, W=700, H=900):
        a = np.zeros((H, W, 3), np.uint8); a[:] = (50, 40, 130); pts = (rng.random((8, 2)) * el).astype(np.int32); B = []
        for y in range(30, H - 2 * el, int(el * 2.3)):
            for x in range(30, W - 2 * el, int(el * 2.5)):
                m = np.zeros((el, el, 3), np.uint8); m[:] = (50, 40, 130); cv2.fillPoly(m, [pts], (200, 180, 60)); cv2.circle(m, (el // 3, el // 2), el // 6, (30, 120, 30), -1)
                m = np.ascontiguousarray(np.rot90(m, int(rng.integers(4)))); a[y:y + el, x:x + el] = m; B.append([x / W, y / H, el / W, el / H])
        return cv2.GaussianBlur((a + rng.normal(0, 15, a.shape)).clip(0, 255).astype(np.uint8), (0, 0), 1.0), B
    def hits(c, B): return sum(any(b[0] <= x["box"][0] + x["box"][2] / 2 <= b[0] + b[2] and b[1] <= x["box"][1] + x["box"][3] / 2 <= b[1] + b[3] for b in B) for x in c)
    a, B = rug(70)
    t0 = time.time(); c1 = RM.find_repeats(a, B[0], boxes=B[:6], algo="v1"); d1 = time.time() - t0
    t0 = time.time(); c2 = RM.find_repeats(a, B[0], boxes=B[:6], algo="v2"); d2 = time.time() - t0
    assert hits(c2, B[6:]) >= hits(c1, B[6:]) - 1 and hits(c2, B[6:]) >= len(B[6:]) - 2, f"v2 нашёл {hits(c2, B[6:])}, v1 {hits(c1, B[6:])} из {len(B) - 6}"
    assert d2 < d1, f"v2 не быстрее при 6 образцах: {d2:.2f} с против {d1:.2f} с"
    a, B = rug(20)                                       # сотни мелких повторов
    c2 = RM.find_repeats(a, B[0], boxes=B[:1], algo="v2")
    assert hits(c2, B[1:]) >= 50, f"мелкие элементы с сотнями повторов: v2 нашёл {hits(c2, B[1:])}"
    j = get(ctx['url'] + 'api/open/rep_eval')
    assert j['algo'] in ('v1', 'v2') and 'running' not in j and 'run' in j, j


@test
def t59_edge_card_v1236(pg, ctx):
    """v12.36: карточка «край» в /m — приходит у целого ковра с подтверждёнными углами, рисует 12 ручек (по 3 на сторону) на картинке,
    перетаскивание ручки меняет правку, «готово» пишет edge_fix и убирает ковёр из очереди"""
    import urllib.request as U, json as _j, cv2, numpy as np
    D = os.path.join(ctx['app'], 'data')
    img = np.full((900, 700, 3), (60, 60, 60), np.uint8); cv2.rectangle(img, (100, 100), (600, 800), (30, 30, 160), -1); cv2.imwrite(f'{D}/rug_img/edgerug.jpg', img)
    r = _j.load(open(f'{D}/rugs/frieze_1.json', encoding='utf-8'))
    r2 = {**r, 'id': 'edgerug', 'title': 'Edge rug', 'w': 700, 'h': 900, 'marks': [], 'same': [], 'same_ok': {}, 'micro': {}, 'frame': 'whole', 'frame_by': 'micro',
          'layout': {'quad': [[1 / 7, 1 / 9], [6 / 7, 1 / 9], [6 / 7, 8 / 9], [1 / 7, 8 / 9]], 'quad_status': 'confirmed', 'bounds': [], 'rings': [], 'status': 'auto', 'carpet': [.14, .11, .72, .78]}}
    for k in ('dup_of', 'hidden', 'split_into', 'split_from', 'multi', 'lib_img', 'edge', 'edge_fix'): r2.pop(k, None)
    _j.dump(r2, open(f'{D}/rugs/edgerug.json', 'w', encoding='utf-8'), ensure_ascii=False); time.sleep(1.5)
    get = lambda path: _j.load(U.urlopen(ctx['url'] + path))
    j = get('api/micro/next?n=12&k=edge')
    assert [t for t in j['tasks'] if t['rid'] == 'edgerug'], f"нет карточки «край»: {[t['rid'] for t in j['tasks']]} left={j['left']}"
    pg.goto(ctx['url'] + 'm?k=edge'); pg.wait_for_timeout(1500)
    for _ in range(12):
        if pg.evaluate("S.tasks[S.i]&&S.tasks[S.i].rid") == 'edgerug': break
        pg.evaluate("S.i++;show()"); pg.wait_for_timeout(300)
    pg.wait_for_selector('.kxh', timeout=15000); pg.wait_for_timeout(800)
    n = pg.evaluate("document.querySelectorAll('.kxh').length"); assert n == 12, f'ручек {n}, ожидалось 12'
    im = pg.query_selector('#kx_ew img').bounding_box()
    hs = [pg.query_selector(f'.kxh[data-k="{k}"][data-i="1"]').bounding_box() for k in ('top', 'bot', 'left', 'right')]
    assert all(im['x'] - 20 <= h['x'] <= im['x'] + im['width'] + 20 and im['y'] - 20 <= h['y'] <= im['y'] + im['height'] + 20 for h in hs), f'ручки вне картинки: {im} {hs}'
    pg.screenshot(path=os.path.join(OUT, 't59_edge.png'))
    h = hs[0]; pg.mouse.move(h['x'] + 13, h['y'] + 13); pg.mouse.down(); pg.mouse.move(h['x'] + 13, h['y'] + 13 + im['height'] * 0.04, steps=5)
    assert pg.evaluate("!$('kx_loupe').hidden && !!$('kx_loupe').querySelector('img')"), 'нет лупы при перетаскивании ручки'
    pg.screenshot(path=os.path.join(OUT, 't59_edge_loupe.png')); pg.mouse.up()
    assert pg.evaluate("$('kx_loupe').hidden"), 'лупа не прячется после перетаскивания'
    pg.click('#kx_ed'); pg.wait_for_timeout(1500)
    f = get('api/rugs/edgerug').get('edge_fix') or {}
    assert f.get('top') and f['top'][1] > 0.01, f'правка края не записалась: {f}'
    assert not [t for t in get('api/micro/next?n=12&k=edge')['tasks'] if t['rid'] == 'edgerug'], 'отвеченный ковёр вернулся в очередь «край»'
    errors_ok(ctx, '/m край v12.36')


@test
def t60_xfind_card_v1236(pg, ctx):
    """v12.36: ночной поиск по коврам → карточка /m «на других коврах»: образец + находки; ✓ ставит подтверждённую рамку с названием образца,
    неотмеченные пишутся «не он», отвеченное не возвращается; /api/open/xfind показывает итог"""
    import urllib.request as U, json as _j, shutil as _sh
    D = os.path.join(ctx['app'], 'data'); base = _j.load(open(f'{D}/rugs/frieze_1.json', encoding='utf-8'))
    for rid in ('xf_q', 'xf_a', 'xf_b'):
        r = {**base, 'id': rid, 'title': 'XF ' + rid, 'same': [], 'same_ok': {}, 'micro': {}, 'marks': []}
        for k in ('dup_of', 'hidden', 'split_into', 'split_from', 'multi', 'lib_img', 'same_group'): r.pop(k, None)
        if rid == 'xf_q': r['marks'] = [{'id': 'mq', 'box': [.3, .3, .1, .1], 'group': 'gq', 'status': 'confirmed', 'motif_text': 'тест-гёль', 'zone': 'field'}]
        _sh.copy(f'{D}/rug_img/frieze_1.jpg', f'{D}/rug_img/{rid}.jpg'); _j.dump(r, open(f'{D}/rugs/{rid}.json', 'w', encoding='utf-8'), ensure_ascii=False)
    _j.dump({'t': int(time.time()), 'items': [{'q': {'rid': 'xf_q', 'g': 'gq', 'box': [.3, .3, .1, .1], 'n': 1},
             'c': [{'rid': 'xf_a', 'box': [.5, .5, .1, .1], 's': 0.9}, {'rid': 'xf_b', 'box': [.2, .6, .1, .1], 's': 0.7}]}]}, open(f'{D}/xfind_night.json', 'w'))
    time.sleep(1.5)
    get = lambda path: _j.load(U.urlopen(ctx['url'] + path))
    j = get('api/micro/next?n=12&k=xfind'); ts = [t for t in j['tasks'] if t['type'] == 'xfind']
    assert ts and len(ts[0]['items']) == 2, f"нет карточки «на других коврах»: {j['left']}"
    pg.goto(ctx['url'] + 'm?k=xfind'); pg.wait_for_selector('.grid.tw .tl', timeout=15000); pg.wait_for_timeout(600)
    pg.screenshot(path=os.path.join(OUT, 't60_xfind.png'))
    pg.click('.grid.tw .tl[data-i="0"]'); pg.wait_for_timeout(200); pg.click('#a1'); pg.wait_for_timeout(1500)
    ma = [m for m in get('api/rugs/xf_a')['marks'] if m.get('machine') == 'xfind']
    assert ma and ma[0]['status'] == 'confirmed' and ma[0]['motif_text'] == 'тест-гёль', f'✓ не поставил рамку с названием: {ma}'
    assert not [m for m in get('api/rugs/xf_b')['marks'] if m.get('machine') == 'xfind'], 'неотмеченная находка получила рамку'
    assert not [t for t in get('api/micro/next?n=12&k=xfind')['tasks'] if t['type'] == 'xfind'], 'отвеченная карточка вернулась'
    o = get('api/open/xfind'); assert o['groups'] == 1 and o['answers'] == {'yes': 1, 'no': 1}, o
    errors_ok(ctx, '/m на других коврах v12.36')


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='cps_test_'); app = os.path.join(tmp, 'cps')
    shutil.copytree(CPS, app, ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc', 'rug_lib', 'rug_img', 'rugs', 'rug_thumbs'))
    make_rugs(os.path.join(app, 'data')); make_color_fig(os.path.join(app, 'data')); make_crops(os.path.join(app, 'data')); make_auto_sheet(os.path.join(app, 'data'))
    port = free_port(); url = f'http://127.0.0.1:{port}/'
    env = {**os.environ, 'CPS_RUGS_TEST': '1', 'CPS_PASSWORD': '', 'CPS_GITHUB_TOKEN': '', 'CPS_NO_SCHED': '1', 'CPS_MOTIF_FAKE': '1', 'CPS_SAM': '0', 'CPS_VLM_MOCK': '1', 'CPS_TW_MIN_SAME': '2'}   # CPS_NO_SCHED: автоматические задачи не запускаются сами
    log = open(os.path.join(OUT, 'server.log'), 'w')
    srv = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend:app', '--port', str(port)], cwd=app, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = fail = 0
    try:
        for _ in range(240):
            try: get(url + 'api/rugs'); break
            except Exception: time.sleep(1)
        else: raise SystemExit('сервер не запустился — см. cps/tests/out/server.log')
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            br = p.chromium.launch(**({'executable_path': os.environ['CPS_CHROME'], 'args': ['--no-sandbox']} if os.environ.get('CPS_CHROME') else {})); pg = br.new_page(viewport={'width': 1600, 'height': 950})
            ctx = {'url': url, 'errors': [], 'app': app}
            pg.on('pageerror', lambda e: ctx['errors'].append(str(e)))
            pg.goto(url); pg.wait_for_timeout(2000)
            for t in TESTS:
                if ARGS.only and ARGS.only not in t.__name__: continue
                try:
                    t(pg, ctx); ok += 1; print('  ✓', t.__name__)
                except Exception as e:
                    fail += 1; print('  ✗', t.__name__, '—', (str(e) or type(e).__name__)[:300])
                    if not isinstance(e, AssertionError): traceback.print_exc(limit=2)
                    try: pg.screenshot(path=f'{OUT}/FAIL_{t.__name__}.png'); pg.keyboard.press('Escape'); pg.evaluate("document.getElementById('modal').hidden=true")
                    except Exception: pass
            br.close()
    finally:
        srv.terminate(); srv.wait(10)
        if ARGS.keep: print('данные теста:', tmp)
        else: shutil.rmtree(tmp, ignore_errors=True)
    print(f'\nпрошло {ok}, упало {fail}. Скриншоты: cps/tests/out/')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
