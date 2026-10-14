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
import os, sys, json, time, shutil, socket, subprocess, tempfile, traceback, urllib.request, argparse
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
    def star(a, cx, cy, r, col):     # v11.4: один и тот же «узор» на трёх коврах — для поиска по всем коврам
        pts = [(int(cx + (r if i % 2 == 0 else r * .45) * np.cos(i * np.pi / 8)), int(cy + (r if i % 2 == 0 else r * .45) * np.sin(i * np.pi / 8))) for i in range(16)]
        cv2.fillPoly(a, [np.array(pts)], col); cv2.circle(a, (cx, cy), int(r * .25), (250, 250, 250), -1)
    STARS = {0: (150, 200, (30, 200, 230)), 3: (230, 140, (200, 90, 40)), 5: (110, 300, (90, 220, 90))}
    for i in range(6):
        a = carpet(300 + 10 * i, 420, 10 + i)
        if i in STARS: star(a, *STARS[i][:2], 34, STARS[i][2])
        save(f"vam_{i}", a, title=f"V&A rug {i}", museum="V&A", cat={"src": "vam"})
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
    pg.keyboard.press('c'); pg.wait_for_timeout(200); assert pg.evaluate("R.sel.status") == 'confirmed', 'C не подтверждает элемент'
    pg.keyboard.press('x'); pg.wait_for_timeout(200); assert pg.evaluate("R.sel.status") == 'rejected', 'X не отклоняет элемент'
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
    pg.click('#tab-pipe'); pg.wait_for_timeout(600)
    pg.keyboard.press('Shift+Slash'); pg.wait_for_timeout(400)
    assert pg.is_visible('#modal .kbgrid'), '«?» не открывает шпаргалку клавиш'
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300); assert not pg.is_visible('#modal'), 'Esc не закрывает шпаргалку'
    pg.click('#railKeys'); pg.wait_for_timeout(300); assert pg.is_visible('#modal .kbgrid'), 'кнопка «Клавиши» не открывает шпаргалку'
    pg.click('#mClose'); pg.wait_for_timeout(200)
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
    assert pg.is_visible('#catalog .kpirow'), 'нет цифр сверху'
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
def t15_rugs_overview(pg, ctx):
    pg.click('#tab-rugs'); pg.wait_for_timeout(1500)
    if pg.evaluate("R.view") == 'edit': pg.evaluate("rugBack()"); pg.wait_for_timeout(600)
    assert pg.is_visible('#rgTree .ptree') and pg.is_visible('#rgKpi') and pg.is_visible('#rgPeek'), 'у «Ковров» нет дерева / цифр / карточки справа'
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
    pg.check('#rgFBig'); pg.wait_for_timeout(400)
    assert pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)") == ['kerimov_vol3_p100_1'], 'чип «есть фото крупнее» не работает'
    pg.uncheck('#rgFBig'); pg.wait_for_timeout(400)
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
    assert pg.is_visible('#rgKpiComp [data-cacc="title"]'), 'в обзоре нет «принять по названию»'
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
    pg.click('#rgKinds [data-kind="items"]'); pg.wait_for_timeout(400)
    assert pg.evaluate("[...document.querySelectorAll('.rgcard')].map(c=>c.dataset.id)") == ['am_bag'], 'фильтр «Изделия» не работает'
    pg.click('#rgKinds [data-kind="carpets"]'); pg.wait_for_timeout(300)
    # скрыть
    pg.evaluate("rugOpen('vam_4')"); pg.wait_for_timeout(1400)
    pg.click('#rgTbMenu'); pg.wait_for_timeout(200); pg.click('#rgTbHide'); pg.wait_for_timeout(300); pg.click('#modal [data-hr="0"]'); pg.wait_for_timeout(1500)
    assert get(u + 'api/rugs/vam_4').get('hidden'), 'фото не скрылось'
    assert 'vam_4' not in pg.evaluate("R.list.filter(r=>rugKindOf(r)==='carpets').map(r=>r.id)"), 'скрытое фото осталось среди ковров'
    pg.click('#rgTbMenu'); pg.wait_for_timeout(200); pg.click('#rgTbHide'); pg.wait_for_timeout(1200)
    assert not get(u + 'api/rugs/vam_4').get('hidden'), '«вернуть» не вернуло фото'
    # автоповторы после новой рамки — только «точно такие же»
    pg.evaluate("rugOpen('border_rep')"); pg.wait_for_timeout(1500)
    pg.evaluate("R.rug.marks=[];rugSaveSoon();"); pg.wait_for_timeout(900)
    pg.evaluate("(()=>{const m={id:'mA',box:[20/300,120/620,80/300,80/620],zone:'border',status:'proposed',motif:'',motif_text:'',scheme:'',group:'',angle:0,flip:0};R.rug.marks.push(m);rugSelect(m);rugFindRepeats(m,true);})()")
    pg.wait_for_timeout(3500)
    mk = pg.evaluate("R.rug.marks.filter(m=>m.machine).map(m=>m.certainty)")
    assert mk and all(c == 'sure' for c in mk), f'после новой рамки добавлены неуверенные повторы: {mk}'
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
    pg.click('#rgDupChk'); pg.wait_for_timeout(1200)
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
    pg.click('#rgTypesBtn'); pg.wait_for_timeout(1500)
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
    pg.click('#tab-rugs'); pg.wait_for_timeout(800)
    pg.evaluate("rugOpen('kerimov_vol3_p100_1')"); pg.wait_for_timeout(1500)
    tw = '#rgStrip .rgtw[data-open="met_123"]'
    assert pg.is_visible(tw + ' [data-vd="sim"]'), 'в ленте двойников нет кнопок ✓ ≈ ✗'
    pg.click(tw + ' [data-vd="sim"]'); pg.wait_for_timeout(1200)
    ok = (get(u + 'api/rugs/kerimov_vol3_p100_1').get('same_ok') or {}).get('met_123') or {}
    assert ok.get('similar') and ok.get('ok') is False, f'«≈ похожи» не записалось: {ok}'
    assert (pg.text_content(tw + ' .st') or '').strip() == '≈', 'значок «≈» не появился на миниатюре'
    assert pg.evaluate("R.rug.id") == 'kerimov_vol3_p100_1', 'клик по кнопке открыл другой ковёр'
    pg.click(tw + ' [data-vd="ok"]'); pg.wait_for_timeout(1200)
    assert ((get(u + 'api/rugs/met_123').get('same_ok') or {}).get('kerimov_vol3_p100_1') or {}).get('ok') is True, '✓ не записалось у второго ковра'
    pg.click(tw + ' [data-vd="ok"]'); pg.wait_for_timeout(1200)
    assert 'met_123' not in (get(u + 'api/rugs/kerimov_vol3_p100_1').get('same_ok') or {}), 'повторный клик не снял отметку'
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
    assert abs((tb['x'] + tb['width'] / 2) - (v['x'] + v['width'] / 2)) < 30 and tb['y'] - v['y'] < 30, 'панель инструментов не над фото посередине'
    assert pg.evaluate("document.getElementById('rgFindRep').disabled"), 'кнопки элемента активны без выбранной рамки'
    assert pg.evaluate("document.querySelectorAll('#rgPassBtn,#rgPassBtn2').length") == 1, 'кнопка «Паспорт» на экране не одна'
    title = pg.evaluate("R.rug.title")
    for sel in ('#rgLeft', '#rgSide', '#sbar'):
        assert title not in (pg.text_content(sel) or ''), f'название ковра повторяется в {sel}'
    assert not pg.is_visible('#rgMenu'); pg.click('#rgTbMenu'); pg.wait_for_timeout(200); assert pg.is_visible('#rgMenu'), 'меню «⋯» не открывается'
    pg.mouse.click(v['x'] + 20, v['y'] + v['height'] - 20); pg.wait_for_timeout(200); assert not pg.is_visible('#rgMenu'), 'меню не закрывается кликом мимо'
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


# ---------------- запуск ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='cps_test_'); app = os.path.join(tmp, 'cps')
    shutil.copytree(CPS, app, ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc', 'rug_lib', 'rug_img', 'rugs', 'rug_thumbs'))
    make_rugs(os.path.join(app, 'data')); make_color_fig(os.path.join(app, 'data')); make_crops(os.path.join(app, 'data')); make_auto_sheet(os.path.join(app, 'data'))
    port = free_port(); url = f'http://127.0.0.1:{port}/'
    env = {**os.environ, 'CPS_RUGS_TEST': '1', 'CPS_PASSWORD': '', 'CPS_GITHUB_TOKEN': '', 'CPS_NO_SCHED': '1', 'CPS_MOTIF_FAKE': '1'}   # CPS_NO_SCHED: автоматические задачи не запускаются сами
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
            br = p.chromium.launch(); pg = br.new_page(viewport={'width': 1600, 'height': 950})
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
