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
    for i in range(6): save(f"vam_{i}", carpet(300 + 10 * i, 420, 10 + i), title=f"V&A rug {i}", museum="V&A", cat={"src": "vam"})


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
    for t in ('tab-pipe', 'tab-sheet', 'tab-queue', 'tab-rugs'):
        pg.click('#' + t); pg.wait_for_timeout(900)
        errors_ok(ctx, 'вкладка ' + t)
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


# ---------------- запуск ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='cps_test_'); app = os.path.join(tmp, 'cps')
    shutil.copytree(CPS, app, ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc', 'rug_lib', 'rug_img', 'rugs', 'rug_thumbs'))
    make_rugs(os.path.join(app, 'data')); make_color_fig(os.path.join(app, 'data')); make_crops(os.path.join(app, 'data'))
    port = free_port(); url = f'http://127.0.0.1:{port}/'
    env = {**os.environ, 'CPS_RUGS_TEST': '1', 'CPS_PASSWORD': '', 'CPS_GITHUB_TOKEN': ''}
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
