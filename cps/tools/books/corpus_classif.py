"""Сверка классификаций из текстов книг (2026-11-13): Керимов 1984 (перепечатан в сборнике симпозиума 2007, с. 14–16 az / 24–25 en)
и Мурадов 2013 (с. 28–56 и 110–127) — против модели (carpet_schools.json) и против подписей к фото во всех источниках.
Списки названий выписаны из текста вручную (OCR слишком шумный для регулярок), написание — азербайджанское, как в модели.
Запуск: python3 corpus_classif.py  → site/model/sources/CLASSIFICATIONS_2026-11-13.json. Правило 4: расхождения фиксируются, не решаются."""
import os, sys, json, re, glob, difflib, unicodedata, collections
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
C = json.load(open(REPO + '/site/model/out/concepts.json')); CON = {c['id']: c for c in C}
CS = json.load(open(REPO + '/site/model/carpet_schools.json'))['items']
def fold(s):
    s = unicodedata.normalize('NFKD', (s or '').lower().replace('ə','e').replace('ı','i').replace('ğ','g').replace('ş','s').replace('ç','c').replace('ö','o').replace('ü','u'))
    return re.sub(r'[^a-z0-9а-я]+', '', ''.join(ch for ch in s if not unicodedata.combining(ch)))
NAME = {}
for c in C:
    for n in [c['headword']] + [x['v'] for x in c.get('names', [])]: NAME.setdefault(fold(n), c['id'])
KEYS = list(NAME)
ALIAS = {'uqah': 'ugax', 'ugah': 'ugax', 'qesed': 'qashad', 'goradil': 'geradil', 'qedi': 'qadi', 'bexsayis': 'baxshayish'}     # то же слово в другом написании (сверено по рус. именам понятий)
GENERIC = {'namazliq', 'ovchuluq', 'ejdahali', 'agacli'}     # названия по сюжету/назначению: ткут в разных школах, расхождение школ не ошибка
def match(n):
    f = fold(n)
    if f in ALIAS: return ALIAS[f], 'alias'
    if f in NAME: return NAME[f], 'exact'
    m = difflib.get_close_matches(f, KEYS, 1, 0.9)
    return (NAME[m[0]], 'fuzzy:' + m[0]) if m and len(f) >= 5 else (None, None)
TYPE_OF = {'guba':'guba-shirvan','shirvan':'guba-shirvan','baku':'guba-shirvan','ganja':'ganja-gazakh','gazakh':'ganja-gazakh','karabakh':'karabakh',
           'nakhchivan':'karabakh','tabriz':'tabriz','ardabil':'tabriz','irevan':None}

K84 = {'id': 'kerimov_1984_in_symposium_2007', 'short': 'Керимов 1984 (перепечатка в сборнике симпозиума 2007)',
 'where': 'symposium_2007_paris: с. 14–16 файла (az), 24–25 (en); в конце статьи указано: из «Rugs & Carpets from the Caucasus», Aurora, Ленинград, 1984',
 'levels': 'тип → группа → подгруппа (район)',
 'groups': [
  ('guba','Quba','Quba',['Köhnə Quba','Quba','Alpan','Xırdagül-çiçi','Sırt-çiçi','Alçagül-çiçi','Qollu-çiçi','Qımıl','Qədim minarə','Hacıqayıb']),
  ('guba','Quba','Dəvəçi',['Qaraqaşlı','Şahnəzərli','Mollakamallı','Ləcədi','Pirəbədil','Herat-Pirəbədil','Zeyvə','Zağlı','Əlixanlı','Bilici','Uqah','Çarax']),
  ('guba','Quba','Qonaqkənd',['Orduc','Afurca','Yerfi','Cek','Qırız','Cimi','Xaşı','Qonaqkənd','Arsalan','Xan','Salmasöyüd']),
  ('shirvan','Şirvan','Şamaxı',['Şirvan','Qobustan','Şamaxı','İsrafil','Ərçiman']),
  ('shirvan','Şirvan','Mərəzə',['Mərəzə','Nabur','Çuxanlı','Cağırlı','Cəmcəmli']),
  ('shirvan','Şirvan','Ağsu',['Bico','Qəşəd','Pirhəsənli']),
  ('shirvan','Şirvan','Kürdəmir',['Kürdəmir','Şilyan','Sor-sor']),
  ('shirvan','Şirvan','Hacıqabul',['Hacıqabul','Şirəlibəy']),
  ('shirvan','Şirvan','(похожи на ширванские)',['Qəbələ','Salyan']),
  ('baku','Bakı','Abşeron',['Bakı','Xilə-buta','Xilə-əfşan','Suraxanı','Novxanı','Goradil','Qala','Fatmayı']),
  ('baku','Bakı','Xızı',['Fındıqan','Qədi']),
  ('ganja','Gəncə','Gəncə',['Gəncə','Qədim minarə','Samux']),
  ('ganja','Gəncə','Goranboy',['Çaylı','Şadılı','Faxralı']),
  ('ganja','Gəncə','Gədəbəy',['Gədəbəy','Çıraqlı','Qaraqoyunlu']),
  ('gazakh','Qazax','Qazax',['Qazax','Salahlı','Şıxlı','Kəmərli','Dəmirçilər','Qaymaqlı','Göyçəli','Dağkəsəmən','Öysüzlü']),
  ('gazakh','Qazax','Borçalı',['Borçalı','Qarayazı','Qaçağan','Qaraçöp']),
  ('karabakh','Qarabağ','Bərdə',['Bərdə','Xanqərvənd','Aran','Qoca','Buynuz','Dəryanur','Açma-yumma','Şabalıd-buta','Balıq']),
  ('karabakh','Qarabağ','Ağcabədi',['Ləmbəran','Qarabağ','Xantirmə']),
  ('karabakh','Şuşa','',['Malıbəyli','Ləmpə','Bağçadagüllər','Bulud','Saxsıdagüllər','Nəlbəkigül']),
  ('karabakh','Cəbrayıl','',['Xanlıq','Qaraqoyunlu','Qubadlı','Kürd','Qasımuşağı','Bəhmənli','Muğan','Talış']),
  ('tabriz','Təbriz','',['Təbriz','Bəxşayiş','Qaraca','Gəravan','Heris','Ləçək-turunc','Əfşan','Ağaclı','Ovçuluq','Dörd fəsil']),
  ('ardabil','Ərdəbil','',['Ərdəbil','Şeyx Səfi','Şah Abbası','Sərabi','Zəncan','Mir','Açma-yumma'])],
 'tech': {'guba-shirvan': {'area_m2': '1–4 (редко 10–15)', 'knots_per_m2': '160 000–300 000 (до 359 000)', 'pile_mm': '3–6'},
          'ganja-gazakh': {'area_m2': '3–10', 'knots_per_m2': '60 000–120 000', 'pile_mm': '6–12'},
          'karabakh': {'area_m2': '2–20 (до 25–30)', 'knots_per_m2': '90 000–160 000 (до 200 000)', 'pile_mm': '6–10'},
          'tabriz': {'area_m2': '1–40 (до 50)', 'knots_per_m2': '60 000–360 000 (редко 400 000)', 'pile_mm': '2–15'}},
 'notes': ['В азерб. тексте в подгруппе «Gəncə» стоит «Sumax» — по смыслу и по Керимову т. II это «Samux» (ошибка перевода/набора сборника); англ. страница с этой частью в скане не читается.',
           'В англ. тексте в подгруппе Quba есть «Golluchichi», в азерб. — нет (пропуск при переводе).',
           '«Qaraqoyunlu» назван дважды: при Гедабее (сёла Чайкенд и Гёлькенд, «в южном Иджеване») и в Джебраильской группе Карабаха.',
           '«Açma-yumma» назван дважды: Барда (Карабах) и Ардебильская группа.', '«Qədim minarə» назван дважды: Губа и Гянджа.',
           'Нахчыванские ковры по технике и рисунку отнесены к подгруппе Барда (Карабахский тип). Сёла Гусара (Хиль, Ясаб, Имамкуликенд, Зейхур): ковры «характерны для Дагестана».']}

M13 = {'id': 'muradov_2013', 'short': 'Мурадов 2013, «Azərbaycan xalça sənəti»',
 'where': 'intangible_2851: с. 28–56 (девять групп), 110–127 (характеристика группы + образец)', 'levels': 'девять групп (школ), без типов и подгрупп',
 'groups': [
  ('baku','Bakı','',['Novxanı','Suraxanı','Fatmayı','Goradil','Xilə-buta','Xilə-əfşan','Bakı','Qala','Fındıqan','Qədi']),
  ('guba','Quba','',['Qımıl','Qonaqkənd','Sırt-çiçi','Əjdahalı','Qədim minarə','Bilici','Qaraqaşlı','Mollakamallı','Pirəbədil','Təkyə','Uqah','Zeyvə','Şahnəzərli','Həsənqala','Afurca','Cek','Alpan','Cimi','Hacıqayıb','Hil','Yerfi','Xaşı']),
  ('shirvan','Şirvan','',['Şilyan','Kürdəmir','Sor-sor','Nabur','Mərəzə','Ərçiman','Qobustan','Şirvan','Məkkə-Mədinə','Qəşəd','Bico','Şamaxı','Pirhəsənli','Qəbələ','Salyan']),
  ('ganja','Gəncə','',['Qaraqoyunlu','Qaracəmirli','Faxralı','Topalhəsənli','Mollabayramlı','Samux','Çaylı','Tirməsayağı']),
  ('gazakh','Qazax-Borçalı','',['Muğanlı','Ulduzlu Qazax','Şıxlı','Daş Salahlı','Öysüzlü','Qaraçöp','Ləmbəli','Qarayazı','Borçalı','Faxralı','Qazax','Namazlıq','Qədirqə kilim','Qoçbuynuz kilim','Daraxlı kilim']),
  ('irevan','İrəvan','',['Ləmbəli']),
  ('karabakh','Qarabağ','',['Çələbi','Malıbəyli','Açma-yumma','Bəhmənli','Xanqərvənd','Xanlıq','Qasımuşağı','Qoca','Şabalıd-buta']),
  ('nakhchivan','Naxçıvan','',['Açma-yumma','Qoca','Buynuz','Dəryanur','Balıq','Şabalıd-buta','Namazlıq']),
  ('tabriz','Təbriz','',['Təbriz','Bəxşayiş','Qaraca','Gəravan','Heris','Ovçuluq','Ləçək-turunc','Əfşan']),
  ('ardabil','Təbriz / Ərdəbil','',['Ərdəbil','Şeyx Səfi','Sərabi','Zəncan'])],
 'tech': {'baku': {'density_text': '36×60 … 40×70 (с. 28)', 'density_spec': '38×38–45×45 на дм², 145–200 тыс./м², ворс 5–7 мм, 100×150–110×180 см (с. 111)'},
          'guba': {'density_text': '30×60, 35×55, 38×50, 40×45, 40×60, 35×70 (с. 32)'},
          'shirvan': {'density_text': 'ворсовые 35×60, 37×50, 40×55; безворсовые 22×60, 18×50, 15×60, 33×50, 30×55 (с. 34)'},
          'ganja': {'density_text': '25×40, 27×35, 32×30 (с. 37)', 'density_spec': '35×35–40×40 на дм², ворс 6–8 мм (с. 117)'},
          'gazakh': {'density_text': '20×35, 25×30, 27×32 (с. 40)', 'density_spec': '25×25–35×35, 60–120 тыс./м², ворс до 12 мм (с. 119)'},
          'irevan': {'density_text': '22×32, 25×35 (в OCR «25x55»), 29×30 (с. 44)'},
          'karabakh': {'density_text': '23×36, 27×30, 29×32; 90–160 тыс./м² (до 200 тыс.); ворс 6–10 мм; 2–20 м² (до 25–30) (с. 45)'},
          'nakhchivan': {'density_text': '25×40 … 27×36, 29×30; 87–100 тыс./м²; ворс 6–10 мм; 6–12 м² (с. 48, 50)', 'density_spec': '35×35 на дм², ворс 6–8 мм, 100×150–160×260 см (с. 125)'},
          'tabriz': {'density_text': 'от 30×30 до 80×80 (с. 56; в OCR «30x80»)'}},
 'notes': ['Иреван выделен отдельной (девятой) группой; сказано, что по рисунку он ближе всего к Газаху, Гяндже, Карабаху, Нахчывану, Восточной Анатолии и Тебризу (с. 44).',
           'Нахчыван выделен отдельной группой, но перечень его ковров (Açma-yumma, Qoca, Buynuz, Dəryanur, Balıq, Şabalıd buta) дословно совпадает с подгруппой «Барда» у Керимова 1984; сказано, что по технике он близок Карабаху (с. 48–49).',
           'Числа о Карабахе (90–160 тыс. узлов, ворс 6–10 мм, 2–20 м²) и перечень Тебриз/Ардебиль совпадают с Керимовым 1984: текст 2013 г. в этих местах не самостоятельное свидетельство.',
           '«Faxralı» назван и в Гянджинской (с. 37), и в Газах-Борчалинской группе (с. 40; образец на с. 119–120 подписан «Borçalı mahalı, Faxralı kəndi»): это два разных села — в Геранбое и в Борчалы.',
           '«Ləmbəli»: в тексте — Газах-Борчалы (с. 40), а подпись к фото на с. 24 — «İrəvan qrupu». Расхождение внутри одной книги.',
           'Плотность в одной и той же книге дана дважды и не сходится для Гянджи (25×40…32×30 против 35×35–40×40) и Нахчывана (25×40…29×30 против 35×35): в главах — «узлов на дм по ширине × по длине», в приложении — другие числа. Брать числа только с указанием страницы.',
           '«Çələbi» = «Bərdə»: старые ткачи называют ковёр «Барда», главный центр — село Челебиляр Бардинского района; с XIX в. ткали и в Газахе (с. 123).',
           '«Borçalı» в народе называют «Çanaqlı bağa» (по крупному гёлю); «Qədirqə kilim» в Газахе называют «Qara ağac» (с. 40–41).',
           'В губинском перечне (с. 32) шесть названий OCR не прочитал («Kukal», «huts», «Zeyid», «нах», «Tum km», «айс») — нужна сверка по скану.']}

def norm_school(p):
    s = p.get('school'); raw = s.get('raw') if isinstance(s, dict) else None
    if raw: return {'talysh': None, 'sheki': 'shirvan', 'shaki': 'shirvan'}.get(raw, raw)
    g = (p.get('group') or '').lower()
    for k, v in (('караб','karabakh'),('ширв','shirvan'),('губ','guba'),('куб','guba'),('баку','baku'),('гянд','ganja'),('газах','gazakh'),('казах','gazakh'),('тебр','tabriz'),('нахч','nakhchivan'),('ирев','irevan')):
        if k in g: return v
    return None

out = {'_about': 'Классификации ковров из текстов книг и их сверка с моделью и с подписями к фото. Составлено 2026-11-13 по распознанному тексту; человек не проверял. Правило 4: расхождения записаны, не решены.',
       'classifications': [], 'vs_model': [], 'new_names': [], 'captions_vs_lists': [], 'summary': {}}
where = collections.defaultdict(list)     # concept → [(src, school, group, sub)]
for S in (K84, M13):
    G = []
    for sch, grp, sub, names in S['groups']:
        row = {'school': sch, 'group_az': grp, 'subgroup_az': sub, 'names': []}
        for n in names:
            cid, how = match(n)
            if cid and CON[cid]['type'] not in ('composition', 'object') and 'composition' not in (CON[cid].get('roles') or {}) and 'object' not in (CON[cid].get('roles') or {}):
                how = 'не ковёр в модели: ' + CON[cid]['type']      # совпало со словом-элементом; оставляем, но помечаем
            row['names'].append({'name': n, 'concept': cid, 'match': how})
            if cid: where[cid].append((S['id'], sch, grp, sub))
            else: out['new_names'].append({'name': n, 'source': S['id'], 'school': sch, 'group': grp, 'subgroup': sub})
            if cid:
                k = CS.get(cid) or {}; t = TYPE_OF.get(sch)
                mg = k.get('group') or k.get('type')
                st = 'нет данных в модели' if not k.get('type') else ('совпадает' if (k.get('group') == sch or (not k.get('group') and k['type'] == t) or (sch in ('nakhchivan','ardabil') and k['type'] == t)) else ('тот же тип, другая группа' if k['type'] == t else 'расходится'))
                out['vs_model'].append({'concept': cid, 'name': n, 'source': S['id'], 'text_school': sch, 'text_group': grp + (' / ' + sub if sub else ''), 'model': {'type': k.get('type'), 'group': k.get('group'), 'confidence': k.get('confidence')}, 'status': st})
        G.append(row)
    out['classifications'].append({'id': S['id'], 'short': S['short'], 'where': S['where'], 'levels': S['levels'], 'groups': G, 'tech': S['tech'], 'notes': S['notes']})

# подписи к фото (все источники) против перечней
for f in sorted(glob.glob(REPO + '/site/model/sources/*/plates.json')):
    sid = f.split('/')[-2]
    for p in json.load(open(f))['plates']:
        cid = p.get('concept')
        if not cid:
            cid, _ = match(p.get('name') or p.get('type_name') or '')
        sch = norm_school(p)
        if not cid or not sch or cid not in where or p.get('modern'): continue
        lst = where[cid]; ok = any(s == sch for _, s, _, _ in lst)
        same_type = any(TYPE_OF.get(s) and TYPE_OF.get(s) == TYPE_OF.get(sch) for _, s, _, _ in lst)
        if cid in GENERIC and not ok: st = 'общее название (ткут в разных школах)'
        elif ok: st = 'совпадает'
        elif sch == 'irevan': st = 'подпись: Иреван (у Керимова такой группы нет)'
        elif same_type: st = 'тот же тип, другая группа'
        else: st = 'расходится'
        out['captions_vs_lists'].append({'plate': p.get('id') or f"{sid}:p{p.get('pdf_page')}", 'source': sid, 'page': p.get('pdf_page'), 'concept': cid, 'caption_name': p.get('name') or p.get('type_name'),
                                         'caption_school': sch, 'lists': sorted({f'{a}: {s}' for a, s, _, _ in lst}), 'status': st})
cnt = lambda rows: dict(collections.Counter(r['status'] for r in rows))
out['summary'] = {'names_in_lists': sum(len(g['names']) for c in out['classifications'] for g in c['groups']), 'vs_model': cnt(out['vs_model']), 'new_names': len(out['new_names']),
                  'captions_checked': len(out['captions_vs_lists']), 'captions': cnt(out['captions_vs_lists'])}
dst = REPO + '/site/model/sources/CLASSIFICATIONS_2026-11-13.json'
json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out['summary'], ensure_ascii=False, indent=1))
