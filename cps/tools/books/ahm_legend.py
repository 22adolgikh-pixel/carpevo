"""Полный перечень предметов альбома Музея истории Азербайджана (файл 2885) из английской легенды (с. 21 файла): 75 номеров.
Раньше в источнике было только 6 фото с прочитанной подписью. Запуск: CORPUS=… python3 ahm_legend.py → site/model/sources/ahm_kovry/legend.json"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import importlib.util
sp = importlib.util.spec_from_file_location('cc', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus_classif.py'))
src = open(sp.origin, encoding='utf-8').read().split("\nK84 = ")[0]; ns = {'__file__': sp.origin}; exec(src, ns)      # только fold/match/CS из соседнего скрипта
match, CS, CON = ns['match'], ns['CS'], ns['CON']
L = {json.loads(l)['page']: ' '.join(json.loads(l)['text'].split()) for l in open(os.path.join(os.environ.get('CORPUS', '.'), 'intangible_2885_text.jsonl'), encoding='utf-8')}
t = L[21].split('LEGENDS', 1)[1].replace('­ ', '').replace('­', '')
t = re.sub(r'(?<=\w)- (?=\w)', '', t)
parts = re.split(r'(?:(?<=\s)|^)(\d{1,2})\. (?=[A-Z])', ' ' + t)
EN = {'bidjo': 'Bico', 'kazakh': 'Qazax', 'khanlyg': 'Xanlıq', 'golluchikhi': 'Qollu-çiçi', 'gollu chichi': 'Qollu-çiçi', 'kobystan': 'Qobustan', 'shykhly': 'Şıxlı', 'shilyan': 'Şilyan',
      'shakhnazerli': 'Şahnəzərli', 'shakhnezerli': 'Şahnəzərli', 'nakhchyvan': 'Naxçıvan', 'sirtchichi': 'Sırt-çiçi', 'ovchulug': 'Ovçuluq', 'gyryz': 'Qırız', 'gashed': 'Qəşəd',
      'seniye': 'Sənnə', 'garabag': 'Qarabağ', 'khila-buta': 'Xilə-buta', 'salmasoyud': 'Salmasöyüd', 'three brothers': None, 'khyrda gyul': 'Xırdagül', 'molla kamally': 'Mollakamallı',
      'geratpirebedil': 'Herat-Pirəbədil', 'shirvan': 'Şirvan', 'sattar-khan': None, 'khilafshan': 'Xilə-əfşan', 'khan cheshnya': 'Xançeşni', 'zili': 'Zili'}
SCH = {'shirvan': 'shirvan', 'kazakh': 'gazakh', 'karabakh': 'karabakh', 'kuba': 'guba', 'tebriz': 'tabriz', 'baku': 'baku', 'iran': None, 'southern azerbaijan': 'tabriz'}
TYPE_OF = {'guba': 'guba-shirvan', 'shirvan': 'guba-shirvan', 'baku': 'guba-shirvan', 'gazakh': 'ganja-gazakh', 'karabakh': 'karabakh', 'tabriz': 'tabriz'}
KW = [('travelling bag', 'перемётная сума'), ('carpet bag', 'ковровая сумка (чанта)'), ('mufrash', 'мафраш'), ('horsecloth', 'попона'), ('coverlet', 'попона'), ('curtain', 'занавес'), ('runner', 'дорожка'),
      ('ritual rug', 'молитвенный ковёр (намазлыг)'), ('pileless', 'безворсовый ковёр'), ('pictorial', 'сюжетный ковёр'), ('piled carpet', 'ворсовый ковёр')]
items = []
for i in range(1, len(parts) - 1, 2):
    no = int(parts[i]); s = re.sub(r'\s+', ' ', parts[i + 1]).strip()
    low = re.sub(r'(?<=\b[a-z]) (?=[a-z]{3,})', '', s.lower())      # «P ictorial», «S irtchichi»: разрыв после первой буквы
    low = low.replace('p ictorial', 'pictorial').replace('s irtchichi', 'sirtchichi').replace('r itual', 'ritual').replace('z ili', 'zili')
    cen = re.search(r'(\d{2})th century', low); tech = re.search(r'\((zili|kilim|verni|piled) ?(?:technique)?\)', low)
    sch_raw = next((k for k in SCH if re.search(r'\b' + k + r'\.?$', low.rstrip('. ')) or (k + ', southern') in low), None)
    if 'southern azerbaijan' in low: sch_raw = 'southern azerbaijan' if 'tebriz' not in low else 'tebriz'
    name = next((v for k, v in EN.items() if re.search(r'(^|part of |\. )' + re.escape(k) + r' (piled|pictorial|pileless)', low)), None)
    kind = next((ru for k, ru in KW if k in low), 'ковёр')
    cid, how = match(name) if name else (None, None)
    sch = SCH.get(sch_raw) if sch_raw else None
    k = CS.get(cid or '') or {}
    st = None
    if cid and sch and k.get('type'):
        st = 'совпадает' if (k.get('group') == sch or (not k.get('group') and k['type'] == TYPE_OF[sch])) else ('тот же тип, другая группа' if k['type'] == TYPE_OF[sch] else 'расходится')
    items.append({'no': no, 'legend_en': s, 'kind': kind, 'fragment': low.startswith('part of'), 'name': name, 'concept': cid, 'century': ({'18': 'XVIII', '19': 'XIX', '20': 'XX'}[cen.group(1)] + ' в.') if cen else None,
                  'technique': tech.group(1) if tech else ('piled' if 'piled' in low else None), 'school': sch, 'school_raw': sch_raw, 'vs_model': st})
out = {'_about': 'Полная легенда альбома «Ковры и ковровые изделия» (Музей истории Азербайджана АН АзССР): 75 предметов по английскому списку на с. 21 файла 2885. Текст — М. Кулиева, фото — Г. Гусейнзаде, общая ред. — П. А. Азизбекова. Инвентарных номеров и размеров в альбоме нет. Разбор автоматический, человек не проверял.',
       'items': items}
json.dump(out, open(REPO + '/site/model/sources/ahm_kovry/legend.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import collections
print(len(items), 'с названием', sum(1 for x in items if x['name']), 'с понятием', sum(1 for x in items if x['concept']), 'школа', sum(1 for x in items if x['school']), collections.Counter(x['vs_model'] for x in items))
for x in items:
    if x['vs_model'] not in (None, 'совпадает') or (x['name'] and not x['concept']) or not x['kind']: print(x['no'], x['legend_en'], '|', x['name'], x['concept'], x['school'], x['vs_model'])
print([x['no'] for x in items] == list(range(1, 76)))
