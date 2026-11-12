"""Словник энциклопедии т. II из полного OCR (ocr_pages.json) → site/model/sources/tagiyeva_enc_2015_v2/glossary_entries.json
и восстановление типа/школы в подписях (enc_parsed.json), потерянных при первом OCR."""
import os, sys, json, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_enc import fold, NAME, REPO, WORK, parse
P = json.load(open(os.path.join(WORK, 'ocr_pages.json')))
HDR = re.compile(r'(?m)^([A-ZƏÖÜÇŞĞİI][A-ZƏÖÜÇŞĞİI \-,“”"]{2,50}?)\s*(\([^)\n]{0,80}\))?\s*[-–—]\s+(?=[a-zəöüçşğıA-ZƏÖÜÇŞĞİ(“"])')
SEC = [(13, 156, 'безворсовые ковры'), (157, 234, 'ковровые изделия'), (235, 315, 'каймы и орнаменты'), (316, 329, 'авторские ковры'),
       (330, 405, 'сырьё, красители, техника')]
full = ''; offs = []
for p in P:
    if p['page'] < 13 or p['page'] > 405: continue
    t = re.sub(r'(?m)^\s*(\d{1,3}|AZƏRBAYCAN XALÇASI ENSİKLOPEDİYASI)\s*$', '', p['text'])
    offs.append((len(full), p['page'])); full += t + '\n'
def page_at(i): return max(pg for o, pg in offs if o <= i)
ms = [m for m in HDR.finditer(full) if 'ENSİKLOPEDİYASI' not in m.group(1)]
out = []
for k, m in enumerate(ms):
    end = ms[k + 1].start() if k + 1 < len(ms) else len(full)
    txt = re.sub(r'-\n\s*', '', full[m.end():end]); txt = re.sub(r'\s+', ' ', txt).strip()
    # подписи к фото внутри статьи убрать
    txt = re.sub(r'[^.]*\bMaterial:[^.]*\.\s*Ölçü[^.]*\.(?:\s*Azərbaycan Xalça Muzeyi\.)?(?:\s*DDK\s*\d+\.?)?', ' ', txt).strip()
    head = re.sub(r'\s+', ' ', m.group(1)).strip(' ,'); pg = page_at(m.start())
    see = re.findall(r'Bax\s+([A-ZƏÖÜÇŞĞİ][\wəöüçşğı\- ]{2,30})', txt)
    cid = NAME.get(fold(head)); prop = None
    if not cid:
        import difflib
        mm = difflib.get_close_matches(fold(head), list(NAME), 1, 0.92)
        if mm and len(fold(head)) >= 4: prop = NAME[mm[0]]
    out.append({'term': head.capitalize() if False else head, 'alt': (m.group(2) or '').strip('()') or None, 'page': pg,
                'section': next((s for a, b, s in SEC if a <= pg <= b), None), 'concept': cid, 'concept_proposed': prop, 'new_candidate': not (cid or prop),
                'text_az': txt[:4000], 'chars': len(txt), 'see': see or None})
d = os.path.join(REPO, 'site/model/sources/tagiyeva_enc_2015_v2')
json.dump({'_about': 'Словарные статьи энциклопедии т. II (azərb., OCR tesseract 250 dpi, человеком не вычитано). concept — понятие модели по точному совпадению '
                     'заголовка; new_candidate — понятия нет. Текст — сырьё для описаний (site/model/descriptions), сам на сайт не выкладывается.',
           'entries': out}, open(f'{d}/glossary_entries.json', 'w'), ensure_ascii=False, indent=1)
print('статей', len(out), '| связано', sum(1 for e in out if e['concept']), '| предложено', sum(1 for e in out if e['concept_proposed']), '| новых', sum(e['new_candidate'] for e in out))
print('предложено:', [(e['term'], e['concept_proposed']) for e in out if e['concept_proposed']])
print('короткие (подозрительные):', [e['term'] for e in out if e['chars'] < 40][:40])
print(collections.Counter(e['section'] for e in out))
print('длинные:', [(e['term'], e['chars']) for e in sorted(out, key=lambda e: -e['chars'])[:25]])
print('связаны:', [e['term'] for e in out if e['concept']][:80])
# --- восстановление подписей по полному OCR страницы
enc = json.load(open(os.path.join(WORK, 'enc_parsed.json'))); T = {p['page']: p['text'] for p in P}; fixed = collections.Counter()
for o in enc:
    if o['kind'] != 'item' or (o.get('obj') and o.get('school')): continue
    key = (o.get('inv') or '').replace('DDK ', '')
    size = 'x'.join(str(int(v)) for v in o['size_cm'][:2]) if o.get('size_cm') else None
    t = re.sub(r'-\n', '', T.get(o['page'], '')); t1 = re.sub(r'\s+', ' ', t)
    pos = [m.start() for m in re.finditer(r'DDK\s*%s\b' % re.escape(key), t1)] if key else []
    if not pos and size: pos = [m.end() for m in re.finditer(re.escape(size), t1)]
    if len(pos) != 1: continue
    seg = t1[max(0, pos[0] - 230):pos[0] + 12]
    cut = [m.end() for m in re.finditer(r'DDK\s*\d+\.?|Muzeyi\.\s(?!DDK)', seg[:-14])]
    seg = seg[cut[-1]:] if cut else seg
    p2 = parse(seg)
    for k in ('obj', 'school', 'school_ru', 'name', 'author', 'year'):
        if p2.get(k) and not o.get(k): o[k] = p2[k]; fixed[k] += 1
    if p2.get('obj') or p2.get('school'): o['caption_full'] = seg.strip()
    if o.get('name') and not o.get('concept'): o['concept'] = NAME.get(fold(o['name']))
    if o.get('obj') and not o.get('concept_obj'): o['concept_obj'] = NAME.get(fold(o['obj']))
json.dump({o['id']: {k: o.get(k) for k in ('obj', 'school', 'school_ru', 'name', 'author', 'year', 'concept', 'concept_obj', 'caption_full') if o.get(k)} for o in enc if o.get('caption_full')},
          open(os.path.join(WORK, 'enc_fix.json'), 'w'), ensure_ascii=False, indent=0)   # parse_enc при импорте пересоздаёт enc_parsed.json, поэтому правки отдельным файлом
it = [o for o in enc if o['kind'] == 'item']
print('восстановлено', dict(fixed), '| без типа', sum(1 for o in it if not o.get('obj')), '| без школы', sum(1 for o in it if not o.get('school')), 'из', len(it))
