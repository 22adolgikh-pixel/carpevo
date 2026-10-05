# build_concepts.py — машинная сборка модели «понятия» (вариант B, 2026-10-05) из данных сайта v44+.
#
# Вход:  data.json сайта (terms, sources, carriers, aliases) и выгрузка pixel_schemes (JSON-список документов).
# Выход (в out/): concepts.json, relations.json, term_map.json (старый id термина → id понятия),
#                 schemes_relink.json, review_queue.csv, report.md.
#
# Правила (всё, что машина делает сама, получает status "auto"; всё сомнительное — в очередь проверки):
#  1. Ключ записи — чтение по книге (book.reading, сверка 27.09) или canonical_az; строго по азербайджанским буквам
#     (ı ≠ i, q ≠ g, ə ≠ e — именно смешение ı/i привело к Baca→Bacı), без регистра, пробелов и дефисов.
#  2. Записи одного вида (узор / ковёр-композиция / изделие) с одинаковым ключом → одно понятие,
#     если значения не противоречат; если значения явно разные (омонимы: başlıq «заголовок» / «наконечник») → очередь.
#  3. Одинаковое значение при разных ключах → кандидат в синонимы (очередь), похожее написание → кандидат-вариант.
#  4. Составные: «X və Y» → consists_of X, Y; «X Y» → kind_of Y (главное слово последнее), related X.
#  5. Производные на -lı/-li/-lu/-lü/-lar/-lər → derived_from основы.
#  6. Ковёр/композиция с тем же ключом, что узор → named_after.
#  7. Схемы перепривязываются к понятиям через term_map; двойные привязки к дублям схлопываются;
#     имя схемы, не совпадающее ни с одним названием понятия, → очередь.
import json, re, sys, csv, os, unicodedata, collections as C
from difflib import SequenceMatcher

AZ_LAT = str.maketrans({'ə': 'e', 'ı': 'i', 'ö': 'o', 'ü': 'u', 'ğ': 'g', 'ş': 'sh', 'ç': 'ch'})


def low(s):
    s = unicodedata.normalize('NFC', s or '').strip()
    s = s.replace('I', 'ı').replace('İ', 'i')            # азербайджанская (турецкая) пара I/ı, İ/i
    return s.lower()


def key(s):
    """Строгий ключ: только буквы, азербайджанские буквы сохраняются."""
    s = low(s).split(' / ')[0].split(' or ')[0]
    return re.sub(r'[^0-9a-zəıöüğşçâ]', '', s)


def words(s):
    s = low(s).split(' / ')[0]
    return [w for w in re.split(r'[^0-9a-zəıöüğşçâ]+', s) if w]


def ascii_id(s):
    return re.sub(r'[^a-z0-9]+', '-', low(s).translate(AZ_LAT)).strip('-') or 'x'


def mtoks(m):
    m = low(m)
    return {w for w in re.split(r'[^a-zа-яё0-9]+', m) if len(w) > 2}


def tmean(t):
    """Значения записи: перевод Керимова по сверке (надёжнее) + ru_meaning, если запись не помечена как неверно переведённая."""
    b = t.get('book') or {}
    ms = mtoks(b.get('kerimov_translation') or '')
    st = b.get('status') or {}
    if not (st.get('wrong_translation') or st.get('wrong_name')) or not ms:
        ms |= mtoks(t.get('ru_meaning') or '')
    return ms


def agree(a, b):
    """Значения согласны: общее слово или общая основа (4 буквы); пустое значение не спорит."""
    if not a or not b: return True
    return bool(a & b) or any(x[:4] == y[:4] for x in a for y in b)


def ru_meaning(p):
    """Значение: если сверка отметила неверный перевод — берём перевод Керимова, иначе ru_meaning."""
    for t in p:
        b = t.get('book') or {}
        if (b.get('status') or {}).get('wrong_translation') and b.get('kerimov_translation'):
            return b['kerimov_translation']
    return next((t['ru_meaning'] for t in p if t.get('ru_meaning')), None) or \
        next(((t.get('book') or {}).get('kerimov_translation') for t in p if (t.get('book') or {}).get('kerimov_translation')), None)


# Легенда т. I разобрана по колонкам с ошибкой: двухсловное название «Ara | хашийе Серединная кайма …» разрезано —
# второе слово названия (аз. кириллица) попало в начало перевода. Склеиваем обратно.
CONT = {'хашийе': 'haşiyə', 'хашиjе': 'haşiyə', 'су': 'su', 'нахышы': 'naxışı', 'гюлю': 'gülü', 'золаглар': 'zolaqlar', 'бута': 'buta'}


def full_scheme_name(s):
    nm = (s.get('name') or '').split('|')[0].strip()
    tr = (s.get('translation') or '').strip()
    if nm and tr:
        w = tr.split()[0].lower()
        if w in CONT and (('|' in (s.get('name') or '')) or tr.split()[0][0].islower()):
            return nm + ' ' + CONT[w]
    return nm


ETYPE = {'ornament': 'ornament', 'carpet': 'composition', 'object': 'object'}
SUFFIX = ('lı', 'li', 'lu', 'lü', 'lar', 'lər')
POSS = (('si', ''), ('sı', ''), ('su', ''), ('sü', ''), ('yi', 'k'), ('yı', 'q'), ('ği', 'k'), ('ğı', 'q'))


def main(data_path, schemes_path, out):
    os.makedirs(out, exist_ok=True)
    D = json.load(open(data_path))
    T = D['terms']
    S = [s for s in json.load(open(schemes_path)) if s.get('id')]
    queue = []   # (тип, понятия/записи, что не так, предложение)

    # ---- 1–2. группировка записей в понятия ----
    groups = C.defaultdict(list)
    for t in T:
        rd = (t.get('book') or {}).get('reading')
        k = key(rd) or key(t.get('canonical_az')) or key(t['id'])
        groups[(ETYPE.get(t['kind'], t['kind']), k)].append(t)

    concepts, term_map = {}, {}
    used_ids = set()
    for (et, k), ts in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        parts, rules = [ts], {}
        if len(ts) > 1:
            # ядро — записи, у которых собственное написание совпадает с чтением (или отличается опечаткой);
            # «переименованные» сверкой (canonical далёк от чтения) присоединяются, только если значение согласно
            sim = lambda t: SequenceMatcher(None, key(t.get('canonical_az')), k).ratio()
            core = [t for t in ts if sim(t) >= 0.8] or [max(ts, key=sim)]
            cm = set().union(*[tmean(t) for t in core])
            parts, extra = [list(core)], []
            for t in ts:
                if t in core: continue
                if agree(tmean(t), cm): parts[0].append(t); rules[t['id']] = 'renamed_by_reading'
                else: extra.append(t)
            for t in extra:
                parts.append([t])
            if extra:
                queue.append(('омоним?', ' | '.join(f"{t['id']} «{t.get('canonical_az')}» «{(t.get('book') or {}).get('kerimov_translation') or t.get('ru_meaning') or '—'}»" for t in ts),
                              f'по сверке все читаются как «{k}», но значения расходятся', 'слить или оставить разными понятиями'))
            core_ms = [tmean(t) for t in core]
            if len(core) > 1 and any(not agree(a, b) for i, a in enumerate(core_ms) for b in core_ms[i + 1:]):
                queue.append(('слито, значения расходятся', ' | '.join(f"{t['id']} «{t.get('ru_meaning') or '—'}»" for t in core),
                              'одно написание — слито автоматически', 'проверить: одно понятие или омонимы'))
        for p in parts:
            main_t = sorted(p, key=lambda t: (key(t.get('canonical_az')) != k, bool(re.search(r'-\d+$', t['id'])),
                                              -(((t.get('book') or {}).get('status') or {}).get('ok', 0)), -t['mention_count'], len(t['id'])))[0]
            rd = (main_t.get('book') or {}).get('reading')
            head = (rd.split(' / ')[0] if rd and '?' not in rd else None) or main_t.get('canonical_az') or main_t['id']
            cid = main_t['id'] if et == 'ornament' or main_t['id'].endswith(('-carpet', '-object')) else main_t['id']
            while cid in used_ids: cid += '-x'
            used_ids.add(cid)
            names, roles, att, schools, srcs, meanings = [], C.Counter(), [], C.Counter(), C.Counter(), []

            def addn(v, lang, kind, src=None):
                v = (v or '').strip()
                if v and not any(n['v'].lower() == v.lower() and n['lang'] == lang for n in names):
                    names.append({'v': v, 'lang': lang, 'kind': kind, **({'src': src} if src else {})})
            addn(head, 'az', 'headword')
            for t in p:
                b = t.get('book') or {}
                if b.get('reading'):
                    for r in b['reading'].split(' / '): addn(r, 'az', 'book_reading', 'kerimov')
                st = b.get('status') or {}
                addn(t.get('canonical_az'), 'az', 'misreading' if t['id'] in rules and all(k.startswith('kerimov') for k in (t.get('source_counts') or {})) else 'record')   # ошибочное чтение: только для поиска/редиректа
                for v in t.get('canonical_az_variants') or []: addn(v, 'az', 'variant')
                for v in (t.get('names') or {}).get('az') or []: addn(v, 'az-cyrl' if re.search('[а-яәөүҹһғҝ]', v.lower()) else 'az', 'attested')
                for v in (t.get('names') or {}).get('ru') or []: addn(v, 'ru', 'attested')
                if b.get('kerimov_translation'): addn(b['kerimov_translation'], 'ru', 'kerimov_translation', 'kerimov')
                for lang in ('ru', 'en'):
                    m = t.get(f'{lang}_meaning')
                    if m and m not in meanings: meanings.append(m)
                    if m: addn(m, lang, 'meaning_rejected' if (st.get('wrong_translation') and b.get('kerimov_translation')) or t['id'] in rules else 'meaning')
                roles.update(b.get('roles') or {})
                if not b.get('roles'):
                    for tab in t.get('tabs') or []: roles[{'motifs': 'motif', 'elements': 'element', 'borders': 'border', 'carpets': 'composition'}[tab]] += 0
                schools.update(s for s in t.get('schools_mentioned_canonical') or [] if s in ('guba-shirvan', 'karabakh', 'ganja-gazakh', 'tabriz', 'baku', 'shirvan', 'guba', 'nakhchivan', 'gazakh'))
                srcs.update(t.get('source_counts') or {})
                for m in t.get('mentions') or []:
                    att.append({'term': t['id'], 'source_id': m.get('source_id'), 'where': m.get('source'),
                                'page': m.get('page_printed') or m.get('pdf_page'), 'role': m.get('role'),
                                'quote': m.get('quote'), 'note': m.get('origin_notes')})
                term_map[t['id']] = cid
                for o in t.get('old_ids') or []: term_map.setdefault(o, cid)
            ext = [r for t in p for r in (t.get('external_references') or [])]
            concepts[cid] = {
                'id': cid, 'type': et, 'key': k, 'headword': head,
                'meaning': {'ru': ru_meaning(p),
                            'en': next((t['en_meaning'] for t in p if t.get('en_meaning')), None)},
                'names': names, 'roles': dict(roles), 'schools': dict(schools), 'sources': dict(srcs),
                'attestations': att, 'records': [t['id'] for t in p], 'external_references': ext,
                'merge': {'status': 'auto' if len(p) > 1 else 'single',
                          'rule': ('renamed_by_reading' if any(t['id'] in rules for t in p) else 'same_word') if len(p) > 1 else None,
                          'renamed': [t['id'] for t in p if t['id'] in rules]},
            }
    for c in concepts.values():
        if c['merge']['renamed']:
            queue.append(('слито по чтению Керимова', f"{c['id']} «{c['headword']}» ← " + ', '.join(c['records']),
                          'записи ' + ', '.join(c['merge']['renamed']) + ' сверка прочла как это слово', 'выборочно проверить (низкий приоритет)'))
    for a, v in (D.get('aliases') or {}).items():
        if v in term_map: term_map.setdefault(a, term_map[v])

    # индексы
    by_key = C.defaultdict(list)
    for c in concepts.values(): by_key[(c['type'], c['key'])].append(c['id'])
    orn_key = {c['key']: c['id'] for c in concepts.values() if c['type'] == 'ornament' and len(by_key[('ornament', c['key'])]) == 1}
    word_key = {}
    for c in concepts.values():
        if c['type'] == 'ornament':
            ws = words(c['headword'])
            if len(ws) == 1: word_key.setdefault(ws[0], c['id'])

    rel = []
    def R(a, t, b, why, status='auto'):
        if a != b and not any(r['from'] == a and r['to'] == b and r['type'] == t for r in rel):
            rel.append({'from': a, 'type': t, 'to': b, 'status': status, 'why': why})

    def find_word(w):
        if w in word_key: return word_key[w]
        for suf, rep in POSS:                     # haşiyəsi → haşiyə, çiçəyi → çiçək
            if w.endswith(suf) and len(w) > len(suf) + 2:
                b = w[:-len(suf)] + rep
                if b in word_key: return word_key[b]
        return None

    # ---- 4. составные ----
    for c in concepts.values():
        if c['type'] != 'ornament': continue
        ws = [w for w in words(c['headword'])]
        if len(ws) < 2: continue
        raw = low(c['headword'])
        if re.search(r'\b(və ya|yaxud|or)\b|/', raw):
            # «X və ya Y», «X or Y» — это два названия, а не составной узор
            for w in re.split(r'\s+(?:və ya|yaxud|or)\s+|\s*/\s*', raw):
                x = find_word(key(w)) if len(words(w)) == 1 else None
                if x: R(c['id'], 'alt_name_of', x, f'«{c["headword"]}»: «или» — другое название', 'proposed')
            continue
        if 'və' in ws or 'ilə' in ws:
            for w in ws:
                if w in ('və', 'ilə'): continue
                x = find_word(w)
                if x: R(c['id'], 'consists_of', x, f'«{c["headword"]}»: часть «{w}»')
        else:
            hx = find_word(ws[-1])
            if hx: R(c['id'], 'kind_of', hx, f'«{c["headword"]}»: главное слово «{ws[-1]}»')
            for w in ws[:-1]:
                x = find_word(w)
                if x: R(c['id'], 'related', x, f'«{c["headword"]}»: содержит «{w}»')

    # ---- 5. производные ----
    for c in concepts.values():
        if c['type'] != 'ornament': continue
        ws = words(c['headword'])
        if len(ws) != 1: continue
        w = ws[0]
        for suf in SUFFIX:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                b = w[:-len(suf)]
                if b in word_key and word_key[b] != c['id']:
                    R(c['id'], 'derived_from', word_key[b], f'{w} = {b} + -{suf}'); break

    # ---- 6. композиции, названные по узору ----
    for c in concepts.values():
        if c['type'] in ('composition', 'object') and c['key'] in orn_key:
            R(c['id'], 'named_after', orn_key[c['key']], 'то же слово')

    # ---- 3. кандидаты в синонимы / варианты написания ----
    bym = C.defaultdict(list)
    for c in concepts.values():
        if c['type'] == 'ornament' and c['meaning']['ru']:
            bym[re.sub(r'\s+', ' ', low(c['meaning']['ru']))].append(c['id'])
    for m, ids in bym.items():
        if len(ids) < 2: continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = concepts[ids[i]], concepts[ids[j]]
                sim = SequenceMatcher(None, a['key'], b['key']).ratio()
                t = 'spelling_variant' if sim >= 0.75 else 'synonym'
                R(a['id'], t, b['id'], f'одно значение «{m}», сходство написания {sim:.2f}', 'proposed')
                queue.append(({'spelling_variant': 'вариант написания?', 'synonym': 'синоним?'}[t], f"{a['id']} «{a['headword']}» ↔ {b['id']} «{b['headword']}»", f'значение «{m}»',
                              'слить в одно понятие' if t == 'spelling_variant' else 'синонимы: слить или связать'))

    # ---- 7. схемы ----
    relink, sch_by_c = [], C.defaultdict(list)
    for s in S:
        old = s.get('site_ids') or []
        new = []
        for i in old:
            cid = term_map.get(i)
            if cid and cid not in new: new.append(cid)
        ok_name = True
        if new and s.get('name'):
            sk = key(s['name'])
            # для ПРОВЕРКИ уже существующей привязки сравниваем мягко (ı/i, ə/e, x/q сведены): имена схем — старая транслитерация
            fold = lambda z: z.translate(str.maketrans('əıöüğşçxq', 'eiougscgg'))
            ok_name = any(SequenceMatcher(None, fold(sk), fold(key(n['v']))).ratio() >= 0.6
                          for cid in new for n in concepts[cid]['names'] if n['lang'] in ('az', 'az-cyrl'))
            if not ok_name:
                queue.append(('схема: имя ≠ понятие', f"{s['id']} «{s['name']}» (табл. {s.get('table')}, рис. {s.get('fig')})",
                              'привязана к ' + ', '.join(f"{c} «{concepts[c]['headword']}»" for c in new), 'проверить привязку'))
        if not new and s.get('name'):
            k = key(full_scheme_name(s))
            cand = orn_key.get(k)
            if cand:
                new = [cand]
                relink.append({'scheme': s['id'], 'old': old, 'new': new, 'why': 'имя схемы = понятие (строго)', 'status': 'auto'})
        if new != old and not any(r['scheme'] == s['id'] for r in relink):
            relink.append({'scheme': s['id'], 'old': old, 'new': new, 'why': 'слияние дублей' if len(new) < len(old) else 'новый id понятия', 'status': 'auto'})
        for cid in new: sch_by_c[cid].append(s['id'])
    for cid, l in sch_by_c.items(): concepts[cid]['schemes'] = l

    # ---- 8. непривязанные схемы с именем: предложить понятие по похожему написанию + согласному значению ----
    fold = lambda z: z.translate(str.maketrans('əıöüğşçxq', 'eiougscgg'))
    nidx = [(fold(key(n['v'])), c['id']) for c in concepts.values() if c['type'] == 'ornament'
            for n in c['names'] if n['lang'] in ('az', 'az-cyrl') and n['kind'] not in ('misreading',)]
    linked = {sid for l in sch_by_c.values() for sid in l}
    proposals = []
    for s in S:
        if s['id'] in linked or not (s.get('name') or '').strip() or '?' in s['name']: continue
        nm = s['name'].split('|')[0]
        k = fold(key(nm))
        if len(k) < 3: continue
        best = max(((SequenceMatcher(None, k, n).ratio(), cid) for n, cid in nidx if n), default=(0, None))
        r, cid = best
        if not cid or r < 0.8: continue
        tm = mtoks(s.get('translation') or '')
        cm = mtoks(concepts[cid]['meaning'].get('ru') or '') | {w for n in concepts[cid]['names'] if n['lang'] == 'ru' for w in mtoks(n['v'])}
        ok = bool(tm) and bool(cm) and agree(tm, cm)
        if ok or r >= 0.92:
            proposals.append({'scheme': s['id'], 'concept': cid, 'ratio': round(r, 2), 'meaning_agrees': ok})
            R2 = 'имя похоже ({:.2f}){}'.format(r, ', значение согласно' if ok else ', значение не сверено')
            queue.append(('схема → понятие?', f"{s['id']} «{s['name']}» «{s.get('translation') or '—'}» → {cid} «{concepts[cid]['headword']}» «{concepts[cid]['meaning'].get('ru') or '—'}»",
                          R2, 'привязать' if ok else 'проверить'))

    # ---- запись ----
    J = lambda n, o: json.dump(o, open(os.path.join(out, n), 'w'), ensure_ascii=False, indent=1)
    J('scheme_link_proposals.json', proposals)
    J('concepts.json', list(concepts.values())); J('relations.json', rel); J('term_map.json', term_map); J('schemes_relink.json', relink)
    with open(os.path.join(out, 'review_queue.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['тип', 'что', 'почему', 'предложение', 'решение'])
        for q in queue: w.writerow(list(q) + [''])

    # отчёт
    ct = C.Counter(c['type'] for c in concepts.values())
    merged = [c for c in concepts.values() if len(c['records']) > 1]
    rt = C.Counter((r['type'], r['status']) for r in rel)
    qt = C.Counter(q[0] for q in queue)
    with_s = sum(1 for c in concepts.values() if c.get('schemes'))
    L = [f'# Сборка понятий — отчёт', '',
         f'Записей-терминов: {len(T)} → понятий: {len(concepts)} (узоры {ct["ornament"]}, композиции/ковры {ct["composition"]}, изделия {ct["object"]}).',
         f'Слито автоматически (одно чтение по книге): {len(merged)} понятий из {sum(len(c["records"]) for c in merged)} записей.', '',
         '## Связи', '| тип | auto | предложено |', '|---|---|---|']
    for t in sorted({t for t, _ in rt}):
        L.append(f'| {t} | {rt[(t, "auto")]} | {rt[(t, "proposed")]} |')
    L += ['', '## Очередь проверки', '| тип | сколько |', '|---|---|'] + [f'| {t} | {n} |' for t, n in qt.most_common()]
    L += ['', f'## Схемы', f'Схем: {len(S)}; привязаны к понятиям: {sum(1 for s in S if any(r["scheme"] == s["id"] and r["new"] for r in relink) or (s.get("site_ids") and not any(r["scheme"] == s["id"] for r in relink)))}; понятий со схемой: {with_s}.',
          f'Изменённых привязок: {len(relink)} (схлопнуты дубли: {sum(1 for r in relink if r["why"] == "слияние дублей")}, новые по имени: {sum(1 for r in relink if r["why"].startswith("имя"))}).', '',
          '## Примеры слияний']
    for c in sorted(merged, key=lambda c: -len(c['records']))[:25]:
        L.append(f'- **{c["headword"]}** «{c["meaning"]["ru"] or "—"}» ← ' + ', '.join(c['records']))
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:30]))


if __name__ == '__main__':
    main(*sys.argv[1:4])
