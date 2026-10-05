# apply_decisions.py — применить решения по очереди проверки к результату build_concepts.py.
# python3 apply_decisions.py DECISIONS.json OUT_DIR
# Решения (decisions_*.json) приняты по данным сверки с Керимовым и тексту книги (ИИ, со ссылками), статус «checked_ai»:
#   merge — слить понятия пункта в primary; separate — снять предложенную связь; related — связь relation (checked_ai);
#   attach/reattach — схема → target; detach — отвязать; keep — слияние подтверждено; unsure — остаётся в очереди;
#   set_meaning — заменить значение понятия target на meaning {ru, en} (по легенде/переводу Керимова).
# Несколько файлов решений применяются по очереди: python3 apply_decisions.py A.json out && python3 apply_decisions.py B.json out
# Пишет обратно concepts.json, relations.json, term_map.json, review_queue.csv (+ колонки «решение», «обоснование»), decisions_log.json.
import json, sys, os, csv, collections as C


def main(dec_path, out):
    L = lambda n: json.load(open(os.path.join(out, n)))
    cs = {c['id']: c for c in L('concepts.json')}
    rel = L('relations.json'); tm = L('term_map.json')
    decs = json.load(open(dec_path))
    alias = {}                                   # слитое id → primary

    def res(x):
        while x in alias: x = alias[x]
        return x

    log = C.Counter()
    for d in decs:
        k = d['decision']
        ids = [res(x) for x in d.get('concepts') or [] if res(x) in cs]
        if k == 'merge' and d.get('primary'):
            p = res(d['primary'])
            if p not in cs: log['merge_missing_primary'] += 1; continue
            for x in ids:
                if x == p or x not in cs: continue
                a, b = cs[p], cs.pop(x)
                for n in b['names']:
                    if not any(n['v'].lower() == m['v'].lower() and n['lang'] == m['lang'] for m in a['names']):
                        a['names'].append(n)
                a['attestations'] += b['attestations']; a['records'] += b['records']
                for f in ('roles', 'schools', 'sources'):
                    for kk, v in (b.get(f) or {}).items(): a[f][kk] = a[f].get(kk, 0) + v
                a['schemes'] = list(dict.fromkeys((a.get('schemes') or []) + (b.get('schemes') or [])))
                a['external_references'] = (a.get('external_references') or []) + (b.get('external_references') or [])
                if not a['meaning'].get('ru'): a['meaning'] = b['meaning']
                alias[x] = p
            cs[p]['merge'] = {**cs[p].get('merge', {}), 'status': 'checked_ai', 'decision_n': d['n']}
            log['merge'] += 1
        elif k == 'keep':
            for x in ids: cs[x]['merge'] = {**cs[x].get('merge', {}), 'status': 'checked_ai', 'decision_n': d['n']}
            log['keep'] += 1
        elif k == 'related' and len(ids) >= 2:
            p = res(d.get('primary') or ids[0]); others = [x for x in ids if x != p]
            for x in others:
                rel.append({'from': x, 'type': d.get('relation') or 'related', 'to': p, 'status': 'checked_ai', 'why': d.get('reason_ru')})
            log['related'] += 1
        elif k == 'separate':
            s = set(ids)
            rel = [r for r in rel if not (r['status'] == 'proposed' and r['from'] in s and r['to'] in s)]
            log['separate'] += 1
        elif k in ('attach', 'reattach', 'detach'):
            for sid in d.get('schemes') or []:
                for c in cs.values():
                    if sid in (c.get('schemes') or []) and (k != 'attach'): c['schemes'].remove(sid)
                if k in ('attach', 'reattach') and d.get('target') and res(d['target']) in cs:
                    t = cs[res(d['target'])]; t['schemes'] = list(dict.fromkeys((t.get('schemes') or []) + [sid]))
            log[k] += 1
        elif k == 'set_meaning' and d.get('target') and res(d['target']) in cs:
            t = cs[res(d['target'])]
            t['meaning'] = {**(t.get('meaning') or {}), **(d.get('meaning') or {}), 'status': 'checked_ai', 'why': d.get('reason_ru')}
            log[k] += 1
        else:
            log[k] += 1
    # связи и term_map — на выжившие id; дубли связей и петли убрать; предложенные связи внутри слитого — убрать
    seen, rel2 = set(), []
    for r in rel:
        r = {**r, 'from': res(r['from']), 'to': res(r['to'])}
        if r['from'] == r['to'] or r['from'] not in cs or r['to'] not in cs: continue
        key = (r['from'], r['type'], r['to'])
        if key in seen: continue
        seen.add(key); rel2.append(r)
    tm = {k: res(v) for k, v in tm.items()}
    for x, p in alias.items(): tm.setdefault(x, p)
    # очередь: решённые пункты помечаются, unsure остаются открытыми
    qp = os.path.join(out, 'review_queue.csv')
    # дополнительные решения (n = 's1'…) очередь не трогают
    Q = list(csv.DictReader(open(qp))) if any(isinstance(d.get('n'), int) for d in decs) else None
    byn = {d['n']: d for d in decs}
    for i, q in enumerate(Q or []):
        d = byn.get(i)
        q['решение'] = (d['decision'] + ((' → ' + (d.get('primary') or d.get('target') or '')) if (d.get('primary') or d.get('target')) else '')) if d else ''
        q['обоснование'] = (d.get('reason_ru') or '') + (' [' + '; '.join(d.get('evidence') or []) + ']' if d and d.get('evidence') else '') if d else ''
        q['уверенность'] = d.get('confidence', '') if d else ''
        q['статус'] = 'открыт' if (not d or d['decision'] == 'unsure') else 'решено ИИ, ждёт подтверждения'
    if Q is not None:
      with open(qp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['тип', 'что', 'почему', 'предложение', 'решение', 'обоснование', 'уверенность', 'статус'])
        w.writeheader()
        for q in Q: w.writerow({k: q.get(k, '') for k in w.fieldnames})
    J = lambda n, o: json.dump(o, open(os.path.join(out, n), 'w'), ensure_ascii=False, indent=1)
    J('concepts.json', list(cs.values())); J('relations.json', rel2); J('term_map.json', tm)
    lp = os.path.join(out, 'decisions_log.json')
    prev = json.load(open(lp)) if os.path.exists(lp) and Q is None else {}
    prev = prev if 'files' in prev else {'files': {}}
    prev['files'][os.path.basename(dec_path)] = {'applied': dict(log), 'merged_away': alias}
    J('decisions_log.json', prev)
    print('применено:', dict(log), '| понятий:', len(cs), '| слито:', len(alias))


if __name__ == '__main__':
    main(*sys.argv[1:3])
