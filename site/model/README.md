# Модель «понятия» для Ковровое ДНК (вариант B, 2026-10-05)

`build_concepts.py DATA.json SCHEMES.json OUT/` собирает из записей-терминов сайта понятия, названия, свидетельства и связи.
Предложение по структуре — проектный документ `claude/site_structure_proposal_2026-10-05.md`.

Входы прогона 2026-10-05: data.json сайта v44 (1409 терминов), выгрузка `pixel_schemes` (938 схем).

out/:
- `concepts.json` — понятия: headword, meaning, names[{v, lang, kind}] (kind: headword / book_reading / record / misreading / attested / meaning / meaning_rejected / kerimov_translation), roles, schools, sources, attestations, records (старые id), schemes, merge{status, rule, renamed}.
- `relations.json` — связи {from, type, to, status auto|proposed, why}: consists_of, kind_of, related, derived_from, named_after, alt_name_of, synonym, spelling_variant.
- `term_map.json` — старый id термина (и старые алиасы) → id понятия.
- `schemes_relink.json` — изменения привязок схем.
- `review_queue.csv` — очередь проверки человеком (колонка «решение»).
- `report.md` — цифры.

Ключ слияния — чтение по сверке с Керимовым (book.reading) или canonical_az, строго по азербайджанским буквам (ı≠i, q≠g, ə≠e).
