"""v11.2 — центр событий и автоматические задачи.

Студия работает сама, человек только решает:
  • «Нужно решение» — что ждёт человека сейчас (пороги дообучения 10 / 100, листы, нарезанные машиной, двойники ковров,
    надёжные рисунки, неподтверждённые красители…). Считается заново раз в минуту поставщиками (provider) из backend.py;
    «скрыть» прячет пункт, пока его состояние (sig) не изменится.
  • «Система» — что сделано само (ночной прогон, обучение, перекраска, отчёт за неделю): data/events.jsonl.
  • «Люди» — кто что правил (по журналу data/activity.jsonl).
  • Автоматические задачи — правило + «без вопросов» (делать само) или «спросить» (появляется в «Нужно решение» с кнопкой).
    Настройки и отметки — data/events_state.json.
Время — местное (CPS_TZ_HOURS, по умолчанию +4, Баку).
"""
import os, json, time, threading
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=float(os.environ.get("CPS_TZ_HOURS", "4"))))
NO_RUN = bool(os.environ.get("CPS_NO_SCHED"))         # тесты: задачи не запускаются сами, только считаются
_cfg = {}
_prov = []            # поставщики пунктов «Нужно решение»
_auto = {}            # автоматические задачи (порядок — как зарегистрированы)
_lock = threading.RLock()
_cache = {"at": 0, "items": []}


def now(): return datetime.now(TZ)


def _p(name): return os.path.join(_cfg["data"], name)


def _state():
    try: return json.load(open(_p("events_state.json"), encoding="utf-8"))
    except Exception: return {}


def _save_state(st):
    p = _p("events_state.json"); tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


# ---------------------------------------------------------------- «Система»: журнал событий
def add(title, body="", kind="system", link=None, level="info"):
    ev = {"t": int(time.time()), "kind": kind, "title": title, "body": body, "level": level}
    if link: ev["link"] = link
    try:
        with _lock, open(_p("events.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception as e: print("events.add", e)
    return ev


def recent(n=150, since=0):
    try: lines = open(_p("events.jsonl"), encoding="utf-8").read().splitlines()[-2000:]
    except Exception: return []
    out = []
    for l in reversed(lines):
        try: e = json.loads(l)
        except Exception: continue
        if e.get("t", 0) < since: break
        out.append(e)
        if len(out) >= n: break
    return out


# ---------------------------------------------------------------- «Нужно решение»
def provider(fn):
    _prov.append(fn); return fn


def decisions(force=False):
    if not force and time.time() - _cache["at"] < 55 and _cache["at"]: return _cache["items"]
    items = []
    for fn in _prov:
        try: items += [x for x in (fn() or []) if x]
        except Exception as e: print("events provider", getattr(fn, "__name__", fn), e)
    st = _state(); A = st.get("auto") or {}
    for aid, a in _auto.items():                         # задачи в режиме «спросить»: готово к запуску → пункт с кнопкой
        c = {**a["defaults"], **(A.get(aid) or {})}
        if not c.get("enabled") or c.get("auto") or not a.get("ask"): continue
        try: r = a["ask"](c)
        except Exception as e: print("events ask", aid, e); r = None
        if r: items.append({"key": "auto:" + aid, "sig": str(r.get("sig", "")), "level": "can", "group": "auto", "title": r["title"],
                            "body": r.get("body", ""), "action": {"do": "run", "arg": aid, "label": r.get("label", "Запустить")}})
    dis = st.get("dismissed") or {}
    for it in items: it["hidden"] = dis.get(it["key"]) == str(it.get("sig", ""))
    items.sort(key=lambda x: (x["level"] != "need", x.get("order", 50)))
    _cache.update(at=time.time(), items=items)
    return items


# ---------------------------------------------------------------- автоматические задачи
def automation(aid, title, rule, run, due=None, ask=None, enabled=True, auto=True, hour=None, weekday=None):
    """run(cfg) → строка-итог (или None, если делать нечего); due(cfg, now) → True, когда пора (для «без вопросов»);
    ask(cfg) → {"title","body","sig"} — пункт «Нужно решение», если задача в режиме «спросить» и есть что делать."""
    d = {"enabled": enabled, "auto": auto}
    if hour is not None: d["hour"] = hour
    if weekday is not None: d["weekday"] = weekday
    _auto[aid] = {"id": aid, "title": title, "rule": rule, "run": run, "due": due, "ask": ask, "defaults": d}


def automations():
    st = _state(); A = st.get("auto") or {}; out = []
    for aid, a in _auto.items():
        c = {**a["defaults"], **(A.get(aid) or {})}
        out.append({"id": aid, "title": a["title"], "rule": a["rule"] if isinstance(a["rule"], str) else a["rule"](c),
                    "enabled": bool(c.get("enabled")), "auto": bool(c.get("auto")), "hour": c.get("hour"), "weekday": c.get("weekday"),
                    "last_run": c.get("last_run"), "last_result": c.get("last_result", "")})
    return out


def set_auto(aid, **kw):
    if aid not in _auto: return None
    with _lock:
        st = _state(); A = st.setdefault("auto", {}); c = A.setdefault(aid, {})
        for k in ("enabled", "auto"):
            if k in kw and kw[k] is not None: c[k] = bool(kw[k])
        for k in ("hour", "weekday"):
            if kw.get(k) is not None: c[k] = int(kw[k])
        _save_state(st)
    _cache["at"] = 0
    return True


def run_auto(aid, who="автомат"):
    a = _auto[aid]; st = _state(); c = {**a["defaults"], **((st.get("auto") or {}).get(aid) or {})}
    try: res = a["run"](c)
    except Exception as e: res = f"ошибка: {e}"
    with _lock:
        st = _state(); A = st.setdefault("auto", {}).setdefault(aid, {})
        A["last_run"] = now().strftime("%Y-%m-%d %H:%M"); A["last_result"] = res or "делать нечего"
        _save_state(st)
    if res: add(f"{a['title']}: {res}", "запущено " + ("само (без вопросов)" if who == "автомат" else f"по кнопке ({who})"), link={"to": "events"})
    _cache["at"] = 0
    return res


def _loop():
    time.sleep(20)
    while True:
        try:
            decisions(force=True)
            if not NO_RUN:
                st = _state(); A = st.get("auto") or {}; t = now()
                for aid, a in _auto.items():
                    c = {**a["defaults"], **(A.get(aid) or {})}
                    if c.get("enabled") and c.get("auto") and a.get("due") and a["due"](c, t): run_auto(aid)
        except Exception as e: print("events loop", e)
        time.sleep(60)


# ---------------------------------------------------------------- «Люди» и отчёт
def people(days=7):
    since = time.time() - days * 86400; per = {}
    try: lines = open(_p("activity.jsonl"), encoding="utf-8").read().splitlines()[-20000:]
    except Exception: lines = []
    for l in lines:
        try: j = json.loads(l)
        except Exception: continue
        if j.get("t", 0) < since or j.get("do") not in ("edit", "login"): continue
        u = j.get("user") or "?"; day = datetime.fromtimestamp(j["t"], TZ).strftime("%Y-%m-%d")
        P = per.setdefault((day, u), {"day": day, "user": u, "figs": set(), "rugs": set(), "other": 0, "last": 0})
        P["last"] = max(P["last"], j["t"])
        w = str(j.get("what") or "").split(" ", 1)[-1]
        if w.startswith("work/"): P["figs"].add(w[5:])
        elif w.startswith("rugs/") and w.count("/") == 1: P["rugs"].add(w[5:])
        elif j["do"] == "edit": P["other"] += 1
    out = []
    for P in sorted(per.values(), key=lambda x: -x["last"]):
        if not (P["figs"] or P["rugs"] or P["other"]): continue
        out.append({"day": P["day"], "user": P["user"], "figs": len(P["figs"]), "rugs": len(P["rugs"]), "other": P["other"],
                    "last": datetime.fromtimestamp(P["last"], TZ).strftime("%H:%M")})
    return out


def report(days=7, extra=None):
    """Отчёт за период: люди + что сделала система + цифры от backend (extra(since) → dict)."""
    since = time.time() - days * 86400
    ppl = {}
    for p in people(days):
        q = ppl.setdefault(p["user"], {"user": p["user"], "figs": 0, "rugs": 0, "days": 0})
        q["figs"] += p["figs"]; q["rugs"] += p["rugs"]; q["days"] += 1
    sysev = [e for e in recent(500, since) if e.get("kind") == "system"]
    r = {"days": days, "from": datetime.fromtimestamp(since, TZ).strftime("%d.%m"), "to": now().strftime("%d.%m"),
         "people": sorted(ppl.values(), key=lambda x: -(x["figs"] + x["rugs"])), "system": len(sysev),
         "system_titles": [e["title"] for e in sysev[:8]],
         "open": [{"title": d["title"], "level": d["level"]} for d in decisions() if not d.get("hidden")][:8]}
    if extra:
        try: r.update(extra(since))
        except Exception as e: r["error"] = str(e)
    return r


def report_text(r):
    s = [f"Неделя {r['from']}–{r['to']}."]
    if r.get("done_new") is not None: s.append(f"Принято рисунков: {r['done_new']} (всего готово {r.get('done_total', '?')} из {r.get('figs_total', '?')}).")
    if r.get("sheets_new"): s.append(f"Нарезано листов: {r['sheets_new']}.")
    if r.get("marks_confirmed"): s.append(f"Подтверждено рамок на коврах: {r['marks_confirmed']}.")
    if r["people"]: s.append("Люди: " + "; ".join(f"{p['user']} — рисунков {p['figs']}, ковров {p['rugs']}" for p in r["people"]) + ".")
    if r["open"]: s.append("Ждёт решения: " + "; ".join(d["title"] for d in r["open"][:4]) + ".")
    return " ".join(s)


# ---------------------------------------------------------------- подключение к студии
def install(app, data_dir, act=None, extra=None):
    """act(item, user) → dict — выполнить кнопку пункта «Нужно решение» на сервере (обучить, прогон); иначе пункт открывает экран."""
    from fastapi import Body, Request, Query
    from fastapi.responses import JSONResponse
    _cfg.update(data=data_dir, act=act, extra=extra)

    @app.on_event("startup")
    def _ev_start():
        threading.Thread(target=_loop, daemon=True).start()

    def _who(request):
        return getattr(request.state, "user", None) or "local"

    @app.get("/api/events")
    def api_events(request: Request):
        st = _state(); seen = (st.get("seen") or {}).get(_who(request), 0)
        sysl = recent(150)
        for e in sysl: e["new"] = e["t"] > seen
        items = decisions()
        return {"decisions": items, "system": sysl, "people": people(7), "automations": automations(),
                "need": sum(1 for d in items if d["level"] == "need" and not d["hidden"]),
                "can": sum(1 for d in items if d["level"] != "need" and not d["hidden"]),
                "unseen": sum(1 for e in sysl if e["new"]), "tz": f"UTC{TZ.utcoffset(None).total_seconds() / 3600:+.0f}"}

    @app.get("/api/events/count")
    def api_events_count(request: Request):
        st = _state(); seen = (st.get("seen") or {}).get(_who(request), 0); items = decisions()
        return {"need": sum(1 for d in items if d["level"] == "need" and not d["hidden"]),
                "can": sum(1 for d in items if d["level"] != "need" and not d["hidden"]),
                "unseen": sum(1 for e in recent(150, seen + 1))}

    @app.post("/api/events/seen")
    def api_events_seen(request: Request):
        with _lock:
            st = _state(); st.setdefault("seen", {})[_who(request)] = int(time.time()); _save_state(st)
        return {"ok": True}

    @app.post("/api/events/dismiss")
    def api_events_dismiss(body: dict = Body(...)):
        key = str(body.get("key") or ""); sig = body.get("sig")
        with _lock:
            st = _state(); D = st.setdefault("dismissed", {})
            if sig is None: D.pop(key, None)
            else: D[key] = str(sig)
            _save_state(st)
        _cache["at"] = 0
        return {"ok": True}

    @app.post("/api/events/act")
    def api_events_act(request: Request, body: dict = Body(...)):
        key = str(body.get("key") or "")
        it = next((d for d in decisions(force=True) if d["key"] == key), None)
        if not it: return JSONResponse({"error": "этого пункта уже нет — обновите список"}, 404)
        a = it.get("action") or {}
        if a.get("do") == "run" and a.get("arg") in _auto:
            res = run_auto(a["arg"], _who(request)); return {"ok": True, "result": res}
        if act:
            r = act(it, _who(request))
            if r is not None: _cache["at"] = 0; return {"ok": True, **r}
        return {"ok": True, "open": a.get("arg")}

    @app.post("/api/automation")
    def api_automation(body: dict = Body(...)):
        if not set_auto(str(body.get("id")), enabled=body.get("enabled"), auto=body.get("auto"), hour=body.get("hour"), weekday=body.get("weekday")):
            return JSONResponse({"error": "нет такой задачи"}, 404)
        return {"ok": True, "automations": automations()}

    @app.get("/api/events/report")
    def api_events_report(days: int = Query(7)):
        r = report(max(1, min(90, days)), _cfg.get("extra")); r["text"] = report_text(r)
        return r
