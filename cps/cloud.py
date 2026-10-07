# cloud.py — пользователи, пароли, регистрация и журнал действий (v10.19).
#
# Кто входит: все, кто пришёл не напрямую с самого компьютера (через nginx на сервере, туннель Cloudflare).
# На самом Mac (http://localhost:8000, без заголовков прокси) студия открывается без входа, как раньше —
# работа записывается на пользователя «local» (роль admin).
#
# Роли: admin — всё + список людей и журнал; editor — правит и запускает обработку; viewer — только смотрит.
# Новая регистрация → viewer; повышает до editor администратор (экран «Люди и журнал»).
# Пароли хранятся только в виде соль+PBKDF2 в data/users.json. Первый запуск создаёт dolgikh (admin), lika (editor),
# guest1 (viewer) со случайными паролями → data/INITIAL_PASSWORDS.txt (в бэкап не идёт; прочитать и удалить).
# Журнал — data/activity.jsonl: одна строка на каждое изменяющее действие.
import os, re, json, time, hmac, hashlib, secrets, threading
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
USERS = os.path.join(DATA, "users.json")
ACTIVITY = os.path.join(DATA, "activity.jsonl")
SECRET = os.path.join(DATA, "auth_secret")
INIT_PW = os.path.join(DATA, "INITIAL_PASSWORDS.txt")
COOKIE = "cps_session"
TTL = 60 * 60 * 24 * 30
ROLES = ("admin", "editor", "viewer")
ROLE_RU = {"admin": "админ", "editor": "редактор", "viewer": "наблюдатель"}
INITIAL = (("dolgikh", "admin"), ("lika", "editor"), ("guest1", "viewer"))
NAME_RE = re.compile(r"^[a-z0-9_.\-]{3,24}$")
ITER = 200_000
# изменяющие запросы, которые журналировать не нужно / которые разрешены наблюдателю
VIEWER_OK = ("/api/dashboard/seen", "/api/logout", "/api/login", "/api/register", "/api/me/password")
NO_LOG = VIEWER_OK + ("/api/users/role", "/api/users/reset")
_lock = threading.Lock()
_fails = {}      # ip → [время неудачных входов]


# ---------- хранилище ----------
def _load():
    try:
        with open(USERS, encoding="utf-8") as f: return json.load(f)
    except Exception: return {}


def _save(u):
    os.makedirs(DATA, exist_ok=True)
    tmp = USERS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(u, f, ensure_ascii=False, indent=1)
    os.replace(tmp, USERS)


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt), ITER).hex()


def _mk(pw, role):
    salt = secrets.token_hex(16)
    return {"salt": salt, "hash": _hash(pw, salt), "role": role, "created": int(time.time()), "last_seen": 0}


def _newpw():
    return "".join(secrets.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(10))


def bootstrap():
    """Первый запуск: создать трёх пользователей. Повторно не трогает существующих."""
    with _lock:
        u = _load()
        if u: return
        lines = []
        for name, role in INITIAL:
            pw = _newpw(); u[name] = _mk(pw, role); lines.append(f"{name}  {pw}  ({role})")
        _save(u)
        with open(INIT_PW, "w", encoding="utf-8") as f:
            f.write("Начальные пароли CPS (смените в приложении: имя вверху → «Сменить пароль»; файл потом удалите)\n" + "\n".join(lines) + "\n")


def _secret():
    try:
        with open(SECRET) as f: return f.read().strip()
    except Exception:
        s = secrets.token_hex(32); os.makedirs(DATA, exist_ok=True)
        with open(SECRET, "w") as f: f.write(s)
        return s


def _sign(msg): return hmac.new(_secret().encode(), msg.encode(), hashlib.sha256).hexdigest()


def _token(name):
    exp = int(time.time()) + TTL
    return f"{name}:{exp}:{_sign(f'{name}:{exp}')}"


def _check(tok):
    try:
        name, exp, sig = tok.rsplit(":", 2)
        if int(exp) < time.time() or not hmac.compare_digest(sig, _sign(f"{name}:{exp}")): return None
        return name if name in _load() else None
    except Exception: return None


def _remote(request: Request):
    h = request.headers
    if "cf-connecting-ip" in h or "cf-ray" in h or "x-forwarded-for" in h or "x-real-ip" in h: return True
    c = request.client.host if request.client else ""
    return c not in ("127.0.0.1", "::1", "localhost", "testclient", "")


def _ip(request):
    return (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else ""))


def current(request: Request):
    """→ (имя, роль) или None."""
    if not _remote(request): return ("local", "admin")
    n = _check(request.cookies.get(COOKIE, ""))
    if not n: return None
    return (n, _load()[n]["role"])


def log(user, action, detail=""):
    try:
        with open(ACTIVITY, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": int(time.time()), "user": user, "do": action, "what": detail}, ensure_ascii=False) + "\n")
    except Exception: pass


def _touch(name):
    u = _load(); r = u.get(name)
    if r and time.time() - r.get("last_seen", 0) > 60:
        r["last_seen"] = int(time.time()); _save(u)


def _describe(method, path, query):
    """Короткое «что сделал» для журнала (без тела запроса)."""
    p = path.removeprefix("/api/")
    return f"{method} {p}" + (f"?{query}" if query and len(query) < 80 else "")


def _rate_ok(ip):
    now = time.time(); l = [t for t in _fails.get(ip, []) if now - t < 600]; _fails[ip] = l
    return len(l) < 8


LOGIN_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Carpet Pattern Studio — вход</title>
<style>body{font-family:system-ui,sans-serif;background:#f4f1ec;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.box{background:#fff;padding:26px 30px;border-radius:12px;box-shadow:0 4px 24px #0002;width:290px}
h1{font-size:18px;margin:0 0 14px}.tabs{display:flex;gap:6px;margin-bottom:14px}
.tabs button{flex:1;margin:0;padding:7px;background:#eee;color:#333;font-size:14px}.tabs button.on{background:#8b2e2e;color:#fff}
input{width:100%;box-sizing:border-box;padding:10px;font-size:16px;border:1px solid #ccc;border-radius:8px;margin-bottom:8px}
button.go{width:100%;padding:10px;font-size:16px;border:0;border-radius:8px;background:#8b2e2e;color:#fff;cursor:pointer}
button{border:0;border-radius:8px;cursor:pointer}.err{color:#b00;font-size:14px;margin-top:10px;min-height:18px}
.hint{color:#777;font-size:12px;margin-top:8px}</style></head><body><div class="box"><h1>Carpet Pattern Studio</h1>
<div class="tabs"><button id="t1" class="on" type="button">Вход</button><button id="t2" type="button">Регистрация</button></div>
<form id="f"><input id="u" placeholder="Имя (латиница)" autocomplete="username" autofocus>
<input id="p" type="password" placeholder="Пароль" autocomplete="current-password">
<input id="p2" type="password" placeholder="Пароль ещё раз" style="display:none" autocomplete="new-password">
<button class="go" id="go">Войти</button><div class="err" id="e"></div>
<div class="hint" id="h" style="display:none">Новый аккаунт сначала может только смотреть — права на правки выдаёт администратор.</div></form></div>
<script>let reg=false;const $=i=>document.getElementById(i);
function mode(r){reg=r;$('t1').className=r?'':'on';$('t2').className=r?'on':'';$('p2').style.display=r?'':'none';$('h').style.display=r?'':'none';$('go').textContent=r?'Зарегистрироваться':'Войти';$('e').textContent=''}
$('t1').onclick=()=>mode(false);$('t2').onclick=()=>mode(true);
$('f').onsubmit=async ev=>{ev.preventDefault();
 if(reg&&$('p').value!==$('p2').value){$('e').textContent='Пароли не совпадают';return}
 const r=await fetch(reg?'/api/register':'/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('u').value,password:$('p').value})});
 const j=await r.json().catch(()=>({}));if(r.ok)location.href='/';else $('e').textContent=j.error||'Ошибка'};</script></body></html>"""


def install(app):
    bootstrap()

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        path = request.url.path
        if path in ("/login", "/api/login", "/api/register"):
            return await call_next(request)
        who = current(request)
        if not who:
            if path.startswith("/api/") or path.startswith("/out/"):
                return JSONResponse({"error": "auth"}, status_code=401)
            return HTMLResponse(LOGIN_HTML, status_code=401)
        request.state.user, request.state.role = who
        mut = request.method in ("POST", "PUT", "DELETE", "PATCH")
        if mut and who[1] == "viewer" and path not in VIEWER_OK:
            return JSONResponse({"error": "Только просмотр: у вашей учётной записи нет прав на правки"}, status_code=403)
        if who[0] != "local": _touch(who[0])
        resp = await call_next(request)
        if mut and path not in NO_LOG and resp.status_code < 400:
            log(who[0], "edit", _describe(request.method, path, request.url.query))
        return resp

    @app.get("/login")
    async def _login_page(): return HTMLResponse(LOGIN_HTML)

    def _set(resp, request, name):
        https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        resp.set_cookie(COOKIE, _token(name), max_age=TTL, httponly=True, secure=https, samesite="lax")
        return resp

    @app.post("/api/login")
    async def _login(request: Request):
        ip = _ip(request)
        if not _rate_ok(ip): return JSONResponse({"error": "Слишком много попыток, подождите 10 минут"}, status_code=429)
        b = await request.json()
        name = str(b.get("name", "")).strip().lower(); pw = str(b.get("password", ""))
        r = _load().get(name)
        if r and hmac.compare_digest(_hash(pw, r["salt"]), r["hash"]):
            log(name, "login", ip)
            return _set(JSONResponse({"ok": True}), request, name)
        _fails.setdefault(ip, []).append(time.time())
        return JSONResponse({"error": "Неверное имя или пароль"}, status_code=401)

    @app.post("/api/register")
    async def _register(request: Request):
        ip = _ip(request)
        if not _rate_ok(ip): return JSONResponse({"error": "Слишком много попыток, подождите 10 минут"}, status_code=429)
        b = await request.json()
        name = str(b.get("name", "")).strip().lower(); pw = str(b.get("password", ""))
        if not NAME_RE.match(name): return JSONResponse({"error": "Имя: 3–24 символа, латиница, цифры, _ . -"}, status_code=400)
        if len(pw) < 6: return JSONResponse({"error": "Пароль не короче 6 символов"}, status_code=400)
        _fails.setdefault(ip, []).append(time.time() - 540)    # регистрации тоже ограничены (≈8 в 10 минут → ~1 в минуту)
        with _lock:
            u = _load()
            if name in u: return JSONResponse({"error": "Такое имя уже занято"}, status_code=409)
            u[name] = _mk(pw, "viewer"); _save(u)
        log(name, "register", ip)
        return _set(JSONResponse({"ok": True}), request, name)

    @app.post("/api/logout")
    async def _logout():
        r = JSONResponse({"ok": True}); r.delete_cookie(COOKIE); return r

    @app.get("/api/me")
    async def _me(request: Request):
        n, role = getattr(request.state, "user", None), getattr(request.state, "role", None)
        return {"name": n, "role": role, "role_ru": ROLE_RU.get(role, role), "local": n == "local"}

    @app.post("/api/me/password")
    async def _chpw(request: Request):
        n = request.state.user
        if n == "local": return JSONResponse({"error": "На своём компьютере пароль не нужен"}, status_code=400)
        b = await request.json()
        with _lock:
            u = _load(); r = u[n]
            if not hmac.compare_digest(_hash(str(b.get("old", "")), r["salt"]), r["hash"]):
                return JSONResponse({"error": "Старый пароль неверен"}, status_code=403)
            new = str(b.get("new", ""))
            if len(new) < 6: return JSONResponse({"error": "Пароль не короче 6 символов"}, status_code=400)
            u[n].update({k: v for k, v in _mk(new, r["role"]).items() if k in ("salt", "hash")}); _save(u)
        log(n, "password", "")
        return {"ok": True}

    def _admin(request): return getattr(request.state, "role", None) == "admin"

    @app.get("/api/users")
    async def _users(request: Request):
        if not _admin(request): return JSONResponse({"error": "Только для администратора"}, status_code=403)
        act = {}
        try:
            with open(ACTIVITY, encoding="utf-8") as f:
                for ln in f:
                    try: e = json.loads(ln)
                    except Exception: continue
                    if e.get("do") == "edit": a = act.setdefault(e["user"], {"n": 0, "last": 0}); a["n"] += 1; a["last"] = e["t"]
        except Exception: pass
        return [{"name": k, "role": v["role"], "role_ru": ROLE_RU[v["role"]], "created": v.get("created"),
                 "last_seen": v.get("last_seen"), "edits": act.get(k, {}).get("n", 0), "last_edit": act.get(k, {}).get("last")}
                for k, v in sorted(_load().items())]

    @app.post("/api/users/role")
    async def _role(request: Request):
        if not _admin(request): return JSONResponse({"error": "Только для администратора"}, status_code=403)
        b = await request.json(); name, role = b.get("name"), b.get("role")
        if role not in ROLES: return JSONResponse({"error": "роль?"}, status_code=400)
        with _lock:
            u = _load()
            if name not in u: return JSONResponse({"error": "нет такого"}, status_code=404)
            if name == request.state.user and role != "admin" and sum(1 for x in u.values() if x["role"] == "admin") < 2:
                return JSONResponse({"error": "Нельзя снять права с единственного администратора"}, status_code=400)
            u[name]["role"] = role; _save(u)
        log(request.state.user, "role", f"{name} → {role}")
        return {"ok": True}

    @app.post("/api/users/reset")
    async def _reset(request: Request):
        if not _admin(request): return JSONResponse({"error": "Только для администратора"}, status_code=403)
        name = (await request.json()).get("name"); pw = _newpw()
        with _lock:
            u = _load()
            if name not in u: return JSONResponse({"error": "нет такого"}, status_code=404)
            u[name].update({k: v for k, v in _mk(pw, u[name]["role"]).items() if k in ("salt", "hash")}); _save(u)
        log(request.state.user, "reset", name)
        return {"ok": True, "password": pw}

    @app.get("/api/activity")
    async def _activity(request: Request, limit: int = 200, user: str = ""):
        if not _admin(request): return JSONResponse({"error": "Только для администратора"}, status_code=403)
        rows = []
        try:
            with open(ACTIVITY, encoding="utf-8") as f:
                for ln in f:
                    try: e = json.loads(ln)
                    except Exception: continue
                    if not user or e.get("user") == user: rows.append(e)
        except Exception: pass
        return rows[-max(1, min(limit, 2000)):][::-1]


if __name__ == "__main__":      # python cloud.py passwd <имя> → новый случайный пароль (если забыли)
    import sys
    bootstrap()
    if len(sys.argv) == 3 and sys.argv[1] == "passwd":
        u = _load()
        if sys.argv[2] not in u: sys.exit("нет такого пользователя: " + ", ".join(u))
        pw = _newpw(); u[sys.argv[2]].update({k: v for k, v in _mk(pw, u[sys.argv[2]]["role"]).items() if k in ("salt", "hash")}); _save(u)
        print(f"Новый пароль для {sys.argv[2]}: {pw}")
    else:
        print("использование: python cloud.py passwd <имя>")
