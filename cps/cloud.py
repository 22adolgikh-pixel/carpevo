# cloud.py — пароль для доступа к студии по ссылке (share.command → туннель Cloudflare).
#
# Пароль спрашивается только у тех, кто пришёл через туннель: такие запросы
# Cloudflare помечает заголовком Cf-Connecting-Ip. На самом Mac
# (http://localhost:8000) студия открывается без пароля, как раньше.
# Без переменной CPS_PASSWORD ничего не включается.
import os, hmac, hashlib
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

PASSWORD = os.environ.get("CPS_PASSWORD", "")
COOKIE = "cps_auth"
_TOKEN = hashlib.sha256(("cps:" + PASSWORD).encode()).hexdigest() if PASSWORD else ""

LOGIN_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Carpet Pattern Studio</title>
<style>body{font-family:system-ui,sans-serif;background:#f4f1ec;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#fff;padding:28px 32px;border-radius:12px;box-shadow:0 4px 24px #0002;min-width:260px}
h1{font-size:18px;margin:0 0 16px}input{width:100%;box-sizing:border-box;padding:10px;font-size:16px;border:1px solid #ccc;border-radius:8px}
button{margin-top:12px;width:100%;padding:10px;font-size:16px;border:0;border-radius:8px;background:#8b2e2e;color:#fff;cursor:pointer}
.err{color:#b00;font-size:14px;margin-top:10px}</style></head><body>
<form method="post" action="/login"><h1>Carpet Pattern Studio</h1>
<input type="password" name="password" placeholder="Пароль" autofocus>
<button>Войти</button>%ERR%</form></body></html>"""


def _remote(request: Request) -> bool:
    return "cf-connecting-ip" in request.headers or "cf-ray" in request.headers


def _ok(request: Request) -> bool:
    return (not PASSWORD) or (not _remote(request)) or \
        hmac.compare_digest(request.cookies.get(COOKIE, ""), _TOKEN)


def install(app):
    if not PASSWORD:
        return

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if request.url.path == "/login" or _ok(request):
            return await call_next(request)
        if request.url.path.startswith("/api/") or request.url.path.startswith("/out/"):
            return JSONResponse({"error": "auth"}, status_code=401)
        return HTMLResponse(LOGIN_HTML.replace("%ERR%", ""), status_code=401)

    @app.post("/login")
    async def _login(request: Request):
        form = await request.form()
        if hmac.compare_digest(str(form.get("password", "")), PASSWORD):
            r = RedirectResponse("/", status_code=303)
            r.set_cookie(COOKIE, _TOKEN, max_age=60 * 60 * 24 * 30, httponly=True,
                         secure=True, samesite="lax")
            return r
        return HTMLResponse(LOGIN_HTML.replace("%ERR%", '<div class="err">Неверный пароль</div>'),
                            status_code=401)
