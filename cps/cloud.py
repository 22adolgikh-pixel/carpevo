# cloud.py — режим «по ссылке» (Hugging Face Space).
# Локально ничего не меняет: всё включается только переменными окружения.
#
#   CPS_PASSWORD   — общий пароль на вход (если пусто — вход без пароля)
#   HF_TOKEN       — токен Hugging Face с правом записи (секрет Space)
#   CPS_DATA_REPO  — датасет HF для хранения работы, например "user/cps-data".
#                    Диск Space стирается при каждом перезапуске, поэтому
#                    data/ и scans/ при старте скачиваются оттуда, а изменения
#                    выгружаются обратно каждые CPS_SYNC_MIN минут (по умолчанию 2).
import os, hmac, hashlib, html
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


def _ok(request: Request) -> bool:
    return (not PASSWORD) or hmac.compare_digest(request.cookies.get(COOKIE, ""), _TOKEN)


def install(app):
    """Пароль на вход + вход через /login. Без CPS_PASSWORD — ничего не делает."""
    if not PASSWORD:
        return

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if request.url.path == "/login" or _ok(request):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "auth"}, status_code=401)
        return HTMLResponse(LOGIN_HTML.replace("%ERR%", ""), status_code=401)

    @app.post("/login")
    async def _login(request: Request):
        form = await request.form()
        if hmac.compare_digest(str(form.get("password", "")), PASSWORD):
            r = RedirectResponse("/", status_code=303)
            # SameSite=None — чтобы вход работал и внутри окна huggingface.co/spaces
            r.set_cookie(COOKIE, _TOKEN, max_age=60 * 60 * 24 * 90, httponly=True,
                         secure=True, samesite="none")
            return r
        return HTMLResponse(LOGIN_HTML.replace("%ERR%", '<div class="err">Неверный пароль</div>'),
                            status_code=401)


# ---------------- хранение работы в датасете HF ----------------
_schedulers = []


def restore(base_dir: str):
    """Скачать сохранённые data/ и scans/ из датасета в base_dir (при старте Space)."""
    repo, token = os.environ.get("CPS_DATA_REPO"), os.environ.get("HF_TOKEN")
    if not (repo and token):
        return
    try:
        from huggingface_hub import HfApi, snapshot_download
        HfApi(token=token).create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        snapshot_download(repo, repo_type="dataset", token=token,
                          allow_patterns=["data/**", "scans/**"], local_dir=base_dir)
        print("cloud: restored from", repo)
    except Exception as e:
        print("cloud: restore failed:", e)


def start_sync(data_dir: str, scans_dir: str):
    """Каждые N минут выгружать изменения data/ и scans/ в датасет."""
    repo, token = os.environ.get("CPS_DATA_REPO"), os.environ.get("HF_TOKEN")
    if not (repo and token):
        return
    from huggingface_hub import CommitScheduler
    os.makedirs(data_dir, exist_ok=True); os.makedirs(scans_dir, exist_ok=True)
    every = float(os.environ.get("CPS_SYNC_MIN", "2"))
    for sub, path in (("data", data_dir), ("scans", scans_dir)):
        _schedulers.append(CommitScheduler(repo_id=repo, repo_type="dataset", folder_path=path,
                                           path_in_repo=sub, every=every, token=token,
                                           private=True, squash_history=False))
    print(f"cloud: syncing to {repo} every {every} min")


def flush():
    for s in _schedulers:
        try:
            s.trigger().result()
        except Exception as e:
            print("cloud: flush failed:", e)
