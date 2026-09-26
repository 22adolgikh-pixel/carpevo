# backup.py — автокопия работы студии в GitHub (ветка cps-data репозитория carpevo).
#
# Включается, если в .env есть CPS_GITHUB_TOKEN (fine-grained токен GitHub
# с правом Contents: Read and write на репозиторий carpevo). Раз в CPS_BACKUP_MIN
# минут (по умолчанию 10) и при остановке студии копирует в ветку cps-data:
#   data/work/  data/sheets/  data/legend.csv  data/i18n.json
# и делает коммит, только если что-то изменилось. История изменений — в GitHub.
# Сканы, кропы и картинки data/out в GitHub НЕ идут: репозиторий публичный,
# а сканы книг защищены авторским правом (кропы и out пересобираются из work).
import os, shutil, subprocess, threading, time

REPO = os.environ.get("CPS_BACKUP_REPO", "22adolgikh-pixel/carpevo")
BRANCH = os.environ.get("CPS_BACKUP_BRANCH", "cps-data")
EVERY = float(os.environ.get("CPS_BACKUP_MIN", "10"))
ITEMS = ("work", "sheets", "legend.csv", "i18n.json")

state = {"enabled": False, "last_ok": None, "last_commit": None, "last_error": None, "running": False}
_lock = threading.Lock()


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=300)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r


def _mirror(src, dst):
    """Копия src → dst: новые/изменённые файлы копируются, удалённые в src — удаляются в dst."""
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst); return
    if not os.path.isdir(src): return
    os.makedirs(dst, exist_ok=True)
    have = set(os.listdir(src))
    for n in os.listdir(dst):
        if n not in have and not n.startswith("."):
            p = os.path.join(dst, n)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for n in have:
        if n.startswith("."): continue
        s, d = os.path.join(src, n), os.path.join(dst, n)
        if os.path.isdir(s): _mirror(s, d)
        elif not os.path.exists(d) or os.path.getmtime(s) != os.path.getmtime(d) or os.path.getsize(s) != os.path.getsize(d):
            shutil.copy2(s, d)


def backup_once(here, data, reason="авто"):
    token = os.environ.get("CPS_GITHUB_TOKEN", "")
    if not token: return {"ok": False, "error": "нет CPS_GITHUB_TOKEN"}
    if not _lock.acquire(blocking=False): return {"ok": False, "error": "уже идёт"}
    state["running"] = True
    try:
        repo = os.path.join(here, ".backup_repo")
        url = os.environ.get("CPS_BACKUP_URL") or f"https://x-access-token:{token}@github.com/{REPO}.git"
        if not os.path.isdir(os.path.join(repo, ".git")):
            shutil.rmtree(repo, ignore_errors=True)
            r = subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, url, repo],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0: raise RuntimeError("clone: " + (r.stderr or "").strip()[-400:].replace(token, "***"))
            _git(repo, "config", "user.name", "CPS autosave")
            _git(repo, "config", "user.email", "cps-autosave@users.noreply.github.com")
        _git(repo, "remote", "set-url", "origin", url)
        for it in ITEMS:
            _mirror(os.path.join(data, it), os.path.join(repo, "data", it))
        _git(repo, "add", "-A")
        if _git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0:
            state.update(last_ok=time.strftime("%Y-%m-%d %H:%M:%S"), last_error=None)
            return {"ok": True, "changed": 0}
        n = len(_git(repo, "diff", "--cached", "--name-only").stdout.split())
        _git(repo, "commit", "-q", "-m", f"CPS: {n} файл(ов) изменено ({reason}, {time.strftime('%Y-%m-%d %H:%M')})")
        if _git(repo, "push", "-q", "origin", BRANCH, check=False).returncode != 0:
            _git(repo, "pull", "-q", "--rebase", "origin", BRANCH)
            _git(repo, "push", "-q", "origin", BRANCH)
        state.update(last_ok=time.strftime("%Y-%m-%d %H:%M:%S"), last_commit=f"{n} файл(ов)", last_error=None)
        return {"ok": True, "changed": n}
    except Exception as e:
        msg = str(e).replace(token, "***"); state["last_error"] = msg
        print("backup:", msg)
        return {"ok": False, "error": msg}
    finally:
        state["running"] = False; _lock.release()


def start(here, data):
    if not os.environ.get("CPS_GITHUB_TOKEN"):
        return
    state["enabled"] = True

    def loop():
        while True:
            backup_once(here, data)
            time.sleep(EVERY * 60)
    threading.Thread(target=loop, daemon=True, name="cps-backup").start()
    print(f"backup: копия в GitHub {REPO}@{BRANCH} каждые {EVERY:g} мин")
