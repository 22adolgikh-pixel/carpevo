# restore.py — восстановление студии на «чистом» (эфемерном) хостинге.
#
# Локально на Mac это не нужно: данные и так лежат на диске между запусками.
# На хостинге (Render/Fly/Cloud Run и т.п.) диск контейнера пропадает при
# каждом перезапуске/деплое, поэтому при старте студия сама:
#   1) подтягивает последний бэкап data/work, data/sheets, legend.csv, i18n.json,
#      site_index.json из ветки cps-data репозитория carpevo (то, что раз в
#      CPS_BACKUP_MIN минут туда пишет backup.py) — если data/work ещё нет;
#   2) скачивает архив сканов со страницы Google Drive (см. CPS_SCANS_ZIP_ID)
#      и распаковывает его в SCANS — если локальной папки scans ещё нет.
#
# Ничего не делает, если данные уже на месте (например, при обычном запуске
# на Mac через start.command) — так что скрипт безопасно вызывать всегда.
import os, io, shutil, subprocess, urllib.request, zipfile, re, http.cookiejar

REPO = os.environ.get("CPS_BACKUP_REPO", "22adolgikh-pixel/carpevo")
BRANCH = os.environ.get("CPS_BACKUP_BRANCH", "cps-data")
SCANS_ZIP_ID = os.environ.get("CPS_SCANS_ZIP_ID", "")  # id файла на Google Drive (публичная ссылка "у кого есть ссылка")


def _git_restore(here, data):
    if os.path.isdir(os.path.join(data, "work")) and os.listdir(os.path.join(data, "work")):
        return {"ok": True, "skipped": "data/work уже на месте"}
    token = os.environ.get("CPS_GITHUB_TOKEN", "")
    url = f"https://x-access-token:{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
    tmp = os.path.join(here, "_restore_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        r = subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, url, tmp],
                            capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return {"ok": False, "error": (r.stderr or r.stdout).strip()[-400:].replace(token, "***")}
        for item in ("work", "sheets", "legend.csv", "i18n.json", "site_index.json", "cell_model.json"):
            s, d = os.path.join(tmp, "data", item), os.path.join(data, item)
            if not os.path.exists(s):
                continue
            os.makedirs(os.path.dirname(d), exist_ok=True) if not os.path.isdir(s) else None
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        return {"ok": True, "restored_from": f"{REPO}@{BRANCH}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _drive_download(file_id, dest_zip):
    """Скачивает публичный (viewer-по-ссылке) файл с Google Drive по его id,
    обходя страницу-предупреждение антивируса для больших файлов. Использует
    один opener с cookiejar на оба запроса — иначе Google не узнаёт вторый
    запрос как продолжение первого и опять отдаёт HTML-страницу вместо файла."""
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cps-restore/1.0)"}
    base = "https://drive.google.com/uc?export=download"

    req = urllib.request.Request(f"{base}&id={file_id}", headers=headers)
    with opener.open(req, timeout=120) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "")

    if "text/html" in ctype:
        token = None
        for c in cj:
            if c.name.startswith("download_warning"):
                token = c.value
        if not token:
            m = re.search(rb'confirm=([0-9A-Za-z_-]+)', data) or re.search(rb'name="confirm"\s+value="([0-9A-Za-z_-]+)"', data)
            token = m.group(1).decode() if m else "t"
        # современный endpoint для подтверждённого скачивания — не всегда
        # требует cookies, но передаём тот же opener на случай, если требует
        url2 = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={token}"
        req2 = urllib.request.Request(url2, headers=headers)
        with opener.open(req2, timeout=300) as r2:
            data = r2.read()
            ctype2 = r2.headers.get("Content-Type", "")
        if "text/html" in ctype2:
            # запасной вариант — тот же трюк, но на старом домене
            req3 = urllib.request.Request(f"{base}&confirm={token}&id={file_id}", headers=headers)
            with opener.open(req3, timeout=300) as r3:
                data = r3.read()

    with open(dest_zip, "wb") as f:
        f.write(data)


def _scans_restore(scans_dir):
    if os.path.isdir(scans_dir) and os.listdir(scans_dir):
        return {"ok": True, "skipped": "scans уже на месте"}
    if not SCANS_ZIP_ID:
        return {"ok": False, "error": "нет CPS_SCANS_ZIP_ID (id zip-архива сканов на Google Drive)"}
    zpath = scans_dir + "_download.zip"
    try:
        _drive_download(SCANS_ZIP_ID, zpath)
        size = os.path.getsize(zpath)
        with open(zpath, "rb") as f:
            head = f.read(4)
        if head != b"PK\x03\x04":
            with open(zpath, "rb") as f:
                preview = f.read(200)
            return {"ok": False, "error": f"скачался не zip ({size} байт), похоже на HTML/ошибку Drive: {preview[:200]!r}"}
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.dirname(scans_dir))
        return {"ok": True, "restored_from": f"drive:{SCANS_ZIP_ID}", "bytes": size}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if os.path.exists(zpath):
            os.remove(zpath)


def restore_all(here, data, scans):
    """Вызывается один раз при старте студии (до backup.start)."""
    r1 = _git_restore(here, data)
    r2 = _scans_restore(scans)
    print("restore: data ←", r1)
    print("restore: scans ←", r2)
    return {"data": r1, "scans": r2}
