# печатает HTTP-код ответа студии на /api/health (200 или 401 — студия жива); запускается внутри контейнера
import urllib.request, urllib.error
try:
    print(urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=5).status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception:
    print(0)
