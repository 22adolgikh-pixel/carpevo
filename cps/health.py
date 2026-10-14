# health.py — v11.10: почему сервер падает. Без доступа к логам контейнера студия сама ведёт журнал:
#  • data/health/mem.log — каждые 10 с: память процесса (RSS), свободная память системы, что считается в очереди тяжёлых расчётов;
#  • data/health/faults.log — аварии на уровне C (OpenCV, torch): faulthandler пишет стек упавшего потока;
#  • при старте: если предыдущий запуск закончился не штатно — событие «Сервер перезапускался» с последней строкой mem.log
#    (если память была на пределе — почти наверняка система убила процесс из-за нехватки памяти).
import os, time, threading, json, faulthandler

_cfg = {}


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"): return int(ln.split()[1]) // 1024
    except Exception: pass
    return 0


def _sys_mb():
    try:
        d = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, v = ln.split(":", 1); d[k] = int(v.split()[0]) // 1024
        return d.get("MemTotal", 0), d.get("MemAvailable", 0)
    except Exception: return 0, 0


def _cg_limit_mb():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v and v != "max" and int(v) < 1 << 50: return int(v) // (1 << 20)
        except Exception: pass
    return 0


def status():
    tot, av = _sys_mb()
    return {"rss_mb": _rss_mb(), "sys_total_mb": tot, "sys_avail_mb": av, "limit_mb": _cg_limit_mb(), "cpus": os.cpu_count(),
            "up_s": int(time.time() - _cfg.get("t0", time.time())), "last_crash": _cfg.get("last_crash")}


def install(app, data, heavy_q=None, events=None):
    d = os.path.join(data, "health"); os.makedirs(d, exist_ok=True)
    _cfg.update(t0=time.time(), d=d)
    try:
        ff = open(os.path.join(d, "faults.log"), "a"); faulthandler.enable(file=ff, all_threads=True)
    except Exception: pass
    run = os.path.join(d, "running.json"); mem = os.path.join(d, "mem.log")
    # предыдущий запуск не закрылся штатно → было падение
    if os.path.exists(run):
        try: prev = json.load(open(run))
        except Exception: prev = {}
        last = ""
        try:
            with open(mem, "rb") as f:
                f.seek(max(0, os.path.getsize(mem) - 600)); last = f.read().decode("utf-8", "ignore").strip().splitlines()[-1]
        except Exception: pass
        _cfg["last_crash"] = {"started": prev.get("started"), "last": last}
        if events:
            @app.on_event("startup")
            def _note():           # события подключаются позже — пишем, когда сервер уже поднялся
                try: events.add("Сервер перезапускался (упал)", "последняя запись о памяти перед падением: " + (last or "нет") + ". Если свободной памяти (avail) почти не оставалось — система убила процесс из-за нехватки памяти.", level="warn")
                except Exception: pass
    json.dump({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid()}, open(run, "w"))

    @app.on_event("shutdown")
    def _bye():
        try: os.remove(run)
        except Exception: pass

    def loop():
        while True:
            try:
                tot, av = _sys_mb(); hq = heavy_q or {}
                line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} rss={_rss_mb()}MB avail={av}/{tot}MB limit={_cg_limit_mb()}MB heavy={hq.get("run") or "-"} wait={hq.get("wait", 0)}\n'
                with open(mem, "a") as f: f.write(line)
                if os.path.getsize(mem) > 2_000_000:            # хвост 1 МБ
                    with open(mem, "rb") as f: f.seek(-1_000_000, 2); tail = f.read()
                    with open(mem, "wb") as f: f.write(tail)
            except Exception: pass
            time.sleep(10)
    threading.Thread(target=loop, daemon=True).start()

    @app.get("/api/health")
    def api_health():
        out = status()
        try:
            with open(mem, "rb") as f:
                f.seek(max(0, os.path.getsize(mem) - 3000)); out["mem_tail"] = f.read().decode("utf-8", "ignore").strip().splitlines()[-12:]
        except Exception: out["mem_tail"] = []
        try:
            fp = os.path.join(d, "faults.log")
            with open(fp, "rb") as f:
                f.seek(max(0, os.path.getsize(fp) - 4000)); out["faults_tail"] = f.read().decode("utf-8", "ignore")[-4000:]
        except Exception: out["faults_tail"] = ""
        return out
