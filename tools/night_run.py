"""Долгий прогон ноутбука без присмотра.

Выполняет ноутбук целиком; все ответы моделей пишутся в кэш, поэтому после сбоя прогон продолжается с того же места.
При ошибке ждёт и запускает снова. Журнал — tom_work/night_run.log, готовый ноутбук — tom_work/night_out.ipynb.

usage: py tools/night_run.py ToM_antifraud.ipynb [ПАРАМЕТР=значение ...]
"""
import datetime
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SRC = sys.argv[1]
OVERRIDES = dict(a.split("=", 1) for a in sys.argv[2:])
WORK = Path("../tom_work") if Path("../tom_work").exists() else Path("tom_work")
LOG, OUT = WORK / "night_run.log", WORK / "night_out.ipynb"
MAX_ATTEMPTS, PAUSE = 12, 180


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_once():
    nb = nbformat.read(SRC, as_version=4)
    original = [c.source for c in nb.cells]
    for c in nb.cells:
        if c.cell_type == "code":
            for k, v in OVERRIDES.items():
                c.source = re.sub(rf"(?m)^{k} = [^#\n]*", lambda m: f"{k} = {v}  ", c.source)
    error = None
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3").execute()      # без ограничения времени на ячейку
    except Exception as e:                                                      # ячейка упала или умерло ядро
        error = f"{type(e).__name__}: {str(e)[-600:]}"
    for c, s in zip(nb.cells, original):
        c.source = s
        if c.cell_type == "code":
            c.outputs = [o for o in c.outputs if not (o.output_type == "stream" and o.name == "stderr" and "Warning" in o.text)]
    nbformat.write(nb, OUT)
    quota = any("дневной лимит исчерпан на всех ключах" in o.get("text", "") for c in nb.cells if c.cell_type == "code"
                for o in c.outputs if o.output_type == "stream")
    return error, quota


def keep_awake():
    """Просит Windows не засыпать, пока работает этот процесс. Настройки питания не меняются."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return True
    except Exception:
        return False


log(f"старт: {SRC}, параметры: {OVERRIDES or 'нет'}; запрет сна на время прогона: {'да' if keep_awake() else 'не удалось'}")
last = None
for attempt in range(1, MAX_ATTEMPTS + 1):
    t0 = time.time()
    error, quota = run_once()
    cached = sum(1 for p in (WORK / "cache").glob("*.jsonl") for _ in open(p, encoding="utf-8"))
    log(f"попытка {attempt}: {(time.time() - t0) / 60:.0f} мин; ответов в кэше всего: {cached}; "
        + ("ОШИБКА " + error if error else "ноутбук выполнен") + ("; дневной лимит исчерпан" if quota else ""))
    if not error:
        break
    if error == last and attempt >= 3:
        log("та же ошибка три раза подряд — останавливаюсь")
        break
    last = error
    time.sleep(PAUSE)
log("конец")
