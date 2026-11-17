"""Из корня BICA: python tools/run_perspective_pilot.py --size 5 --budget 65.
Без --budget разрешено только воспроизведение из кэша. Основной ноутбук не исполняется целиком.
"""
import argparse
import copy
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep

import pandas as pd

from perspectives import pilot_metrics, pilot_predictions


def setup(extra_cells=()):
    # Используем те же определения, ключи кэша и разбиение, что в выполненном ноутбуке.
    cells = json.loads(Path("ToM_antifraud.ipynb").read_text())["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    ns = {}
    for prefix in ("import os, sys, re", "GEMINI_API_KEY =", "def _read_env_file",
                   "class QuotaExhausted", "PUPPET_URL =", "SPEAKER_NAMES =", *extra_cells):
        matches = [s for s in sources if s.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f"Не найдена единственная ячейка {prefix!r}")
        exec(compile(matches[0], f"<notebook: {prefix}>", "exec"), ns)
    return ns


def run(size, budget, workers=3):
    ns = setup()
    model = ns["llm"]
    post = model._post
    used = 0
    statuses = Counter()
    lock = threading.Lock()
    next_attempt = 0.0

    def limited_post(*args):
        nonlocal used, next_attempt
        while True:
            with lock:
                if used >= budget:
                    raise RuntimeError("Лимит исходящих запросов исчерпан")
                wait = next_attempt - monotonic()
                if wait <= 0:
                    used += 1
                    next_attempt = monotonic() + 1.5
                    break
            sleep(wait)
        response = post(*args)
        with lock:
            statuses[response.status_code] += 1
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after", "")
                cooldown = float(retry_after) if retry_after.isdecimal() else 15
                next_attempt = max(next_attempt, monotonic() + cooldown)
            if response.status_code != 200 and statuses[response.status_code] in (1, 5, 20):
                print(f"Google API: HTTP {response.status_code}, ответов с таким статусом: "
                      f"{statuses[response.status_code]}", flush=True)
        return response

    model._post = limited_post
    if not budget:
        def cache_only(*args):
            raise RuntimeError("Нужен новый ответ модели: указан нулевой бюджет")
        model._request = cache_only
    rows = list(ns["pilot"].head(size).itertuples(index=False))
    dest = ns["RESULTS_DIR"] / f"perspectives_dev_{size}"
    dest.mkdir(exist_ok=True)
    start = monotonic()

    def predict(position, row, worker):
        result, trace = pilot_predictions(row, worker, ns["predict_direct"], ns["format_dialog"],
                                          ns["valid_belief"], ns["DIRECT_SYSTEM"])
        record = {"prediction": result, "observations": trace}
        print(f"Готов диалог {position}/{size}", flush=True)
        return record

    try:
        # Один поток на ключ. Кэш, счётчики расхода и блокировка записи общие.
        # Назначение по номеру диалога сохраняется и при повторах запроса.
        keys = list(dict.fromkeys(model.keys)) or [None]
        shards = [list(enumerate(rows, 1))[i::len(keys)] for i in range(min(size, len(keys)))]

        def predict_shard(item):
            i, shard = item
            worker = copy.copy(model)
            worker.keys = [keys[i]] if keys[i] else []
            return [(position, predict(position, row, worker)) for position, row in shard]

        with ThreadPoolExecutor(max_workers=min(workers, len(shards))) as pool:
            batches = list(pool.map(predict_shard, enumerate(shards)))
        records = [record for _, record in sorted(entry for group in batches for entry in group)]
    finally:
        elapsed = monotonic() - start
        if used:
            record = dict(finished=datetime.now(timezone.utc).isoformat(), attempts=used, seconds=round(elapsed, 2),
                          budget=budget, http_statuses=dict(statuses), new_responses=int(model.stats().get("api_calls", pd.Series(dtype=int)).sum()))
            with (dest / "run_costs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        print(f"Исходящих попыток: {used}; секунд: {elapsed:.1f}", flush=True)
    frame = pd.DataFrame(r["prediction"] for r in records)
    frame.to_csv(dest / "predictions.csv", index=False)
    metrics = pilot_metrics(frame)
    metrics.to_csv(dest / "metrics.csv", index=False)
    (dest / "observations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    usage = model.stats()
    usage[["calls", "in_tok", "out_tok"]].to_csv(dest / "usage.csv")
    print(metrics.round(3).to_string(index=False))
    print(usage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=(5, 30), type=int, default=5)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=range(1, 16), default=3)
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("budget must be nonnegative")
    os.chdir(Path(__file__).resolve().parents[1])
    run(args.size, args.budget, args.workers)
