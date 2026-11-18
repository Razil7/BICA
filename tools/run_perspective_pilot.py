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


def setup(extra_cells=(), *, load_puppet=True):
    # Используем те же определения, ключи кэша и разбиение, что в выполненном ноутбуке.
    cells = json.loads(Path("ToM_antifraud.ipynb").read_text())["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    ns = {}
    data_cells = ("PUPPET_URL =", "SPEAKER_NAMES =") if load_puppet else ()
    for prefix in ("import os, sys, re", "GEMINI_API_KEY =", "def _read_env_file",
                   "class QuotaExhausted", *data_cells, *extra_cells):
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
    run_started = datetime.now(timezone.utc).isoformat()
    dest = ns["RESULTS_DIR"] / f"perspectives_dev_{size}"
    dest.mkdir(exist_ok=True)

    def limited_post(*args):
        nonlocal used, next_attempt
        while True:
            with lock:
                if used >= budget:
                    raise RuntimeError("Лимит исходящих запросов исчерпан")
                wait = next_attempt - monotonic()
                if wait <= 0:
                    used += 1
                    # При остановке процесса finally может не выполниться.
                    # Запись до отправки даёт верхнюю границу фактического расхода.
                    with (dest / "attempts.jsonl").open("a", encoding="utf-8") as ledger:
                        ledger.write(json.dumps({"started": run_started, "attempt": used}) + "\n")
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
    start = monotonic()

    def predict(position, row, worker):
        result, trace = pilot_predictions(row, worker, ns["predict_direct"], ns["format_dialog"],
                                          ns["valid_belief"], ns["DIRECT_SYSTEM"])
        record = {"prediction": result, "observations": trace}
        print(f"Готов диалог {position}/{size}", flush=True)
        return record

    try:
        # Назначение ключа сохраняется при повторах. Независимые диалоги
        # могут продолжаться, пока один запрос ожидает повтора после HTTP 503.
        keys = list(dict.fromkeys(model.keys)) or [None]
        clients = []
        for key in keys:
            worker = copy.copy(model)
            worker.keys = [key] if key else []
            clients.append(worker)

        def predict_dialogue(item):
            position, row = item
            return predict(position, row, clients[(position - 1) % len(clients)])

        with ThreadPoolExecutor(max_workers=min(workers, size)) as pool:
            records = list(pool.map(predict_dialogue, enumerate(rows, 1)))
    finally:
        elapsed = monotonic() - start
        if used:
            record = dict(started=run_started, finished=datetime.now(timezone.utc).isoformat(), attempts=used, seconds=round(elapsed, 2),
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
    # После успешного пилота точка продолжения должна отражать готовый набор.
    counts = [len(record["observations"]) for record in records]
    checkpoint = frame.assign(dialogue=range(1, len(frame) + 1), ready=True,
                              observations=counts, total=counts)
    checkpoint.to_csv(dest / "checkpoint.tmp", index=False)
    (dest / "checkpoint.tmp").replace(dest / "checkpoint.csv")
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
