"""Запуск из корня BICA: python tools/run_access_pairs.py [--budget 100]. Ноль = только кэш."""
import argparse
import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep

import pandas as pd

from access_pairs import (baseline_prompt, cases, controller, gold, interpretation_prompt,
                          parse_decision, parse_stance, public_message)
from run_perspective_pilot import setup

SYSTEM = "Выполни задачу по приведённым данным. Ответь в запрошенном формате JSON."
STANCE_FIELDS = {"stance": {"enum": ["old", "current", "unknown"]}}
DECISION_FIELDS = {"world": {"type": "string"}, "seen": {"type": "boolean"},
                   "source": {"type": ["string", "null"]}, **STANCE_FIELDS,
                   "action": {"enum": ["inform", "correct", "acknowledge"]},
                   "public_value": {"type": ["string", "null"]}}


def response_format(fields):
    schema = dict(type="object", properties=fields, required=list(fields), additionalProperties=False)
    return {"type": "json_schema", "json_schema": dict(name="decision", strict=True, schema=schema)}


class BudgetStop(Exception):
    pass


def design():
    queries, plan = {}, []
    for pair, seen, events in cases():
        row = dict(pair=pair, seen=seen, events=events)
        for method in ("direct", "summary"):
            key = f"{pair}.{int(seen)}.{method}"
            queries[key] = baseline_prompt(events, method == "summary"), parse_decision
            row[method] = key
        if events[-1]["kind"] == "speech":
            key = pair + ".interpret"
            q = interpretation_prompt(events), parse_stance
            assert key not in queries or queries[key] == q
            queries[key] = q
            row["controller"] = key
        plan.append(row)
    return queries, plan


def collect(model, keys, queries, budget, dest, systems=None, keep_invalid=False):
    lock = threading.Lock()
    used, next_time = 0, 0.0
    original_post = model._post

    def post(*args):
        nonlocal used, next_time
        with lock:
            if used >= budget:
                raise BudgetStop
            used += 1
            start = max(monotonic(), next_time)
            next_time = start + 1.5
            with (dest / "attempts.jsonl").open("a") as f:
                f.write(json.dumps({"attempt": used, "run": run_id}) + "\n")
        sleep(max(0, start - monotonic()))
        response = original_post(*args)
        if response.status_code != 200:
            print(f"HTTP {response.status_code}", flush=True)
        return response

    def cache_miss(*args):
        raise BudgetStop

    run_id = pd.Timestamp.now(tz="UTC").isoformat()
    clients = []
    for key in keys or [None]:
        client = copy.copy(model)
        client.keys = [key] if key else []
        client._post = post
        client.max_retries = 2
        if not budget or not key:
            client._request = cache_miss
        clients.append(client)

    def answer(item):
        i, key = item
        # Объект модели и ключ этого задания сохраняются между попытками.
        client = clients[i % len(clients)]
        prompt, validate = queries[key]
        try:
            result = client.call_json(prompt, system=(systems or {}).get(key, SYSTEM), validate=validate, retries=1,
                                      max_tokens=256, tag=key)
            return key, result
        except BudgetStop:
            return key, None
        except ValueError as exc:
            if keep_invalid:
                return key, {"invalid": str(exc)}
            return key, None
        except RuntimeError as exc:
            print(f"{key}: {type(exc).__name__}", flush=True)
            return key, None

    pending = list(enumerate(queries))
    answers = {}
    while pending:
        before = used
        with ThreadPoolExecutor(max_workers=3) as pool:
            for key, value in pool.map(answer, pending):
                if value is not None:
                    answers[key] = value
        pending = [(i, k) for i, k in pending if k not in answers]
        print(f"Ответов {len(answers)}/{len(queries)}; HTTP-попыток {used}/{budget}", flush=True)
        if used >= budget or used == before:
            break
    return answers


def evaluate(plan, answers, model):
    rows = []
    for case in plan:
        pair, seen, events = case["pair"], case["seen"], case["events"]
        expected = gold(pair, seen)
        for method in ("direct", "summary", "controller"):
            key = case.get(method)
            if key and key not in answers:
                continue
            audit = {}
            actual = (controller(events, lambda _: answers[key], trace=audit)
                      if method == "controller" else answers[key])
            u = model.usage[key] if key else {}
            # Цена одного логического решения на холодном кэше. Один и тот же
            # контент интерпретации у близнецов физически запрашивается один раз.
            rows.append(dict(pair=pair, exposed=seen, method=method, gold=expected, prediction=actual,
                             trace=audit, public_message=public_message(actual),
                             world_ok=actual["world"] == expected["world"],
                             access_ok=(actual["seen"], actual["source"]) == (seen, expected["source"]),
                             stance_ok=actual["stance"] == expected["stance"],
                             public_ok=all(actual[k] == expected[k] for k in ("action","public_value")),
                             leak=not seen and actual["seen"], exact=actual == expected,
                             calls=u.get("calls",0), tokens=u.get("in_tok",0)+u.get("out_tok",0)))
    return rows


def local_model(ns, formats, schema_identity):
    options = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    model = ns["LLM"]("lmstudio", "google/gemma-4-e4b", keys=["localhost"], min_interval=0,
                      extra={**options, **schema_identity})

    def local_post(key, prompt, system, json_mode, temperature, max_tokens):
        body = dict(model=model.model, temperature=temperature, max_tokens=max_tokens,
                    messages=[dict(role="system", content=system), dict(role="user", content=prompt)],
                    response_format=formats[prompt], **options)
        return ns["requests"].post("http://127.0.0.1:1234/v1/chat/completions", json=body, timeout=120)
    model._post = local_post
    return model


def main(budget, local=False):
    os.chdir(Path(__file__).resolve().parents[1])
    ns = setup(load_puppet=False)
    dest = Path("tom_work/results") / ("access_pairs_local_v1" if local else "access_pairs_v1")
    dest.mkdir(parents=True, exist_ok=True)
    ns["CACHE_DIR"] = dest
    queries, plan = design()
    if local:
        formats = {p: response_format(STANCE_FIELDS if validate is parse_stance else DECISION_FIELDS)
                   for p, validate in queries.values()}
        model = local_model(ns, formats, {"decision_schema": DECISION_FIELDS, "stance_schema": STANCE_FIELDS})
    else:
        model = ns["LLM"](ns["MAIN_PROVIDER"], ns["MAIN_MODEL"], keys=ns["API_KEYS"][ns["MAIN_PROVIDER"]],
                          min_interval=ns["MIN_INTERVAL"], extra=ns["MAIN_EXTRA"], tpm=ns["MAIN_TPM"])
    with (dest / "cases.jsonl").open("w", encoding="utf-8") as f:
        for case in plan:
            sample = {k: case[k] for k in ("pair", "seen", "events")}
            sample["gold"] = gold(case["pair"], case["seen"])
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    answers = collect(model, list(dict.fromkeys(model.keys)), queries, budget, dest)
    rows = evaluate(plan, answers, model)
    complete = len(answers) == len(queries)
    target = dest / ("decisions.jsonl" if complete else "partial.jsonl")
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if not complete:
        raise SystemExit("Часть ответов отсутствует; результат сохранён в partial.jsonl")
    df = pd.DataFrame(rows)
    out = []
    for name, g in df.groupby("method", sort=False):
        out.append(dict(method=name, n=len(g), pairs=len(g)//2, world_ok=g.world_ok.sum(),
                        access_ok=g.access_ok.sum(), stance_ok=g.stance_ok.sum(), public_ok=g.public_ok.sum(),
                        exact_pairs=g.groupby("pair").exact.all().sum(),
                        public_pairs=g.groupby("pair").public_ok.all().sum(),
                        leaks=g.leak.sum(), private_cases=(~g.exposed).sum(),
                        calls=g.calls.sum(), tokens=g.tokens.sum()))
    summary = pd.DataFrame(out)
    summary.to_csv(dest / "metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--local", action="store_true", help="Установленная Gemma E4B через LM Studio на localhost")
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("Лимит должен быть неотрицательным")
    main(args.budget, args.local)
