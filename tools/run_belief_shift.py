"""Проверка выраженного сдвига убеждения на локальной Gemma. По умолчанию только кэш."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from perspectives import SYSTEM, STANCE, SHIFT_LABELS, SHIFT_SYSTEM, belief_trace, observe, shift_request
from run_access_pairs import collect, local_model, response_format
from run_perspective_pilot import setup

DEST = Path("tom_work/results/belief_shift_e4b")
FIELDS = {"decision":{"judgment":{"enum":list(SHIFT_LABELS)}},
          "observation":{"stance":{"enum":list(STANCE)}, "quote":{"type":"string"}}}
FIELDS["evidence"] = {**FIELDS["decision"], "turn":{"type":"integer"}, "quote":{"type":"string"}}


class CaptureObservation:
    """Получить без исполнения точный запрос и валидатор действующего наблюдателя."""
    def call_json(self, prompt, validate, **kwargs):
        return prompt, validate


def synthetic():
    records = json.loads(Path("tools/belief_shift_pairs.json").read_text())
    for case in records:
        for variant in ("control", "change"):
            turns = [dict(speaker="USER", text=case["initial"]), dict(speaker="BOT", text=case["argument"]),
                     dict(speaker="USER", text=case[variant])]
            yield dict(id=case["id"] + "." + variant, pair=case["id"], variant=variant, kind=case["type"],
                       statement=case["statement"], pre=case["pre"], turns=turns, push=case["push"],
                       gold=case[variant + "_gold"])


def request(case, evidence):
    return shift_request(case["statement"], case["pre"], case["turns"],
                         final_only="pair" in case, evidence=evidence)


def direction(value):
    if value == 0:
        return "stable"
    return "up" if value > 0 else "down"


def signal(judgment):
    return "hold" if judgment in ("unknown","stable") else judgment


def main(budget):
    os.chdir(Path(__file__).resolve().parents[1])
    ns = setup()
    samples = list(synthetic())
    natural = ns["dev"].iloc[100:130]
    assert len(natural) == 30 and not set(natural.id) & set(ns["pilot"].id)
    for row in natural.itertuples(index=False):
        samples.append(dict(id=row.id, statement=row.belief_statement, pre=row.pre_belief, turns=row.turns,
                            post=row.post_belief, gold=direction(row.post_belief-row.pre_belief)))
    DEST.mkdir(exist_ok=True, parents=True)
    ns["CACHE_DIR"] = DEST
    queries, formats, systems = {}, {}, {}
    for case in samples:
        for method in (("legacy","direct","grounded") if "pair" in case else ("direct","grounded")):
            key = case["id"] + ":" + method
            if method == "legacy":
                prompt, validate = observe(CaptureObservation(),case["statement"],case["turns"],len(case["turns"])-1)
                schema, system = "observation", SYSTEM
            else:
                prompt, validate = request(case,method=="grounded")
                schema, system = ("evidence" if method=="grounded" else "decision"), SHIFT_SYSTEM
            queries[key] = (prompt,validate)
            systems[key] = system
            formats[prompt] = response_format(FIELDS[schema])

    model = local_model(ns, formats, {"belief_shift_v1":FIELDS})
    answers = collect(model, ["localhost"], queries, budget, DEST, systems=systems, keep_invalid=True)
    rows = []
    for case in samples:
        for method in ("legacy","direct","grounded"):
            key = case["id"] + ":" + method
            if key not in answers:
                continue
            result = answers[key]
            if "invalid" in result:
                pred = "invalid"
            elif method == "legacy":
                rating = belief_trace([result],case["pre"])[-1]["rating"]
                pred = signal(direction(rating-case["pre"]))
            else:
                pred = result["judgment"]
            usage = model.usage[key]
            rows.append({k:v for k,v in case.items() if k!="turns"} |
                        dict(method=method, pred=pred, signal=signal(pred), result=result,
                             calls=usage["calls"], tokens=usage["in_tok"]+usage["out_tok"]))
    complete = len(answers)==len(queries)
    output = DEST / ("predictions.jsonl" if complete else "partial.jsonl")
    output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    if not complete:
        raise SystemExit(f"Сохранён частичный результат {len(answers)}/{len(queries)}")
    paired = paired_metrics(rows)
    paired.to_csv(DEST / "paired.csv", index=False)
    print(paired.to_string(index=False))
    natural = natural_metrics(rows)
    natural.to_csv(DEST / "puppet.csv", index=False)
    print(natural.to_string(index=False))


def paired_metrics(rows):
    measures = []
    for method in ("legacy", "direct", "grounded"):
        group = [r for r in rows if "pair" in r and r["method"] == method]
        changed = [r for r in group if r["variant"] == "change"]
        controls = [r for r in group if r["variant"] == "control"]
        correct = {r["id"]: r["signal"] == signal(r["gold"]) for r in group}
        both = sum(correct[r["pair"] + ".control"] and correct[r["id"]] for r in changed)
        measures.append(dict(
            method=method,
            n=len(group),
            pairs_both=both,
            direction_correct=sum(correct[r["id"]] for r in changed),
            false_updates=sum(r["signal"] in ("up", "down") for r in controls),
            invalid=sum(r["pred"] == "invalid" for r in group),
            tokens=sum(r["tokens"] for r in group),
        ))
    return pd.DataFrame(measures)


def natural_metrics(rows):
    measures = []
    for method in ("direct", "grounded"):
        group = [r for r in rows if "post" in r and r["method"] == method]
        shifts = [r for r in group if r["pred"] in ("up", "down")]
        stable = [r for r in group if r["pred"] == "stable"]
        unknown = [r for r in group if r["pred"] == "unknown"]
        measures.append(dict(
            method=method,
            n=len(group),
            signed=len(shifts),
            direction_correct=sum(r["pred"] == r["gold"] for r in shifts),
            stable=len(stable),
            stable_exact=sum(r["gold"] == "stable" for r in stable),
            unknown=len(unknown),
            unknown_with_nonzero_shift=sum(r["post"] != r["pre"] for r in unknown),
            invalid=sum(r["pred"] == "invalid" for r in group),
            tokens=sum(r["tokens"] for r in group),
        ))
    return pd.DataFrame(measures)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget",type=int,default=0)
    args = p.parse_args()
    if args.budget < 0:
        p.error("budget must be nonnegative")
    main(args.budget)
