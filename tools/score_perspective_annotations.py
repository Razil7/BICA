"""Сверка двух независимых заполненных копий annotation_blank.csv."""
import argparse
import json
from pathlib import Path

import pandas as pd

from perspectives import STANCE


def read_form(path):
    frame = pd.read_csv(path, keep_default_na=False).set_index(["dialogue", "turn"], verify_integrity=True)
    if frame.empty or not frame.stance.isin(STANCE).all():
        raise ValueError("Для каждой строки укажите одну из пяти категорий")
    for row in frame.itertuples():
        if row.stance == "unclear" and row.quote:
            raise ValueError("Для unclear оставьте quote пустым")
        if row.stance != "unclear" and (not row.quote.strip() or row.quote not in row.words):
            raise ValueError(f"Проверьте цитату в строке {row.Index}")
    return frame.sort_index()


def agreement(a, b):
    if not a[["statement", "words", "previous"]].equals(b[["statement", "words", "previous"]]):
        raise ValueError("Разметчики заполнили разные формы")
    observed = (a.stance == b.stance).mean()
    expected = sum(a.stance.eq(s).mean() * b.stance.eq(s).mean() for s in STANCE)
    return observed, (observed - expected) / (1 - expected) if expected < 1 else float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--observations", type=Path, help="observations.jsonl пилота на тех же 30 диалогах")
    args = parser.parse_args()
    a, b = read_form(args.first), read_form(args.second)
    observed, kappa = agreement(a, b)
    print(f"Реплик: {len(a)}, совпадение: {observed:.1%}, каппа Коэна: {kappa:.3f}")
    print(pd.crosstab(a.stance, b.stance))
    print("Расхождения (диалог, реплика):", list(a.index[a.stance != b.stance]))

    if args.observations:
        agent = {}
        for dialogue, line in enumerate(args.observations.read_text().splitlines(), 1):
            for obs in json.loads(line)["observations"]:
                agent[dialogue, obs["turn"]] = obs["stance"]
        consensus = a[a.stance == b.stance]
        matched = pd.Series([agent[i] for i in consensus.index], index=consensus.index, dtype=str)
        print(f"Совпадение агента с согласованной разметкой: {(matched == consensus.stance).sum()}/{len(consensus)}")
        print(pd.crosstab(consensus.stance, matched, rownames=["люди"], colnames=["модель"]))
