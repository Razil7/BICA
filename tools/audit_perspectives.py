"""Локальная сверка сохранённых результатов и слепая форма для двух разметчиков."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_perspective_pilot import setup


def audit():
    ns = setup(("def belief_metrics", "PRESSURE_TYPES =", "FACTS =", "MENTALMANIP_URL =",
                "DETECT_PRESSURE =", "DETECT_STATEMENT ="))
    root = ns["RESULTS_DIR"]
    out = root / "perspectives_audit"
    out.mkdir(exist_ok=True)
    model = ns["llm"]

    def offline(*args):
        raise RuntimeError("Разбора нет в кэше")

    model._request = offline
    ref = pd.read_csv(root / "table_puppet_test_main.csv", index_col=0)
    metrics = []
    for file, method in (("прямое_предсказание", "прямое предсказание"), ("ToM_правило", "ToM-правило"),
                         ("ToM_суждение_LLM", "ToM + суждение LLM")):
        df = pd.read_csv(root / f"puppet_test_{file}.csv")
        assert df.id.is_unique and set(df.id) == set(ns["test"].id)
        loss = (df.pred - df.post_belief).abs()
        assert loss.notna().all() and abs(loss.mean() - ref.loc[method, "MAE"]) < 0.00051
        metrics.append(dict(method=method, n=len(df), mae=loss.mean(),
                            r_shift=(df.pred - df.pre_belief).corr(df.post_belief - df.pre_belief)))
    pd.DataFrame(metrics).to_csv(out / "baseline.csv", index=False)

    # Параметры финального прогона сохранены в выводе ячейки fit_tracker.
    ns["TRACKER_PARAMS"].update(k_push=0, stance_rate=0.1, recv_last=0.3)
    saved = pd.read_csv(root / "det_test_ToM_правило_формула_темы.csv").set_index("id")
    traces = []
    for row in ns["detect_sample"]("test", 30).itertuples(index=False):
        parses = ns["parse_turns_symmetric"](row.turns, mode="steps")
        if not all(parses):
            raise RuntimeError(f"Неполный разбор {row.id}")
        directions = ns["detect_states"](row.turns, parses)
        max_risk = max(state["max_risk"] for state, _ in directions.values())
        assert np.isclose(saved.loc[row.id, "confidence"], max_risk)
        for speaker, (state, series) in directions.items():
            for point, (turn, risk, parts) in zip(state["trajectory"], series, strict=True):
                traces.append(dict(id=row.id, source=row.source, speaker=speaker, turn=turn,
                                   belief=point["credence"], belief_risk=parts["credence"], risk=risk))
    traces = pd.DataFrame(traces)
    assert set(traces.id) == set(saved.index)
    traces.to_csv(out / "detector_trace.csv", index=False)

    # Разметчики получают одинаковую форму без прогнозов, post, ID участника и условия.
    candidates = []
    for dialogue, row in enumerate(ns["pilot"].head(30).itertuples(index=False), 1):
        for index, turn in enumerate(row.turns):
            if turn["speaker"] == "USER":
                candidates.append(dict(dialogue=dialogue, turn=index + 1, statement=row.belief_statement,
                                       previous=row.turns[index-1]["text"] if index else "", words=turn["text"]))
    candidates.sort(key=lambda x: hashlib.sha256(f"554:{x['dialogue']}:{x['turn']}".encode()).digest())
    blind = pd.DataFrame(candidates[:75]).assign(stance="", quote="")
    blind.to_csv(out / "annotation_blank.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(metrics).to_string(index=False))
    print("Детекция:", traces.id.nunique(), "диалогов,", len(traces), "реплик; значения убеждения:",
          traces.belief.unique(), "; вклад в риск:", traces.belief_risk.unique())


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    audit()
