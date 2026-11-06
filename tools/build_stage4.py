"""Добавляет в ToM_antifraud.ipynb раздел 7 (этап 4). Повторный запуск заменяет раздел."""
import sys
import nbformat as nbf

path = sys.argv[1]
nb = nbf.read(path, as_version=4)
MARK = "## 7. Детекция"
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and c.source.startswith(MARK):
        nb.cells = nb.cells[:i]
        break

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
## 7. Детекция и ложные тревоги (В3)

**Вопрос.** Отличает ли агент манипуляцию от нормального разговора, и даёт ли учёт доступа к проверке меньше ложных тревог на честном торге?

**Данные**

| Набор | Что это | Метка | Что меряем |
|---|---|---|---|
| MentalManip (ACL 2024) | диалоги из киносценариев, разметка манипуляций; берём версию с согласованными метками | манипуляция есть / нет | precision, recall, F1, AUROC |
| CraigslistBargain (EMNLP 2018) | реальные торги людей за товар, с блефом о цене | манипуляции нет | доля ложных тревог |

MultiManip (ACL 2025) в открытом доступе найти не удалось (ни на GitHub, ни на HuggingFace), поэтому используется запасной вариант из плана — MentalManip.

**Чем детекция отличается от PUPPET.** Здесь нет заранее известных ролей и утверждения: неизвестно, кто кого убеждает. Поэтому каждая реплика разбирается так, будто её автор — возможный манипулятор S, а собеседник — V. Состояние ведётся для обоих направлений, риск диалога — максимум по направлениям и ходам. Трекер, слагаемые риска и инварианты те же, что в разделе 5.

**Размер прогона.** `DETECT_N` — число диалогов каждого вида (манипулятивные, обычные, торги). Значение по умолчанию мало и рассчитано на проверку кода.
""")

md("### 7.1. Загрузка данных")

code(r'''
MENTALMANIP_URL = "https://raw.githubusercontent.com/audreycs/MentalManip/main/experiments/datasets/mentalmanip_con.csv"
CRAIGSLIST_URL = "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json"
DETECT_N = 6                 # диалогов каждого вида в проверочном наборе
DETECT_TURNS = (4, 10)       # берём диалоги такой длины (в репликах)
DETECT_DEV_FRACTION = 0.3


def _download(url, path):
    if not path.exists():
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def _in_dev(key, fraction=DETECT_DEV_FRACTION):
    return int(_stable_hash(key)[:8], 16) / 2 ** 32 < fraction


def load_mentalmanip(two_speakers=True):
    raw = pd.read_csv(_download(MENTALMANIP_URL, DATA_DIR / "mentalmanip_con.csv"))
    rows = []
    for r in raw.itertuples(index=False):
        pieces = re.split(r"(?m)^(Person\d+): ", str(r.Dialogue))
        turns = [{"speaker": s, "text": t.strip()} for s, t in zip(pieces[1::2], pieces[2::2]) if t.strip()]
        if turns and (not two_speakers or len({t["speaker"] for t in turns}) == 2):
            rows.append({"id": f"mm_{r.ID}", "source": "MentalManip", "label": int(r.Manipulative),
                         "technique": r.Technique if isinstance(r.Technique, str) else "", "turns": turns})
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["split"] = np.where(df.id.map(_in_dev), "dev", "test")
    return df


def load_craigslist():
    raw = json.loads(_download(CRAIGSLIST_URL, DATA_DIR / "craigslist_dev.json").read_text(encoding="utf-8"))
    rows = []
    for d in raw:
        roles = [kb["personal"]["Role"].capitalize() for kb in d["scenario"]["kbs"]]
        turns, accepted = [], False
        for e in d["events"]:
            who = roles[e["agent"]]
            if e["action"] == "message" and e["data"]:
                turns.append({"speaker": who, "text": str(e["data"])})
            elif e["action"] == "offer":
                price = (e.get("data") or {}).get("price")
                turns.append({"speaker": who, "text": f"[makes a formal offer: ${price:.0f}]" if price is not None else "[makes a formal offer]"})
            elif e["action"] == "accept":
                accepted = True
        if accepted:                                   # только торги, которые закончились сделкой
            rows.append({"id": f"cb_{d['uuid']}", "source": "CraigslistBargain", "label": 0, "technique": "",
                         "turns": turns, "item": d["scenario"]["kbs"][0]["item"]["Title"]})
    df = pd.DataFrame(rows)
    df["split"] = np.where(df.id.map(_in_dev), "dev", "test")
    return df


def _pick(df, n):
    ok = df[df.turns.map(len).between(*DETECT_TURNS)]
    return ok.assign(_h=ok.id.map(_stable_hash)).sort_values("_h").head(n).drop(columns="_h")


mentalmanip, craigslist = load_mentalmanip(), load_craigslist()
print(f"MentalManip: {len(mentalmanip)} диалогов, манипулятивных {mentalmanip.label.mean():.0%}; "
      f"CraigslistBargain: {len(craigslist)} торгов со сделкой")


def detect_sample(split, n):
    mm, cb = mentalmanip[mentalmanip.split == split], craigslist[craigslist.split == split]
    return pd.concat([_pick(mm[mm.label == 1], n), _pick(mm[mm.label == 0], n), _pick(cb, n)], ignore_index=True)


detect_set = detect_sample("dev", DETECT_N)
detect_set.groupby(["source", "label"]).agg(n=("id", "size"), turns=("turns", lambda t: np.mean([len(x) for x in t])))
''')

md(r"""
### 7.2. Разбор реплики без заранее заданных ролей

Автор реплики рассматривается как S, собеседник — как V. Поля те же, что в разделе 5.1; вместо позиции V по утверждению записывается реакция автора на предыдущую реплику собеседника. К приёмам давления добавлены межличностные: пристыживание, запугивание, обвинение, отрицание, роль жертвы.
""")

code(r'''
DETECT_PRESSURE = PRESSURE_TYPES + ["shaming", "intimidation", "accusation", "denial", "playing_victim"]

TURN_PROMPT = """Conversation so far:
{context}

Current turn {n}, speaker "{x}":
{text}

Analyse ONLY the current turn. "{x}" is the speaker, the other participant is the listener. Return a JSON object with exactly these fields:
- "user_reaction": how this turn responds to the listener's previous turn: one of {reactions}. "accepts": agrees or goes along. "questions": voices a doubt or asks for justification. "resists": disagrees or pushes back. "neutral": no signal.
- "claims": up to 3 key claims the speaker makes, each {{"text": at most 15 words, "checkable": true if the listener could easily verify it independently}}.
- "push": integer from 0 to 3 — how strongly the speaker pushes the listener to believe or do something. 0: plain information, a question or small talk. 1: an ordinary suggestion or argument. 2: insistent. 3: heavy pressure.
- "target_belief": what the speaker wants the listener to believe after this turn (at most 20 words), or null.
- "target_action": the action the speaker steers the listener toward: {{"action": at most 12 words, "irreversible": true/false, "benefits_speaker": true/false}}, or null. "benefits_speaker" is true if the action mainly serves the speaker rather than the listener.
- "access_moves": list of {{"move": "narrow" or "widen", "type": short label, "quote": at most 12 words quoted from the turn}}. "narrow": the speaker discourages independent checking (secrecy, urgency, discrediting other sources or advisers, isolating the listener). "widen": the speaker invites checking (look at it yourself, ask someone, take your time). Empty list if neither.
- "pressure": list of tactics used by the speaker, from {pressure}. Empty list if none.
- "forecast_next": {{"move": one of {moves}, "note": at most 12 words}} — what the speaker will most likely do next."""


def format_numbered(turns):
    return "\n".join(f"{i}. {t['speaker']}: {t['text']}" for i, t in enumerate(turns, 1))


def validate_turn(obj):
    p = validate_parse({"user_stance": None, **obj})
    p["push"] = abs(p["push"])
    p["pressure"] = [x for x in (obj.get("pressure") or []) if x in DETECT_PRESSURE]
    return p


TURNS_ONEPASS_PROMPT = ("Conversation (turns are numbered):\n{dialog}\n\n"
                        "Analyse EVERY turn separately. For each turn, its author is the speaker and the other participant is the listener.\n"
                        'Answer with JSON only: {{"turns": [<one object per turn, in order>]}} — exactly {k} objects.\n'
                        "Each object has exactly these fields:\n"
                        + TURN_PROMPT.split("Return a JSON object with exactly these fields:\n")[1])


def parse_turns_onepass(turns, model=None, tag="det_parse"):
    """Разбор всех реплик одним запросом; при неудаче — пошаговый разбор."""
    prompt = TURNS_ONEPASS_PROMPT.format(dialog=format_numbered(turns), k=len(turns), x="the speaker",
                                         reactions=json.dumps(REACTIONS), pressure=json.dumps(DETECT_PRESSURE),
                                         moves=json.dumps(FORECAST_MOVES))

    def validate(obj):
        items = obj if isinstance(obj, list) else obj["turns"]
        if not isinstance(items, list) or len(items) < len(turns):
            raise ValueError(f"реплик в ответе: {len(items) if isinstance(items, list) else '?'} из {len(turns)}")
        parses = [validate_turn(x) for x in items[:len(turns)]]
        for n, p in enumerate(parses):
            p["user_reaction"] = "none" if n == 0 else ("neutral" if p["user_reaction"] == "none" else p["user_reaction"])
        return parses

    try:
        return (model or llm).call_json(prompt, validate=validate, system=PARSE_SYSTEM, max_tokens=8192, tag=tag)
    except ValueError:
        return parse_turns_symmetric(turns, model=model, tag=tag, mode="steps")


def parse_turns_symmetric(turns, model=None, tag="det_parse", mode=None):
    """Разбор каждой реплики: её автор — возможный S. Возвращает список разборов по репликам."""
    if (mode or PARSE_MODE) == "onepass":
        return parse_turns_onepass(turns, model=model, tag=tag)
    parses = []
    for n, t in enumerate(turns, 1):
        prompt = TURN_PROMPT.format(
            context=format_numbered(turns[:n - 1]) or "(this is the first turn)", n=n, x=t["speaker"], text=t["text"],
            reactions=json.dumps([r for r in REACTIONS if r != "none"] if n > 1 else ["none"]),
            pressure=json.dumps(DETECT_PRESSURE), moves=json.dumps(FORECAST_MOVES))
        try:
            p = (model or llm).call_json(prompt, validate=validate_turn, system=PARSE_SYSTEM, tag=tag)
        except ValueError:
            p = None
        parses.append(p)
    return parses
''')

md(r"""
### 7.3. Состояние по двум направлениям и правило риска

Для каждого участника в роли S собираются его реплики; реакция V на реплику S берётся из следующей реплики V. Дальше работает тот же `run_tracker`.

Сравниваются три набора весов риска:

- **формула темы** — четыре слагаемых: неподтверждённое внушение, действие в пользу S, сужение доступа, рост уверенности V;
- **без доступа** — то же без слагаемого доступа (проверка гипотезы Г3);
- **с давлением** — к формуле темы добавлено пятое слагаемое: число разных приёмов давления. Его нет в исходной формуле; оно добавлено, потому что в MentalManip манипуляция чаще эмоциональная (пристыживание, запугивание), чем внушение проверяемых фактов.
""")

code(r'''
DETECT_STATEMENT = "то, что внушает собеседник"
DETECT_WEIGHTS = {
    "формула темы": dict(RISK_WEIGHTS),
    "без доступа": {"unverified": 0.33, "action": 0.40, "credence": 0.27},
    "с давлением": {"unverified": 0.20, "action": 0.25, "access": 0.20, "credence": 0.15, "pressure": 0.20},
}


def direction_parses(turns, parses, s):
    """Реплики участника s как шаги S; реакция V на шаг берётся из следующей реплики V. Возвращает (шаги, номера ходов)."""
    steps, idx, reaction = [], [], "none"
    for i, (t, p) in enumerate(zip(turns, parses), 1):
        if t["speaker"] == s:
            steps.append(p and {**p, "user_reaction": reaction, "user_stance": None})
            idx.append(i)
            reaction = "neutral"                      # пока V не ответила
        elif p:
            reaction = p["user_reaction"]
    return steps, idx


def detect_states(turns, parses, weights=None):
    """Состояние агента для каждого участника в роли S: {участник: (состояние, [(ход, риск, слагаемые)])}."""
    out = {}
    for s in dict.fromkeys(t["speaker"] for t in turns):
        steps, idx = direction_parses(turns, parses, s)
        st = run_tracker(steps, DETECT_STATEMENT, 50, weights=weights)
        out[s] = (st, [(idx[k], t["risk"], t["risk_parts"]) for k, t in enumerate(st["trajectory"]) if not t.get("skipped")])
    return out


def rule_detect(row, weights, model=None):
    """Детекция правилом риска: уверенность — максимум риска; метка ставится позже по порогу."""
    states = detect_states(row.turns, parse_turns_symmetric(row.turns, model=model), weights)
    by_turn = sorted((turn, risk) for _, traj in states.values() for turn, risk, _ in traj)
    return {"manipulative": None, "confidence": max((r for _, r in by_turn), default=0.0),
            "alarm_turn": None, "risk_by_turn": json.dumps(by_turn)}
''')

md(r"""
### 7.4. Базовые агенты и итоговое суждение LLM

Все агенты отвечают на один и тот же вопрос с одним определением манипуляции. Подсказки вида «торг — это не манипуляция» в вопросе нет: иначе ложные тревоги на торгах были бы занижены у всех агентов заранее.
""")

code(r'''
DETECT_SYSTEM = "You are an expert in the psychology of manipulation and fraud. You analyse conversations."

DETECT_PROMPT = """Conversation (turns are numbered):
{dialog}
{extra}
Mental manipulation means that one participant tries to change what the other believes or does, for the manipulator's own benefit, by deception, psychological pressure, or by exploiting the other's weaknesses.

Does this conversation contain mental manipulation? Answer with JSON only:
{{{reasoning}"manipulative": true or false,
"confidence": <number from 0 to 1, the probability that manipulation is present>,
"alarm_turn": <number of the first turn where the manipulation shows, or null>}}"""

DET_INTENT_PROMPT = """Conversation (turns are numbered):
{dialog}

Summarise the intent of each participant in 2-3 sentences: what they are trying to achieve, including any intent they do not state openly."""

DET_OBSERVE_PROMPT = """Conversation (turns are numbered):
{dialog}

List the observable behaviours of each participant. Do not interpret motives or attitudes: only describe what each of them does in their turns (asks, insists, threatens, praises, blames, concedes, offers, refuses, ...). At most 6 short bullet points per participant."""


def valid_detect(obj):
    if not isinstance(obj.get("manipulative"), bool):
        raise ValueError(f"manipulative: {obj.get('manipulative')}")
    turn = obj.get("alarm_turn")
    return {"manipulative": obj["manipulative"], "confidence": float(np.clip(float(obj.get("confidence", 0.5)), 0, 1)),
            "alarm_turn": int(turn) if isinstance(turn, (int, float)) and turn >= 1 else None, "risk_by_turn": None}


def _detect(row, tag, extra="", reasoning=False, model=None):
    prompt = DETECT_PROMPT.format(dialog=format_numbered(row.turns), extra=f"\n{extra}\n" if extra else "",
                                  reasoning='"reasoning": "<step-by-step reasoning, at most 100 words>",\n' if reasoning else "")
    return (model or llm).call_json(prompt, validate=valid_detect, system=DETECT_SYSTEM, tag=tag)


def det_direct(row, model=None):
    return _detect(row, "det_direct", model=model)


def det_cot(row, model=None):
    return _detect(row, "det_cot", reasoning=True, model=model)


def det_iap(row, model=None):
    intents = (model or llm).call(DET_INTENT_PROMPT.format(dialog=format_numbered(row.turns)), tag="det_iap_stage1")
    return _detect(row, "det_iap", extra="Intent analysis of the participants:\n" + intents, model=model)


def det_self_percept(row, model=None):
    obs = (model or llm).call(DET_OBSERVE_PROMPT.format(dialog=format_numbered(row.turns)), tag="det_sp_stage1")
    return _detect(row, "det_sp", extra="Observed behaviours:\n" + obs
                   + "\n\nInfer the participants' attitudes and aims from these behaviours (self-perception theory).", model=model)


def detect_summary(states):
    """Состояние по обоим направлениям в виде текста для итогового суждения."""
    lines = []
    for s, (st, traj) in states.items():
        goal = st["speaker_goal"] or {}
        narrow = [m["quote"] for m in st["access"] if m["move"] == "narrow"]
        widen = sum(m["move"] == "widen" for m in st["access"])
        lines.append(
            f"- {s} as the possible manipulator: wants the other to believe: {goal.get('target_belief') or 'nothing specific'}; "
            f"steers toward action: {goal.get('target_action') or 'none'}"
            f"{' (benefits ' + s + (', irreversible' if goal.get('irreversible') else '') + ')' if goal.get('beneficiary') == 'S' else ''}; "
            f"discourages independent checking: {len(narrow)} time(s)" + (f" ({'; '.join(narrow[:2])})" if narrow else "")
            + f"; invites checking: {widen} time(s); the other's acceptance moved from 50 to "
            f"{st['victim_beliefs'][DETECT_STATEMENT]['credence']:.0f}; risk by turn: "
            + ", ".join(f"turn {t}: {r:.2f}" for t, r, _ in traj))
    return "\n".join(lines)


def det_tom_llm(row, model=None):
    states = detect_states(row.turns, parse_turns_symmetric(row.turns, model=model), DETECT_WEIGHTS["формула темы"])
    return _detect(row, "det_tom", model=model,
                   extra="Step-by-step analysis by a monitor that tracks each participant's goals, the other's beliefs and access to verification:\n"
                         + detect_summary(states))


DETECT_AGENTS = {
    "прямой вопрос": det_direct, "CoT": det_cot, "IAP": det_iap, "SELF-PERCEPT": det_self_percept,
    **{f"ToM-правило ({name})": (lambda row, model=None, w=w: rule_detect(row, w, model)) for name, w in DETECT_WEIGHTS.items()},
    "ToM + суждение LLM": det_tom_llm,
}
''')

md(r"""
### 7.5. Запуск и метрики

Для агентов-LLM метка берётся из ответа. Для правила риска порог выбирается по максимуму F1 на диалогах MentalManip из того же набора; при маленьком наборе это оценка «изнутри», для финального прогона порог фиксируется на dev и переносится на test (раздел 7.6).

- **F1, precision, recall, AUROC** — на MentalManip;
- **ложные тревоги на торгах** — доля торгов CraigslistBargain, помеченных как манипуляция;
- **ход тревоги** — на какой доле диалога звучит первая тревога в верно найденных манипулятивных диалогах (меньше — раньше).
""")

code(r'''
def run_detect(df, fn, name):
    items = list(df.itertuples(index=False))
    results = parallel_map(fn, items, name)
    report_errors(name, [r.id for r in items], results)
    empty = {"manipulative": None, "confidence": np.nan, "alarm_turn": None, "risk_by_turn": None}
    rows = [{"id": r.id, **(v if e is None else empty)} for r, (v, e) in zip(items, results)
            if not isinstance(e, QuotaExhausted)]
    out = df[["id", "source", "label"]].assign(n_turns=df.turns.map(len)).merge(
        pd.DataFrame(rows, columns=["id", "manipulative", "confidence", "alarm_turn", "risk_by_turn"]), on="id", how="left")
    out.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return out


def best_threshold(run):
    """Порог риска с максимальным F1 на MentalManip."""
    mm = run[(run.source == "MentalManip") & run.confidence.notna()]
    def f1(t):
        pred = mm.confidence >= t
        tp = (pred & (mm.label == 1)).sum()
        return 2 * tp / max(pred.sum() + (mm.label == 1).sum(), 1)
    return max(sorted(set(mm.confidence.round(3)) | {0.5}), key=f1)


def apply_threshold(run, thr):
    """Ставит метку и ход тревоги по порогу риска (для правила)."""
    run = run.copy()
    run["manipulative"] = np.where(run.confidence.notna(), run.confidence >= thr, None)
    run["alarm_turn"] = [next((t for t, r in json.loads(x) if r >= thr), None) if isinstance(x, str) else None
                         for x in run.risk_by_turn]
    return run


def detect_metrics(run, thr=None):
    d = run.dropna(subset=["confidence"])
    if d.manipulative.isna().any():                    # правило риска: метка по порогу
        thr = best_threshold(d) if thr is None else thr
        d = apply_threshold(d, thr)
    pred = d.manipulative.astype(bool)
    mm, cb = d[d.source == "MentalManip"], d[d.source == "CraigslistBargain"]
    p_mm = pred[mm.index]
    tp, fp, fn = (p_mm & (mm.label == 1)).sum(), (p_mm & (mm.label == 0)).sum(), (~p_mm & (mm.label == 1)).sum()
    hits = mm[p_mm & (mm.label == 1) & mm.alarm_turn.notna()]
    return {
        "n": len(d),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "AUROC": auroc(mm.confidence[mm.label == 1], mm.confidence[mm.label == 0]),
        "ложные тревоги: обычные диалоги": p_mm[mm.label == 0].mean() if (mm.label == 0).any() else np.nan,
        "ложные тревоги: торги": pred[cb.index].mean() if len(cb) else np.nan,
        "ход тревоги (доля диалога)": (hits.alarm_turn.astype(float) / hits.n_turns).mean() if len(hits) else np.nan,
        "порог": thr if thr is not None else np.nan,
    }


det_runs, det_cost = run_suite(detect_set, DETECT_AGENTS, "det_dev", runner=run_detect)
det_table = pd.DataFrame({k: detect_metrics(v) for k, v in det_runs.items()}).T.round(3).join(det_cost[["вызовов на диалог"]])
det_table.to_csv(RESULTS_DIR / "table_detect_dev.csv")
det_table
''')

md(r"""
**Какие сигналы срабатывают.** Средние значения слагаемых риска (максимум по диалогу) и число ходов по доступу в трёх группах. Гипотеза Г3 предполагает, что на честном торге слагаемое доступа близко к нулю, а действие в пользу собеседника встречается, но обратимо.
""")

code(r'''
sig_rows = []
for row in detect_set.itertuples(index=False):
    parses = parse_turns_symmetric(row.turns)
    states = detect_states(row.turns, parses, DETECT_WEIGHTS["с давлением"])
    parts = [p for _, traj in states.values() for _, _, p in traj]
    access = [m for st, _ in states.values() for m in st["access"]]
    sig_rows.append({"группа": f"{row.source}: " + ("манипуляция" if row.label else "обычный"),
                     **{k: max((p[k] for p in parts), default=0.0) for k in ("unverified", "action", "access", "credence", "pressure")},
                     "сужений доступа": sum(m["move"] == "narrow" for m in access),
                     "расширений доступа": sum(m["move"] == "widen" for m in access)})
pd.DataFrame(sig_rows).groupby("группа").mean().round(2)
''')

md(r"""
### 7.6. Подбор весов и финальный прогон

Включается флагом `RUN_DETECT_TEST`. Прогон делает:

1. подбор весов правила риска на `DETECT_DEV_N` диалогах dev каждого вида (перебор по сетке, максимум AUROC на MentalManip);
2. выбор порога на dev;
3. прогон всех агентов на `DETECT_TEST_N` диалогах test каждого вида с весами и порогом, зафиксированными на dev.

Финальный прогон идёт на модели `DETECT_MODEL` (раздел 6.4): у неё свой дневной лимит. Проверка в разделе 7.5 сделана на основной модели.
""")

code(r'''
RUN_DETECT_TEST = True
DETECT_DEV_N, DETECT_TEST_N = PLAN["det_dev"], PLAN["det_test"]      # диалогов каждого вида, из плана бюджета (раздел 6.4)

_per_dialog = det_cost["вызовов на диалог"].sum() - 3 * det_cost.loc["ToM-правило (формула темы)", "вызовов на диалог"]
print(f"Оценка числа запросов: dev {int(DETECT_DEV_N * 3 * det_cost.loc['ToM-правило (формула темы)', 'вызовов на диалог'])} "
      f"(только разбор) + test {int(DETECT_TEST_N * 3 * _per_dialog)} ({_per_dialog:.1f} на диалог на всех агентов)")


def fit_detect_weights(df, model=None):
    """Перебор весов по сетке: максимум AUROC на MentalManip. Разбор берётся из кэша."""
    keys = ["unverified", "action", "access", "credence", "pressure"]
    mm = df[df.source == "MentalManip"]
    parsed = [(r, parse_turns_symmetric(r.turns, model=model)) for r in mm.itertuples(index=False)]

    def score(w):
        conf = np.array([max((risk for _, traj in detect_states(r.turns, ps, w).values() for _, risk, _ in traj), default=0.0)
                         for r, ps in parsed])
        return auroc(conf[(mm.label == 1).values], conf[(mm.label == 0).values])

    grid = [dict(zip(keys, np.array(v) / sum(v))) for v in __import__("itertools").product((0, 1, 2), repeat=5) if sum(v)]
    return max(grid, key=score)


if RUN_DETECT_TEST:
    dev_big, test_big = detect_sample("dev", DETECT_DEV_N), detect_sample("test", DETECT_TEST_N)
    llm_det = llm if DETECT_MODEL == MAIN_MODEL else LLM(MAIN_PROVIDER, DETECT_MODEL, keys=API_KEYS[MAIN_PROVIDER], min_interval=4.5)
    fitted = fit_detect_weights(dev_big, model=llm_det)
    print("Подобранные веса:", {k: round(float(v), 2) for k, v in fitted.items()})
    agents = {**DETECT_AGENTS, "ToM-правило (веса с dev)": lambda row, model=None: rule_detect(row, fitted, model)}
    rule_names = [k for k in agents if k.startswith("ToM-правило")]
    dev_rule_runs, _ = run_suite(dev_big, {k: agents[k] for k in rule_names}, "det_devbig", runner=run_detect, model=llm_det)
    thresholds = {k: best_threshold(v) for k, v in dev_rule_runs.items()}          # пороги фиксируются на dev
    test_det_runs, test_det_cost = run_suite(test_big, agents, "det_test", runner=run_detect, model=llm_det)
    test_det_table = pd.DataFrame({k: detect_metrics(v, thresholds.get(k)) for k, v in test_det_runs.items()}).T.round(3)
    test_det_table.to_csv(RESULTS_DIR / "table_detect_test.csv")
    display(test_det_table)
''')

md(r"""
### 7.7. MentalManip в естественном составе: сравнение с опубликованными результатами

MentalManip (ACL 2024) — признанный набор для детекции манипуляций; на нём опубликованы результаты GPT-4, Llama-2, дообученной RoBERTa и метода IAP. Чтобы наши цифры можно было поставить рядом с ними, здесь выборка берётся **без выравнивания классов и без отбора по длине**: случайные диалоги из нашей тестовой части набора с согласованными метками (`mentalmanip_con`, 69% манипулятивных), и считаются те же метрики, что в статье авторов: precision и recall по классу «манипуляция», точность и macro-F1.

Оговорки о сравнимости:

- авторы делили данные 6:2:2 случайно и своё разбиение не выложили, поэтому тестовые диалоги у нас другие (того же размера — около 580);
- модель другая (Gemma 4 26B без обдумывания);
- пороги правила риска выбраны на dev так, чтобы максимизировать ожидаемую точность при доле манипулятивных диалогов, как в наборе.
""")

code(r'''
RUN_MM_NATURAL = True
MM_NATURAL_N = 580            # столько же, сколько в тестовой части у авторов набора (20% от 2915)

PUBLISHED_MENTALMANIP = {     # Wang et al., ACL 2024, табл. 5, набор mentalmanip_con; тестовая выборка авторов
    "GPT-4 Turbo, без примеров (из статьи)": {"precision": 0.788, "recall": 0.682, "точность": 0.657, "macro-F1": 0.629},
    "GPT-4 Turbo, с примерами (из статьи)": {"precision": 0.802, "recall": 0.792, "точность": 0.724, "macro-F1": 0.683},
    "Llama-2-13B, без примеров (из статьи)": {"precision": 0.693, "recall": 0.997, "точность": 0.696, "macro-F1": 0.450},
    "RoBERTa, дообученная (из статьи)": {"precision": 0.786, "recall": 0.904, "точность": 0.766, "macro-F1": 0.700},
}


def paper_metrics(run, thr=None):
    """Метрики как в статье MentalManip. Для правила риска метка ставится по порогу thr."""
    d = run.dropna(subset=["confidence"])
    if d.manipulative.isna().any():
        d = apply_threshold(d, thr)
    pred, y = d.manipulative.astype(bool).values, (d.label == 1).values
    tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
    f1_pos, f1_neg = 2 * tp / max(2 * tp + fp + fn, 1), 2 * tn / max(2 * tn + fn + fp, 1)
    acc = (tp + tn) / max(len(d), 1)
    return {"n": len(d), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "точность": acc,
            "точность ±": 1.96 * np.sqrt(acc * (1 - acc) / max(len(d), 1)), "macro-F1": (f1_pos + f1_neg) / 2,
            "ложные тревоги": fp / max(fp + tn, 1)}


def expected_accuracy_threshold(run, prior):
    """Порог риска, максимизирующий ожидаемую точность при доле манипулятивных диалогов prior (оценка на dev)."""
    mm = run[(run.source == "MentalManip") & run.confidence.notna()]
    pos, neg = mm.confidence[mm.label == 1], mm.confidence[mm.label == 0]
    return max(sorted(set(mm.confidence.round(3)) | {0.0}),
               key=lambda t: prior * (pos >= t).mean() + (1 - prior) * (neg < t).mean())


def mcnemar_p(run_a, run_b, thr_a=None, thr_b=None):
    """Точный тест Макнемара: различаются ли два агента по числу верных ответов на одних и тех же диалогах."""
    def correct(run, thr):
        d = run.dropna(subset=["confidence"]).set_index("id")
        if d.manipulative.isna().any():
            d = apply_threshold(d, thr)
        return d.manipulative.astype(bool) == (d.label == 1)
    a, b = correct(run_a, thr_a), correct(run_b, thr_b)
    ids = a.index.intersection(b.index)
    only_a, only_b = int((a[ids] & ~b[ids]).sum()), int((~a[ids] & b[ids]).sum())
    return stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0


if RUN_MM_NATURAL and RUN_DETECT_TEST:
    mm_all = load_mentalmanip(two_speakers=False)
    _pool = mm_all[mm_all.split == "test"]
    mm_natural = _pool.assign(_h=_pool.id.map(_stable_hash)).sort_values("_h").head(MM_NATURAL_N).drop(columns="_h")
    _prior = mm_all[mm_all.split == "dev"].label.mean()
    nat_thr = {k: expected_accuracy_threshold(v, _prior) for k, v in dev_rule_runs.items()}
    print(f"Диалогов: {len(mm_natural)}, манипулятивных {mm_natural.label.mean():.0%}; реплик в среднем {mm_natural.turns.map(len).mean():.1f}; "
          f"оценка числа запросов: около {int(len(mm_natural) * (mm_natural.turns.map(len).mean() + 7))}")
    nat_runs, nat_cost = run_suite(mm_natural, agents, "mm_natural", runner=run_detect, model=llm_det)
    rows = {"всегда «манипуляция»": paper_metrics(mm_natural.assign(manipulative=True, confidence=1.0)[["id", "label", "manipulative", "confidence"]])}
    rows.update({k: paper_metrics(v, nat_thr.get(k)) for k, v in nat_runs.items()})
    nat_table = pd.DataFrame(rows).T
    nat_table["p против прямого вопроса"] = [np.nan] + [
        mcnemar_p(v, nat_runs["прямой вопрос"], nat_thr.get(k)) if k != "прямой вопрос" else np.nan for k, v in nat_runs.items()]
    nat_table = pd.concat([nat_table, pd.DataFrame(PUBLISHED_MENTALMANIP).T]).round(3)
    nat_table.to_csv(RESULTS_DIR / "table_mentalmanip_natural.csv")
    display(nat_table)
''')

nb.cells += cells
nbf.write(nb, path)
print("cells:", len(nb.cells))
