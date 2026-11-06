"""Добавляет в ToM_antifraud.ipynb разделы 8–12 (этап 5). Повторный запуск заменяет их.
Текст выводов (раздел 12) берётся из файла conclusions.md рядом со скриптом, если он есть."""
import sys
from pathlib import Path
import nbformat as nbf

path = sys.argv[1]
nb = nbf.read(path, as_version=4)
MARK = "## 8. Абляции"
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and c.source.startswith(MARK):
        nb.cells = nb.cells[:i]
        break

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
## 8. Абляции

Что даёт каждая часть агента. Все варианты считаются на первых `ABL_N` диалогах пилота агента (`tom_pilot`); разбор берётся из кэша, новые запросы нужны только для итогового суждения и для разбора «одним запросом».

| Вариант | Что убрано |
|---|---|
| без траектории | из анализа для итогового суждения убрана траектория убеждений V и её реакции |
| без цели S | убраны толчки, внушаемое убеждение, действие и гипотеза о скрытой выгоде |
| без доступа | убраны ходы по доступу к проверке |
| без исходного убеждения | трекер стартует от 50, а не от исходной оценки V |
| другой режим разбора | если основной режим пошаговый — весь диалог разбирается одним вызовом, и наоборот |

Для ToM-правила отдельно: без учёта реакции V, без регрессии к среднему, без исходного убеждения.
""")

code(r'''
def predict_tom_variant(row, parses, parts=("trajectory", "goal", "access"), use_pre=True, tag="tom_final", model=None):
    """Итоговое суждение по состоянию с заданными блоками анализа."""
    state = puppet_state(row, parses, use_pre=use_pre)
    prompt = FINAL_PROMPT.format(statement=row.belief_statement, pre=row.pre_belief, dialog=format_dialog(row.turns),
                                 analysis=state_summary(state, parses, row.belief_statement, parts))
    return (model or llm).call_json(prompt, validate=valid_belief, system=FINAL_SYSTEM, tag=tag)


# другой режим разбора: если основной — пошаговый, проверяем разбор одним запросом, и наоборот
ABL_PARSE_MODE = "steps" if PARSE_MODE == "onepass" else "onepass"
ABL_PARSE_LABEL = "  разбор одним запросом" if ABL_PARSE_MODE == "onepass" else "  пошаговый разбор"

ABLATIONS = {
    "ToM + суждение LLM (полный)": lambda r: predict_tom_variant(r, pilot_parses[r.id]),
    "  без траектории": lambda r: predict_tom_variant(r, pilot_parses[r.id], parts=("goal", "access"), tag="abl"),
    "  без цели S": lambda r: predict_tom_variant(r, pilot_parses[r.id], parts=("trajectory", "access"), tag="abl"),
    "  без доступа": lambda r: predict_tom_variant(r, pilot_parses[r.id], parts=("trajectory", "goal"), tag="abl"),
    "  без исходного убеждения": lambda r: predict_tom_variant(r, pilot_parses[r.id], use_pre=False, tag="abl"),
    ABL_PARSE_LABEL: lambda r: predict_tom_variant(r, tom_parses(r, mode=ABL_PARSE_MODE), tag="abl"),
}
ABL_N = 50            # диалогов для абляций (первые из пилота агента); каждый стоит 6 запросов
abl_set = tom_pilot[tom_pilot.id.isin(pilot_parses)].head(ABL_N)
abl_runs, abl_cost = run_suite(abl_set, ABLATIONS, "abl")


def rule_run(df, params=None, use_pre=True, drift=DRIFT):
    pred = [run_tracker(pilot_parses[r.id], r.belief_statement, r.pre_belief, params=params, use_pre=use_pre,
                        drift=drift)["victim_beliefs"][r.belief_statement]["credence"] for r in df.itertuples(index=False)]
    return df.assign(pred=pred)


# Правило: параметры подобраны на этих же диалогах, поэтому все варианты считаются «внутри выборки» и сравнимы только между собой
abl_runs.update({
    "ToM-правило (полное, внутри выборки)": rule_run(abl_set),
    "  без реакции V": rule_run(abl_set, params={**TRACKER_PARAMS, "recv": {k: 1.0 for k in TRACKER_PARAMS["recv"]}}),
    "  без регрессии к среднему": rule_run(abl_set, drift=None),
    "  без исходного убеждения ": rule_run(abl_set, use_pre=False),
})
abl_table = metrics_table(abl_runs)
abl_table.to_csv(RESULTS_DIR / "table_ablations.csv")
abl_table
''')

md(r"""
## 9. Вторая семья моделей

Проверка, что результаты не привязаны к одной модели. Прогоняются прямое предсказание и ToM-агент на первых диалогах пилота агента; рядом для сравнения — основная модель на тех же диалогах. Параметры ToM-правила взяты те же, что подобраны на основной модели.

| Модель | Откуда | Объём |
|---|---|---|
| `gemini-3.5-flash-lite` | тот же ключ Gemini | 30 диалогов; дневной лимит 500 запросов на ключ |
| `openai/gpt-oss-20b` | Groq | 5 диалогов; мешает дневной лимит токенов Groq |

В теме вторая семья — Llama через Groq, но по доступным ключам моделей Llama на Groq нет.
""")

code(r"""
SECOND_MODELS = [
    {"provider": "gemini", "model": "gemini-3.5-flash-lite", "interval": 4.5, "per_key": 1, "n": 30},
    {"provider": "groq", "model": "openai/gpt-oss-20b", "extra": {"reasoning_effort": "low"}, "interval": 4.5, "per_key": 1, "n": 5},
]

second_agents = {k: BELIEF_AGENTS[k] for k in ("прямое предсказание", "ToM-правило", "ToM + суждение LLM")}
main_ref = {"прямое предсказание": tom_runs["прямое предсказание"], "ToM-правило": tom_runs["ToM-правило (CV)"],
            "ToM + суждение LLM": tom_runs["ToM + суждение LLM"]}
second_llms, second_runs, second_tables = {}, {}, {}
_workers = WORKERS
for cfg in SECOND_MODELS:
    provider, model_name, n = cfg["provider"], cfg["model"], cfg["n"]
    if not API_KEYS.get(provider) or not n or model_name == MAIN_MODEL:
        print(f"{model_name}: пропущено.")
        continue
    m = LLM(provider, model_name, keys=API_KEYS[provider], min_interval=cfg["interval"], extra=cfg.get("extra"), tpm=cfg.get("tpm"))
    sset = tom_pilot.head(n)
    WORKERS = cfg["per_key"] * len(API_KEYS[provider])  # parallel_map берёт число потоков отсюда
    runs, cost = run_suite(sset, second_agents, "dev_" + re.sub(r"[^\w]+", "_", model_name), model=m)
    WORKERS = _workers
    second_llms[model_name], second_runs[model_name] = m, runs
    second_tables[model_name] = metrics_table(runs).join(cost[["вызовов на диалог", "токенов на диалог"]])
    second_tables[f"{MAIN_MODEL} (те же {len(sset)} диалогов)"] = metrics_table({k: v[v.id.isin(sset.id)] for k, v in main_ref.items()})
if second_tables:
    second_table = pd.concat(second_tables)
    second_table.to_csv(RESULTS_DIR / "table_second_model.csv")
    display(second_table)
""")

md(r"""
## 10. Статистика

- **Бутстрап-интервалы** (95%) для r сдвига и MAE: диалоги перевыбираются с возвращением.
- **Парные сравнения**: два агента сравниваются на одних и тех же диалогах; p-значение — парный бутстрап разности метрик (двустороннее).
- **Поправка Холма** на множественные сравнения внутри каждого семейства: «агенты против прямого предсказания» (гипотеза Г1) и «абляции против полного агента».
""")

code(r'''
N_BOOT = 5000


def _metric(pre, post, pred, name):
    d_true, d_pred = post - pre, pred - pre
    if name == "MAE":
        return np.mean(np.abs(pred - post))
    return np.corrcoef(d_pred, d_true)[0, 1] if np.std(d_pred) > 1e-9 and np.std(d_true) > 1e-9 else np.nan


def _arrays(run):
    d = run.dropna(subset=["pred"]).set_index("id")
    return d.pre_belief, d.post_belief, d.pred


def bootstrap_ci(run, name, n_boot=N_BOOT, seed=SEED):
    pre, post, pred = (x.values for x in _arrays(run))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pre), (n_boot, len(pre)))
    vals = np.array([_metric(pre[i], post[i], pred[i], name) for i in idx])
    return np.nanpercentile(vals, [2.5, 97.5])


def paired_test(run_a, run_b, name, n_boot=N_BOOT, seed=SEED):
    """Разность метрики (a − b) на общих диалогах и двустороннее p-значение парного бутстрапа."""
    (pre, post, pa), (_, _, pb) = _arrays(run_a), _arrays(run_b)
    ids = pa.index.intersection(pb.index)
    pre, post, pa, pb = pre[ids].values, post[ids].values, pa[ids].values, pb[ids].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (n_boot, len(ids)))
    diffs = np.array([_metric(pre[i], post[i], pa[i], name) - _metric(pre[i], post[i], pb[i], name) for i in idx])
    diffs = diffs[~np.isnan(diffs)]
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return _metric(pre, post, pa, name) - _metric(pre, post, pb, name), p


def holm(pvals):
    """Поправка Холма: возвращает скорректированные p-значения в исходном порядке."""
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def ci_table(runs):
    rows = {}
    for name, run in runs.items():
        m = belief_metrics(run.pre_belief, run.post_belief, run.pred)
        (r_lo, r_hi), (m_lo, m_hi) = bootstrap_ci(run, "r"), bootstrap_ci(run, "MAE")
        rows[name] = {"n": m["n"], "r сдвига": round(m["r_сдвиг"], 2), "r: 95% ДИ": f"[{r_lo:.2f}; {r_hi:.2f}]",
                      "MAE": round(m["MAE"], 1), "MAE: 95% ДИ": f"[{m_lo:.1f}; {m_hi:.1f}]"}
    return pd.DataFrame(rows).T


def comparison_table(runs, baseline):
    """Каждый агент против baseline: разности r и MAE, p-значения и поправка Холма."""
    names = [k for k in runs if k != baseline]
    res = {k: (*paired_test(runs[k], runs[baseline], "r"), *paired_test(runs[k], runs[baseline], "MAE")) for k in names}
    out = pd.DataFrame(res, index=["Δr", "p (r)", "ΔMAE", "p (MAE)"]).T
    out["p Холм (r)"], out["p Холм (MAE)"] = holm(out["p (r)"].values), holm(out["p (MAE)"].values)
    return out[["Δr", "p (r)", "p Холм (r)", "ΔMAE", "p (MAE)", "p Холм (MAE)"]].round(3)


# Сравнение с абляциями идёт на диалогах, где абляции посчитаны
stat_runs = {k: tom_runs[k] for k in ("прямое предсказание", "ToM-правило (CV)", "ToM + суждение LLM")}
display(ci_table({"регрессия к среднему": tom_runs["регрессия к среднему"], **stat_runs}))
print("Агенты против прямого предсказания (Δ = агент − прямое; для MAE отрицательное значение — агент лучше):")
vs_direct = comparison_table(stat_runs, "прямое предсказание")
vs_direct.to_csv(RESULTS_DIR / "table_stats_vs_direct.csv")
display(vs_direct)
print("Абляции против полного агента (Δ = вариант − полный):")
_llm_abl = {k: v for k, v in abl_runs.items() if not k.strip().startswith("ToM-правило") and "реакции" not in k
            and "регрессии" not in k and not k.endswith(" ")}
vs_full = comparison_table(_llm_abl, "ToM + суждение LLM (полный)")
vs_full.to_csv(RESULTS_DIR / "table_stats_ablations.csv")
vs_full
''')

md(r"""
### 10.1. Как различить небольшой эффект при малом числе запросов

Сдвиг убеждения — очень шумная величина: стандартное отклонение сдвига около 22 пунктов, а лучший агент снижает ошибку на 1–2 пункта. Ниже — приёмы, которые повышают чувствительность сравнения. Все они считаются по уже сделанным прогнозам и запросов не требуют.

**Повышают мощность**

1. **Парный тест по ошибкам на каждом диалоге** вместо сравнения двух корреляций. Ошибки двух агентов на одном диалоге сильно связаны, и в разности общий шум вычитается. Считаются три теста: знаковый ранговый Уилкоксона (устойчив к выбросам), парный t-тест и перестановочный тест со сменой знаков (точный, не опирается на нормальность).
2. **Односторонняя гипотеза**, объявленная заранее: «агент точнее прямого предсказания». При том же уровне значимости различимая разница меньше на 12%.
3. **Регрессионная поправка на ковариаты** (обобщение CUPED; та же идея, что в prediction-powered inference). Разность ошибок частично объясняется величинами, известными до разговора для всех диалогов набора: насколько исходное убеждение далеко от середины шкалы, само исходное убеждение, длина диалога. Их вклад вычитается из разности без смещения оценки, потому что средние ковариат по всему набору известны точно. Дисперсия уменьшается в (1 − R²) раз.
4. **Оптимальное распределение выборки по слоям** (распределение Неймана). Если разброс разности ошибок в разных слоях разный (например, у людей с крайними исходными убеждениями он больше), то из таких слоёв выгодно брать больше диалогов, а среднее считать с весами слоёв. Оценка остаётся несмещённой.
5. **Объединение независимых прогонов** (метод Стоуффера): z-статистики с разных наборов диалогов или с разных моделей складываются с весами √n. Два прогона, каждый из которых сам по себе незначим, вместе могут дать значимый результат.
6. **Последовательный анализ.** Прогон идёт по дням; после каждого дня можно смотреть на результат и остановиться раньше. Чтобы многократные проверки не завышали долю ложных находок, используются границы О'Брайена — Флеминга: на ранних просмотрах порог строже, на последнем почти обычный.

**Отвечают на вопрос точнее**

7. **Проверка добавочной информации.** Вопрос «даёт ли явная модель жертвы что-то сверх прямого предсказания» проверяется регрессией: реальный сдвиг ~ прогноз прямого + прогноз агента. Значимый коэффициент при прогнозе агента означает, что агент несёт информацию, которой нет у прямого предсказания, даже если его собственная ошибка не меньше.
8. **Ансамбль.** Среднее двух прогнозов считается бесплатно; если оно лучше каждого из них — это тот же довод в единицах ошибки.
9. **Проверка эквивалентности** (два односторонних теста, TOST). Если преимущества нет, это тоже нужно показать, а не просто «не нашли разницу»: проверяется, что разность ошибок лежит внутри заранее заданного коридора ±1 пункт.

**Защищают от ложной находки**

10. **Бутстрап по кластерам.** Диалогов много, а утверждений всего 25: диалоги об одном утверждении не независимы. Интервал считается перевыборкой утверждений целиком. Он честнее и обычно шире обычного.

Плюс неравное распределение диалогов между агентами (раздел 6.4): оно даёт самый большой выигрыш.
""")

code(r'''
EQUIV_MARGIN = 1.0             # коридор эквивалентности для MAE, пунктов шкалы


def _pair_frame(run_a, run_b):
    """Ошибки двух агентов на общих диалогах и ковариаты, известные до разговора."""
    a, b = run_a.dropna(subset=["pred"]).set_index("id"), run_b.dropna(subset=["pred"]).set_index("id")
    ids = a.index.intersection(b.index)
    meta = puppet.set_index("id").loc[ids]
    return pd.DataFrame({"d": (a.pred - a.post_belief).abs()[ids] - (b.pred - b.post_belief).abs()[ids],
                         "extremity": (meta.pre_belief - 50).abs(), "pre": meta.pre_belief,
                         "n_turns": meta.turns.map(len), "statement": meta.belief_statement})


def _covariates(pool):
    return pd.DataFrame({"extremity": (pool.pre_belief - 50).abs(), "pre": pool.pre_belief, "n_turns": pool.turns.map(len)})


def paired_loss_test(run_a, run_b, n_perm=10000, seed=SEED):
    """Парные тесты по ошибкам на диалогах. Гипотеза: агент a точнее агента b (односторонняя)."""
    d = _pair_frame(run_a, run_b).d.values
    signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, len(d)))
    return {"n": len(d), "ΔMAE (a − b)": d.mean(),
            "p Уилкоксон": stats.wilcoxon(d, alternative="less").pvalue if np.any(d != 0) else 1.0,
            "p t-тест": stats.ttest_1samp(d, 0, alternative="less").pvalue,
            "p перестановочный": ((signs * d).mean(axis=1) <= d.mean()).mean()}


def adjusted_diff(run_a, run_b, pool, covs=("extremity", "pre", "n_turns")):
    """Разность ошибок с регрессионной поправкой на ковариаты. pool — набор, из которого взяты диалоги."""
    f = _pair_frame(run_a, run_b)
    X = f[list(covs)].values - _covariates(pool)[list(covs)].mean().values       # центрируем по среднему набора
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(f)), X], f.d.values, rcond=None)
    resid = f.d.values - np.c_[np.ones(len(f)), X] @ beta
    se_raw = f.d.std(ddof=1) / np.sqrt(len(f))
    se_adj = np.sqrt(resid @ resid / max(len(f) - len(covs) - 1, 1) / len(f))
    return {"ΔMAE без поправки": f.d.mean(), "ст. ошибка": se_raw, "ΔMAE с поправкой": beta[0], "ст. ошибка с поправкой": se_adj,
            "R² ковариат": 1 - (resid @ resid) / ((f.d - f.d.mean()) @ (f.d - f.d.mean())),
            "p с поправкой (одностор.)": stats.norm.cdf(beta[0] / se_adj)}


def cluster_ci(run_a, run_b, n_boot=N_BOOT, seed=SEED):
    """95% интервал для ΔMAE: бутстрап по утверждениям (кластерам), а не по диалогам."""
    f = _pair_frame(run_a, run_b)
    groups = [g.d.values for _, g in f.groupby("statement")]
    rng = np.random.default_rng(seed)
    vals = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"утверждений": len(groups), "ΔMAE": f.d.mean(), "95% ДИ по кластерам": f"[{lo:.2f}; {hi:.2f}]"}


def equivalence_test(run_a, run_b, margin=EQUIV_MARGIN):
    """TOST: лежит ли ΔMAE внутри коридора ±margin. Малое p — агенты практически равны по ошибке."""
    d = _pair_frame(run_a, run_b).d.values
    se = d.std(ddof=1) / np.sqrt(len(d))
    p = max(stats.t.sf((d.mean() + margin) / se, len(d) - 1), stats.t.cdf((d.mean() - margin) / se, len(d) - 1))
    return {"коридор": f"±{margin}", "ΔMAE": d.mean(), "p эквивалентности": p}


def incremental_validity(run_base, run_new):
    """Регрессия: реальный сдвиг ~ прогноз base + прогноз new. Несёт ли new информацию сверх base?"""
    a, b = run_base.dropna(subset=["pred"]).set_index("id"), run_new.dropna(subset=["pred"]).set_index("id")
    ids = a.index.intersection(b.index)
    y = (a.post_belief - a.pre_belief)[ids].values
    X = np.c_[np.ones(len(ids)), (a.pred - a.pre_belief)[ids].values, (b.pred - b.pre_belief)[ids].values]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    t = beta / np.sqrt(np.diag(res @ res / (len(y) - 3) * np.linalg.inv(X.T @ X)))
    return {"n": len(y), "коэф. при base": beta[1], "коэф. при new": beta[2], "t (new)": t[2],
            "p (new, одностор.)": stats.t.sf(t[2], len(y) - 3), "R² только base": np.corrcoef(X[:, 1], y)[0, 1] ** 2,
            "R² base + new": 1 - res @ res / ((y - y.mean()) @ (y - y.mean())),
            "корреляция прогнозов": np.corrcoef(X[:, 1], X[:, 2])[0, 1]}


def ensemble(run_a, run_b):
    """Среднее двух прогнозов на общих диалогах."""
    b = run_b.set_index("id").pred
    out = run_a[run_a.id.isin(b.index)].copy()
    out["pred"] = (out.pred.values + b[out.id].values) / 2
    return out


def stouffer(results):
    """Объединение независимых прогонов: results — список (ΔMAE, ст. ошибка, n). Возвращает z и одностороннее p."""
    z = np.array([m / se for m, se, _ in results])
    w = np.sqrt([n for _, _, n in results])
    z_all = (w * z).sum() / np.sqrt((w ** 2).sum())
    return {"z": z_all, "p (одностор.)": stats.norm.cdf(z_all)}


def detectable_effect(sd_pair, n, power=0.8, one_sided=True, alpha=0.05):
    """Наименьшая разница MAE, которую различит парный тест при n диалогах."""
    z_a = stats.norm.ppf(1 - alpha if one_sided else 1 - alpha / 2)
    return (z_a + stats.norm.ppf(power)) * sd_pair / np.sqrt(n)


def obf_bounds(looks, one_sided=True):
    """Границы О'Брайена — Флеминга для z-статистики при равных порциях данных (уровень 0.05)."""
    const = {True: {1: 1.645, 2: 1.678, 3: 1.710, 4: 1.733, 5: 1.751},
             False: {1: 1.960, 2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040}}[one_sided][looks]
    return [round(float(const * np.sqrt(looks / k)), 3) for k in range(1, looks + 1)]


def neyman_allocation(pool, pair_frame, n, bins=(0, 20, 40, 50)):
    """Сколько диалогов брать из каждого слоя по удалённости исходного убеждения от середины.
    pair_frame — парные разности на уже посчитанных диалогах (оценка разброса по слоям)."""
    labels = [f"{a}–{b}" for a, b in zip(bins[:-1], bins[1:])]
    cut = lambda x: pd.cut(x, bins, labels=labels, include_lowest=True)
    size = cut((pool.pre_belief - 50).abs()).value_counts().reindex(labels)
    sd = pair_frame.groupby(cut(pair_frame.extremity), observed=False).d.std(ddof=1).reindex(labels).fillna(pair_frame.d.std(ddof=1))
    share = size * sd / (size * sd).sum()
    out = pd.DataFrame({"диалогов в наборе": size, "разброс разности": sd.round(1),
                        "доля набора": (size / size.sum()).round(2), "доля выборки": share.round(2),
                        "взять": np.minimum((share * n).round().astype(int), size)})
    # насколько оптимальное распределение точнее пропорционального: отношение дисперсий оценки среднего
    w = size / size.sum()
    out.attrs["выигрыш"] = float((w * sd).sum() ** 2 / (w * sd ** 2).sum())
    return out


def power_report(runs, pool, agent="ToM + суждение LLM", base="прямое предсказание"):
    display(pd.DataFrame({f"{agent} против «{base}» (одностор.)": paired_loss_test(runs[agent], runs[base])}).round(3).T)
    display(pd.DataFrame({"поправка на ковариаты": adjusted_diff(runs[agent], runs[base], pool)}).round(3).T)
    display(pd.DataFrame({"интервал по кластерам": cluster_ci(runs[agent], runs[base]),
                          "эквивалентность": equivalence_test(runs[agent], runs[base])}).T)
    display(pd.DataFrame({"добавочная информация агента": incremental_validity(runs[base], runs[agent])}).round(3).T)
    display(metrics_table({base: runs[base], agent: runs[agent], "ансамбль (среднее двух)": ensemble(runs[base], runs[agent])}))


print(f"Пилот, {len(tom_pilot)} диалогов dev:")
power_report(tom_runs, dev)
''')

code(r'''
# Какую разницу можно различить и как брать выборку. Оценки разброса — по 30 диалогам пилота, грубые.
pf = _pair_frame(tom_runs["ToM + суждение LLM"], tom_runs["прямое предсказание"])
sd_pair = pf.d.std(ddof=1)
r2 = max(adjusted_diff(tom_runs["ToM + суждение LLM"], tom_runs["прямое предсказание"], dev)["R² ковариат"], 0)
alloc = neyman_allocation(test, pf, PLAN["n_main"])
gain = alloc.attrs["выигрыш"]
mde = pd.DataFrame({n: {"двусторонний тест": detectable_effect(sd_pair, n, one_sided=False),
                        "односторонний": detectable_effect(sd_pair, n),
                        "+ поправка на ковариаты": detectable_effect(sd_pair * np.sqrt(1 - r2), n),
                        "+ распределение по слоям": detectable_effect(sd_pair * np.sqrt((1 - r2) * gain), n)}
                    for n in sorted({100, 200, PLAN["n_main"], len(test)})}).T.round(2)
mde.index.name = "диалогов"
print(f"Наименьшая различимая разница MAE (мощность 80%). Разброс парной разности {sd_pair:.1f}; "
      f"ковариаты объясняют {r2:.0%} её дисперсии; распределение по слоям уменьшает дисперсию до {gain:.0%}.")
display(mde)
print("Распределение выборки главной пары по слоям (удалённость исходного убеждения от середины шкалы):")
display(alloc)
print("Границы О'Брайена — Флеминга для z (односторонний тест, уровень 0.05): "
      + "; ".join(f"{k} просмотра: {obf_bounds(k)}" for k in (2, 3)))

if "main_runs" in globals():                           # финальный прогон сделан — те же проверки на test
    print(f"Test, главная пара, {len(main_set)} диалогов:")
    power_report(main_runs, test)
''')

md(r"""
## 11. Графики и разбор ошибок
""")

code(r'''
show = {"регрессия к среднему": tom_runs["регрессия к среднему"], **stat_runs}
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a, name, label in ((ax[0], "r", "r сдвига (выше — лучше)"), (ax[1], "MAE", "MAE (ниже — лучше)")):
    pts = [_metric(*(x.values for x in _arrays(run)), name) for run in show.values()]
    cis = np.array([bootstrap_ci(run, name) for run in show.values()])
    y = np.arange(len(show))[::-1]
    a.errorbar(pts, y, xerr=[np.array(pts) - cis[:, 0], cis[:, 1] - np.array(pts)], fmt="o", capsize=4, color="C0")
    a.set_yticks(y); a.set_yticklabels(list(show)); a.set_xlabel(label); a.grid(axis="x", alpha=0.3)
fig.suptitle(f"PUPPET, dev, {len(tom_pilot)} диалогов; отрезки — 95% бутстрап-интервалы", fontsize=10)
plt.tight_layout(); plt.show()
''')

code(r'''
# Разбивка по доменам, типу выгоды и условию: MAE прямого предсказания и агента (группы маленькие)
meta = puppet.set_index("id")[["topic", "incentive_type", "valence"]]
err = pd.DataFrame({
    "прямое": (tom_runs["прямое предсказание"].set_index("id").pred - tom_runs["прямое предсказание"].set_index("id").post_belief).abs(),
    "ToM + LLM": (tom_runs["ToM + суждение LLM"].set_index("id").pred - tom_runs["ToM + суждение LLM"].set_index("id").post_belief).abs(),
}).join(meta)
for col, title in (("topic", "домен"), ("incentive_type", "тип выгоды"), ("valence", "знак скрытой выгоды")):
    t = err.groupby(col).agg(n=("прямое", "size"), MAE_прямое=("прямое", "mean"), MAE_ToM=("ToM + LLM", "mean")).round(1)
    t["ToM лучше на"] = (t.MAE_прямое - t.MAE_ToM).round(1)
    t.index.name = title
    display(t)
''')

code(r'''
# Самые большие ошибки агента: что было в диалоге
rows = []
tom_pred, dir_pred = tom_runs["ToM + суждение LLM"].set_index("id").pred, tom_runs["прямое предсказание"].set_index("id").pred
for r in tom_pilot.itertuples(index=False):
    ps = [p for p in pilot_parses[r.id] if p]
    reacts = pd.Series([p["user_reaction"] for p in ps]).value_counts()
    rows.append({"утверждение": r.belief_statement[:45], "условие": r.valence, "до": r.pre_belief, "после": r.post_belief,
                 "прямое": dir_pred[r.id], "ToM": tom_pred[r.id], "ошибка ToM": abs(tom_pred[r.id] - r.post_belief),
                 "сумма толчков": sum(p["push"] for p in ps), "принял": reacts.get("accepts", 0),
                 "спорил": reacts.get("resists", 0) + reacts.get("questions", 0)})
worst = pd.DataFrame(rows).sort_values("ошибка ToM", ascending=False).head(8).round(0)
worst
''')

code(r'''
# Траектории убеждений V по трекеру: три диалога с наибольшим реальным сдвигом
big = tom_pilot.assign(a=tom_pilot.delta.abs()).sort_values("a", ascending=False).head(3)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
for a, r in zip(ax, big.itertuples(index=False)):
    st = puppet_state(r, pilot_parses[r.id])
    traj = [t for t in st["trajectory"] if not t.get("skipped")]
    if not traj:                                       # диалог не разобран — рисовать нечего
        continue
    a.plot([0] + [t["turn"] for t in traj], [r.pre_belief] + [t["credence"] for t in traj], marker="o", label="трекер")
    a.scatter([traj[-1]["turn"] + 0.6], [r.post_belief], marker="*", s=160, color="C3", label="реальная оценка после", zorder=3)
    a.scatter([0], [r.pre_belief], marker="s", color="k", label="оценка до", zorder=3)
    a.set(title=f"{r.belief_statement[:38]}…\n{r.valence}", xlabel="шаг", ylim=(-3, 103)); a.grid(alpha=0.3)
ax[0].set_ylabel("уверенность V"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
''')

md(r"""
## 12. Русский демо-набор

Десять коротких диалогов: шесть по публично описанным схемам (предупреждения банков и Банка России) и четыре обычных разговора с похожими внешними признаками — деньги, срочность, коды. Набор нужен для качественной демонстрации и не входит в основную оценку. Новых сценариев мошенничества здесь нет: только типовые схемы, о которых банки сами предупреждают клиентов.

S — собеседник, V — человек, которого защищает агент.
""")

code(r'''
RU_STATEMENT = "Собеседник — тот, за кого себя выдаёт, и ему можно доверять"


def _dlg(*lines):
    return [{"speaker": "S" if i % 2 == 0 else "V", "text": t} for i, t in enumerate(lines)]


RU_DEMO = [
    ("служба безопасности банка", 1, DEMO_SCAM),
    ("«безопасная сделка» на Авито", 1, _dlg(
        "Здравствуйте, я по объявлению о коляске. Беру без торга, но я в другом городе, оформим доставку.",
        "Хорошо, а как оплата?",
        "Я уже оплатил через безопасную сделку. Перейдите по ссылке, которую я прислал, и введите данные карты, чтобы получить деньги.",
        "Там просят номер карты и код с обратной стороны.",
        "Да, это для зачисления. Потом придёт СМС с кодом, введите его тоже. Только быстрее, ссылка действует десять минут.")),
    ("«родственник в беде»", 1, _dlg(
        "Мам, это я, пишу с чужого номера. Я попал в аварию, сбил человека.",
        "Господи! Ты цел? Где ты?",
        "Цел. Тут следователь, он говорит, что можно решить без суда, если сегодня передать 300 тысяч. Только папе не говори и не звони мне, телефон забрали.",
        "У меня нет столько.",
        "Сними сколько есть и отдай курьеру, он приедет через час. Мам, быстрее, иначе меня посадят.")),
    ("«продление договора связи»", 1, _dlg(
        "Добрый день, это ваш оператор связи. Срок действия вашего договора истекает сегодня, номер будет отключён.",
        "Я ничего такого не слышал.",
        "Продлить можно прямо сейчас по телефону, в салон идти не нужно. Вам придёт СМС с кодом, продиктуйте его.",
        "Пришёл код, но там написано «никому не сообщайте».",
        "Это стандартная фраза, она для мошенников. Мне можно, я сотрудник. Диктуйте, осталось две минуты.")),
    ("«инвестиции с гарантией»", 1, _dlg(
        "Здравствуйте! Я персональный аналитик инвестиционной платформы. Наши клиенты получают 30% в месяц с гарантией.",
        "Звучит слишком хорошо.",
        "Понимаю сомнения. Но набор закрывается сегодня, осталось два места. Вносите 50 тысяч на счёт куратора, и завтра увидите прибыль.",
        "Я бы посоветовался с сыном.",
        "Не стоит: родственники обычно отговаривают, потому что не разбираются. Это решение нужно принять самостоятельно и сейчас.")),
    ("«безопасный счёт»", 1, _dlg(
        "Вас беспокоит следователь. На ваше имя прямо сейчас оформляют кредит, в деле замешаны сотрудники вашего банка.",
        "Какой кредит? Я ничего не оформлял.",
        "Чтобы спасти деньги, переведите их на безопасный счёт, я продиктую реквизиты. Разговор секретный: за разглашение предусмотрена статья.",
        "А можно я сначала схожу в банк?",
        "Ни в коем случае. В отделении работают сообщники. Оставайтесь на линии и идите к банкомату.")),
    ("настоящий банк", 0, DEMO_BANK),
    ("торг на Авито с блефом", 0, _dlg(
        "Здравствуйте, велосипед ещё продаётся? Больше пяти тысяч не дам, у меня есть другой вариант за четыре с половиной.",
        "Продаётся. За пять не отдам, он почти новый. Шесть.",
        "Давайте пять с половиной. Я приеду, посмотрю, прокачусь, и если всё в порядке — заберу сегодня.",
        "Хорошо, приезжайте, посмотрите. Оплата на месте.",
        "Договорились, буду в семь. Адрес напишите в чат.")),
    ("курьерская доставка", 0, _dlg(
        "Добрый день, это служба доставки. Ваш заказ приедет завтра с 10 до 14, удобно?",
        "Да, удобно. А что нужно будет для получения?",
        "Курьер попросит код из СМС, когда отдаст посылку вам в руки. До этого код никому называть не нужно. Статус заказа можно проверить в приложении магазина.",
        "Понял, спасибо.",
        "Если время не подойдёт, перенесите доставку в приложении или по телефону на сайте.")),
    ("напоминание из поликлиники", 0, _dlg(
        "Здравствуйте, это регистратура поликлиники. Напоминаем о записи к терапевту завтра в 9:30.",
        "А можно перенести?",
        "Можно. Перезвоните в регистратуру по номеру на сайте поликлиники или перенесите запись через портал.",
        "Хорошо, я посмотрю расписание.",
        "Спасибо. Если не сможете прийти, отмените запись, чтобы время досталось другому пациенту.")),
]

ru_rows = []
for name, label, turns in RU_DEMO:
    parses = parse_dialog(turns, RU_STATEMENT, 50, names=("Собеседник V", "Собеседник S"), speakers=("V", "S"), tag="demo")
    st = run_tracker(parses, RU_STATEMENT, 50)
    goal = st["speaker_goal"] or {}
    ru_rows.append({"диалог": name, "мошенничество": bool(label), "макс. риск": st["max_risk"],
                    "ход тревоги": st["alarm_turn"], "ходов S": len(parses),
                    "сужений доступа": sum(m["move"] == "narrow" for m in st["access"]),
                    "расширений доступа": sum(m["move"] == "widen" for m in st["access"]),
                    "необратимое действие в пользу S": bool(goal.get("irreversible") and goal.get("beneficiary") == "S"),
                    "вера V в конце": st["victim_beliefs"][RU_STATEMENT]["credence"],
                    "I-Public": public_message(st) or "— (агент молчит)"})
ru_table = pd.DataFrame(ru_rows).set_index("диалог")
ru_table.to_csv(RESULTS_DIR / "table_ru_demo.csv")
pd.set_option("display.max_colwidth", 200)
display(ru_table.drop(columns="I-Public"))
display(ru_table[["мошенничество", "I-Public"]])
print(f"Тревога (риск ≥ {RISK_ALARM}): на мошеннических {int(ru_table[ru_table['мошенничество']]['ход тревоги'].notna().sum())} из "
      f"{int(ru_table['мошенничество'].sum())}, на обычных {int(ru_table[~ru_table['мошенничество']]['ход тревоги'].notna().sum())} из "
      f"{int((~ru_table['мошенничество']).sum())}")
''')

md("## 13. Стоимость по всем разделам")

code(r'''
print(f"Основная модель: {llm.provider}/{llm.model}; ответов в кэше: {len(llm.cache)}")
display(llm.stats())
if len(llm.key_calls):
    display(llm.key_stats().to_frame())
for _name, _m in second_llms.items():
    print(f"Модель {_m.provider}/{_name}; ответов в кэше: {len(_m.cache)}")
    display(_m.stats())
''')

conclusions = Path(__file__).with_name("conclusions.md")
md(conclusions.read_text(encoding="utf-8") if conclusions.exists() else "## 14. Итоги\n\n(выводы будут добавлены после прогона)")

nb.cells += cells
nbf.write(nb, path)
print("cells:", len(nb.cells))
