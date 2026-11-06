"""Собирает ToM_antifraud.ipynb (этап 1)."""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# ToM-антифрод: агент с моделью убеждений жертвы

Команда 08-554, направление ToM-2. Описание темы — в `Тема_ToM-антифрод.md`, план и журнал — в `ПЛАН_И_ПРОГРЕСС.md`.

**Как запускать**

1. Откройте ноутбук в Google Colab (Файл → Загрузить блокнот) или локально в Jupyter.
2. Дайте ноутбуку ключи API одним из способов (раздел 1.2).
3. Запускайте ячейки сверху вниз. Все ответы моделей кэшируются, поэтому повторный запуск не тратит вызовы, а прерванный прогон продолжается с того же места.

**Содержание**

1. Настройка: папки, ключи, обёртка LLM
2. Данные PUPPET и разбиение dev/test
3. Метрики
4. Бейзлайн «прямое предсказание» и пилот
""")

md("## 1. Настройка\n\n### 1.1. Параметры и рабочая папка")

code(r"""
import os, sys, re, json, time, hashlib, threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from scipy import stats

try:
    from tqdm.auto import tqdm
except ImportError:                      # без tqdm просто не будет полосы прогресса
    tqdm = lambda it, **kw: it

IN_COLAB = "google.colab" in sys.modules

# ---- параметры, которые можно менять ----
USE_DRIVE = True                 # в Colab хранить кэш и результаты на Google Drive
MAIN_PROVIDER = "gemini"         # "gemini" или "groq"
MAIN_MODEL = "gemma-4-26b-a4b-it"
MAIN_EXTRA = {"thinkingConfig": {"thinkingLevel": "minimal"}}   # Gemma: без обдумывания (иначе 17–20 с на запрос)
MAIN_TPM = 15_000                # лимит входных токенов в минуту на ключ (у Gemma 16 тыс.); None — не учитывать
MIN_INTERVAL = 2.0               # пауза между запросами одного ключа, секунд (у Gemma 30 запросов в минуту)
WORKERS_PER_KEY = 3              # параллельных потоков на один ключ
MAX_WORKERS = None               # всего потоков; None — WORKERS_PER_KEY × число ключей основной модели
PILOT_N = 100                    # размер пилота на dev
SEED = 554

if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK_DIR = Path("/content/drive/MyDrive/ToM_antifraud")
elif IN_COLAB:
    WORK_DIR = Path("/content/ToM_antifraud")
else:
    # кэш, данные и результаты лежат рядом с папкой проекта, а если такой папки нет — внутри неё
    WORK_DIR = Path("../tom_work") if Path("../tom_work").exists() else Path("tom_work")

DATA_DIR, CACHE_DIR, RESULTS_DIR = WORK_DIR / "data", WORK_DIR / "cache", WORK_DIR / "results"
for d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("Рабочая папка:", WORK_DIR)
""")

md(r"""
### 1.2. Ключи API

Ключи ищутся по порядку, используется первый найденный источник:

1. **ячейка ниже** — впишите ключи в кавычки (не отправляйте ноутбук с вписанными ключами);
2. **Colab Secrets** — значок ключа слева, секреты `GEMINI_API_KEY` и `GROQ_API_KEY`;
3. **файл `keys.env`** — в папке `secrets/` рядом с ноутбуком или уровнем выше, либо в рабочей папке на Drive (`ToM_antifraud/keys.env`); образец — `secrets/keys.env.example`;
4. переменные окружения.

Формат `keys.env`: по строке `ИМЯ=значение`. Несколько ключей одного сервиса можно перечислить через запятую: запросы чередуются между ними, поэтому прогон идёт быстрее, а дневные лимиты складываются. Ключ, исчерпавший дневной лимит, выбывает до следующего запуска.

Где взять: Gemini — aistudio.google.com → Get API key; Groq — console.groq.com → API Keys.
""")

code(r'''
GEMINI_API_KEY = ""
GROQ_API_KEY = ""
''')

code(r'''
def _read_env_file(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def find_keys(name, manual=""):
    """Возвращает (список ключей, источник). Значения ключей не печатаются."""
    if manual.strip():
        return _split_keys(manual), "ячейка ноутбука"
    if IN_COLAB:
        try:
            from google.colab import userdata
            v = userdata.get(name)
            if v:
                return _split_keys(v), "Colab Secrets"
        except Exception:
            pass
    for p in (Path("secrets/keys.env"), Path("../secrets/keys.env"), Path("keys.env"), WORK_DIR / "keys.env", WORK_DIR / "secrets" / "keys.env"):
        if p.exists():
            v = _read_env_file(p).get(name)
            if v:
                return _split_keys(v), f"файл {p}"
    if os.environ.get(name):
        return _split_keys(os.environ[name]), "переменная окружения"
    return [], None


def _split_keys(v):
    return [k.strip() for k in v.split(",") if k.strip()]


API_KEYS = {}
for _provider, _name, _manual in (("gemini", "GEMINI_API_KEY", GEMINI_API_KEY), ("groq", "GROQ_API_KEY", GROQ_API_KEY)):
    _keys, _src = find_keys(_name, _manual)
    API_KEYS[_provider] = _keys
    print(f"{_name}: " + (f"найдено ключей: {len(_keys)} ({_src})" if _keys else "НЕ НАЙДЕН"))
''')

md(r"""
### 1.3. Обёртка LLM

Один класс для Gemini и Groq:

- **кэш** — каждый ответ пишется в `cache/<провайдер>__<модель>.jsonl`; одинаковый запрос второй раз не отправляется;
- **паузы и повторы** — выдерживает `MIN_INTERVAL` для каждого ключа отдельно и чередует ключи; при ошибке 429 ключ отдыхает столько, сколько просит сервер, при исчерпании дневного лимита выбывает;
- **параллельность** — диалоги обрабатываются в несколько потоков (`WORKERS_PER_KEY` на каждый ключ); каждый ключ при этом шлёт не чаще одного запроса в `MIN_INTERVAL` секунд и не больше `MAIN_TPM` входных токенов в минуту, так что скорость растёт пропорционально числу ключей;
- **JSON** — `call_json` разбирает ответ, проверяет его функцией `validate` и повторяет запрос при невалидном ответе;
- **учёт стоимости** — по метке `tag` считаются вызовы и токены (и из кэша тоже, чтобы сравнивать агентов по бюджету).
""")

code(r'''
class QuotaExhausted(RuntimeError):
    """Дневной лимит исчерпан на всех ключах."""


def parse_json(text):
    """Достаёт JSON-объект из ответа модели (допускает ```-обёртку и текст вокруг)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("в ответе нет JSON")
        obj = json.loads(m.group(0))
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]                                 # некоторые модели (Gemma) оборачивают объект в список из одного элемента
    return obj


class LLM:
    def __init__(self, provider, model, keys=None, min_interval=4.0, timeout=120, max_retries=8, mock_fn=None, extra=None, tpm=None):
        self.provider, self.model = provider, model
        self.tpm = tpm                               # лимит входных токенов в минуту на ключ (None — не учитывать)
        self.extra = extra or {}                     # доп. параметры: для Groq — поля запроса, для Gemini — поля generationConfig
        self.keys = list(keys or [])
        self.min_interval, self.timeout, self.max_retries = min_interval, timeout, max_retries
        self.mock_fn = mock_fn                       # для provider="mock": функция prompt -> текст
        self._next = {}                              # ключ -> когда им можно отправить следующий запрос
        self._lock = threading.Lock()                # защищает ключи, кэш и счётчики при работе в несколько потоков
        self._key_no = {k: i + 1 for i, k in enumerate(self.keys)}
        self.key_calls = defaultdict(int)            # номер ключа -> запросов за этот запуск
        self._exhausted = False                      # все ключи выбыли по дневному лимиту
        self.usage = defaultdict(lambda: {"calls": 0, "api_calls": 0, "in_tok": 0, "out_tok": 0})
        self.cache_path = CACHE_DIR / f"{provider}__{model.replace('/', '_')}.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.cache[rec["key"]] = rec

    # ---------- публичные методы ----------
    def call(self, prompt, system=None, json_mode=False, temperature=0.0, max_tokens=2048, sample=0, tag="misc"):
        """Текст ответа. sample различает независимые повторы одного запроса (self-consistency)."""
        key = hashlib.sha256(json.dumps(
            [self.provider, self.model, system, prompt, json_mode, temperature, max_tokens, sample]
            + ([self.extra] if self.extra else []),
            ensure_ascii=False).encode("utf-8")).hexdigest()
        rec = self.cache.get(key)
        fresh = rec is None
        if fresh:
            rec = self._request(prompt, system, json_mode, temperature, max_tokens)
            rec["key"] = key
        with self._lock:
            if fresh:
                self.cache[key] = rec
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            u = self.usage[tag]
            u["api_calls"] += int(fresh)
            u["calls"] += 1
            u["in_tok"] += rec["in_tok"]
            u["out_tok"] += rec["out_tok"]
        return rec["text"]

    def call_json(self, prompt, validate=None, retries=2, sample=0, **kw):
        """JSON-ответ, проверенный validate(obj) -> значение (или ValueError). Невалидный ответ — повтор."""
        err = None
        for attempt in range(retries + 1):
            # повтор при нулевой температуре дал бы тот же ответ, поэтому повторные попытки идут с температурой 0.4
            kw_try = kw if attempt == 0 else {**kw, "temperature": max(kw.get("temperature", 0.0), 0.4)}
            text = self.call(prompt, json_mode=True, sample=sample + 1000 * attempt, **kw_try)
            try:
                obj = parse_json(text)
                return validate(obj) if validate else obj
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                err = e
        raise ValueError(f"невалидный JSON после {retries + 1} попыток: {err}")

    def key_stats(self):
        """Сколько запросов ушло через каждый ключ за этот запуск (ключи пронумерованы в порядке из keys.env)."""
        return pd.Series(dict(sorted(self.key_calls.items())), name="запросов за этот запуск", dtype=int).rename_axis("ключ №")

    def stats(self):
        """Таблица стоимости по меткам: логические вызовы, реальные запросы к API, токены."""
        df = pd.DataFrame(self.usage).T
        return df.astype(int) if len(df) else df

    # ---------- запросы ----------
    def _request(self, prompt, system, json_mode, temperature, max_tokens):
        if self.provider == "mock":
            text = self.mock_fn(prompt)
            return {"text": text, "in_tok": len(prompt) // 4, "out_tok": len(text) // 4}
        delay = 5.0
        for _ in range(self.max_retries):
            key = self._acquire_key((len(prompt) + len(system or "")) / 3.8)      # грубая оценка числа входных токенов
            try:
                r = self._post(key, prompt, system, json_mode, temperature, max_tokens)
            except requests.RequestException:
                time.sleep(delay); delay *= 2
                continue
            if r.status_code == 200:
                rec = self._parse(r.json())
                if rec["text"]:
                    return rec
                time.sleep(delay); delay *= 2            # пустой ответ (фильтр, обрыв) — повтор
            elif r.status_code == 429:
                self._on_rate_limit(r, key)
            elif r.status_code >= 500:
                time.sleep(delay); delay *= 2
            else:
                raise RuntimeError(f"{self.provider} {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"{self.provider}: нет ответа после {self.max_retries} попыток")

    def _acquire_key(self, tokens=0.0):
        """Берёт ключ, который освободится раньше всех, бронирует его следующий слот и ждёт своей очереди.
        Каждый ключ отправляет не чаще одного запроса в min_interval секунд, сколько бы потоков ни работало."""
        with self._lock:
            if not self.keys:
                if self._exhausted:
                    raise QuotaExhausted(f"{self.provider}/{self.model}: дневной лимит исчерпан на всех ключах.")
                raise RuntimeError(f"Нет ключа для {self.provider} — см. раздел 1.2")
            key = min(self.keys, key=lambda k: self._next.get(k, 0.0))
            start = max(time.time(), self._next.get(key, 0.0))
            # следующий слот ключа: не раньше паузы и не раньше, чем позволит лимит токенов в минуту
            self._next[key] = start + max(self.min_interval, 60 * tokens / self.tpm if self.tpm else 0.0)
            self.key_calls[self._key_no.get(key, 0)] += 1
        wait = start - time.time()
        if wait > 0:
            time.sleep(wait)
        return key

    def _post(self, key, prompt, system, json_mode, temperature, max_tokens):
        if self.provider == "gemini":
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, **self.extra}}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if json_mode:
                body["generationConfig"]["responseMimeType"] = "application/json"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            return requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=self.timeout)
        if self.provider == "groq":
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            body = {"model": self.model, "messages": messages, "temperature": temperature,
                    "max_completion_tokens": max_tokens, **self.extra}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            return requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"}, json=body, timeout=self.timeout)
        raise ValueError(f"неизвестный провайдер: {self.provider}")

    def _parse(self, data):
        if self.provider == "gemini":
            parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            um = data.get("usageMetadata", {})
            return {"text": text, "in_tok": um.get("promptTokenCount", 0),
                    "out_tok": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)}
        um = data.get("usage", {})
        return {"text": data["choices"][0]["message"].get("content") or "",
                "in_tok": um.get("prompt_tokens", 0), "out_tok": um.get("completion_tokens", 0)}

    def _on_rate_limit(self, r, key):
        body = r.text
        wait = 30.0
        m = re.search(r'"retryDelay":\s*"([\d.]+)s"', body)
        if m:
            wait = float(m.group(1))
        elif r.headers.get("retry-after"):
            wait = float(r.headers["retry-after"])
        daily = bool(re.search(r"PerDay|per day|\(RPD\)|\(TPD\)", body, flags=re.I))
        with self._lock:
            if daily:
                if key in self.keys:
                    self.keys.remove(key)
                    print(f"[{self.provider}] дневной лимит ключа исчерпан, осталось ключей: {len(self.keys)}")
                if not self.keys:
                    self._exhausted = True
                    raise QuotaExhausted(f"{self.provider}/{self.model}: дневной лимит исчерпан на всех ключах. "
                                         "Сделанное сохранено в кэше — запустите ячейку позже, прогон продолжится.")
            else:                                       # минутный лимит: ключ отдыхает, запросы идут через остальные
                self._next[key] = max(self._next.get(key, 0.0), time.time() + min(wait + 1.0, 90.0))

llm = LLM(MAIN_PROVIDER, MAIN_MODEL, keys=API_KEYS[MAIN_PROVIDER], min_interval=MIN_INTERVAL, extra=MAIN_EXTRA, tpm=MAIN_TPM)
WORKERS = MAX_WORKERS or max(1, min(32, WORKERS_PER_KEY * len(API_KEYS[MAIN_PROVIDER])))


def parallel_map(fn, items, desc, workers=None):
    """Применяет fn к каждому элементу в несколько потоков. Возвращает список пар (результат, ошибка) в исходном порядке.
    После исчерпания дневного лимита новые элементы не запускаются: у них ошибка QuotaExhausted."""
    items, stop = list(items), threading.Event()

    def task(x):
        if stop.is_set():
            return None, QuotaExhausted("не запускалось: дневной лимит исчерпан")
        try:
            return fn(x), None
        except QuotaExhausted as e:
            stop.set()
            return None, e
        except Exception as e:
            return None, e

    with ThreadPoolExecutor(max_workers=workers or WORKERS) as pool:
        return list(tqdm(pool.map(task, items), total=len(items), desc=desc))


def report_errors(name, ids, results):
    """Печатает ошибки по диалогам; возвращает число обычных ошибок (без исчерпания лимита)."""
    quota = [e for _, e in results if isinstance(e, QuotaExhausted)]
    failed = [(i, e) for i, (_, e) in zip(ids, results) if e is not None and not isinstance(e, QuotaExhausted)]
    for i, e in failed:
        print(f"[{name}] {i}: {type(e).__name__}: {str(e)[:150]}")
    if quota:
        print(f"[{name}] дневной лимит исчерпан: не обработано диалогов {len(quota)}. Сделанное сохранено в кэше — "
              "запустите ячейку позже, прогон продолжится.")
    return len(failed)


print(f"Модель: {MAIN_PROVIDER}/{MAIN_MODEL}; в кэше ответов: {len(llm.cache)}; параллельных потоков: {WORKERS}")
''')

md("Проверка связи: один короткий запрос (повторный запуск берёт ответ из кэша).")

code(r'''
print(llm.call_json('Return JSON: {"ok": true}', tag="check"))
''')

md(r"""
## 2. Данные PUPPET

Shen et al. (2026), *The Hidden Puppet Master*: люди разговаривают с LLM, у которой в части условий есть скрытая выгода; согласие с утверждением замерено по шкале 0–100 до и после разговора.

**Условия эксперимента** (из кода интерфейса исследования):

| Условие | Манипуляция | Знак скрытой выгоды | Персонализация |
|---|---|---|---|
| C1 | да | отрицательная | да |
| C2 | да | положительная | да |
| C3 | да | отрицательная | нет |
| C4 | да | положительная | нет |
| C5 | нет (контроль) | — | да |
| C6 | нет (контроль) | — | нет |

Оставляем участников, прошедших проверку внимания: 1085 диалогов из 1151. В статье указано 1035 — точный фильтр авторов в репозитории не описан, поэтому числа могут немного отличаться от опубликованных.
""")

code(r'''
PUPPET_URL = "https://raw.githubusercontent.com/mitmedialab/llm-manipulation/main/hidden_puppet_master_dataset.csv"
PUPPET_CSV = DATA_DIR / "hidden_puppet_master_dataset.csv"
DEV_FRACTION = 0.28

CONDITIONS = {  # условие -> (манипуляция, знак скрытой выгоды)
    "C1": (True, "negative"), "C2": (True, "positive"), "C3": (True, "negative"),
    "C4": (True, "positive"), "C5": (False, "none"), "C6": (False, "none"),
}


def parse_turns(text):
    """Строка 'USER: ...\nBOT: ...' -> список {'speaker', 'text'}."""
    pieces = re.split(r"(?m)^(USER|BOT): ", text)
    return [{"speaker": s, "text": t.strip()} for s, t in zip(pieces[1::2], pieces[2::2])]


def _stable_hash(s):
    return hashlib.md5(f"{SEED}:{s}".encode()).hexdigest()


def load_puppet():
    if not PUPPET_CSV.exists():
        r = requests.get(PUPPET_URL, timeout=120)
        r.raise_for_status()
        PUPPET_CSV.write_bytes(r.content)
    raw = pd.read_csv(PUPPET_CSV)
    raw = raw[raw["passed_attention_check"] == True]
    df = pd.DataFrame({
        "id": raw["participant_id"].astype(str),
        "topic": raw["topic"],
        "condition": raw["condition"],
        "manipulative": raw["condition"].map(lambda c: CONDITIONS[c][0]),
        "valence": raw["condition"].map(lambda c: CONDITIONS[c][1]),
        "personalized": raw["personalization"].eq("personalized"),
        "incentive_type": raw["incentiveType"],        # тип скрытой выгоды (нужен для В2)
        "incentive": raw["incentive"],                 # текст скрытой выгоды (пусто в контроле)
        "query": raw["queryText"],
        "belief_statement": raw["belief_statement"],
        "pre_belief": raw["pre_belief"].astype(float),
        "post_belief": raw["post_belief"].astype(float),
        "delta": raw["belief_delta"].astype(float),
        "turns": raw["conversation_parsed"].map(parse_turns),
    })
    # Разбиение dev/test: внутри каждой страты (домен x условие) порядок задаёт хэш id,
    # поэтому оно не зависит от версии библиотек и порядка строк.
    df["_h"] = df["id"].map(_stable_hash)
    df = df.sort_values("_h").reset_index(drop=True)
    rank = df.groupby(["topic", "condition"]).cumcount()
    size = df.groupby(["topic", "condition"])["id"].transform("size")
    df["split"] = np.where(rank < (size * DEV_FRACTION).round(), "dev", "test")
    return df.drop(columns="_h")


puppet = load_puppet()
dev, test = puppet[puppet.split == "dev"], puppet[puppet.split == "test"]
pilot = dev.head(PILOT_N)          # dev уже в случайном (хэш) порядке
print(f"Всего: {len(puppet)}, dev: {len(dev)}, test: {len(test)}, пилот: {len(pilot)}")
puppet.groupby("split")[["pre_belief", "post_belief", "delta"]].mean().round(1)
''')

code(r'''
summary = puppet.groupby("condition").agg(
    n=("id", "size"), manipulative=("manipulative", "first"), valence=("valence", "first"),
    mean_delta=("delta", "mean"), mean_abs_delta=("delta", lambda x: x.abs().mean()),
    bot_turns=("turns", lambda t: np.mean([sum(x["speaker"] == "BOT" for x in d) for d in t])),
).round(2)
display(summary)
display(pd.crosstab(puppet.topic, puppet.split))
''')

md(r"""
## 3. Метрики

Прогноз агента — итоговое убеждение `post` (0–100). Считаем:

- **r(сдвиг), ρ(сдвиг)** — корреляции Пирсона и Спирмена между предсказанным и реальным сдвигом `post − pre`. Это основная метрика.
- **MAE** — средняя абсолютная ошибка по `post` (она же по сдвигу).
- **смещение** — средний предсказанный сдвиг минус средний реальный.
- **смещение |сдвига|** — то же по модулю сдвига: отрицательное значение значит, что модель недооценивает силу изменения убеждений.
- **r(post)** — корреляция по итоговому убеждению; приводится для справки.

**Почему основная метрика — по сдвигу.** Итоговое убеждение сильно зависит от исходного: прогноз «ничего не изменится» (`post = pre`) уже даёт r(post) около 0.7. Поэтому r(post) почти ничего не говорит о качестве модели, и все агенты сравниваются с ориентирами без модели:

- «без изменений» — `post = pre`;
- «средний сдвиг» — ко всем прибавляется средний сдвиг по dev;
- «регрессия к среднему» — сдвиг линейно зависит от исходного убеждения: кто начинал у края шкалы, чаще сдвигается к середине. Коэффициенты оцениваются по dev без проверяемых диалогов.
""")

code(r'''
def belief_metrics(pre, post, pred):
    """Метрики прогноза итогового убеждения. Строки без прогноза пропускаются."""
    pre, post, pred = (np.asarray(x, dtype=float) for x in (pre, post, pred))
    ok = ~np.isnan(pred)
    pre, post, pred = pre[ok], post[ok], pred[ok]
    d_true, d_pred = post - pre, pred - pre

    def corr(fn, a, b):                    # корреляция не определена для константы
        return fn(a, b)[0] if len(a) > 2 and np.std(a) > 1e-9 and np.std(b) > 1e-9 else np.nan

    return {
        "n": int(ok.sum()),
        "r_сдвиг": corr(stats.pearsonr, d_pred, d_true),
        "rho_сдвиг": corr(stats.spearmanr, d_pred, d_true),
        "MAE": np.mean(np.abs(pred - post)) if ok.any() else np.nan,
        "смещение": np.mean(d_pred - d_true) if ok.any() else np.nan,
        "смещение_|сдвига|": np.mean(np.abs(d_pred) - np.abs(d_true)) if ok.any() else np.nan,
        "r_post": corr(stats.pearsonr, pred, post),
    }


def metrics_table(runs, by=None):
    """runs: {название: DataFrame с колонками pre_belief, post_belief, pred}. by — колонка для разбивки."""
    rows = {}
    for name, df in runs.items():
        groups = [(None, df)] if by is None else list(df.groupby(by))
        for g, part in groups:
            rows[name if by is None else (name, g)] = belief_metrics(part.pre_belief, part.post_belief, part.pred)
    return pd.DataFrame(rows).T.round(3)


def fit_drift(reference):
    """Регрессия к среднему: сдвиг = a + b * исходное убеждение. Оценивается по данным, без модели."""
    b, a = np.polyfit(reference.pre_belief, reference.delta, 1)
    return {"a": float(a), "b": float(b)}


def trivial_runs(df, reference=None):
    """Ориентиры без модели. Оцениваются по reference — по умолчанию это dev без диалогов из df."""
    ref = dev[~dev.id.isin(df.id)] if reference is None else reference
    drift = fit_drift(ref)
    return {
        "без изменений": df.assign(pred=df.pre_belief),
        "средний сдвиг": df.assign(pred=df.pre_belief + ref.delta.mean()),   # константа: корреляция не определена
        "регрессия к среднему": df.assign(pred=(df.pre_belief + drift["a"] + drift["b"] * df.pre_belief).clip(0, 100)),
    }


metrics_table(trivial_runs(pilot))
''')

md(r"""
## 4. Бейзлайн «прямое предсказание»

Один запрос на диалог: модель видит утверждение, исходную оценку и весь разговор и сразу называет итоговую оценку. Так устроено прямое предсказание в статье PUPPET.

**Что видят все агенты (одинаковый вход):** утверждение, исходное убеждение, реплики диалога. Условие эксперимента и скрытая выгода агентам не показываются.
""")

code(r'''
SPEAKER_NAMES = {"USER": "User", "BOT": "AI assistant"}


def format_dialog(turns):
    return "\n\n".join(f"{SPEAKER_NAMES[t['speaker']]}: {t['text']}" for t in turns)


def valid_belief(obj, field="post_belief"):
    v = float(obj[field])
    if not 0 <= v <= 100:
        raise ValueError(f"{field} вне диапазона 0–100: {v}")
    return v


DIRECT_SYSTEM = "You are an expert in the psychology of persuasion. You predict how a person's beliefs change after a conversation."

DIRECT_PROMPT = """A person talked with an AI assistant. Before the conversation they rated how much they agree with a statement on a scale from 0 (strongly disagree) to 100 (strongly agree).

Statement: "{statement}"
Rating before the conversation: {pre:.0f}

Conversation:
{dialog}

Predict this person's rating of the same statement right after the conversation.
Answer with JSON only: {{"post_belief": <integer from 0 to 100>}}"""


def predict_direct(row, model=None):
    prompt = DIRECT_PROMPT.format(statement=row.belief_statement, pre=row.pre_belief, dialog=format_dialog(row.turns))
    return (model or llm).call_json(prompt, validate=valid_belief, system=DIRECT_SYSTEM, tag="direct")


def run_predictions(df, predict_fn, name):
    """Прогоняет predict_fn по диалогам. При исчерпании лимита останавливается; сделанное остаётся в кэше."""
    rows = list(df.itertuples(index=False))
    results = parallel_map(predict_fn, rows, name)
    failed = report_errors(name, [r.id for r in rows], results)
    preds = [np.nan if e is not None else v for v, e in results]
    out = df[["id", "split", "topic", "condition", "manipulative", "pre_belief", "post_belief", "delta"]].copy()
    out["pred"] = preds
    out.to_csv(RESULTS_DIR / f"puppet_{name}.csv", index=False)
    print(f"[{name}] прогнозов: {out.pred.notna().sum()} из {len(df)}, ошибок: {failed}")
    return out
''')

md("### 4.1. Пилот на dev")

code(r'''
pilot_direct = run_predictions(pilot, predict_direct, "pilot_direct")
pilot_runs = {**trivial_runs(pilot), "прямое предсказание": pilot_direct}
metrics_table(pilot_runs)
''')

code(r'''
# Разбивка: манипулятивные условия (C1–C4) и контроль (C5–C6)
display(metrics_table({"прямое предсказание": pilot_direct}, by="manipulative"))
# Разбивка по доменам (на 100 диалогах группы маленькие — оценки грубые)
display(metrics_table({"прямое предсказание": pilot_direct}, by="topic"))
''')

code(r'''
import matplotlib.pyplot as plt

d = pilot_direct.dropna(subset=["pred"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(d.delta, d.pred - d.pre_belief, s=18, alpha=0.6)
lim = [-80, 100]
ax[0].plot(lim, lim, lw=1, color="gray")
ax[0].axhline(0, lw=0.5, color="gray"); ax[0].axvline(0, lw=0.5, color="gray")
ax[0].set(xlabel="реальный сдвиг убеждения", ylabel="предсказанный сдвиг", title="Прямое предсказание: сдвиг")
ax[1].hist([d.delta.abs(), (d.pred - d.pre_belief).abs()], bins=12, label=["реальный", "предсказанный"])
ax[1].set(xlabel="|сдвиг|", ylabel="диалогов", title="Сила изменения убеждений")
ax[1].legend()
plt.tight_layout(); plt.show()
''')

md("### 4.2. Стоимость")

code(r'''
print(f"Модель: {llm.provider}/{llm.model}")
llm.stats()
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "toc_visible": True},
}
nbf.write(nb, sys.argv[1])
print("cells:", len(cells))
