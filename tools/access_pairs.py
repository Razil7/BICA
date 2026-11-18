"""Парный тест перехода I-Now → He-Now → I-Public. Сценарии полностью синтетические."""
import json
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Setting:
    topic: str
    old: str
    current: str
    aside: str
    messages: tuple


SETTINGS = {
    "room": Setting("аудитория семинара", "Б-12", "Г-07", "Раздатки будут на столе.",
                    ("Выхожу.", "Подхожу к двери Б-12.", "Уже у двери Г-07.")),
    "verification": Setting("официальный канал проверки уведомлений", "личный кабинет", "раздел «Поддержка»",
                            "В выходные ответ может занять больше времени.",
                            ("Спасибо за помощь.", "Для проверки по привычке открою личный кабинет.",
                             "Статус уведомления посмотрю через раздел «Поддержка».")),
}
MODES = ("status", "neutral", "stale", "current")
ACTOR, OBSERVER, STAFF = "участник", "агент", "координатор"


def cases():
    for domain, mode, received in product(SETTINGS, MODES, (False, True)):
        setting = SETTINGS[domain]
        audience = [OBSERVER, STAFF] + ([ACTOR] if received else [])
        utterance = None if mode == "status" else setting.messages[MODES.index(mode) - 1]
        events = [
            dict(id="e1", kind="bulletin", topic=setting.topic, value=setting.old,
                 seen_by=[OBSERVER, STAFF, ACTOR]),
            dict(id="e2", kind="bulletin", topic=setting.topic, value=setting.current, seen_by=audience),
            dict(id="e3", kind="notice", text=setting.aside, seen_by=[OBSERVER, STAFF, ACTOR]),
        ]
        if utterance is None:
            events.append(dict(id="e4", kind="status", actor=ACTOR, text="Готовится приступить."))
        else:
            events.append(dict(id="e4", kind="speech", actor=ACTOR, text=utterance))
        yield f"{domain}.{mode}", received, events


POLICY = """Агент помогает участнику. seen_by — журнал тех, кто увидел именно это сообщение.
Официальные бюллетени (bulletin) последовательно заменяют значение указанной темы.
Стороннее уведомление (notice) не обновляет тему. Обрабатывай только приведённые события.
Доступ к бюллетеню и собственная высказанная позиция участника — отдельные поля.
Даже увиденное сообщение само по себе не доказывает понимания. Участник может назвать
новое значение, даже если этот бюллетень ему не показывали.
Правило ответа: если участник выражает устаревшую позицию, поправь его (correct).
Если он сам называет актуальное значение, достаточно acknowledge. Если позиции нет,
при отсутствии доступа сообщи значение (inform), при наличии доступа достаточно acknowledge.
В correct и inform передай точное актуальное значение, в acknowledge поставь public_value=null.
Верни JSON с полями world (актуальное значение дословно), seen (видел ли участник последний
бюллетень по теме, true/false), source (id этого бюллетеня, если видел, иначе null),
stance (old/current/unknown по собственным словам участника), action (correct/inform/acknowledge),
public_value (строка или null)."""


def transcript(events):
    lines = []
    for e in events:
        if e["kind"] == "bulletin":
            body = f'Официальный бюллетень. Тема: {e["topic"]}. Значение: {e["value"]}.'
        elif e["kind"] == "notice":
            body = "Стороннее уведомление. " + e["text"]
        else:
            body = f'{e["actor"]}, ' + ("реплика" if e["kind"] == "speech" else "запись статуса") + ": " + e["text"]
        visibility = " Увидели: " + ", ".join(e["seen_by"]) + "." if "seen_by" in e else ""
        lines.append(e["id"] + ". " + body + visibility)
    return "\n".join(lines)


def baseline_prompt(events, structured):
    # Одна инструкция и одни исходные события. В сводке нет вычисленных ответов.
    data = json.dumps(events, ensure_ascii=False) if structured else transcript(events)
    return POLICY + "\n\nЖурнал событий:\n" + data


def interpretation_prompt(events):
    bulletins = [e for e in events if e["kind"] == "bulletin"]
    utterance = events[-1]["text"]
    return (f'Тема: {bulletins[-1]["topic"]}. Предыдущее значение: {bulletins[-2]["value"]}. '
            f'Текущее значение: {bulletins[-1]["value"]}. Реплика участника: {utterance}\n'
            'Какое значение участник сам использует или считает действующим? По приветствию, '
            'благодарности и простому подтверждению получения позиция неизвестна. '
            'Верни JSON {"stance":"old|current|unknown"}. Выбери одно значение.')


def parse_stance(obj):
    if obj["stance"] not in ("old", "current", "unknown"):
        raise ValueError("неизвестная категория позиции")
    return obj["stance"]


def parse_decision(obj):
    parse_stance(obj)
    if type(obj["seen"]) is not bool or not isinstance(obj["world"], str):
        raise ValueError("некорректный тип world/seen")
    if obj["action"] not in ("correct", "inform", "acknowledge"):
        raise ValueError("неизвестный тип ответа")
    for key in ("source", "public_value"):
        if obj[key] is not None and not isinstance(obj[key], str):
            raise ValueError("неверный тип " + key)
    return {k: obj[k] for k in ("world", "seen", "source", "stance", "action", "public_value")}


def check_access(events, bulletin, source):
    """MetaMon проверяет основание переноса текущего бюллетеня в модель участника."""
    evidence = next((e for e in events if e["id"] == source), None)
    if evidence is None:
        return False, "no_evidence"
    if evidence["kind"] != "bulletin" or evidence["id"] != bulletin["id"]:
        return False, "not_current_bulletin"
    if ACTOR not in evidence["seen_by"]:
        return False, "recipient_absent"
    return True, "delivery_confirmed"


def controller(events, interpret, trace=None):
    bulletin = next(e for e in reversed(events) if e["kind"] == "bulletin" and OBSERVER in e["seen_by"])
    candidate = next((e["id"] for e in reversed(events) if e["kind"] == "bulletin"
                      and e["topic"] == bulletin["topic"] and ACTOR in e["seen_by"]), None)
    seen, reason = check_access(events, bulletin, candidate)
    stance = interpret(interpretation_prompt(events)) if events[-1]["kind"] == "speech" else "unknown"
    if stance == "old":
        action = "correct"
    elif stance == "current" or seen:
        action = "acknowledge"
    else:
        action = "inform"
    if trace is not None:
        trace.update(world_source=bulletin["id"], proposed_source=candidate, monitor=reason,
                     stance_source=events[-1]["id"] if stance != "unknown" else None,
                     model_requested=events[-1]["kind"] == "speech")
    return dict(world=bulletin["value"], seen=seen, source=bulletin["id"] if seen else None,
                stance=stance, action=action, public_value=None if action == "acknowledge" else bulletin["value"])


def public_message(decision):
    first = {"correct": "Уточню.", "inform": "Передаю обновление.", "acknowledge": "Принято."}
    value = decision["public_value"]
    return first[decision["action"]] + (f" Актуальное значение — {value}." if value is not None else "")


# Золотые метки определены конструкцией сценариев независимо от реализации контроллера.
ACTIONS = {"status": ("inform", "acknowledge"), "neutral": ("inform", "acknowledge"),
           "stale": ("correct", "correct"), "current": ("acknowledge", "acknowledge")}


def gold(pair, received):
    domain, mode = pair.split(".")
    value = SETTINGS[domain].current
    action = ACTIONS[mode][int(received)]
    return dict(world=value, seen=received, source="e2" if received else None,
                stance={"stale":"old", "current":"current"}.get(mode, "unknown"),
                action=action, public_value=None if action == "acknowledge" else value)
