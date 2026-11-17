"""Регрессии на минимальных парах, включая последнюю реплику и смену тезиса."""
import unittest
from types import SimpleNamespace

from perspectives import belief_trace, detect_with_positions, observe, pilot_predictions


class Reply:
    def __init__(self, responses):
        self.responses = iter(responses)

    def call_json(self, prompt, validate, **kwargs):
        return validate(next(self.responses))


class PerspectivesTest(unittest.TestCase):
    def test_minimal_pairs(self):
        for stance, words, expected in [("agrees", "Я в это верю", 62.5),
                                        ("disagrees", "Я этому не верю", 37.5),
                                        ("acknowledges", "Я понял", 50),
                                        ("accepts_action", "Я сделаю это", 50),
                                        ("unclear", "А где?", 50)]:
            with self.subTest(stance=stance):
                model = Reply([dict(stance=stance, quote=words)])
                obs = observe(model, "Отделение открыто", [dict(speaker="V", text=words)], 0)
                self.assertEqual(belief_trace([obs], 50)[0]["rating"], expected)
                self.assertIsNone(belief_trace([obs])[0]["rating"])

    def test_wrong_speaker_quote(self):
        turns = [dict(speaker="S", text="Я уверен"), dict(speaker="V", text="Спасибо")]
        with self.assertRaises(ValueError):
            observe(Reply([dict(stance="agrees", quote="Я уверен")]), "Это банк", turns, 1)

    def test_terminal_turn_and_new_target(self):
        turns = [dict(speaker=s, text=t) for s, t in (
            ("S", "A"), ("V", "нет"), ("S", "A"), ("V", "я согласен"),
            ("S", "B"), ("V", "ладно"))]
        parses = [dict(target_belief=t["text"] if t["speaker"] == "S" else None) for t in turns]
        def parse(ts, **kwargs):
            self.assertEqual(kwargs["mode"], "steps")
            return parses[:len(ts)]
        def states(ts, ps, weights):
            return {s: (None, [(i, 0, {"action": 0.5}) for i, t in enumerate(ts, 1) if t["speaker"] == s])
                    for s in ("S", "V")}
        model = Reply([dict(stance="disagrees", quote="нет"), dict(stance="agrees", quote="я согласен"),
                       dict(stance="acknowledges", quote="ладно")])
        histories, trace = detect_with_positions(turns, model, parse, states, {"action": 0.3, "credence": 0.2})
        self.assertEqual([p["risk"] for p in trace], [0.15, 0.15, 0.15, 0.35, 0.15, 0.15])
        self.assertIsNone(histories["V", "B"][-1]["last_stance"])
        short = Reply([dict(stance="disagrees", quote="нет"), dict(stance="agrees", quote="я согласен")])
        _, prefix = detect_with_positions(turns[:4], short, parse, states, {"action": 0.3, "credence": 0.2})
        self.assertEqual(prefix, trace[:4])

    def test_review_budget_and_no_post_in_prompt(self):
        calls = []
        answers = dict(stance_v2=dict(stance="agrees", quote="I agree with the claim"),
                       quotes_v2=45, trace_v2=40, review_v2=55, metamon_v2=60)

        class Model:
            def call_json(self, prompt, tag, validate, **kwargs):
                calls.append((tag, prompt))
                value = answers[tag]
                return validate(value if isinstance(value, dict) else {"post_belief": value})

        row = SimpleNamespace(id="demo", pre_belief=30, post_belief=97, belief_statement="The room is open",
                              turns=[dict(speaker="USER", text="I agree with the claim")])
        result, _ = pilot_predictions(row, Model(), lambda row, model: 45, lambda turns: turns[0]["text"],
                                      lambda obj: obj["post_belief"], "test")
        self.assertTrue(result["flagged"])
        self.assertEqual([tag for tag, _ in calls], list(answers))
        self.assertTrue(all("97" not in prompt for _, prompt in calls))
        self.assertEqual((result["review"], result["metamon"]), (55, 60))
