import copy
import unittest

from run_belief_shift import request, signal, synthetic


class BeliefShiftTest(unittest.TestCase):
    def setUp(self):
        self.control, self.change = list(synthetic())[:2]

    def test_contrast_changes_only_the_last_words(self):
        cases = list(synthetic())
        self.assertEqual(len(cases), 24)
        for control, change in zip(cases[::2], cases[1::2]):
            for key in ("pair", "pre", "statement", "push"):
                self.assertEqual(control[key], change[key])
            self.assertEqual(control["turns"][:-1], change["turns"][:-1])
            self.assertNotEqual(control["turns"][-1], change["turns"][-1])
            self.assertEqual(signal(control["gold"]), "hold")
            self.assertIn(change["gold"], ("up", "down"))

    def test_target_and_outcome_cannot_enter_either_prompt(self):
        altered = copy.deepcopy(self.change)
        altered.update(gold="down", post=2, variant="control", push="down")
        for grounded in (False, True):
            self.assertEqual(request(self.change, grounded)[0], request(altered, grounded)[0])

    def test_quote_must_come_from_correct_speaker_and_turn(self):
        _, validate = request(self.change, True)
        for i in (0, 1):
            with self.assertRaises(ValueError):
                validate(dict(judgment="up", turn=i + 1, quote=self.change["turns"][i]["text"]))
        with self.assertRaises(ValueError):
            validate(dict(judgment="up", turn=3, quote="вымышленная цитата"))
        text = self.change["turns"][2]["text"]
        self.assertEqual(validate(dict(judgment="up", turn=3, quote=text))["quote"], text)

    def test_abstention_remains_distinct_from_stable_and_invalid(self):
        _, validate = request(self.control, True)
        unknown = validate(dict(judgment="unknown", turn=0, quote=""))
        self.assertEqual(unknown, dict(judgment="unknown", turn=0, quote=""))
        self.assertEqual(signal("stable"), signal("unknown"))
        self.assertNotEqual(signal("invalid"), signal("unknown"))
