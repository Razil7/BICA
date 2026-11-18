import copy
import unittest

from access_pairs import cases, check_access, controller, gold, interpretation_prompt, parse_decision
from run_access_pairs import design


class AccessPairTest(unittest.TestCase):
    def test_only_recipient_changes(self):
        scenarios = list(cases())
        self.assertEqual(len(scenarios), 16)
        for a, b in zip(scenarios[::2], scenarios[1::2]):
            self.assertEqual(a[:2], (b[0], False))
            self.assertTrue(b[1])
            records = copy.deepcopy(b[2])
            records[1]["seen_by"].remove("участник")
            self.assertEqual(records, a[2])
            if records[-1]["kind"] == "speech":
                self.assertEqual(interpretation_prompt(a[2]), interpretation_prompt(b[2]))

    def test_controller_routing_and_reference(self):
        for pair, seen, events in cases():
            expected = gold(pair, seen)
            with self.subTest(pair=pair, seen=seen):
                calls = []
                def oracle(prompt):
                    calls.append(prompt)
                    return expected["stance"]
                self.assertEqual(controller(events, oracle), expected)
                self.assertEqual(len(calls), int(events[-1]["kind"] == "speech"))

    def test_evidence_id_and_queries(self):
        _, _, events = list(cases())[1]
        events[1]["id"] = "updated-id"
        self.assertEqual(controller(events, None)["source"], "updated-id")
        self.assertEqual(len(design()[0]), 38)
        with self.assertRaises(ValueError):
            parse_decision(dict(world="A", seen="false", source=None, stance="unknown",
                                action="acknowledge", public_value=None))

    def test_private_update_cannot_be_attributed_via_other_sources(self):
        _, _, events = list(cases())[6]  # private bulletin, speaker names the current value
        for candidate in ("e1", "e2", "e3", "e4", None):
            with self.subTest(candidate=candidate):
                self.assertFalse(check_access(events, events[1], candidate)[0])
        trace = {}
        result = controller(events, lambda _: "current", trace)
        self.assertEqual((result["seen"], result["stance"], result["action"]),
                         (False, "current", "acknowledge"))
        self.assertEqual(trace["stance_source"], "e4")
