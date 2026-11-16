import argparse
import unittest

from experiments.agent_vs_model.gemma27 import plan, summarize


class Gemma27DoorTests(unittest.TestCase):
    def test_matched_behavior_plan(self):
        args = argparse.Namespace(design='capability', baseline_only=True, families=8,
                                  seed=1042, capability_wordings=['original', 'explicit'],
                                  max_new_tokens=12)
        trials, summary = plan(args)
        self.assertEqual(len(trials), 128)
        self.assertEqual(summary['scored_forwards'], 128)
        self.assertEqual(sum(t['wording'] == 'explicit' for t in trials), 32)
        self.assertEqual(len({t['request_key'] for t in trials}), 104)
        args.baseline_only = False
        with self.assertRaises(ValueError):
            plan(args)

    def test_summary_separates_wordings(self):
        rows = [dict(task='instrumental', wording=w, request_key=w, family='f',
                     rule_order='even_first', parity=p, p_even_lr=value,
                     correct_label='L', parsed={'answer': answer},
                     distribution={'choice_probability_mass': 0.9})
                for w, answer in [('original', 'L'), ('explicit', 'R')]
                for p, value in [('even', 0.8), ('odd', 0.2)]]
        cells = {c['wording']: c for c in summarize(rows)['cells']}
        self.assertEqual(cells['original']['correct_rate'], 1)
        self.assertEqual(cells['explicit']['correct_rate'], 0)
        self.assertAlmostEqual(cells['original']['paired_parity_effect'], 0.6)


if __name__ == '__main__':
    unittest.main()
