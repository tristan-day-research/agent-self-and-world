import argparse
import unittest

from experiments.agent_vs_model.gemma4 import plan


class Gemma4DoorTests(unittest.TestCase):
    def test_matched_plan(self):
        args = argparse.Namespace(design='capability', baseline_only=True, families=8,
                                  seed=1042, capability_wordings=['original', 'explicit'],
                                  max_new_tokens=12)
        trials, details = plan(args)
        self.assertEqual(len(trials), 128)
        self.assertEqual(details['model'], 'google/gemma-4-31B-it')
        self.assertFalse(details['thinking'])
        self.assertEqual(len({t['request_key'] for t in trials}), 104)


if __name__ == '__main__':
    unittest.main()
