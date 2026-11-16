import argparse
import unittest

from experiments.agent_vs_model.qwen35 import encode_sae, plan, summarize


class QwenDoorTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(baseline_only=True, design='capability', families=4, seed=42,
                      capability_wordings=['original'],
                      qwen_sae_layers=[16, 32, 48], qwen_lens_layers=[16, 32, 48],
                      qwen_attention_layers=[15, 31, 47, 63], attention=True,
                      qwen_lens='r')
        values.update(changes)
        return argparse.Namespace(**values)

    def test_bounded_capability_plan(self):
        trials, summary = plan(self.args())
        self.assertEqual(len(trials), 48)
        self.assertEqual(summary['scored_forwards'], 48)
        self.assertEqual(summary['short_generations'], 48)
        self.assertEqual(summary['lens_kind'], 'r')
        paired, paired_plan = plan(self.args(seed=1042, families=8,
                                            capability_wordings=['original', 'explicit']))
        self.assertEqual(len(paired), 128)
        self.assertEqual(paired_plan['scored_forwards'], 128)
        original = next(t for t in paired if t['task'] == 'instrumental' and t['wording'] == 'original')
        explicit = next(t for t in paired if t['task'] == 'instrumental' and t['wording'] == 'explicit')
        self.assertEqual(original['code'], explicit['code'])
        self.assertEqual(original['correct_label'], explicit['correct_label'])
        self.assertIn('Determine whether its final digit is even or odd', explicit['messages'][-1]['content'])
        self.assertNotIn('Determine whether its final digit is even or odd', original['messages'][-1]['content'])
        crossed, crossed_summary = plan(self.args(design='crossed'))
        self.assertEqual(len(crossed), 128)
        self.assertEqual(crossed_summary['scored_forwards'], 96)
        with self.assertRaises(ValueError):
            plan(self.args(baseline_only=False))
        with self.assertRaises(ValueError):
            plan(self.args(qwen_sae_layers=[16, 16]))

    def test_topk_sae_feature_ids(self):
        import torch
        encoder = torch.eye(64)
        bias = torch.zeros(64)
        values = encode_sae(torch.arange(64).float(), encoder, bias)
        self.assertEqual(values['top_features'][0]['feature'], 63)
        self.assertEqual(values['top_features'][0]['activation'], 63)
        self.assertEqual(len(values['top_features']), 20)

    def test_capability_summary(self):
        rows = [dict(condition='capability', task='mapping', family='fam000',
                     request_key='a', parsed={'answer': 'R'}, correct_label='R',
                     correct_procedure='R', distribution={'choice_probability_mass': 0.9})]
        cell = summarize(rows)['cells'][0]
        self.assertEqual(cell['correct_rate'], 1)
        self.assertEqual(cell['mean_choice_mass'], .9)


if __name__ == '__main__':
    unittest.main()
