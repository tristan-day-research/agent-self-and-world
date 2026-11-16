"""Small offline checks; pretrained capture must be validated on the pod."""

import unittest
from types import SimpleNamespace

from experiments.gemma4_suite.door_speaker_attention import design, plan


class DoorSpeakerAttentionTests(unittest.TestCase):
    def test_factorial_design_and_prompts(self):
        rows = design(2, 1046)
        self.assertEqual(len(rows), 192)
        self.assertEqual(len({r['id'] for r in rows}), 192)
        self.assertEqual({r['speaker'] for r in rows}, {'assistant', 'roleplay', 'story'})
        self.assertEqual({r['task'] for r in rows},
                         {'self_report', 'repeat', 'instrumental', 'free_response'})
        for row in rows:
            system, user = (m['content'] for m in row['messages'])
            self.assertEqual(row['code'] in system, row['code_available'])
            self.assertEqual('Door code: not supplied.' in system, not row['code_available'])
            self.assertIn('Mara knows' if row['attributed_knowledge'] else 'Mara does not know', user)
            for excerpt in row['spans'].values():
                self.assertEqual((system + user).count(excerpt), 1)
        self.assertEqual(plan(SimpleNamespace(door_families=2, seed=1046,
                                              attention_layers=[10, 30, 50]))['rows'], 192)

    def test_attention_hooks_capture_and_cleanup(self):
        import torch
        from torch import nn
        from core.gemma4_attention import record_boundary_attention, summarize_attention

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(_attn_implementation='eager')
                self.q_proj = nn.Linear(4, 4, bias=False)
                self.q_norm = nn.Identity()

            def forward(self, x):
                q = self.q_norm(self.q_proj(x).reshape(1, 3, 2, 2))
                weights = torch.zeros(1, 2, 3, 3)
                weights[:, :, -1, :] = torch.tensor([.2, .3, .5])
                return q, weights

        layer = SimpleNamespace(self_attn=FakeAttention())
        with record_boundary_attention([layer], [0], prompt_length=3) as records:
            layer.self_attn(torch.ones(1, 3, 4))
        self.assertEqual(records[0]['weights'].shape, (2, 3))
        self.assertEqual(len(layer.self_attn._forward_hooks), 0)
        self.assertEqual(len(layer.self_attn.q_proj._forward_hooks), 0)
        self.assertEqual(len(layer.self_attn.q_norm._forward_hooks), 0)
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
        summary = summarize_attention(records, {'code_field': [1]}, [10, 11, 12], tokenizer)
        self.assertAlmostEqual(summary['0']['heads'][0]['span_mass']['code_field'], .3)


if __name__ == '__main__':
    unittest.main()
