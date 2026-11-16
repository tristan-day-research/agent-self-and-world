"""Offline guards for the bounded Gemma-4 protocol."""
import unittest
from types import SimpleNamespace

from experiments.gemma4_suite.main import design, plan
from experiments.gemma4_suite.door_mechanisms import door_design
from experiments.gemma4_suite.speaker_design import prompts
from experiments.gemma4_suite.speaker_mechanisms import _direction, _feature_candidates


class Gemma4SuiteDesignTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(sections=['speaker', 'door', 'confidence'],
            families=4, seed=1043, speaker_set='both', questions_per_domain=1,
            personas=['default_assistant', 'chemist', 'historian', 'artist'],
            sae_layer=30, sae=True, jlens=True, jlens_layers=[30, 45, 52],
            queries=True, query_layers=[22, 30, 45, 52], max_new_tokens=96)

    def test_planned_counts_and_crossed_prompt(self):
        args = self.args()
        rows = design(args)
        self.assertEqual(plan(args)['sections'], {'speaker': 27, 'door': 128, 'confidence': 40})
        self.assertEqual(len(rows), len({r['id'] for r in rows}))
        door = [r for r in rows if r['section'] == 'door']
        self.assertEqual({r['condition'] for r in door},
                         {'present_knows', 'present_doesnt_know',
                          'absent_knows', 'absent_doesnt_know'})
        for row in door:
            record = row['messages'][0]['content']
            self.assertEqual('Door code: not supplied.' in record, not row['code_available'])
            if row['task'] == 'instrumental':
                self.assertIn('Determine whether its final digit is even or odd.',
                              row['messages'][-1]['content'])

    def test_gemma3_feature_ids_are_not_in_plan(self):
        self.assertNotIn('features', plan(self.args()))

    def test_followup_family_and_character_splits(self):
        door = door_design(16, 1044)
        self.assertEqual(len(door), 256)
        self.assertEqual(len({r['family'] for r in door}), 16)
        self.assertEqual({r['task'] for r in door}, {'self_report', 'instrumental'})
        speaker = prompts()
        self.assertEqual(len(speaker), 128)
        self.assertEqual({split: len([r for r in speaker if r['split'] == split])
                          for split in ('train', 'validation', 'steer_test')},
                         {'train': 64, 'validation': 32, 'steer_test': 32})
        keys = {(r['character'], r['prose'], r['viewpoint']) for r in speaker
                if r['kind'] == 'viewpoint'}
        self.assertEqual(len(keys), 32)
        systems = {split: next(r['messages'][0]['content'] for r in speaker
                               if r['split'] == split and r['kind'] == 'speaker_setting'
                               and r['setting'] == 'story')
                   for split in ('train', 'validation', 'steer_test')}
        self.assertEqual(len(set(systems.values())), 3)
        viewpoints = {split: next(r['messages'][-1]['content'].split('Viewpoint: ')[-1]
                                 for r in speaker if r['split'] == split and
                                 r['kind'] == 'viewpoint' and r['viewpoint'] == 'immersed')
                      for split in ('train', 'validation', 'steer_test')}
        self.assertEqual(len(set(viewpoints.values())), 3)

    def test_train_only_speaker_direction_and_feature_candidate(self):
        import torch
        rows = prompts()
        vectors = {r['id']: {30: torch.tensor([2., 0.]) if r.get('viewpoint') == 'immersed'
                              else torch.tensor([1., 0.]) if r['kind'] == 'speaker_setting'
                              and r['setting'] == 'story' else torch.tensor([0., 0.])} for r in rows}
        unit, stats = _direction(rows, vectors, 30, 'viewpoint', 'train', 'immersed', 'detached')
        self.assertTrue(torch.allclose(unit, torch.tensor([1., 0.])))
        self.assertEqual(stats['n_pairs'], 8)
        _, third_stats = _direction(rows, vectors, 30, 'speaker_setting', 'train',
                                    'story', 'assistant', 'third')
        self.assertEqual(third_stats['n_pairs'], 8)
        sparse = {r['id']: {'indices': torch.tensor([42], dtype=torch.int32),
                            'values': torch.tensor([3.] if r.get('viewpoint') == 'immersed' else [0.])}
                  for r in rows if r['kind'] == 'viewpoint'}
        self.assertEqual(_feature_candidates(rows, sparse)[0]['feature_id'], 42)
