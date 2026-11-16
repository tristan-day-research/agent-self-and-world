"""Protect provenance, review blinding and the full-width loading audit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.probes import read_probes


class ValidationTests(unittest.TestCase):
    def test_confirmation_plan_is_offline_and_has_expected_bounds(self):
        from run import main
        stream = io.StringIO()
        with patch('core.gemma.load_gemma', side_effect=AssertionError('must not load')), \
             patch('core.runs.run_context', side_effect=AssertionError('must not write')), \
             contextlib.redirect_stdout(stream):
            self.assertEqual(main(['ism', '--dry-run', '--prompt-file',
                'experiments/ism/probes/confirmation.jsonl', '--strengths', '-2', '2',
                '--temperatures', '0', '1', '1', '--random-control']), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan['settings'], {'assistant': 4, 'roleplay': 4, 'story': 4})
        self.assertEqual(plan['evaluations'], 504)
        self.assertEqual(len(plan['prompt_provenance']['sha256']), 64)
        validation, _ = read_probes('experiments/ism/probes/confirmation.jsonl')
        heldout, _ = read_probes('experiments/ism/probes/heldout.jsonl')
        from core.calibration import read_reference_prompts
        reference, _ = read_reference_prompts('experiments/ism/reference_prompts.jsonl')
        messages = lambda rows: {json.dumps(p['messages'], sort_keys=True) for p in rows}
        self.assertFalse(messages(validation) & messages(heldout))
        self.assertFalse(messages(validation) & messages(reference))

    def test_invalid_probes_rejected_before_loading(self):
        valid = {'id': 'a', 'condition': 'story', 'source': 'authored',
                 'messages': [{'role': 'user', 'content': 'Write a story.'}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'probes.jsonl'
            for rows in ([valid, valid], [{**valid, 'source': ''}],
                         [{**valid, 'messages': [{'role': 'assistant', 'content': 'hello'}]}]):
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):
                    read_probes(p)

    def test_review_key_roundtrip_and_no_condition_leakage(self):
        from core.review import export_review
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            run.mkdir()
            rows = [{'messages': [{'role': 'user', 'content': 'Tell a story'}],
                     'text': f'Reply {i}', 'prompt_id': 'p', 'strength': strength,
                     'steering_features': [133], 'condition': 'steered'}
                    for i, strength in enumerate([-2, 2])]
            (run/'outputs.json').write_text(json.dumps({'results': rows}))
            dest = Path(tmp)/'review'
            result = export_review([run], dest)
            self.assertEqual(result['ratings_completed'], 0)
            cards = [json.loads(line) for line in (dest/'cards.jsonl').read_text().splitlines()]
            keys = json.loads((dest/'private_key.json').read_text())['key']
            for c in cards:
                self.assertEqual(set(c), {'id', 'messages', 'reply'})
                self.assertEqual(c['reply'], rows[keys[c['id']]['result_index']]['text'])
            ratings = [json.loads(line) for line in (dest/'ratings.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['assistant_framing'] is None for r in ratings))
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                export_review([run], dest)

    def test_full_width_audit_detects_wrong_selected_weights(self):
        import torch
        from safetensors.torch import save_file
        from core.sae import SelectedJumpReLU
        from core.sae_audit import audit_selected_sae
        generator = torch.Generator().manual_seed(12)
        weights = {'w_enc': torch.randn(8, 20, generator=generator),
                   'w_dec': torch.randn(20, 8, generator=generator),
                   'b_enc': torch.randn(20, generator=generator),
                   'b_dec': torch.randn(8, generator=generator),
                   'threshold': torch.ones(20)}
        ids = [2, 17]
        sae = SelectedJumpReLU(ids, weights['w_enc'][:, ids], weights['b_enc'][ids],
                              weights['threshold'][ids], weights['w_dec'][ids])
        residuals = torch.randn(30, 8, generator=generator)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'params.safetensors'
            save_file(weights, path)
            result = audit_selected_sae(sae, residuals, path)
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['sample_positions']), 16)
            self.assertEqual(result['full_width'], 20)
            sae.w_enc += 3
            with self.assertRaises(AssertionError):
                audit_selected_sae(sae, residuals, path)
