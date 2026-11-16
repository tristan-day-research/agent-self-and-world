"""Offline tests for scientific measurement invariants and the migrated CLI."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ConfidenceTests(unittest.TestCase):
    def test_conditional_entropy_can_be_sharp_while_choice_mass_is_tiny(self):
        import torch
        from core.choice_metrics import distribution_metrics,label_token_ids
        m=distribution_metrics(torch.tensor([100.,1.,0.,-2.]),{'A':1,'B':2},'B')
        self.assertLess(m['choice_probability_mass'],1e-30)
        self.assertEqual(m['choice_argmax'],'A')
        self.assertFalse(m['unconstrained_argmax_is_choice'])
        self.assertAlmostEqual(m['correct_choice_nll'],-math.log(m['choice_probabilities']['B']))
        equal=distribution_metrics(torch.zeros(5),{'A':1,'B':2,'C':3,'D':4})
        self.assertAlmostEqual(equal['choice_entropy_normalized'],1.)
        self.assertAlmostEqual(equal['choice_entropy_nats'],math.log(4))
        class Tokenizer:
            def encode(self,letter,**kwargs): return [1,2]
        with self.assertRaisesRegex(ValueError,'exactly one'):
            label_token_ids(Tokenizer(),['A','B'])
        with self.assertRaises(ValueError):
            distribution_metrics(torch.zeros(5),{'A':1,'B':1})

    def test_shared_questions_and_neutral_controls_preserve_the_originals(self):
        from experiments.confidence_entropy.design import load_design,ROOT,prompt_texts
        q,p,t,provenance=load_design(ROOT/'questions.json',ROOT/'protocol.json',2,42,persona_set='both')
        self.assertEqual(len(q),10)
        self.assertEqual(len(p),10)
        self.assertEqual(len({x['id'] for x in q}),10)
        chemist=next(x for x in p if x['id']=='chemist')
        neutral=next(x for x in p if x['id']=='chemist_neutral')
        self.assertIn('especially confident',chemist['system_prompt'])
        self.assertNotIn('especially confident',neutral['system_prompt'])
        self.assertIn('deep expertise in chemistry',neutral['system_prompt'])
        self.assertEqual(q,load_design(ROOT/'questions.json',ROOT/'protocol.json',2,42,persona_set='both')[0])
        texts=prompt_texts(q[0],t)
        self.assertEqual(len(set(texts.values())),4)
        self.assertTrue(texts['answer'].endswith(': '))
        self.assertNotIn(q[0]['correct_answer']+' is correct',texts['answer'])
        all_q,_,_,_=load_design(ROOT/'questions.json',ROOT/'protocol.json',None,42)
        self.assertEqual(len(all_q),150)

    def test_difference_in_differences_removes_shared_steering_effect(self):
        from core.confidence_analysis import summarize_confidence
        def row(persona,strength,h):
            return {'decision':'answer','question_id':'q','domain':'art','persona':persona,
                    'strength':strength,'direction_kind':'sae','expert_match':persona=='artist',
                    'distribution':{'choice_entropy_nats':h,'choice_entropy_normalized':h,
                    'correct_choice_probability':.5,'correct_choice_nll':1.,'multiclass_brier':.5,
                    'choice_probability_mass':1.,'choice_argmax_correct':True,'unconstrained_argmax_is_choice':True}}
        with tempfile.TemporaryDirectory() as tmp:
            result=summarize_confidence([row('default_assistant',None,.8),row('artist',None,.6),
                                        row('default_assistant',1.,.7),row('artist',1.,.5)],tmp)
            self.assertAlmostEqual(result['interactions'][0]['difference_in_differences']['choice_entropy_nats'],0.)

    def test_paired_capture_zero_upstream_and_direct_final(self):
        import torch
        from transformers import Gemma3ForCausalLM
        from core.decision import measure_decision
        from experiments.ism.main import toy_setup,toy_sae
        from experiments.confidence_entropy.main import SmokeTokenizer
        model,_,inputs,direction=toy_setup(0)
        cfg=model.config
        cfg.num_hidden_layers=3
        cfg.layer_types=['sliding_attention','full_attention','sliding_attention']
        model=Gemma3ForCausalLM(cfg).eval()
        layer=model.model.layers[1]
        lenses={0:(model.model.layers[0],torch.eye(32)),1:(layer,torch.eye(32)),2:(model.model.layers[2],None)}
        common=dict(model=model,inputs=inputs,layer=layer,sae=toy_sae(0),direction=direction,
                    choices={'A':2,'B':3},tokenizer=SmokeTokenizer(),correct_label='A',lenses=lenses)
        baseline=measure_decision(strength=None,**common)
        zero=measure_decision(strength=0,**common)
        steered=measure_decision(strength=2,**common)
        self.assertEqual(baseline['distribution'],zero['distribution'])
        up=lambda r: next(p['_vector_observed'] for p in r['jspace'] if p['layer']==0)
        torch.testing.assert_close(up(baseline),up(steered),rtol=0,atol=0)
        self.assertNotEqual(baseline['sae_boundary_post'],steered['sae_boundary_post'])
        self.assertEqual(len(steered['jspace']),4)
        self.assertTrue(all(not block._forward_hooks for block in model.model.layers))

    def test_cli_smoke_artifacts_and_failed_run_resume(self):
        from run import main
        from core import runs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            flags=['confidence_entropy','--smoke','--device','cpu','--random-control','--generate-responses']
            with patch.object(runs,'PROJECT_ROOT',root):
                main(flags)
                directory,=(root/'outputs/_smoke/confidence_entropy').iterdir()
                outputs=json.loads((directory/'outputs.json').read_text())
                self.assertEqual(len(outputs['results']),48)
                self.assertTrue(all('open_response' in r and 'coded_response' in r for r in outputs['results']))
                self.assertTrue(all('open_sae_generated' in r for r in outputs['results']))
                self.assertTrue((directory/'jspace_effects.json').exists())
                geometry=json.loads((directory/'persona_jspace.json').read_text())['contrasts']
                self.assertTrue(geometry)
                self.assertTrue(all(r['interaction_vector_l2']==0 for r in geometry if r['layer']==0 and r['phase']=='pre'))
                original=outputs['results']
                self.assertTrue(all(p['delta_l2']==0 for p in json.loads((directory/'jspace_effects.json').read_text())['comparisons'] if p['layer']==0 and p['phase']=='pre'))
                # Simulate interruption after the first cell, retaining its exact rows/vectors.
                (directory/'outputs.json').write_text(json.dumps({**outputs,'results':original[:2]}))
                status=json.loads((directory/'status.json').read_text())
                status['state']='interrupted'
                (directory/'status.json').write_text(json.dumps(status))
                main(flags+['--resume-directory',str(directory)])
                resumed=json.loads((directory/'outputs.json').read_text())['results']
                self.assertEqual(resumed,original)
                self.assertEqual(json.loads((directory/'status.json').read_text())['resume_count'],1)
                with self.assertRaisesRegex(ValueError,'active or completed'):
                    main(flags+['--resume-directory',str(directory)])
                self.assertFalse((directory/'report.md').exists())

    def test_dry_run_has_no_inference_or_run_directory(self):
        from run import main
        stream=io.StringIO()
        with patch('core.gemma.load_gemma',side_effect=AssertionError('must not load')), \
             patch('experiments.confidence_entropy.main.run_context',side_effect=AssertionError('must not write')), \
             contextlib.redirect_stdout(stream):
            main(['confidence_entropy','--dry-run'])
        plan=json.loads(stream.getvalue())
        self.assertEqual(plan['scored_forward_passes'],800)
        self.assertEqual(plan['optional_generation_calls'],0)

    def test_resume_rejects_changed_parameters(self):
        from core import runs
        with tempfile.TemporaryDirectory() as tmp,patch.object(runs,'PROJECT_ROOT',Path(tmp)):
            with self.assertRaises(RuntimeError):
                with runs.run_context('confidence_entropy',{'strength':1}) as run:
                    directory=run.directory
                    raise RuntimeError('test interruption')
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                with runs.run_context('confidence_entropy',{'strength':2},directory): pass

    def test_pretrained_branch_with_tiny_real_wrapper_and_reused_calibration(self):
        import torch
        from transformers import Gemma3Config,Gemma3ForConditionalGeneration,SiglipVisionConfig
        from experiments.ism.main import toy_setup,toy_sae
        from experiments.confidence_entropy.main import SmokeTokenizer
        from core.gemma import MODEL_ID,MODEL_REVISION,SAE_ID,SAE_REVISION,SAE_PATH
        from core import runs
        from run import main
        text,_,_,_=toy_setup(4)
        text.config.num_hidden_layers=12
        text.config.layer_types=['sliding_attention','full_attention']*6
        vision=SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                                 num_attention_heads=2,image_size=16,patch_size=8)
        model=Gemma3ForConditionalGeneration(Gemma3Config(text_config=text.config.to_dict(),
            vision_config=vision.to_dict(),image_token_index=1000,boi_token_index=1001,
            eoi_token_index=1002,mm_tokens_per_image=4)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            calibration=root/'calibration.json'
            calibration.write_text(json.dumps({'kind':'reference_set_peaks','complete':True,
                'model':MODEL_ID,'model_revision':MODEL_REVISION,'sae':SAE_ID,
                'sae_revision':SAE_REVISION,'sae_path':SAE_PATH,'layer':9,'dtype':'float32',
                'feature_peaks':{'133':2.,'4360':1.}}))
            with patch.object(runs,'PROJECT_ROOT',root), \
                 patch('core.gemma.load_gemma',return_value=(model,SmokeTokenizer(),model.model.language_model.layers[9])), \
                 patch('core.gemma.load_selected_sae',return_value=(toy_sae(1),{})), \
                 patch('core.jlens.load_jacobians',return_value=({l:torch.eye(32) for l in [8,9,10]},{})):
                main(['confidence_entropy','--device','cpu','--dtype','float32','--domains','art',
                      '--questions-per-domain','1','--personas','default_assistant','chemist',
                      '--persona-set','original','--calibration-file',str(calibration),
                      '--jlens-layers','8','9','10','11'])
            directory,=(root/'outputs/confidence_entropy').iterdir()
            rows=json.loads((directory/'outputs.json').read_text())['results']
            self.assertEqual(len(rows),16)
            self.assertEqual({p['layer'] for p in rows[0]['jspace']},{8,9,10,11})
            self.assertTrue(all(r['correct_answer'] is not None for r in rows if r['decision']=='answer'))
