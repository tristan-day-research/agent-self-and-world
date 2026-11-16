"""Offline causal-instrument and factorial-design checks; no downloads."""
import json,tempfile,re
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import torch
from core import runs

class TinyTokenizer:
    chat_template='synthetic word-offset template'
    def decode(self,ids,**kwargs):return ' '.join(f'tok{i}' for i in ids)
    def apply_chat_template(self,messages,**kwargs):return '\n'.join(m['role']+': '+m['content'] for m in messages)+'\nmodel:'
    def __call__(self,text,return_offsets_mapping=False,**kwargs):
        spans=[m.span() for m in re.finditer(r'\S+',text)]
        ids=[1+sum(map(ord,text[a:b]))%30 for a,b in spans]
        if return_offsets_mapping:return {'input_ids':ids,'offset_mapping':spans}
        class Inputs(dict):
            def to(self,device):return Inputs({k:v.to(device) for k,v in self.items()})
        t=torch.tensor([ids]);return Inputs(input_ids=t,attention_mask=torch.ones_like(t))


def setup_model():
    from transformers import Gemma3TextConfig,Gemma3ForCausalLM
    torch.manual_seed(4)
    config=Gemma3TextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=12,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,layer_types=['full_attention']*12,
        max_position_embeddings=1024,bos_token_id=1,eos_token_id=31,pad_token_id=0)
    model=Gemma3ForCausalLM(config).eval();model.set_attn_implementation('eager')
    return model,model.model.layers[9]

class SpeakerTests(TestCase):
    def test_factorial_design_and_category_coverage(self):
        from core.probes import read_probes
        from core.paper_features import feature_manifest
        from collections import Counter
        rows,_=read_probes('experiments/ism/speaker_probes.jsonl')
        self.assertEqual(len(rows),24)
        self.assertEqual(len({tuple(sorted(p['factors'].items())) for p in rows}),24)
        self.assertEqual(Counter(f['category'] for f in feature_manifest()),
            {'ISM':2,'Narrative':4,'Tone':1,'Concept':1,'A-traits':6,'A-nature':5,'A-summon':4})
        from core.speaker_analysis import align_spans
        for p in rows:
            formatted=TinyTokenizer().apply_chat_template(p['messages'])
            groups=align_spans(TinyTokenizer(),formatted,p['attention_spans'])
            self.assertTrue(all(groups.values()))

    def run_tiny(self,root,flags):
        from run import main
        from experiments.ism.main import toy_sae
        model,layer=setup_model()
        with patch.object(runs,'PROJECT_ROOT',root),patch('core.gemma.load_gemma',return_value=(model,TinyTokenizer(),layer)),patch('core.gemma.load_selected_sae',return_value=(toy_sae(4),{})):
            main(['ism','--device','cpu','--dtype','float32','--monitor','ism','--feature','133',
                '--attention','--attention-layers','8','9','10','11','--teacher-force-baseline',
                '--query-layer','11','--query-head','1','--query-control-head','0','--max-new-tokens','2',*flags])
        directory,=(root/'outputs/ism').iterdir()
        return directory,json.loads((directory/'outputs.json').read_text())['results']

    def test_rescue_pipeline_self_invariance_and_locality(self):
        from core.attention_analysis import load_tensors
        with tempfile.TemporaryDirectory() as temp:
            directory,rows=self.run_tiny(Path(temp),['--prompt','Introduce yourself.','--strengths','-1','1','--query-rescue'])
            self.assertEqual(len(rows),10)
            assessment=json.loads((directory/'speaker_assessment.json').read_text())
            self.assertEqual(len(assessment['query_rescues']),6)
            baseline=rows[0]
            for row in rows:
                p=row.get('query_intervention')
                if not p:continue
                original=next(r for r in rows if r['strength']==row['strength'] and not r.get('query_intervention'))
                if p['kind']=='self':
                    self.assertEqual(row['token_ids'],original['token_ids'])
                    self.assertEqual(row['fixed_continuation']['continuation_likelihood'],original['fixed_continuation']['continuation_likelihood'])
                for phase in ['prompt','fixed']:
                    owner=lambda r:r if phase=='prompt' else r['fixed_continuation']
                    current=load_tensors(directory,owner(row)['attention'])
                    before=load_tensors(directory,owner(original)['attention'])
                    for key in ['k_raw_norm','k_normalized_norm','v_norm']:
                        torch.testing.assert_close(current['L11.'+key],before['L11.'+key],rtol=0,atol=0)
                    if p['kind']=='restore':
                        b=load_tensors(directory,owner(baseline)['attention'])
                        torch.testing.assert_close(current['L11.q_normalized'][1],b['L11.q_normalized'][1],rtol=0,atol=0)
                        torch.testing.assert_close(current['L11.q_normalized'][0],before['L11.q_normalized'][0],rtol=0,atol=0)

    def test_person_swap_pipeline_and_cross_setting_analysis(self):
        probes=[json.loads(line) for line in Path('experiments/ism/speaker_probes.jsonl').read_text().splitlines()]
        probes=[p for p in probes if p['factors']['character']=='mira' and p['factors']['prose']=='plain']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'probes.jsonl';source.write_text(''.join(json.dumps(p)+'\n' for p in probes))
            directory,rows=self.run_tiny(root,['--prompt-file',str(source),'--strengths','0','--query-person-swap'])
            self.assertEqual(len(rows),30)
            assessment=json.loads((directory/'speaker_assessment.json').read_text())
            self.assertEqual(len(assessment['query_rescues']),18)
            self.assertEqual(len(assessment['cross_setting_comparisons']),48)
            self.assertTrue(assessment['semantic_attention'])
            for row in rows:
                p=row.get('query_intervention')
                if not p:continue
                if p['kind']=='person_self':
                    base=next(r for r in rows if r['prompt_id']==row['prompt_id'] and r['strength'] is None and not r.get('query_intervention'))
                    self.assertEqual(row['token_ids'],base['token_ids'])
                else:
                    self.assertNotEqual(p['donor_person'],row['design_factors']['person'])
                    donor=next(r for r in rows if r['prompt_id']==p['donor_prompt_id'] and not r.get('query_intervention'))
                    self.assertEqual({k:v for k,v in donor['design_factors'].items() if k!='person'},
                        {k:v for k,v in row['design_factors'].items() if k!='person'})

    def test_fixed_likelihood_alignment(self):
        from core.assessment import assess_fixed_continuation
        from experiments.ism.main import toy_sae
        model,layer=setup_model();inputs=TinyTokenizer()('A short question')
        tokens=[3,4,5]
        result=assess_fixed_continuation(model,layer,inputs,tokens,torch.zeros(32),None,'all',toy_sae(4),TinyTokenizer())
        fixed=torch.cat([inputs['input_ids'],torch.tensor([tokens[:-1]])],dim=1)
        with torch.inference_mode():logits=model(input_ids=fixed,use_cache=False).logits[0,len(inputs['input_ids'][0])-1:]
        expected=-torch.nn.functional.cross_entropy(logits,torch.tensor(tokens),reduction='none')
        torch.testing.assert_close(torch.tensor(result['continuation_likelihood']['token_log_probabilities']),expected)
