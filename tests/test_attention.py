import tempfile
import unittest
from pathlib import Path
import torch
from core.attention import record_attention, save_attention_artifact, compare_attention
from core.steering import steer_layer
from experiments.ism.main import toy_setup

OPTIONS={'layers':[0,1], 'max_positions':3,'stride':1}

class AttentionTests(unittest.TestCase):
    def test_observer_no_effect_zero_control_and_causal_order(self):
        model,layer,inputs,direction=toy_setup(12)
        model.set_attn_implementation("eager")
        with torch.inference_mode(): plain=model(**inputs,use_cache=False).logits
        records=[]
        for strength in [None,0,1]:
            with record_attention(model,inputs['input_ids'],3,OPTIONS) as record:
                if strength is None:
                    with torch.inference_mode(): output=model(**inputs,use_cache=False).logits
                    torch.testing.assert_close(output,plain,rtol=0,atol=0)
                else:
                    with steer_layer(layer,direction,strength),torch.inference_mode():
                        model(**inputs,use_cache=False)
            records.append(record)
        for key,value in records[0]['_tensors'].items():
            torch.testing.assert_close(value,records[1]['_tensors'][key],rtol=0,atol=0)
            if key.startswith('L0.'):
                torch.testing.assert_close(value,records[2]['_tensors'][key],rtol=0,atol=0)
        self.assertGreater((records[0]['_tensors']['L1.q_rotary']-records[2]['_tensors']['L1.q_rotary']).abs().max(),0)
        rows=compare_attention(records[0]['_tensors'],records[1]['_tensors'])
        self.assertTrue(all(r['attention_js_nats']==0 and r['q_l2']==0 for r in rows))
        self.assertEqual(len(layer.self_attn._forward_hooks),0)

    def test_masks_gqa_positions_and_saved_artifact(self):
        model,layer,_,_=toy_setup(5)
        model.set_attn_implementation("eager")
        ids=torch.tensor([[1,2,3,4,5,6]])
        # Shorten local window to exercise it. Toy has GQA (2 Q heads / 1 KV head).
        layer.self_attn.sliding_window=2
        model.config.sliding_window=2
        with record_attention(model,ids,3,OPTIONS) as record,torch.inference_mode():
            model(input_ids=ids,use_cache=False)
        self.assertEqual(record['query_positions'],[2,3,4])
        for name, tensor in record['_tensors'].items():
            if name.endswith('.pattern'):
                for i,p in enumerate(record['query_positions']):
                    self.assertEqual(float(tensor[:,i,p+1:].abs().sum()),0.)
                torch.testing.assert_close(tensor.sum(-1),torch.ones_like(tensor.sum(-1)))
        self.assertEqual(record['_tensors']['L0.v_norm'].shape[0],1)
        with tempfile.TemporaryDirectory() as temp:
            result={'attention':record}
            save_attention_artifact(result,temp,0)
            saved=torch.load(Path(temp)/record['tensor_file'],weights_only=True)
            self.assertIn('L0.q_rotary',saved)
            self.assertNotIn('_tensors',record)

    def test_query_only_patch_holds_keys_values_fixed(self):
        from core.attention import patch_queries
        model,layer,inputs,direction=toy_setup(13)
        model.set_attn_implementation('eager')
        with record_attention(model,inputs['input_ids'],3,OPTIONS) as baseline,torch.inference_mode():
            model(**inputs,use_cache=False)
        records=[]
        from contextlib import nullcontext
        for patch in [False,True]:
            patcher=patch_queries(model.model.layers[1].self_attn,
                baseline['_tensors']['L1.q_normalized'],[2]) if patch else nullcontext()
            with patcher,record_attention(model,inputs['input_ids'],3,OPTIONS) as record,steer_layer(layer,direction,1),torch.inference_mode():
                model(**inputs,use_cache=False)
            records.append(record)
        for name in ['k_raw_norm','k_normalized_norm','v_norm']:
            torch.testing.assert_close(records[0]['_tensors']['L1.'+name],records[1]['_tensors']['L1.'+name],rtol=0,atol=0)
        torch.testing.assert_close(records[1]['_tensors']['L1.q_normalized'],baseline['_tensors']['L1.q_normalized'],rtol=0,atol=0)
        self.assertGreater((records[0]['_tensors']['L1.pattern']-records[1]['_tensors']['L1.pattern']).abs().max(),0)

    def test_analysis_rejects_different_text(self):
        import copy
        from core.attention_analysis import analyze_run
        from core.runs import write_json
        model,layer,inputs,_=toy_setup(7)
        model.set_attn_implementation('eager')
        with record_attention(model,inputs['input_ids'],3,OPTIONS) as record,torch.inference_mode():
            model(**inputs,use_cache=False)
        results=[{'prompt_id':'same','strength':strength,'attention':copy.deepcopy(record)} for strength in [None,0]]
        with tempfile.TemporaryDirectory() as temp:
            for i,r in enumerate(results): save_attention_artifact(r,temp,i)
            write_json(Path(temp)/'outputs.json',{'results':results})
            self.assertEqual(analyze_run(temp)['exact_zero_tensor_checks'],1)
            results[1]['attention']['input_ids'][0]+=1
            write_json(Path(temp)/'outputs.json',{'results':results})
            with self.assertRaisesRegex(ValueError,'nonidentical'):
                analyze_run(temp)

    def test_cached_decoding_skipped_and_cleanup_on_exception(self):
        model,layer,inputs,_=toy_setup(7)
        model.set_attn_implementation("eager")
        with record_attention(model,inputs['input_ids'],3,OPTIONS) as record,torch.inference_mode():
            model.generate(**inputs,max_new_tokens=3,pad_token_id=0)
        self.assertEqual(len(record['rows']),4)
        self.assertGreater(record['skipped_cached_calls'],0)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with record_attention(model,inputs['input_ids'],3,OPTIONS):
                raise RuntimeError('test')
        self.assertEqual(len(layer.self_attn.q_norm._forward_hooks),0)

if __name__=='__main__': unittest.main()
