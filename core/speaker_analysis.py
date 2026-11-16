"""Speaker feature summaries, aligned semantic attention and query-rescue evidence.

These are descriptive measurements, never automatically authored research reports.
"""
from collections import defaultdict,Counter
from pathlib import Path
import itertools,json,re,statistics as st
from core.paper_features import feature_manifest
from core.runs import write_json


def align_spans(tokenizer,formatted,spans):
    encoded=tokenizer(formatted,add_special_tokens=False,return_offsets_mapping=True)
    if 'offset_mapping' not in encoded:raise ValueError('Semantic attention spans require a fast tokenizer with offsets')
    offsets=encoded['offset_mapping']
    groups={}
    for label,text in spans.items():
        intervals=[(m.start(),m.end()) for m in re.finditer(re.escape(text),formatted)]
        if not intervals:raise ValueError(f'Attention span absent from formatted prompt: {label}')
        groups[label]=[i for i,(a,b) in enumerate(offsets) if b>a and any(a<end and b>start for start,end in intervals)]
        if not groups[label]:raise ValueError(f'Attention span has no aligned tokens: {label}')
    return groups


def assess_speakers(directory):
    from core.attention_analysis import load_tensors
    import torch
    directory=Path(directory)
    results=json.loads((directory/'outputs.json').read_text())['results']
    feature_rows=[]; span_rows=[]; raw={}; query_rescues=[]
    manifest=feature_manifest()
    for i,result in enumerate(results):
        text=result.get('text','')
        result_marker={'first_person_pronouns':len(re.findall(r'\b(?:I|me|my|mine|we|us|our|ours)\b',text,re.I)),
                       'third_person_pronouns':len(re.findall(r'\b(?:he|she|they|him|her|them|his|their|theirs)\b',text,re.I))}
        # Store diagnostics separately; never rewrite the model's outputs.json.
        for space,owner in [('free',result),('fixed',result.get('fixed_continuation',{}))]:
            rows=owner.get('token_trace',[])
            for token_source in ['prompt','generated']:
                selected=[row for row in rows if row['source']==token_source]
                for feature in manifest:
                    key=str(feature['feature']) if feature['layer']==9 else feature['key']
                    values=[row.get('features_after_steering',{}).get(key,row['features'].get(key)) for row in selected]
                    values=[v for v in values if v is not None]
                    if values:
                        feature_rows.append({'result_index':i,'space':space,'token_source':token_source,
                            **feature,'n_positions':len(values),'mean':st.mean(values),'peak':max(values),
                            'active_fraction':sum(v>0 for v in values)/len(values)})
        record=result.get('attention')
        if record and result.get('attention_span_positions'):
            tensors=load_tensors(directory,record)
            if result["strength"] is None and not result.get("query_intervention"):
                raw[i]={k:v for k,v in tensors.items() if k.endswith(".q_normalized")}
            for name,pattern in tensors.items():
                if not name.endswith('.pattern'):continue
                layer=int(name.split('.')[0][1:])
                for head in range(pattern.shape[0]):
                    for label,positions in result['attention_span_positions'].items():
                        span_rows.append({'result_index':i,'layer':layer,'head':head,'span':label,
                            'mass':float(pattern[head,0,positions].sum()),'n_span_tokens':len(positions)})
        result_marker.update(result_index=i)
        result['_speaker_markers']=result_marker
    categories=[]
    grouped=defaultdict(list)
    for row in feature_rows:grouped[(row['result_index'],row['space'],row['token_source'],row['category'])].append(row)
    for (i,space,source,category),rows in grouped.items():
        categories.append({'result_index':i,'space':space,'token_source':source,'category':category,
            'n_features_observed':len(rows),'mean_feature_active_fraction':st.mean(r['active_fraction'] for r in rows)})
    # Prompt factors match within character/person/prose; context prefixes intentionally differ.
    families=defaultdict(list)
    for i,r in enumerate(results):
        f=r.get('design_factors')
        if f and r['strength'] is None and not r.get('query_intervention') and i in raw:
            families[(f['character'],f['person'],f['prose'])].append(i)
    cross=[]
    masses={(r['result_index'],r['layer'],r['head'],r['span']):r for r in span_rows}
    for family,indexes in families.items():
        for left,right in itertools.combinations(indexes,2):
            for name,q in raw[left].items():
                if not name.endswith('.q_normalized'):continue
                layer=int(name.split('.')[0][1:]); other=raw[right][name]
                for head in range(q.shape[0]):
                    diffs={span:masses[(right,layer,head,span)]['mass']-masses[(left,layer,head,span)]['mass']
                           for span in results[left]['attention_span_positions'] if (right,layer,head,span) in masses}
                    cross.append({'family':family,'left_result_index':left,'right_result_index':right,
                        'left_setting':results[left]['prompt_condition'],'right_setting':results[right]['prompt_condition'],
                        'layer':layer,'head':head,'query_cosine_before_rope':float(torch.nn.functional.cosine_similarity(q[head,0],other[head,0],dim=0)),
                        'semantic_span_mass_differences':diffs})
    for i,r in enumerate(results):
        intervention=r.get('query_intervention')
        if not intervention:continue
        base=next((b for b in results if b['prompt_id']==r['prompt_id'] and b['strength'] is None and not b.get('query_intervention')),None)
        original=next((b for b in results if b['prompt_id']==r['prompt_id'] and b['strength']==r['strength']
            and b.get('direction_kind')==r.get('direction_kind') and b.get('steering_features')==r.get('steering_features') and not b.get('query_intervention')),None)
        if base is None or original is None:raise ValueError('Query rescue lacks original/baseline')
        l=r['fixed_continuation']['continuation_likelihood'];o=original['fixed_continuation']['continuation_likelihood'];b=base['fixed_continuation']['continuation_likelihood']
        query_rescues.append({'result_index':i,**intervention,
            'output_tokens_changed_from_steered':r['token_ids']!=original['token_ids'],
            'first_output_token_changed_from_steered':r['token_ids'][:1]!=original['token_ids'][:1],
            'fixed_mean_nll':l['mean_nll'],'fixed_mean_nll_change_from_steered':l['mean_nll']-o['mean_nll'],
            'first_token_logprob_change_from_steered':l['first_token_log_probability']-o['first_token_log_probability'],
            'baseline_first_token_logprob':b['first_token_log_probability']})
    output={'scope':'Published feature exemplars; category labels are author interpretations, not validated aggregate detectors.',
        'coverage':dict(Counter(f['category'] for f in manifest)),
        'category_note':'Category summary averages active fractions across observed exemplars, never sums incomparable magnitudes. Primary-layer values use post-addition coefficients when present.',
        'features':feature_rows,'categories':categories,'literal_person_markers':[r['_speaker_markers'] for r in results],
        'semantic_attention':span_rows,'cross_setting_comparisons':cross,
        'cross_setting_note':'Different prefixes by design; pre-RoPE query cosine and explicitly aligned semantic spans. Overlapping spans are not a probability partition. No raw-index cross-prompt JS.',
        'query_rescues':query_rescues}
    write_json(directory/'speaker_assessment.json',output)
    return output

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();result=assess_speakers(args.directory)
    print(f"Saved speaker_assessment.json; {len(result['features'])} feature summaries, {len(result['cross_setting_comparisons'])} setting contrasts, {len(result['query_rescues'])} query patches")
