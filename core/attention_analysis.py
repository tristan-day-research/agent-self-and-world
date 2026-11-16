"""Offline, identical-prefix attention comparisons; no model or downloads.

Usage: python -m core.attention_analysis outputs/ism/RUN
"""
import json
from pathlib import Path
from core.attention import compare_attention
from core.runs import write_json


def load_tensors(directory, record):
    import torch
    path=(Path(directory)/record['tensor_file']).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError('Attention tensor path escapes run directory')
    return torch.load(path,map_location='cpu',weights_only=True)


def analyze_run(directory):
    directory=Path(directory)
    results=json.loads((directory/'outputs.json').read_text())['results']
    baselines={r['prompt_id']:r for r in results if r['strength'] is None and not r.get('query_intervention')}
    comparisons=[]; zero_checks=0; largest=None; largest_score=-1
    for i,result in enumerate(results):
        if result['strength'] is None and not result.get('query_intervention'):
            continue
        baseline=baselines.get(result['prompt_id'])
        if baseline is None:
            raise ValueError('Missing baseline for attention comparison')
        for phase in ['prompt','fixed']:
            a=(baseline if phase=='prompt' else baseline.get('fixed_continuation',{})).get('attention')
            b=(result if phase=='prompt' else result.get('fixed_continuation',{})).get('attention')
            if a is None and b is None:
                continue
            if a is None or b is None:
                raise ValueError('Attention missing on one member of pair')
            for field in ['input_ids','query_positions','layers']:
                if a[field]!=b[field]:
                    raise ValueError(f'Cannot compare nonidentical attention {field}')
            before=load_tensors(directory,a); after=load_tensors(directory,b)
            if result['strength']==0:
                import torch
                for key in before:
                    torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
                zero_checks+=1
            rows=compare_attention(before,after)
            for row in rows:
                row['position']=a['query_positions'][row['query_index']]
            comparisons.append({'result_index':i,'prompt_id':result['prompt_id'], 'phase':phase,
                'strength':result['strength'],'features':result.get('steering_features',[]),
                'direction_kind':result.get('direction_kind','sae'),'query_intervention':result.get('query_intervention'),'rows':rows})
            best=max(rows,key=lambda row:row['attention_js_nats'])
            if result['strength']!=0 and best['attention_js_nats']>largest_score:
                largest_score=best['attention_js_nats']
                largest=(i,phase,a,b,best,before[f"L{best['layer']}.pattern"][best['head'],best['query_index']],
                         after[f"L{best['layer']}.pattern"][best['head'],best['query_index']])
    output={'note':'Observational identical-text effects, not head importance or mediation. Cached decode excluded. JS uses renormalized stored rows.',
            'exact_zero_tensor_checks':zero_checks,'comparisons':comparisons}
    write_json(directory/'attention_effects.json',output)
    if largest is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        i,phase,a,b,best,p,q=largest
        # A selected example, not an aggregate over unlike prompts/conditions.
        indexes=(q-p).abs().topk(min(16,len(p))).indices.sort().values.tolist()
        labels=[f'{j}: '+(a.get('tokens',[str(t) for t in a['input_ids']])[j]).replace('\n','↵')[:24] for j in indexes]
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        x=list(range(len(indexes)))
        ax.bar([j-.2 for j in x],p[indexes].tolist(),.4,label='Baseline')
        ax.bar([j+.2 for j in x],q[indexes].tolist(),.4,label='Intervention')
        ax.set_xticks(x,labels,rotation=65,ha='right'); ax.set_ylabel('Attention weight'); ax.legend()
        ax.set_title(f'Largest observed attention JS (exploratory selection)\nResult {i}, {phase}, L{best["layer"]} H{best["head"]}, query {best["position"]}')
        folder=directory/'figures'; folder.mkdir(exist_ok=True)
        fig.savefig(folder/'attention_example.png',dpi=140); plt.close(fig)
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=analyze_run(args.directory)
    print(f"Saved attention_effects.json: {len(result['comparisons'])} pairs; {result['exact_zero_tensor_checks']} exact zero checks")
