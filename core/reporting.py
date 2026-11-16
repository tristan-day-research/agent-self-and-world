"""CPU-only evidence from saved traces; agent authors report.md separately: python -m core.reporting RUN_DIRECTORY."""
import argparse
import json
import os
from pathlib import Path

from core.paper_features import feature_manifest
from core.runs import write_json


def feature_key(key):
    return key if key.startswith("L") else f"L9:{key}"


def summarize(result, feature_field="features"):
    """Keep prompt and generated statistics separate; zero is an inactive latent."""
    summary = {}
    for source in ("prompt", "generated"):
        rows = [row for row in result.get("token_trace", []) if row["source"] == source]
        keys = sorted({key for row in rows for key in row.get(feature_field, {})})
        summary[source] = {}
        for key in keys:
            values = [row[feature_field][key] for row in rows if key in row.get(feature_field, {})]
            summary[source][feature_key(key)] = {
                "count": len(values), "active": sum(value > 0 for value in values),
                "fraction": sum(value > 0 for value in values) / len(values),
                "mean": sum(values) / len(values), "peak": max(values),
                "mean_when_active": (sum(values) / sum(value > 0 for value in values)
                                     if any(value > 0 for value in values) else 0.0),
            }
    return summary


def condition(result, metadata):
    if result.get("query_intervention"):
        p=result["query_intervention"]
        return f"{p['kind']} L{p['layer']}H{p['head']} (strength={result['strength']})"
    if result["strength"] is None:
        return "baseline"
    if result["strength"] == 0:
        return "zero-hook"
    features = result.get("steering_features", metadata.get("steering_features", []))
    prefix = "random matched to " if result.get("direction_kind") == "random" else ""
    if result.get("direction_kind") == "sae_control":
        prefix = f"SAE-control {result.get('control_source_feature')} matched to "
    index = result.get("control_index")
    suffix = f" #{index+1}" if index is not None and index > 0 else ""
    return prefix + "+".join(map(str, features)) + suffix + f" @ {result['strength']:+g}"


def generate_evidence(directory, plots=True):
    """Rebuild supporting artifacts. report.md is reserved for agent-authored analysis."""
    import hashlib
    from core.behavior import generation_metrics
    directory = Path(directory)
    def read(name, default):
        path = directory/name
        return json.loads(path.read_text()) if path.exists() else default
    outputs = read('outputs.json', {})
    results = outputs.get('results', [])
    metadata = read('metadata.json', {})
    status = read('status.json', {})
    stats = [summarize(r) for r in results]
    assessment = {'results': [
        {'prompt_id': r.get('prompt_id', 'synthetic'), 'condition': condition(r, metadata),
         'temperature': r.get('temperature', 0), 'seed': r.get('seed', 0),
         'direction_kind': r.get('direction_kind', 'sae'),
         'generation_metrics': generation_metrics(r.get('token_ids', []), r.get('text', '')),
         'control_index': r.get('control_index'), 'control_source_feature': r.get('control_source_feature'),
         'features': s, 'features_after_steering': summarize(r, 'features_after_steering'),
         'fixed_continuation_features': summarize(r.get('fixed_continuation', {})),
         'fixed_continuation_features_after_steering': summarize(r.get('fixed_continuation', {}), 'features_after_steering')}
        for r, s in zip(results, stats)]}
    write_json(directory/'assessment.json', assessment)
    controls = []
    baselines = [(i, r) for i, r in enumerate(results) if r['strength'] is None and not r.get('query_intervention')]
    for _, base in baselines:
        pid = base.get('prompt_id', 'synthetic')
        zero = next((r for r in results if r.get('prompt_id', 'synthetic') == pid and r['strength'] == 0), None)
        controls.append({'prompt_id': pid, 'zero_available': zero is not None,
                         'equal': None if zero is None else {key: base.get(key) == zero.get(key)
                         for key in ('token_ids', 'token_trace', 'jspace', 'fixed_continuation')}})
    brief = {'run_id': directory.name, 'state': status.get('state', 'unknown'),
             'kind': outputs.get('kind'), 'saved_completions': len(results), 'controls': controls,
             'outputs_sha256': hashlib.sha256((directory/'outputs.json').read_bytes()).hexdigest()
                               if (directory/'outputs.json').exists() else None,
             'lens_vector_files': sorted({r['jspace_vectors_file'] for r in results if r.get('jspace_vectors_file')}),
             'artifacts': ['outputs.json', 'metadata.json', 'config.json', 'assessment.json',
                           'generations.md', 'magnitudes.md', 'measurements.md'],
             'report': {'path': 'report.md', 'exists': (directory/'report.md').exists(),
                        'authoring': 'Codex/Claude reviews saved evidence and writes report.md; scripts never write it'}}
    write_json(directory/'analysis_brief.json', brief)
    geometry = None
    if brief['lens_vector_files']:
        from core.lens_analysis import vector_effects
        geometry = vector_effects(results, directory)
        write_json(directory/'jspace_effects.json', geometry)
    magnitude_lines = ['# Recorded ISM magnitudes', '',
        '| Prompt / condition | 133 mean before | after | peak before | after | 4360 mean before | after | peak before | after |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r, before in zip(results, stats):
        after = summarize(r, 'features_after_steering')
        numbers = []
        for key in ('L9:133', 'L9:4360'):
            for metric in ('mean', 'peak'):
                for summary in (before, after):
                    value = summary['generated'].get(key, {}).get(metric)
                    numbers.append('—' if value is None else f'{value:.3f}')
        magnitude_lines.append(f"| {r.get('prompt_id', 'synthetic')} / {condition(r, metadata)} | " + ' | '.join(numbers) + ' |')
    (directory/'magnitudes.md').write_text('\n'.join(magnitude_lines)+'\n')
    generations = ['# Saved replies and vocabulary readouts', '']
    for r in results:
        generations += [f"## {r.get('prompt_id', 'synthetic')} — {condition(r, metadata)}", '',
                        r.get('text', str(r['token_ids'])), '']
        for point in r.get('jspace', []):
            def tokens(name):
                return ', '.join(repr(item['token']) for item in point.get(name, [])[:3])
            description = (f"observed {tokens('top_tokens')}" if 'top_tokens' in point else
                           f"before {tokens('pre_current_steering_top_tokens')}; after {tokens('post_current_steering_top_tokens')}")
            generations.append(f"- Block {point.get('layer', 9)}, position {point['position']} ({point['source']}, "
                               f"{point.get('lens_kind', 'jacobian')}): {description}.")
        generations.append('')
    (directory/'generations.md').write_text('\n'.join(generations))
    if plots and any(r.get('token_trace') for r in results):
        _plot_evidence(directory, results, metadata, stats, baselines)
    if plots and geometry and any(item['space'] == 'fixed' for item in geometry['comparisons']):
        brief['jspace_fixed_plot_files'] = _plot_fixed_jspace(directory, results, metadata, geometry)
        write_json(directory/'analysis_brief.json', brief)
    lines = ['# Machine-generated measurements', '',
             'Supporting data only. The research report is authored separately in report.md.', '',
             f"Run: `{directory.name}` · state: `{brief['state']}` · saved completions: {len(results)}", '',
             f"Model: `{metadata.get('model', 'unknown')}` · coefficient mode: `{metadata.get('coefficient_mode', 'unknown')}`", '',
             '| Baseline / zero pair | Tokens | SAE trace | Lens readout | Fixed continuation |',
             '|---|---|---|---|---|']
    for control in controls:
        values = ['missing']*4 if control['equal'] is None else [str(v) for v in control['equal'].values()]
        lines.append(f"| {control['prompt_id']} | " + ' | '.join(values) + ' |')
    if outputs.get('kind') == 'reference_calibration':
        lines += ['', 'Reference-set peaks (not authors\' corpus peaks):', '',
                  '| Feature | Peak |', '|---|---:|']
        lines += [f'| {key} | {value:.4f} |' for key, value in outputs.get('feature_peaks', {}).items()]
    lines += ['', '[Exact settings](config.json) · [Provenance](metadata.json) · '
              '[Feature statistics](assessment.json) · [Magnitudes](magnitudes.md) · '
              '[Replies](generations.md) · [Analysis manifest](analysis_brief.json)']
    (directory/'measurements.md').write_text('\n'.join(lines)+'\n')
    return directory/'measurements.md'


def _plot_evidence(directory, results, metadata, stats, baselines):
    figures = directory/'figures'
    figures.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(figures/'.mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME', str(figures/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    keys = sorted({k for s in stats for k in s['generated']}, key=lambda k: tuple(map(int, k[1:].split(':'))))
    labels = {f['key']: f['label'] for f in feature_manifest()}
    if keys:
        matrix = np.array([[s['generated'].get(k, {}).get('fraction', np.nan) for s in stats] for k in keys])
        fig, ax = plt.subplots(figsize=(max(10,len(results)*.22), max(4.5,len(keys)*.3)))
        im = ax.imshow(matrix, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set_yticks(range(len(keys)), [f"{k} {labels.get(k,'')}" for k in keys], fontsize=8)
        ax.set_xticks(range(len(results)), [f"{r.get('prompt_id','toy')} / {condition(r,metadata)}" for r in results], rotation=90, fontsize=7)
        fig.colorbar(im, ax=ax, label='Active-token fraction')
        fig.tight_layout();fig.savefig(figures/'feature_activity.png',dpi=140);plt.close(fig)
    if not baselines:
        return
    fig, axes = plt.subplots(len(baselines),1,figsize=(10,max(3,len(baselines)*2.2)),squeeze=False)
    for ax, (_,base) in zip(axes[:,0],baselines):
        rows = base.get('token_trace',[])
        for key in ('133','4360'):
            ax.plot([r['position'] for r in rows], [r['features'].get(key,r['features'].get(f'L9:{key}',0)) for r in rows],label=key)
        boundary=next((r['position'] for r in rows if r['source']=='generated'),len(rows))
        ax.axvline(boundary-.5,color='black',linestyle='--',linewidth=.8)
        ax.set_title(base.get('prompt_id','synthetic'),fontsize=10);ax.set_ylabel('Activation');ax.legend(fontsize=8)
    axes[-1,0].set_xlabel('Token position; dashed line marks first processed reply token')
    fig.tight_layout();fig.savefig(figures/'ism_timeline.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    for row,key in enumerate(('L9:133','L9:4360')):
        for column,metric in enumerate(('mean','peak')):
            ax=axes[row,column]
            ax.bar(range(len(baselines)),[stats[i]['generated'].get(key,{}).get(metric,0) for i,_ in baselines])
            ax.set_xticks(range(len(baselines)),[r.get('prompt_id','toy') for _,r in baselines],rotation=55,ha='right',fontsize=7)
            ax.set_title(f'{key}: {metric} activation (baseline)');ax.set_ylabel('Raw feature activation')
    fig.tight_layout();fig.savefig(figures/'ism_magnitude.png',dpi=140);plt.close(fig)


def _plot_fixed_jspace(directory, results, metadata, geometry):
    """Relative change to same-layer baseline on the identical final prompt token."""
    import re
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = directory/'figures'
    figures.mkdir(exist_ok=True)
    paths = []
    prompt_ids = list(dict.fromkeys(p['prompt_id'] for p in geometry['comparisons'] if p['space']=='fixed'))
    for pid in prompt_ids:
        rows = [p for p in geometry['comparisons'] if p['space']=='fixed' and p['source']=='prompt'
                and p['prompt_id']==pid and p['phase'] in ('post','observed')]
        if not rows:
            continue
        indices = sorted({p['result_index'] for p in rows})
        layers = sorted({p['layer'] for p in rows})
        values = np.full((len(indices),len(layers)),np.nan)
        for point in rows:
            value = point['relative_delta_l2']
            if value is not None:
                values[indices.index(point['result_index']),layers.index(point['layer'])] = value
        fig,ax=plt.subplots(figsize=(max(7,len(layers)),max(4,len(indices)*.3)))
        im=ax.imshow(values,aspect='auto',cmap='viridis')
        ax.set_yticks(range(len(indices)),[condition(results[i],metadata) for i in indices],fontsize=7)
        ax.set_xticks(range(len(layers)),layers);ax.set_xlabel('Block')
        ax.set_title(f'{pid}: same-prompt J-vector change vs baseline')
        fig.colorbar(im,ax=ax,label='L2 change / baseline vector norm (within each layer)')
        fig.tight_layout()
        relative=Path('figures')/('jspace_fixed_'+re.sub(r'[^a-zA-Z0-9_-]','_',pid)+'.png')
        fig.savefig(directory/relative,dpi=140);plt.close(fig)
        paths.append(str(relative))
    return paths


def generate_report(directory):
    """Backward-compatible entry point; now produces evidence, never report.md."""
    return generate_evidence(directory)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(generate_evidence(parser.parse_args().directory))
