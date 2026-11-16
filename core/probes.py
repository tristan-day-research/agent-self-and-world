"""Validated text-only evaluation probes and content-addressed provenance."""
import hashlib
import json
from pathlib import Path


def read_probes(path):
    raw = Path(path).read_bytes()
    samples = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not samples or any(not isinstance(p.get('id'), str) or not p['id'] for p in samples):
        raise ValueError('Probes need nonempty string IDs')
    if len({p['id'] for p in samples}) != len(samples):
        raise ValueError('Probe IDs must be unique')
    for p in samples:
        if p.get('condition') not in {'assistant', 'roleplay', 'story', 'assistant_emotional'}:
            raise ValueError('Probe condition must describe a supported setting')
        messages = p.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Each probe needs messages')
        if any(not isinstance(m, dict) or m.get('role') not in {'system', 'user', 'assistant'}
               or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages):
            raise ValueError('Probe messages must have valid roles and nonempty text')
        if messages[-1]['role'] != 'user':
            raise ValueError('Evaluation probes must end in a user message')
        if not isinstance(p.get('source'), str) or not p['source']:
            raise ValueError('Each probe needs explicit source provenance')
    return tuple(samples), {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                            'count': len(samples), 'sources': sorted({p['source'] for p in samples})}


def evaluation_plan(prompts, groups, strengths, temperatures, random_count=0, sae_control_count=0):
    """Compute cost bounds without loading model weights or contacting a pod."""
    n_strengths = len(set(s for s in strengths if s != 0))
    directions = len(groups) * (1 + random_count + sae_control_count)
    counts = {setting: sum(p['condition'] == setting for p in prompts)
              for setting in sorted({p['condition'] for p in prompts})}
    return {'prompts': len(prompts), 'settings': counts, 'groups': groups,
            'temperatures': temperatures, 'directions': directions,
            'nonzero_strengths': list(dict.fromkeys(s for s in strengths if s != 0)),
            'evaluations': len(prompts) * len(temperatures) * (2 + directions*n_strengths),
            'baseline_zero_per_prompt_trial': 2,
            'scope': 'evaluated generations only; calibration and fixed-text passes are additional'}
