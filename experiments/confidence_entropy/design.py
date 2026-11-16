"""Migrated prompt protocol and shared, domain-balanced question selection."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
CONFIDENCE_OPTIONS = {'S': '<5%', 'T': '5-10%', 'U': '10-20%', 'V': '20-40%',
                      'W': '40-60%', 'X': '60-80%', 'Y': '80-90%', 'Z': '>90%'}
CONFIDENCE_MIDPOINTS = {'S': .025, 'T': .075, 'U': .15, 'V': .3,
                        'W': .5, 'X': .7, 'Y': .85, 'Z': .95}
EXPERT_DOMAINS = {'chemist': 'chemistry', 'historian': 'history', 'artist': 'art'}


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def load_design(questions_file, protocol_file, per_domain, seed, domains=None, selected_personas=None,
                persona_set='original'):
    payload, question_hash = read_json(questions_file)
    protocol, protocol_hash = read_json(protocol_file)
    questions = payload['questions']
    if not questions or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Questions need unique IDs')
    for q in questions:
        if not isinstance(q.get('choices'), dict) or len(q['choices']) < 2 or q['correct_answer'] not in q['choices']:
            raise ValueError(f"Invalid choices/answer for {q.get('id')}")
        if any(not isinstance(k,str) or len(k)!=1 or not k.isupper() for k in q['choices']):
            raise ValueError('Choice labels must be single uppercase letters')
    domain_names = sorted({q['domain'] for q in questions}) if domains is None else domains
    if len(set(domain_names)) != len(domain_names) or set(domain_names)-{q['domain'] for q in questions}:
        raise ValueError('Unknown or repeated domains')
    rng = random.Random(seed)
    sample = []
    for domain in domain_names:
        candidates = [q for q in questions if q['domain']==domain]
        if per_domain is None:
            sample.extend(candidates)
        else:
            if per_domain < 1 or per_domain > len(candidates):
                raise ValueError(f'Invalid per-domain sample count for {domain}')
            sample.extend(rng.sample(candidates, per_domain))
    original = protocol['personas']
    names = list(original) if selected_personas is None else selected_personas
    if len(set(names)) != len(names) or set(names)-set(original):
        raise ValueError('Unknown or repeated personas')
    if 'default_assistant' not in names:
        raise ValueError('Include default_assistant for the persona × steering interaction')
    personas = []
    for name in names:
        spec = {'id':name, 'base_persona':name, 'prompt_variant':'original',
                'system_prompt':original[name]['system_prompt'], 'expert_domain':EXPERT_DOMAINS.get(name)}
        if persona_set in {'original','both'} or name not in EXPERT_DOMAINS:
            personas.append(spec)
        if persona_set in {'neutral','both'} and name in EXPERT_DOMAINS:
            # Only omit the explicit confidence suggestion; keep expertise and perspective.
            prompt = spec['system_prompt'].replace(
                f" and are especially confident in {EXPERT_DOMAINS[name]}-related topics.", '.')
            if prompt == spec['system_prompt']:
                raise ValueError(f'Could not locate confidence wording in {name}')
            personas.append({**spec, 'id':name+'_neutral', 'prompt_variant':'neutral_confidence_wording',
                             'system_prompt':prompt})
    return sample, personas, protocol['prompts'], {
        'questions_sha256':question_hash, 'protocol_sha256':protocol_hash,
        'question_ids':[q['id'] for q in sample], 'sampling':'same domain-balanced sample for every persona/condition',
        'original_dataset_source_sha256':payload.get('source_sha256'),
        'original_protocol_source_sha256':protocol.get('source_sha256')}


def prompt_texts(question, templates):
    choices = question['choices']
    labels = sorted(choices)
    formatted = '\n'.join(f'  {k}: {choices[k]}' for k in labels)
    options = ', '.join(labels[:-1])+f', or {labels[-1]}' if len(labels)>2 else ' or '.join(labels)
    common = {'question_text':question['question'], 'formatted_options':formatted, 'options_str':options}
    full = question.get('original_prompt', question['question']+'\n\n'+formatted)
    confidence_labels = list(CONFIDENCE_OPTIONS)
    return {
        'answer': templates['ce_forced_choice_setup']+'\n\n'+templates['ce_forced_choice'].format(**common),
        'stated_confidence': templates['ce_stated_confidence_setup']+'\n\n'+templates['ce_stated_confidence'].format(
            **common, confidence_formatted_options='\n'.join(f'  {k}: {v}' for k,v in CONFIDENCE_OPTIONS.items()),
            confidence_options_str=', '.join(confidence_labels[:-1])+f', or {confidence_labels[-1]}'),
        'open_answer': templates['ce_open_ended'].format(question_prompt=full),
        'open_confidence': templates['ce_confidence_open_ended'].format(question_prompt=full)}
