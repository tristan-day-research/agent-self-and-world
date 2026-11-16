"""Paired persona × steering effects; evidence only, never an authored report."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from core.runs import write_json


def summarize_confidence(results, directory):
    rows = [r for r in results if r['decision']=='answer']
    def condition(r):
        if r['strength'] is None: return 'baseline'
        if r['strength']==0: return 'zero-hook'
        return r['direction_kind']+f":{r['strength']:+g}"
    lookup = {(r['question_id'],r['persona'],condition(r)):r for r in rows}
    metrics = ['choice_entropy_nats','choice_entropy_normalized','correct_choice_probability',
               'correct_choice_nll','multiclass_brier','choice_probability_mass']
    interactions = []
    for r in rows:
        if r['persona']=='default_assistant' or r['strength'] in (None,0): continue
        q,p,c = r['question_id'],r['persona'],condition(r)
        keys = [(q,p,'baseline'),(q,'default_assistant',c),(q,'default_assistant','baseline')]
        if any(k not in lookup for k in keys): continue
        p0,default_s,default_0 = [lookup[k] for k in keys]
        differences = {m: (r['distribution'][m]-default_s['distribution'][m])
                          -(p0['distribution'][m]-default_0['distribution'][m]) for m in metrics}
        interactions.append({'question_id':q,'persona':p,'domain':r['domain'],
                             'expert_match':r['expert_match'],'condition':c,
                             'difference_in_differences':differences})
    groups = defaultdict(list)
    for r in rows: groups[(r['persona'],r['domain'],condition(r))].append(r)
    summaries = [{'persona':p,'domain':d,'condition':c,'n_questions':len(rr),
                  **{m:statistics.mean(r['distribution'][m] for r in rr) for m in metrics},
                  'conditional_choice_accuracy':statistics.mean(r['distribution']['choice_argmax_correct'] for r in rr),
                  'first_token_format_rate':statistics.mean(r['distribution']['unconstrained_argmax_is_choice'] for r in rr)}
                 for (p,d,c),rr in groups.items()]
    out = {'scope':'descriptive paired effects; questions are shared across personas and conditions',
           'interaction_formula':'(H_persona,steer - H_default,steer) - (H_persona,baseline - H_default,baseline)',
           'summaries':summaries,'interactions':interactions}
    confidence = {(r['question_id'],r['persona'],condition(r)):r for r in results if r['decision']=='stated_confidence'}
    out['paired_stated_confidence'] = [
        {'question_id':r['question_id'],'persona':r['persona'],'condition':condition(r),
         'answer_entropy_normalized':r['distribution']['choice_entropy_normalized'],
         'conditional_answer_correct':r['distribution']['choice_argmax_correct'],
         'conditional_expected_confidence_midpoint':confidence[(r['question_id'],r['persona'],condition(r))].get('conditional_expected_confidence_midpoint'),
         'first_token_confidence_midpoint':confidence[(r['question_id'],r['persona'],condition(r))].get('first_token_confidence_midpoint'),
         'confidence_letter_mass':confidence[(r['question_id'],r['persona'],condition(r))]['distribution']['choice_probability_mass']}
        for r in rows if (r['question_id'],r['persona'],condition(r)) in confidence]
    write_json(Path(directory)/'assessment.json',out)
    Path(directory,'measurements.md').write_text(
        '# Confidence experiment evidence\n\n'
        'Answer entropy is uncertainty conditional on the listed answer-letter tokens, not a direct measure of confidence or knowledge. '
        'Inspect answer-letter mass, correctness, NLL/Brier and task compliance as well.\n\n'
        f'{len(rows)} answer distributions; {len(interactions)} paired persona × steering contrasts. '
        'See assessment.json for domain summaries and per-question interactions. '
        'No model-authored interpretation or semantic immersion rating is generated.\n')
    return out


def persona_geometry(results,directory):
    """Persona × steering vector contrast at a shared question/decision boundary.

    Persona prefixes differ intentionally. This is a controlled input contrast,
    not the identical-full-prefix comparison in core.lens_analysis.vector_effects.
    """
    import torch
    directory=Path(directory).resolve()
    vectors={}
    lookup={}
    for row in results:
        if not row.get('jspace_vectors_file'): continue
        key=(row['question_id'],row['persona'],row['decision'],row['direction_kind'],row['strength'])
        path=(directory/row['jspace_vectors_file']).resolve()
        if not path.is_relative_to(directory): raise ValueError('Vector path escapes run directory')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=1: raise ValueError('Unsupported vector artifact schema')
        vectors[key]={(p['layer'],p['phase']):p['vector'] for p in payload['spaces']['free']
                      if p['position']==len(row['input_ids'])-1}
        lookup[key]=row
    contrasts=[]
    for key,current in vectors.items():
        q,persona,decision,kind,strength=key
        if persona=='default_assistant' or strength in (None,0): continue
        other_keys=[(q,'default_assistant',decision,kind,strength),
                    (q,persona,decision,'sae',None),(q,'default_assistant',decision,'sae',None)]
        if any(k not in vectors for k in other_keys): continue
        default_s,persona_0,default_0=[vectors[k] for k in other_keys]
        for layer_phase,value in current.items():
            if any(layer_phase not in v for v in (default_s,persona_0,default_0)): continue
            shift=value-default_s[layer_phase]
            baseline_shift=persona_0[layer_phase]-default_0[layer_phase]
            baseline_distance=float(baseline_shift.norm())
            contrasts.append({'question_id':q,'persona':persona,'decision':decision,
                              'direction_kind':kind,'strength':strength,
                              'layer':layer_phase[0],'phase':layer_phase[1],
                              'baseline_persona_distance_l2':baseline_distance,
                              'steered_persona_distance_l2':float(shift.norm()),
                              'change_in_persona_distance_l2':float(shift.norm())-baseline_distance,
                              'interaction_vector_l2':float((shift-baseline_shift).norm())})
    return {'scope':'same question/decision boundary; different persona prefixes by design; within-layer raw transported vectors',
            'interaction_formula':'(v_persona,steer - v_default,steer) - (v_persona,baseline - v_default,baseline)',
            'caution':'Distance is not an immersion score; layer-9 linear additions can cancel in this contrast.',
            'contrasts':contrasts}
