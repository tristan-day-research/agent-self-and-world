"""Factorial persona × SAE steering experiment on shared multiple-choice items."""
import argparse
import hashlib
from contextlib import nullcontext
from importlib.metadata import version
import json
import math
from pathlib import Path

from core.runs import run_context, write_json
from experiments.confidence_entropy.design import (
    ROOT, CONFIDENCE_OPTIONS, CONFIDENCE_MIDPOINTS, load_design, prompt_texts)


def configure_parser(parser):
    from core.attention import configure_attention
    configure_attention(parser)
    parser.add_argument('--questions-file',type=Path,default=ROOT/'questions.json')
    parser.add_argument('--protocol-file',type=Path,default=ROOT/'protocol.json')
    parser.add_argument('--personas',nargs='+')
    parser.add_argument('--persona-set',choices=['original','neutral','both'],default='both')
    parser.add_argument('--domains',nargs='+')
    parser.add_argument('--questions-per-domain',type=int,default=2)
    parser.add_argument('--all-questions',action='store_true')
    parser.add_argument('--features',type=int,nargs='+',choices=[133,4360],default=[133])
    parser.add_argument('--strengths',type=float,nargs='+',default=[-2.,2.])
    parser.add_argument('--coefficient-mode',choices=['raw','reference-alpha'],default='reference-alpha')
    parser.add_argument('--calibration-file',type=Path)
    parser.add_argument('--calibration-prompts',type=Path,
                        help='Defaults to the independent 16-prompt reference set when no calibration file is given')
    parser.add_argument('--random-control',action='store_true')
    parser.add_argument('--jlens',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--jlens-layers',type=int,nargs='+',default=[8,9,10,17,22,29,32,33])
    parser.add_argument('--jlens-revision',default='e166deb5ad540932100ef803c8c9bcc2a5453694')
    parser.add_argument('--save-jspace-vectors',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--generate-responses',action='store_true',
                        help='Also generate both original free-text prompts and the two short coded responses')
    parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--dtype',choices=['auto','float32','bfloat16','float16'],default='auto')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume-directory',type=Path,
                        help='Resume a failed/interrupted run with identical flags and source hashes')


def plan_for(args,questions,personas):
    conditions = 2 + len(set(s for s in args.strengths if s!=0))*(2 if args.random_control else 1)
    cells = len(questions)*len(personas)*conditions
    return {'questions':len(questions),'personas':len(personas),'conditions':conditions,
            'question_persona_condition_cells':cells,'scored_forward_passes':2*cells,
            'optional_generation_calls':4*cells if args.generate_responses else 0,
            'reference_generations':16 if not args.smoke and args.coefficient_mode=='reference-alpha' and not args.calibration_file else 0,
            'strengths':args.strengths,'features':args.features,'coefficient_mode':args.coefficient_mode,
            'jlens':args.jlens,'jlens_positions':'final prompt position, separate answer/confidence contexts',
            'question_domains':sorted({q['domain'] for q in questions}),
            'note':'Planning only. No model loaded, no paid launch. Custom calibration count may differ.'}


class SmokeTokenizer:
    """Only for random-model plumbing; not a real tokenizer or scientific probe."""
    chat_template='synthetic smoke template'
    def encode(self,text,**kwargs):
        if len(text)==1 and 'A'<=text<='Z': return [2+ord(text)-ord('A')]
        if len(text)==1 and '0'<=text<='9': return [2+ord(text)-ord('0')]
        positions=list(range(0,len(text),max(1,len(text)//10)))[:10]
        return [1]+[3+(ord(text[i])*31+i)%29 for i in positions]
    def decode(self,ids,**kwargs): return ' '.join(f'tok{i}' for i in ids)
    def apply_chat_template(self,messages,**kwargs):
        return '\n'.join(m['role']+': '+m['content'] for m in messages)+'\nassistant:'
    def __call__(self,text,**kwargs):
        import torch
        class Inputs(dict):
            def to(self,device): return Inputs({k:v.to(device) for k,v in self.items()})
        ids=torch.tensor([self.encode(text)])
        return Inputs(input_ids=ids,attention_mask=torch.ones_like(ids))


def main(args):
    from core.attention import attention_options, save_attention_artifact
    attention = attention_options(args)
    if attention and args.smoke:
        attention = {**attention, "layers": [0, 1]}
    if not args.strengths or any(not math.isfinite(s) for s in args.strengths):
        raise ValueError('Strengths must be finite')
    if len(set(args.features))!=len(args.features) or len(set(args.jlens_layers))!=len(args.jlens_layers):
        raise ValueError('Do not repeat features or lens layers')
    if args.questions_per_domain<1 or args.max_new_tokens<1:
        raise ValueError('Counts must be positive')
    if args.calibration_file and args.calibration_prompts:
        raise ValueError('Choose a calibration file or prompts, not both')
    if args.coefficient_mode=='raw' and (args.calibration_file or args.calibration_prompts):
        raise ValueError('Raw coefficients do not use reference calibration')
    questions,personas,templates,provenance=load_design(
        args.questions_file,args.protocol_file,None if args.all_questions else args.questions_per_domain,
        args.seed,args.domains,args.personas,args.persona_set)
    if args.smoke:
        questions=questions[:2]; personas=personas[:2]
    plan=plan_for(args,questions,personas)
    if args.dry_run:
        print(json.dumps({**plan,'provenance':provenance},indent=2)); return 0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume_directory'}
    with run_context(args.experiment,config,args.resume_directory) as run:
        import torch
        from core.choice_metrics import label_token_ids
        from core.decision import measure_decision
        from core.gemma import MODEL_ID,MODEL_REVISION,SAE_ID,SAE_REVISION,SAE_PATH,LAYER
        from core.lens_analysis import save_vector_artifact,vector_effects
        from core.confidence_analysis import summarize_confidence
        from core.steering import steer_layer
        if args.smoke:
            from experiments.ism.main import toy_setup,toy_sae
            model,layer,_inputs,_direction=toy_setup(args.seed)
            if attention:
                model.set_attn_implementation("eager")
            sae=toy_sae(args.seed); tokenizer=SmokeTokenizer(); device='cpu'; dtype='float32'
            peaks={f:1. for f in args.features}
            layer_id=0
            lenses={0:(layer,torch.eye(32)),1:(model.model.layers[1],None)} if args.jlens else None
            model_meta={'model':'synthetic random tiny Gemma; not scientific results'}
            lens_meta={'kind':'synthetic identity/direct-final'}
        else:
            from core.gemma import load_gemma,load_selected_sae
            from core.device import resolve_device
            device=resolve_device(args.device)
            dtype=args.dtype if args.dtype!='auto' else ('float32' if device=='cpu' else 'bfloat16')
            sae,sae_config=load_selected_sae([133,4360]); sae.to(device)
            model,tokenizer,layer=load_gemma(device,dtype); layer_id=LAYER
            model_meta={'model':MODEL_ID,'model_revision':MODEL_REVISION,'sae':SAE_ID,
                        'sae_revision':SAE_REVISION,'sae_path':SAE_PATH,'layer':LAYER,'dtype':dtype}
            if args.coefficient_mode=='reference-alpha':
                from core.calibration import collect_reference_peaks,read_reference_prompts,validated_peaks
                if args.resume_directory and (run.directory/'calibration.json').exists():
                    artifact=json.loads((run.directory/'calibration.json').read_text())
                elif args.calibration_file:
                    artifact=json.loads(args.calibration_file.read_text())
                else:
                    source=args.calibration_prompts or ROOT.parent/'ism/reference_prompts.jsonl'
                    samples,digest=read_reference_prompts(source)
                    artifact=collect_reference_peaks(model,tokenizer,layer,sae,samples,digest,device,
                                                     50,args.seed,model_meta,run.directory/'calibration.json')
                peaks=validated_peaks(artifact,args.features,model_meta)
                write_json(run.directory/'calibration.json',artifact)
            else:
                peaks={f:1. for f in args.features}
            lenses=None; lens_meta=None
            if args.jlens:
                from core.jlens import load_jacobians
                count=len(model.model.language_model.layers)
                if any(l<0 or l>=count for l in args.jlens_layers):
                    raise ValueError(f'Lens layers must be in 0..{count-1}')
                matrices,lens_meta=load_jacobians([l for l in args.jlens_layers if l!=count-1],sae.w_enc.shape[0],args.jlens_revision)
                lenses={l:(model.model.language_model.layers[l],None if l==count-1 else matrices[l].to(device))
                        for l in args.jlens_layers}
        direction=sum(sae.w_dec[sae.feature_ids.index(f)]*peaks[f] for f in args.features).to(device)
        directions=[('sae',direction)]
        if args.random_control:
            from core.assessment import matched_random_direction
            directions.append(('random',matched_random_direction(direction,args.seed+1000)))
        conditions=[('sae',direction,None),('sae',direction,0.)]+[
            (kind,d,s) for kind,d in directions for s in dict.fromkeys(args.strengths) if s!=0]
        metadata={**model_meta,'device':device,'dtype':dtype,'synthetic':args.smoke,
                  'question_provenance':provenance,'personas':personas,'plan':plan_for(args,questions,personas),
                  'chat_template':tokenizer.chat_template,'jlens':lens_meta,
                  'feature_peaks':{str(k):v for k,v in peaks.items()},'direction_norm':float(direction.norm()),
                  'random_control_seed':args.seed+1000 if args.random_control else None,
                  'label_policy':'bare uppercase letter; exactly one unique token each; no variant-max or truncation',
                  'sampling_temperature_for_distributions':1.,'scope':'all processed tokens',
                  'versions':{name:version(name) for name in ['torch','transformers']}}
        project=ROOT.parents[1]
        metadata['measurement_code_sha256']={str(path.relative_to(project)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),project/'core/decision.py',project/'core/choice_metrics.py',
                         project/'core/attention.py',project/'core/jlens.py',project/'core/steering.py',ROOT/'design.py']}
        if args.resume_directory and (run.directory/'metadata.json').exists():
            previous=json.loads((run.directory/'metadata.json').read_text())
            for key in ['question_provenance','measurement_code_sha256','dtype','model','feature_peaks']:
                if previous.get(key)!=metadata.get(key):
                    raise ValueError(f'Resume provenance differs at {key}')
        write_json(run.directory/'metadata.json',metadata)
        write_json(run.directory/'questions.json',questions)
        results=json.loads((run.directory/'outputs.json').read_text())['results'] if args.resume_directory and (run.directory/'outputs.json').exists() else []
        completed={(r['prompt_id'],r['direction_kind'],r['strength']) for r in results}
        def checkpoint():
            write_json(run.directory/'outputs.json',{'kind':'confidence_entropy','synthetic':args.smoke,'results':results})
        for question in questions:
            texts=prompt_texts(question,templates)
            for persona in personas:
                for kind,current,strength in conditions:
                    for decision in ['answer','stated_confidence']:
                        prompt_id=f"{question['id']}:{persona['id']}:{decision}"
                        if (prompt_id,kind,strength) in completed:
                            continue
                        messages=[{'role':'system','content':persona['system_prompt']},
                                  {'role':'user','content':texts[decision]}]
                        formatted=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                        inputs=tokenizer(formatted,add_special_tokens=False,return_tensors='pt').to(device)
                        choices=question['choices'] if decision=='answer' else CONFIDENCE_OPTIONS
                        choice_ids=label_token_ids(tokenizer,list(choices))
                        result=measure_decision(model,inputs,layer,sae,current,strength,choice_ids,tokenizer,
                            question['correct_answer'] if decision=='answer' else None,lenses,args.save_jspace_vectors,attention=attention)
                        result.update(question_id=question['id'],domain=question['domain'],persona=persona['id'],
                            base_persona=persona['base_persona'],prompt_variant=persona['prompt_variant'],
                            expert_match=(persona['expert_domain']==question['domain']) if persona['expert_domain'] else None,
                            decision=decision,prompt_id=prompt_id,
                            messages=messages,formatted_prompt=formatted,choice_token_ids=choice_ids,
                            correct_answer=question['correct_answer'] if decision=='answer' else None,
                            condition='baseline' if strength is None else 'steered',strength=strength,
                            direction_kind=kind,steering_features=args.features if strength else [])
                        if decision=='stated_confidence':
                            probs=result['distribution']['choice_probabilities']
                            label=result['distribution']['unconstrained_argmax_label']
                            result['conditional_expected_confidence_midpoint']=sum(probs[k]*v for k,v in CONFIDENCE_MIDPOINTS.items())
                            result['first_token_confidence_midpoint']=CONFIDENCE_MIDPOINTS.get(label)
                        if args.generate_responses:
                            # All four legacy contexts stay independent; no preceding answer/reasoning leaks in.
                            for name,text,limit in [('coded',formatted,8),
                                ('open',tokenizer.apply_chat_template([messages[0],{'role':'user','content':texts['open_answer' if decision=='answer' else 'open_confidence']}],
                                    tokenize=False,add_generation_prompt=True),3 if args.smoke else args.max_new_tokens)]:
                                prompt_inputs=tokenizer(text,add_special_tokens=False,return_tensors='pt').to(device)
                                from core.decision import generate_diagnostic
                                diagnostic=generate_diagnostic(model,prompt_inputs,layer,sae,current,strength,tokenizer,limit)
                                result[name+'_response']=diagnostic['text']
                                result[name+'_sae_generated']=diagnostic['sae_generated']
                                result[name+'_formatted_prompt']=text
                        save_attention_artifact(result,run.directory,len(results),tokenizer)
                        if args.save_jspace_vectors and lenses:
                            save_vector_artifact(result,run.directory,len(results))
                        if strength==0:
                            baseline=next(r for r in reversed(results) if r['prompt_id']==result['prompt_id'] and r['strength'] is None)
                            for key in ['distribution','sae_boundary_pre','sae_boundary_post','jspace']:
                                if result[key]!=baseline[key]: raise RuntimeError(f'Zero-hook mismatch in {key}')
                        results.append(result); checkpoint()
                    run.logger.info('Completed %s | %s | %s | strength=%s',question['id'],persona['id'],kind,strength)
        if attention:
            from core.attention_analysis import analyze_run
            analyze_run(run.directory)
        summarize_confidence(results,run.directory)
        if args.save_jspace_vectors and lenses:
            write_json(run.directory/'jspace_effects.json',vector_effects(results,run.directory))
            from core.confidence_analysis import persona_geometry
            write_json(run.directory/'persona_jspace.json',persona_geometry(results,run.directory))
    return 0
