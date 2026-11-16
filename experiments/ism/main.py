import argparse
from importlib.metadata import version
import math
import json
from pathlib import Path

from core.device import resolve_device
from core.gemma import MODEL_ID, MODEL_REVISION, SAE_ID, SAE_REVISION, SAE_PATH, LAYER
from core.runs import run_context, write_json
from core.sae import SelectedJumpReLU, flatten_record, record_sae_features
from core.assessment import generate_comparison, token_trace
from experiments.ism.prompts import prompt_specs


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def feature_peak(value):
    try:
        feature, peak = value.split("=", 1)
        feature, peak = int(feature), float(peak)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("use FEATURE=POSITIVE_PEAK") from error
    if feature not in {133, 4360} or not math.isfinite(peak) or peak <= 0:
        raise argparse.ArgumentTypeError("feature must be 133/4360 and peak must be positive")
    return feature, peak


def configure_parser(parser):
    parser.add_argument("--preset", choices=["attention-table6","query-rescue-table6","speaker-styles","speaker-styles-ism","speaker-person-swap"], help="Prepared Table 6 attention protocol; expanded by run.py")
    from core.attention import configure_attention
    configure_attention(parser)
    parser.add_argument("--query-person-swap", action="store_true", help="Swap a boundary Q from the matched opposite-person prompt, plus control/self patches")
    parser.add_argument("--query-rescue", action="store_true", help="Add candidate/control/self Q patches to target steering conditions")
    parser.add_argument("--query-layer", type=int, default=11)
    parser.add_argument("--query-head", type=int, default=7)
    parser.add_argument("--query-control-head", type=int, default=0)
    parser.add_argument("--feature", type=int, choices=(133, 4360), nargs="+",
                        help="Override default separate 133, 4360, and joint comparisons")
    parser.add_argument("--prompt-set", choices=("table6", "table7", "paper"), default="paper")
    parser.add_argument("--prompt-file", type=Path,
                        help="Validated JSONL probes with IDs, settings, messages and source provenance")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print evaluation counts and probe provenance; no model, download or run directory")
    parser.add_argument("--audit-sae", action="store_true",
                        help="Check selected encoder/decoder against full-width official formula on first prompt")
    parser.add_argument("--monitor", choices=("ism", "paper"), default="paper",
                        help="paper records 23 explicitly numbered features at layers 9/17/22")
    parser.add_argument("--random-control", action="store_true",
                        help="Also test one seeded norm-matched random direction per steering group")
    parser.add_argument("--random-control-count", type=positive_int, default=1,
                        help="Independent isotropic directions per group when --random-control is enabled")
    parser.add_argument("--sae-control-count", type=int, default=0,
                        help="Random SAE decoder directions per group, norm-matched and recorded")
    parser.add_argument("--prompt-id", action="append", help="Select published prompt IDs from --prompt-set")
    parser.add_argument("--record-feature", type=int, choices=(133, 4360), nargs="+",
                        default=[133, 4360], help="SAE features saved at every traced token")
    parser.add_argument("--strengths", type=finite_float, nargs="+", default=[-50.0, 0.0, 50.0],
                        help="Raw coefficients, or alpha in paper-alpha mode")
    parser.add_argument("--coefficient-mode", choices=("raw", "paper-alpha", "reference-alpha"), default="raw")
    parser.add_argument("--feature-peak", action="append", type=feature_peak, default=[],
                        metavar="FEATURE=PEAK", help="Corpus peak m_f required for paper-alpha mode")
    parser.add_argument("--calibration-prompts", type=Path,
                        help="Collect unsteered reference peaks from a JSONL messages file")
    parser.add_argument("--calibration-file", type=Path, help="Reuse a completed reference calibration JSON")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--calibration-max-new-tokens", type=positive_int, default=50)
    parser.add_argument("--temperatures", type=finite_float, nargs="+", default=[0.0],
                        help="One trial per entry; seed=--seed+trial index, e.g. 0 1 1")
    parser.add_argument("--teacher-force-baseline", action="store_true",
                        help="Also measure every intervention on the same baseline continuation")
    parser.add_argument("--prompt", action="append", help="Custom user prompt; default uses Tables 6 and 7")
    parser.add_argument("--interactive", action="store_true", help="Keep weights loaded for new prompts")
    parser.add_argument("--jlens", action="store_true",
                        help="Apply multi-layer Gemma J-Lens (~433 MB download, reused cache)")
    parser.add_argument("--jlens-layers", type=int, nargs="+", default=[8, 9, 10, 17, 22, 29, 32, 33],
                        help="Zero-based blocks; 33 is the direct final-layer vocabulary readout")
    parser.add_argument("--jlens-stride", type=positive_int, default=16,
                        help="Sample every Nth generated position plus boundaries; 1 reads every position")
    parser.add_argument("--jlens-revision", default="e166deb5ad540932100ef803c8c9bcc2a5453694")
    parser.add_argument("--jlens-top-k", type=positive_int, default=10)
    parser.add_argument("--save-jspace-vectors", action="store_true",
                        help="Save sampled transported vectors for same-prefix geometric comparisons")
    parser.add_argument("--max-new-tokens", type=positive_int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scope", choices=("all", "last"), default="all")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")


def toy_setup(seed):
    """A tiny random Gemma text model and synthetic direction; no downloads."""
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(seed)
    config = Gemma3TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=16,
                             max_position_embeddings=64, sliding_window=32,
                             layer_types=["sliding_attention", "full_attention"],
                             pad_token_id=0, bos_token_id=1, eos_token_id=None)
    model = Gemma3ForCausalLM(config).eval()
    direction = torch.randn(32)
    direction /= direction.norm()
    inputs = {"input_ids": torch.tensor([[1, 4, 7]]),
              "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    return model, model.model.layers[0], inputs, direction


class ToyTokenizer:
    def decode(self, ids, **_kwargs):
        return " ".join(f"tok{token}" for token in ids)


def toy_sae(seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return SelectedJumpReLU((133, 4360), torch.randn(32, 2, generator=generator),
                            torch.zeros(2), torch.zeros(2),
                            torch.randn(2, 32, generator=generator))


def main(args):
    from core.attention import attention_options, save_attention_artifact
    attention = attention_options(args)
    if attention and args.smoke:
        attention = {**attention, "layers": [0, 1]}
    if (args.query_rescue or args.query_person_swap) and not (args.attention and args.teacher_force_baseline):
        raise ValueError('Query rescue requires --attention and --teacher-force-baseline')
    if (args.query_rescue or args.query_person_swap) and args.smoke:
        args.query_layer,args.query_head,args.query_control_head=1,1,0
    if (args.query_rescue or args.query_person_swap) and (args.query_head==args.query_control_head or min(args.query_layer,args.query_head,args.query_control_head)<0):
        raise ValueError('Query rescue needs nonnegative indices and distinct candidate/control heads')
    groups = [args.feature] if args.feature else [[133], [4360], [133, 4360]]
    steering_features = list(dict.fromkeys(f for group in groups for f in group))
    if (args.feature and len(set(args.feature)) != len(args.feature)) or len(set(args.record_feature)) != len(args.record_feature):
        raise ValueError("Do not repeat features; use strength to change their contribution")
    if args.smoke and args.interactive:
        raise ValueError("Interactive mode is for real model runs; omit --smoke")
    if any(t < 0 for t in args.temperatures):
        raise ValueError("Temperatures must be nonnegative")
    if args.sae_control_count < 0:
        raise ValueError("SAE control count must be nonnegative")
    if args.save_jspace_vectors and not (args.jlens or args.smoke):
        raise ValueError("Saving J-Space vectors requires --jlens")
    if args.prompt_file and (args.prompt or args.interactive):
        raise ValueError("Choose a probe file or custom/interactive prompts")
    if args.prompt_id and (args.prompt or args.interactive):
        raise ValueError("--prompt-id selects published prompts; omit custom/interactive prompts")
    evaluation_prompts = prompt_specs(args.prompt, args.prompt_set)
    prompt_provenance = {"kind": "inline" if args.prompt else "published_examples",
                         "prompt_set": None if args.prompt else args.prompt_set}
    if args.prompt_file:
        from core.probes import read_probes
        evaluation_prompts, prompt_provenance = read_probes(args.prompt_file)
    if args.prompt_id:
        missing = set(args.prompt_id) - {p["id"] for p in evaluation_prompts}
        if missing:
            raise ValueError(f"Unknown prompt IDs in selected set: {sorted(missing)}")
        evaluation_prompts = tuple(p for p in evaluation_prompts if p["id"] in args.prompt_id)
    person_donors = {}
    if args.query_person_swap:
        if args.query_rescue:raise ValueError('Run person swaps and ISM rescue separately for interpretable conditions')
        for probe in evaluation_prompts:
            f=probe.get('factors',{})
            matches=[other for other in evaluation_prompts if other.get('factors',{}).get('person')!=f.get('person')
                and all(other.get('factors',{}).get(k)==f.get(k) for k in ['character','setting','prose'])]
            if f.get('person') not in {'first','third'} or len(matches)!=1:
                raise ValueError('Person swap needs complete matched first/third probe pairs')
            person_donors[probe['id']]=matches[0]
    if args.teacher_force_baseline and args.scope != "all":
        raise ValueError("Teacher-forced comparison requires --scope all for comparable positions")
    if args.calibration_prompts and args.calibration_file:
        raise ValueError("Choose a calibration prompts file or an existing calibration artifact")
    if not args.smoke:
        if args.calibration_only and not (args.calibration_prompts or args.calibration_file):
            raise ValueError("--calibration-only requires --calibration-prompts or --calibration-file")
        if args.coefficient_mode == "reference-alpha" and not (args.calibration_prompts or args.calibration_file):
            raise ValueError("reference-alpha requires a documented reference calibration")
        if args.coefficient_mode != "reference-alpha" and not args.calibration_only and (args.calibration_prompts or args.calibration_file):
            raise ValueError("Use reference-alpha with reference calibration; it is not paper-corpus calibration")
        if args.coefficient_mode == "paper-alpha" and set(steering_features) - dict(args.feature_peak).keys():
            raise ValueError("paper-alpha requires an explicit --feature-peak for every steered feature")
    if args.dry_run:
        from core.probes import evaluation_plan
        plan = evaluation_plan(evaluation_prompts, groups, args.strengths, args.temperatures,
                               args.random_control_count if args.random_control else 0,
                               args.sae_control_count)
        if args.query_rescue:
            extra = len(evaluation_prompts)*len(args.temperatures)*len(groups)*len(set(s for s in args.strengths if s!=0))*3
            plan['unpatched_evaluations'] = plan['evaluations']
            plan['query_patch_evaluations'] = extra
            plan['evaluations'] += extra
        if args.query_person_swap:
            extra = len(evaluation_prompts)*len(args.temperatures)*3
            plan['query_person_swap_evaluations'] = extra
            plan['donor_forward_passes'] = len(evaluation_prompts)*len(args.temperatures)
            plan['evaluations'] += extra
        plan.update(prompt_provenance=prompt_provenance,
                    max_new_tokens=args.max_new_tokens,
                    max_evaluated_new_tokens=plan["evaluations"]*args.max_new_tokens,
                    fixed_continuation_passes=plan["evaluations"] if args.teacher_force_baseline else 0,
                    calibration_prompts=str(args.calibration_prompts) if args.calibration_prompts else None,
                    coefficient_mode=args.coefficient_mode, jlens=args.jlens,
                    note="Planning only; not proof of checkpoint access, numerical parity or successful inference")
        print(json.dumps(plan, indent=2))
        return 0
    config_snapshot = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with run_context(args.experiment, config_snapshot) as run:
        results = []
        if args.smoke:
            run.logger.info("Synthetic CPU smoke: tiny random Gemma/SAE/J-Lens; no downloads")
            model, layer, inputs, direction = toy_setup(args.seed)
            if attention:
                model.set_attn_implementation("eager")
            sae = toy_sae(args.seed)
            import torch
            toy_tokenizer = ToyTokenizer()
            toy_jacobian = torch.eye(32)
            toy_lenses = {0: (layer, toy_jacobian), 1: (model.model.layers[1], None)}
            results = list(generate_comparison(
                model, layer, inputs, direction, [0, -1, 1], args.scope, 3, args.seed,
                sae=sae, tokenizer=toy_tokenizer, jacobian=toy_jacobian,
                text_model=model, jlens_top_k=3, steering_layer_index=0,
                lens_layers=toy_lenses, jlens_stride=1, save_vectors=args.save_jspace_vectors, attention=attention))
            if args.query_rescue or args.query_person_swap:
                from core.query_interventions import query_sources,capture_boundary_query
                from core.assessment import assess_fixed_continuation
                unpatched=list(results);reference=unpatched[0]['token_ids']
                common=dict(sae=sae,tokenizer=toy_tokenizer,jacobian=toy_jacobian,text_model=model,
                    jlens_top_k=3,steering_layer_index=0,lens_layers=toy_lenses,jlens_stride=1,
                    save_vectors=args.save_jspace_vectors,attention=attention)
                def smoke_fixed(item,patch=None):
                    return assess_fixed_continuation(model,layer,inputs,reference,direction,item['strength'],
                        args.scope,sae,toy_tokenizer,lens_layers=toy_lenses,jacobian=toy_jacobian,
                        steering_layer_index=0,jlens_top_k=3,jlens_stride=1,
                        save_vectors=args.save_jspace_vectors,attention=attention,query_patch=patch)
                for item in unpatched:item['fixed_continuation']=smoke_fixed(item)
                donors=query_sources(unpatched[0],1)
                for item in unpatched:
                    if args.query_rescue and item['strength'] not in (None,0):
                        current=query_sources(item,1)
                        variants=[('restore',donors,1),('control_restore',donors,0),('self',current,1)]
                    elif args.query_person_swap and item['strength'] is None:
                        other={key:value.clone() for key,value in inputs.items()};other['input_ids'][0,-1]=4
                        opposite=capture_boundary_query(model,other,1)
                        source={'prompt':{'source':opposite,'positions':[2]},'fixed':{'source':opposite,'positions':[2]}}
                        variants=[('person_swap',source,1),('person_control_swap',source,0),('person_self',donors,1)]
                    else:continue
                    for label,source,head in variants:
                        patch={phase:{'layer':1,'heads':[head],**{k:source[phase][k] for k in ['source','positions']}} for phase in ['prompt','fixed']}
                        patched=next(generate_comparison(model,layer,inputs,direction,
                            [] if item['strength'] is None else [item['strength']],args.scope,3,args.seed,
                            include_controls=item['strength'] is None,query_patch=patch['prompt'],**common))
                        patched['fixed_continuation']=smoke_fixed(patched,patch['fixed'])
                        patched['query_intervention']={'kind':label,'layer':1,'head':head,'scope':'synthetic smoke only'}
                        if label in {'self','person_self'} and (patched['token_ids']!=item['token_ids'] or patched['fixed_continuation']['continuation_likelihood']!=item['fixed_continuation']['continuation_likelihood']):
                            raise RuntimeError('Synthetic self-query patch changed outputs')
                        results.append(patched)
            for index, result in enumerate(results):
                result.update(prompt_id="synthetic", input_ids=inputs["input_ids"][0].tolist(),
                              text=toy_tokenizer.decode(result["token_ids"]))
                if args.teacher_force_baseline and "fixed_continuation" not in result:
                    from core.assessment import assess_fixed_continuation
                    result["fixed_continuation"] = assess_fixed_continuation(
                        model, layer, inputs, results[0]["token_ids"], direction, result["strength"],
                        args.scope, sae, toy_tokenizer, lens_layers=toy_lenses,
                        jacobian=toy_jacobian, steering_layer_index=0, jlens_top_k=3,
                        jlens_stride=1, save_vectors=args.save_jspace_vectors, attention=attention)
                save_attention_artifact(result, run.directory, index, toy_tokenizer)
                if args.save_jspace_vectors:
                    from core.lens_analysis import save_vector_artifact
                    save_vector_artifact(result, run.directory, index)
            write_json(run.directory / "outputs.json", {"kind": "synthetic_smoke", "results": results})
            write_json(run.directory / "metadata.json", {
                "model": "random tiny Gemma3ForCausalLM", "layer": 0, "device": "cpu",
                "dtype": "float32", "features": "synthetic SAE", "strengths": [-1, 0, 1],
                "jlens": "synthetic identity matrix", "max_new_tokens": 3,
                "input_ids": inputs["input_ids"].tolist(),
                "versions": {name: version(name) for name in ("torch", "transformers")},
            })
            if attention:
                from core.attention_analysis import analyze_run
                analyze_run(run.directory)
            from core.speaker_analysis import assess_speakers
            assess_speakers(run.directory)
            run.logger.info("Synthetic generation, activation capture, steering, and J readout passed")
            return 0

        from core.gemma import load_gemma, load_selected_sae

        device = resolve_device(args.device)
        dtype = args.dtype if args.dtype != "auto" else ("float32" if device == "cpu" else "bfloat16")
        run.logger.info("Loading %s on %s (%s); first use downloads weights", MODEL_ID, device, dtype)
        from core.paper_features import PAPER_FEATURES, feature_manifest
        from core.conditions import control_feature_ids, direction_plan, direction_manifest
        excluded = {feature for features in PAPER_FEATURES.values() for feature in features}
        sae_controls = control_feature_ids(args.sae_control_count, 65536, excluded, args.seed + 2000)
        selected_features = list(dict.fromkeys([*args.record_feature, *steering_features,
                                 *(PAPER_FEATURES[9] if args.monitor == "paper" else []), *sae_controls]))
        sae, sae_config = load_selected_sae(selected_features)
        model, tokenizer, layer = load_gemma(device, dtype)
        sae.to(device)
        calibration_provenance = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                                  "sae": SAE_ID, "sae_revision": SAE_REVISION,
                                  "sae_path": SAE_PATH, "layer": LAYER,
                                  "dtype": dtype}
        calibration = None
        if args.calibration_prompts or args.calibration_file:
            from core.calibration import collect_reference_peaks, read_reference_prompts, validated_peaks
            if args.calibration_prompts:
                samples, digest = read_reference_prompts(args.calibration_prompts)
                evaluation_messages = [list(p["messages"]) for p in evaluation_prompts]
                if any(s["messages"] in evaluation_messages for s in samples):
                    raise ValueError("Calibration/evaluation prompts overlap; use an independent reference set")
                run.logger.info("Collecting reference peaks on %d independent authored prompts", len(samples))
                calibration = collect_reference_peaks(
                    model, tokenizer, layer, sae, samples, digest, device,
                    args.calibration_max_new_tokens, args.seed, calibration_provenance,
                    run.directory / "calibration.json")
            else:
                calibration = json.loads(args.calibration_file.read_text())
                write_json(run.directory / "calibration.json", calibration)
            reference_peaks = validated_peaks(calibration, steering_features, calibration_provenance)
            if args.calibration_only:
                write_json(run.directory / "metadata.json", calibration_provenance)
                write_json(run.directory / "outputs.json", {"kind": "reference_calibration", "results": [],
                                                           "feature_peaks": reference_peaks})
                run.logger.info("Reference calibration saved; this is not the authors' corpus peak")
                return 0
        monitors, extra_configs = {}, {}
        if args.monitor == "paper":
            for layer_id in (17, 22):
                encoder, config = load_selected_sae(list(PAPER_FEATURES[layer_id]), layer=layer_id)
                module = model.get_submodule(f"model.language_model.layers.{layer_id}")
                monitors[layer_id] = (module, encoder.to(device))
                extra_configs[str(layer_id)] = config
        peak_map = dict(args.feature_peak)
        if args.coefficient_mode == "reference-alpha":
            if args.feature_peak:
                raise ValueError("Reference peaks come only from the calibration artifact")
            peak_map = reference_peaks
        if args.coefficient_mode == "paper-alpha":
            missing = set(steering_features) - set(peak_map)
            if missing:
                raise ValueError(f"paper-alpha requires --feature-peak for {sorted(missing)}")
        directions = [sum(sae.w_dec[selected_features.index(feature)] *
                          (peak_map[feature] if args.coefficient_mode != "raw" else 1.0)
                          for feature in group).to(device) for group in groups]
        comparisons_plan = direction_plan(
            groups, directions, sae, args.seed,
            args.random_control_count if args.random_control else 0, sae_controls)
        jacobian = jlens_metadata = None
        if args.jlens:
            from core.jlens import load_jacobians
            run.logger.info("Loading public Gemma J-Lens; first use downloads about 433 MB")
            block_count = model.config.text_config.num_hidden_layers
            if len(set(args.jlens_layers)) != len(args.jlens_layers) or any(l < 0 or l >= block_count for l in args.jlens_layers):
                raise ValueError(f"Lens layers must be unique integers in 0..{block_count-1}")
            fitted = [l for l in args.jlens_layers if l != block_count-1]
            matrices, jlens_metadata = load_jacobians(fitted, sae.w_enc.shape[0], args.jlens_revision)
            lens_layers = {l: (model.get_submodule(f"model.language_model.layers.{l}"),
                                None if l == block_count-1 else matrices[l].to(device))
                           for l in args.jlens_layers}
            jacobian = lens_layers[LAYER][1] if LAYER in lens_layers else None
            jlens_metadata.update(readout_layers=args.jlens_layers, direct_final_layer=block_count-1,
                                  generated_position_stride=args.jlens_stride)
        else:
            lens_layers = None
        write_json(run.directory / "metadata.json", {
            "model": MODEL_ID, "model_revision": MODEL_REVISION, "sae": SAE_ID,
            "sae_revision": SAE_REVISION, "sae_path": SAE_PATH, "sae_config": sae_config,
            "hook": f"model.language_model.layers.{LAYER}.output", "device": device, "dtype": dtype,
            "num_hidden_layers": getattr(getattr(model.config, "text_config", model.config), "num_hidden_layers"),
            "steering_features": steering_features, "steering_groups": groups,
            "recorded_features": selected_features, "extra_sae_configs": extra_configs,
            "feature_manifest": feature_manifest() if args.monitor == "paper" else [],
            "decoder_row_norms": {str(feature): float(sae.w_dec[selected_features.index(feature)].norm())
                                  for feature in selected_features},
            "formula": "h += strength * sum(scale_f * original_decoder_row_f)",
            "coefficient_mode": args.coefficient_mode, "feature_peaks": peak_map,
            "calibration": ("independent reference-set peaks; not exact paper corpus"
                            if args.coefficient_mode == "reference-alpha" else
                            "provided corpus peaks; strength is alpha"
                            if args.coefficient_mode == "paper-alpha"
                            else "raw coefficients; paper corpus peak not reproduced"),
            "jlens": jlens_metadata, "chat_template": tokenizer.chat_template,
            "jlens_post_capture": "actual post-hook model output",
            "random_control": args.random_control,
            "random_control_seeds": [spec["seed"] for spec in comparisons_plan if spec["kind"] == "random"],
            "sae_control_features": sae_controls, "directions": direction_manifest(comparisons_plan),
            "saved_jspace_vectors": args.save_jspace_vectors,
            "temperatures": args.temperatures, "trial_seeds": [args.seed+i for i in range(len(args.temperatures))],
            "prompt_provenance": prompt_provenance,
            "do_sample": any(t > 0 for t in args.temperatures), "use_cache": True,
            "versions": {name: version(name) for name in
                         ("torch", "transformers", "huggingface-hub", "safetensors")},
        })
        prompts = evaluation_prompts
        prompt_index = 0
        audited_sae = False
        while True:
            for prompt in prompts:
                formatted = tokenizer.apply_chat_template(
                    list(prompt["messages"]), tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to(device)
                from core.speaker_analysis import align_spans
                span_positions = align_spans(tokenizer,formatted,prompt['attention_spans']) if prompt.get('attention_spans') else {}
                if args.audit_sae and not audited_sae:
                    import torch
                    from huggingface_hub import hf_hub_download
                    from core.sae_audit import audit_selected_sae
                    params_path = hf_hub_download(SAE_ID, f"{SAE_PATH}/params.safetensors",
                                                 revision=SAE_REVISION, local_files_only=True)
                    with record_sae_features(layer, sae, capture_residuals=True) as audit_record, torch.inference_mode():
                        model(**inputs, use_cache=False)
                    _, residuals = flatten_record(audit_record)
                    audit = audit_selected_sae(sae, residuals, params_path)
                    write_json(run.directory / "sae_audit.json", audit)
                    run.logger.info("Full-width SAE formula/loading audit passed: max error %s",
                                    audit["max_absolute_error"])
                    audited_sae = True
                def comparisons():
                    from core.query_interventions import query_sources,patch_specification
                    from core.attention import decoder_layers
                    for trial_index, temperature in enumerate(args.temperatures):
                        reference_tokens = None
                        donors = None
                        if args.query_rescue or args.query_person_swap:
                            blocks=decoder_layers(model)
                            if args.query_layer>=len(blocks) or args.query_layer<=LAYER:
                                raise ValueError('Query rescue must target a downstream block')
                            heads=blocks[args.query_layer].self_attn.config.num_attention_heads
                            if max(args.query_head,args.query_control_head)>=heads:
                                raise ValueError('Query head out of range')
                            if attention['layers']!=[-1] and args.query_layer not in attention['layers']:
                                raise ValueError('Record the query-rescue block in --attention-layers')
                        common=dict(sae=sae,tokenizer=tokenizer,jacobian=jacobian,text_model=model,
                            jlens_top_k=args.jlens_top_k,monitors=monitors,lens_layers=lens_layers,
                            steering_layer_index=LAYER,jlens_stride=args.jlens_stride,
                            temperature=temperature,save_vectors=args.save_jspace_vectors,attention=attention)
                        def fixed(item,direction,patch=None):
                            from core.assessment import assess_fixed_continuation
                            return assess_fixed_continuation(model,layer,inputs,reference_tokens,direction,item['strength'],
                                args.scope,sae,tokenizer,monitors,lens_layers=lens_layers,jacobian=jacobian,
                                steering_layer_index=LAYER,jlens_top_k=args.jlens_top_k,jlens_stride=args.jlens_stride,
                                save_vectors=args.save_jspace_vectors,attention=attention,query_patch=patch)
                        def annotate(item,spec):
                            item.update(steering_features=spec['features'] if item['strength'] else [],
                                direction_kind=spec['kind'],control_index=spec['control_index'],
                                control_source_feature=spec.get('source_feature'),control_seed=spec.get('seed'),
                                trial_index=trial_index,temperature=temperature,seed=args.seed+trial_index)
                        for group_index,spec in enumerate(comparisons_plan):
                            direction=spec['direction']
                            for item in generate_comparison(model,layer,inputs,direction,args.strengths,args.scope,
                                    args.max_new_tokens,args.seed+trial_index,include_controls=group_index==0,**common):
                                if item['strength'] is None:reference_tokens=item['token_ids']
                                if args.teacher_force_baseline:item['fixed_continuation']=fixed(item,direction)
                                if (args.query_rescue or args.query_person_swap) and item['strength'] is None:
                                    donors=query_sources(item,args.query_layer)
                                current = query_sources(item,args.query_layer) if args.query_rescue and item['strength'] not in (None,0) and spec['kind']=='sae' else None
                                annotate(item,spec)
                                yield item
                                if args.query_person_swap and item['strength'] is None:
                                    import torch
                                    from core.query_interventions import capture_boundary_query
                                    donor=person_donors[prompt['id']]
                                    donor_formatted=tokenizer.apply_chat_template(list(donor['messages']),tokenize=False,add_generation_prompt=True)
                                    donor_inputs=tokenizer(donor_formatted,add_special_tokens=False,return_tensors='pt').to(device)
                                    opposite=capture_boundary_query(model,donor_inputs,args.query_layer)
                                    position=inputs['input_ids'].shape[1]-1
                                    for label,source,head in [('person_swap',opposite,args.query_head),
                                            ('person_control_swap',opposite,args.query_control_head),
                                            ('person_self',donors['prompt']['source'],args.query_head)]:
                                        patch={'layer':args.query_layer,'heads':[head],'positions':[position],'source':source}
                                        patched=next(generate_comparison(model,layer,inputs,direction,[],args.scope,
                                            args.max_new_tokens,args.seed+trial_index,include_controls=True,query_patch=patch,**common))
                                        patched['fixed_continuation']=fixed(patched,direction,patch)
                                        patched['condition']='query_patched'
                                        patched['query_intervention']={'kind':label,'layer':args.query_layer,'head':head,
                                            'donor_prompt_id':donor['id'] if label!='person_self' else prompt['id'],
                                            'donor_person':donor['factors']['person'] if label!='person_self' else prompt['factors']['person'],
                                            'donor_input_ids':donor_inputs['input_ids'][0].tolist() if label!='person_self' else inputs['input_ids'][0].tolist(),
                                            'alignment':'final prompt query after q_norm, before recipient RoPE; other factors matched',
                                            'generation_scope':'final prompt position only; cached decode unchanged',
                                            'fixed_scope':'final prompt position only'}
                                        annotate(patched,spec)
                                        if label=='person_self' and (patched['token_ids']!=item['token_ids'] or patched['fixed_continuation']['continuation_likelihood']!=item['fixed_continuation']['continuation_likelihood']):
                                            raise RuntimeError('Person self-patch changed outputs/likelihood')
                                        yield patched
                                if current is None:continue
                                for label,source,head in [('restore',donors,args.query_head),
                                        ('control_restore',donors,args.query_control_head),('self',current,args.query_head)]:
                                    prompt_patch=patch_specification(source['prompt'],args.query_layer,head,inputs['input_ids'][0].tolist())
                                    patched=next(generate_comparison(model,layer,inputs,direction,[item['strength']],args.scope,
                                        args.max_new_tokens,args.seed+trial_index,include_controls=False,query_patch=prompt_patch,**common))
                                    fixed_ids=inputs['input_ids'][0].tolist()+reference_tokens[:-1]
                                    fixed_patch=patch_specification(source['fixed'],args.query_layer,head,fixed_ids)
                                    patched['fixed_continuation']=fixed(patched,direction,fixed_patch)
                                    patched['query_intervention']={'kind':label,'layer':args.query_layer,'head':head,
                                        'donor':'unsteered_same_text' if label!='self' else 'same_steered_text',
                                        'generation_scope':'final prompt position only; cached decode unchanged',
                                        'fixed_scope':'all sampled attention query positions'}
                                    annotate(patched,spec)
                                    if label=='self':
                                        if patched['token_ids']!=item['token_ids']:raise RuntimeError('Self-query patch changed output tokens')
                                        if patched['fixed_continuation']['continuation_likelihood']!=item['fixed_continuation']['continuation_likelihood']:
                                            raise RuntimeError('Self-query patch changed continuation likelihood')
                                    yield patched
                for result in comparisons():
                    result.update(attention_span_positions=span_positions, design_factors=prompt.get("factors"), attention_span_texts=prompt.get("attention_spans"), prompt_index=prompt_index,
                                  prompt_id=(f"{prompt['id']}_{prompt_index}" if args.interactive else prompt["id"]),
                                  prompt_condition=prompt["condition"], messages=prompt["messages"],
                                  prompt_source=prompt.get("source", "custom" if args.prompt else "Paper Tables 6/7"),
                                  formatted_prompt=formatted,
                                  input_ids=inputs["input_ids"][0].tolist(),
                                  text=tokenizer.decode(result["token_ids"], skip_special_tokens=True))
                    save_attention_artifact(result, run.directory, len(results), tokenizer)
                    result["generation_limit"] = {
                        "max_new_tokens": args.max_new_tokens,
                        "reached_token_cap": len(result["token_ids"]) >= args.max_new_tokens,
                        "last_token_is_eos": (result["token_ids"][-1] in
                            ([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int)
                             else (model.generation_config.eos_token_id or []))) if result["token_ids"] else False,
                    }
                    if len(args.temperatures) > 1:
                        result["prompt_id"] += f"_trial{result['trial_index']}"
                    if args.save_jspace_vectors:
                        from core.lens_analysis import save_vector_artifact
                        save_vector_artifact(result, run.directory, len(results))
                    results.append(result)
                    write_json(run.directory / "outputs.json",
                               {"kind": "exploratory_ism", "results": results})
                    run.logger.info("Prompt %d | %s | features=%s | direction=%s | strength=%s\n%s", prompt_index,
                                    result["condition"], result["steering_features"], result["direction_kind"],
                                    result["strength"], result["text"])
                prompt_index += 1
            if not args.interactive:
                break
            try:
                prompt_text = input("\nNext prompt (blank to finish): ").strip()
            except EOFError:
                break
            if not prompt_text:
                break
            prompts = prompt_specs([prompt_text])
        if attention:
            from core.attention_analysis import analyze_run
            analyze_run(run.directory)
        from core.speaker_analysis import assess_speakers
        assess_speakers(run.directory)
    return 0
