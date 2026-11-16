# Minimum ISM replication before a new research pilot

Keep Gemma-3-4B-IT, GemmaScope 2 65k/medium, layer-9 features 133/4360, and the
Table 6/7 prompts. Target a focused functional replication of ISM before pursuing
AAC, perspective or agenthood hypotheses. Rediscovering all 529 latents is not a
prerequisite.

## Targets and review criteria

1. **Natural activation signatures.** Inspect token traces, firing fraction,
   mean, mean when active, and peak. Check sparse onset-associated 4360 versus
   sustained 133. Keep sample-level activation density separate from token-level
   firing frequency; nine prompts cannot reproduce population density estimates.
2. **Table 6 causal effects.** Compare baseline/zero, separate 133/4360 and joint
   steering at both polarities. Review Assistant preambles, stage directions,
   narrative elaboration, character viewpoint and task/identity preservation.
   Changed words alone do not replicate immersed-versus-detached behavior.
3. **Table 7 emotional Assistant behavior.** Review all six prompts, including
   null/opposite effects. Does negative steering reduce theatrical presentation
   while retaining helpful responses? Does positive steering increase it?
4. **Controls.** Require exact baseline/zero token agreement per seed, actual
   before/after SAE measurements, and norm-matched random directions. Compare
   each intervention on the identical baseline continuation as well as its free
   reply. Look for repeatable directional effects before interpreting circuits.

If expected effects do not appear, check calibration, sampling, formatting,
checkpoint revisions and intervention placement. Stronger steering is not itself
better replication: review coherence and task performance too.

## Calibration provenance

Paper §2.4.1 uses `h += alpha * corpus_peak_f * decoder_f`, with `alpha = +/-2`.
The exact corpus peaks and full corpus are not supplied in the PDF. `paper-alpha`
continues to require explicitly supplied peaks.

`reference-alpha` uses our authored, independent **16-prompt** reference set in
`experiments/ism/reference_prompts.jsonl`: Assistant, emotional Assistant, roleplay
and story contexts, excluding evaluation prompts. Reference replies use 50 new
tokens, matching the paper's dataset-generation length. This is a small proxy,
not the authors' corpus or a converged estimate of its maximum. A larger reference
set may produce larger peaks.

`calibration.json` saves texts, replies, token IDs, sample maxima, source SHA256,
model/SAE revisions, dtype, seed and token cap before evaluation. Peaks include
all processed prompt/reply positions; the final emitted token is not processed.
Reused calibration must be complete and match model/SAE/layer/dtype. Raw defaults
remain unchanged.

## First bounded run

From the Mac project root:

```bash
run-experiment ism --device cuda --prompt-set table6 \
  --coefficient-mode reference-alpha \
  --calibration-prompts experiments/ism/reference_prompts.jsonl \
  --strengths -2 2 --max-new-tokens 256 \
  --random-control --teacher-force-baseline --jlens
```

This makes 16 unsteered reference generations, then **42 evaluated completions**
(3 prompts × 14 conditions), plus a teacher-forced forward pass per completion.
It costs more than the earlier raw-strength run. No paid job is launched by setup.
Review the report before a larger sweep. Joint directions sum individually scaled
rows; their norms differ from single directions. Random controls match each
group's total norm, and do not by themselves establish statistical synergy.

## Follow-up after review

Reuse the calibration on the pod to keep strengths fixed:

```bash
run-experiment ism --device cuda --prompt-set table7 \
  --coefficient-mode reference-alpha \
  --calibration-file outputs/ism/<calibration-run-id>/calibration.json \
  --strengths -2 2 --max-new-tokens 256 \
  --temperatures 0 1 1 --random-control --teacher-force-baseline --jlens
```

This is **252 completions**: greedy and two temperature-1 trials with seeds
`seed`, `seed+1`, `seed+2`. Sampling explicitly uses top-p 1, top-k 0; the authors'
full sampler configuration is unknown. Use `--temperatures 0` for an 84-completion
first pass. Trial IDs remain distinct. Reuse requires the file on the pod;
local `outputs/` is excluded from sync-up. To save only calibration, pass
`--calibration-only --calibration-prompts experiments/ism/reference_prompts.jsonl`.

## Multi-layer J-Lens extension

The language model has **34 blocks (0–33)**. The public checkpoint has fitted
J matrices for 0–32. Layer 33 uses the actual final residual/vocabulary head
without fitted transport. Default readouts are **8, 9, 10, 17, 22, 29, 32, 33**;
`--jlens-layers` selects any subset. One revision-pinned checkpoint is loaded once.

Read the last prompt position, first processed generated position, every 16th
generated position thereafter, and the last processed position. Configure
`--jlens-stride`; 1 reads every processed generated position. Layer 9 has actual
before/after-current-addition readouts. Other layers have observed outputs,
including upstream intervention effects where applicable. Layer 8 on the identical
prompt is an upstream control. Later free-generation readouts remain confounded
by changed text. New runs can record both SAE and J-Lens on teacher-forced text;
`--save-jspace-vectors` adds within-layer same-prefix geometric comparisons.

Reports include mean/peak magnitude figures and post-steering/fixed-continuation
statistics in `assessment.json`. Vocabulary top-token lists remain a limited
J-Space assessment unless vectors are saved. New vector metrics cover cosine and
L2 effects, not full covariance/subspace analysis. Fitting R-Lens/J++ is later
work; no exact-model fitted R/J++ artifact was found in the public repos checked.
