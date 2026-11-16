# ISM confirmation: prepared, not launched

Goal: establish a repeatable behavioral intervention before interpreting J-Space
or information-access experiments. This is a functional reconstruction, not an
exact reproduction of the authors' prompts, corpus peaks or checkpoint revisions.

## What the audit found

Code review against paper §2.4.1 and the official Gemma Scope 2 tutorial finds the
expected conventions: original decoder rows, corpus/reference-peak scaling,
post-block layer 9, all processed positions during prefill and cached decoding,
and JumpReLU encoding without subtracting decoder bias or normalizing input.
Gemma's saved chat template merges system text into the user turn; our saved Lyra
prompt reflects that template. This is observable formatting, not proof of the
authors' exact formatting. No obvious formula mismatch has been identified.

New `--audit-sae` runs one unsteered first-prompt forward, then compares selected
encoding with full-width encoding from the raw pinned checkpoint using the
official formula on up to 16 actual residual positions. It checks threshold masks
and exact original decoder rows, and saves `sae_audit.json` with checkpoint SHA256.
An error aborts before evaluated generations. It needs roughly an additional
0.7 GB CPU RAM for the full encoder, plus temporary tensors. This check is not an
independent SAELens execution or author-version parity. Only the synthetic audit
has run locally; pretrained weights are absent locally. No pretrained pass claimed.

Primary references:
- Paper §2.4.1 / §3.3.1 / Appendix H: https://arxiv.org/html/2608.07852v2
- Official tutorial's `JumpReLUSAE.encode`: https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r
- Upstream loader inspected: https://github.com/decoderesearch/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py

Remaining fidelity gaps: authors' full steering probes, original corpus peaks,
exact model/SAE revisions and sampling defaults are not established. Do not
substitute Table 1 mean maxima for corpus peaks. An exact upstream-library parity
check is still distinct from our new full-width formula/loading check.

## Existing outputs and scoring

`outputs/ism/20261107T023612Z-ba1f9307/behavioral_review/` contains 138 shuffled
replies from the two recent runs, blank `ratings.jsonl`, `rubric.md`, and a private
mapping. Share only cards/rubric/blank ratings with a reviewer. Lock ratings before
joining the key. The prompts themselves remain visible; conditions/features and
activations are hidden. Reviewers who already studied the results are not fully
blinded. No semantic ratings or independent review have been completed.

The condition-aware `nonblind_notes.md` identifies informative examples and a task
confound in joint-negative robot output. Continued storytelling, missing stage
directions, or zero encoded coefficients are not sufficient success/failure tests.

For future results, generate fresh material locally with:

```bash
python -m core.review outputs/ism/RUN_ID --destination outputs/ism/RUN_ID/behavioral_review
```

## Bounded next run

- Same pinned Gemma-3-4B-IT and layer-9 65k medium SAE; explicit BF16 on CUDA.
- 12 probes: four Assistant, four Roleplay, four Story. Three are Table 6 examples;
  nine are explicitly project-authored. File `experiments/ism/probes/confirmation.jsonl`.
- Separate 133, separate 4360, and joint, alpha −2/+2 using independent reference
  peaks. Recollect the existing 16-reference calibration for portable provenance;
  do not rely on syncing ignored output files. These peaks remain proxies.
- Temperatures 0/1/1 and trial seeds 0/1/2. Each trial resets the same seed for
  each intervention and its baseline; this is paired sampling, not identical text.
- One norm-matched isotropic direction for each of the three feature groups,
  two signs; same vectors reused across probes/trials. Three directions are a
  limited comparison, not a null distribution. High-norm controls may degenerate.
- Baseline and zero-hook per prompt/trial; cap 256 emitted tokens. Cap hits and
  EOS flags are saved to distinguish truncation from completed answers.
- Monitor all 23 paper features. J-Lens, saved J vectors and teacher-forced passes
  are explicitly disabled in this behavioral confirmation to reduce cost. They
  remain available for follow-up after choosing a reliable condition.

Total: **504 evaluated generations** (216 target, 216 random-control, 36 baseline,
36 zero-hook), plus **16 reference generations** and one unsteered audit forward.
At most 129,024 evaluated new tokens plus 800 reference new tokens. Unknown early
EOS and GPU throughput prevent a firm time estimate; provision roughly 30–90
minutes as a planning estimate, not a measured bound or spending limit.
The previous vector-heavy 96-evaluation run took 18m21s; this plan omits that lens
work but has more and longer generations, including stochastic trials.

CPU-only plan, which does not create a run or access a pod:

```bash
python run.py ism --dry-run --device cuda --dtype bfloat16 \
  --prompt-file experiments/ism/probes/confirmation.jsonl \
  --coefficient-mode reference-alpha \
  --calibration-prompts experiments/ism/reference_prompts.jsonl \
  --strengths -2 2 --temperatures 0 1 1 --seed 0 \
  --random-control --random-control-count 1 --audit-sae --max-new-tokens 256
```

**Paid launch, only after explicit OPS approval:**

```bash
run-experiment ism --device cuda --dtype bfloat16 \
  --prompt-file experiments/ism/probes/confirmation.jsonl \
  --coefficient-mode reference-alpha \
  --calibration-prompts experiments/ism/reference_prompts.jsonl \
  --strengths -2 2 --temperatures 0 1 1 --seed 0 \
  --random-control --random-control-count 1 --audit-sae --max-new-tokens 256
```

## How we will decide what to do next

Freeze the rubric before reviewing new outputs. Assess helper/editor framing,
character portrayal, literary presentation, task fulfillment and degradation
separately; avoid a composite score that assumes the hypothesis. Pair comparisons
within prompt/trial. Inspect positive and negative effects without assuming both
must work or be symmetric. Report consistency for all four prompts per setting
and all trials, alongside quality and truncation. The paper's 10-of-12 consistency
rule is useful context, but our bespoke rubric is not its steerinterp judge.

Choose a candidate only if the framing/presentation shift is coherent across
Roleplay and Story and across seeds without relying on severe degradation. Report
how many paired outputs support the proposed effect, do not select only attractive
examples, and compare against coherent controls. If controls produce the same
signature, call specificity unresolved. If only negative steering is reliable,
use it as a suppression intervention; do not claim reliable positive induction.

Six fresh authored probes in `experiments/ism/probes/heldout.jsonl` are reserved
for subsequent confirmation after choosing a setting. They are not used for
calibration or strength tuning. If ±2 has mixed quality, document the failure and
then explicitly plan a 1–2 sweep rather than silently altering this run.

No claim about subjective perspective follows from passing these behavioral tests.
