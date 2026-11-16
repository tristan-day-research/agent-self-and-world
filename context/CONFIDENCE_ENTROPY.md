# Persona × ISM confidence experiment

## Question and interpretation

Does steering the ISM-associated Gemma feature 133 change how strongly a persona
alters its distribution over answers, especially inside versus outside its
claimed expertise? The primary test is an interaction, not an overall entropy
change from steering:

`[H(persona, steer) − H(default, steer)] − [H(persona, baseline) − H(default, baseline)]`

A nonzero value indicates a changed persona effect on answer uncertainty on this
question. Its sign need not always be negative. Lower entropy can accompany wrong
answers, answer-label bias, poor formatting or degradation. It does not establish
knowledge, genuine self-confidence, subjective immersion, or agenthood. Any output
change necessarily has computational causes; “underlying computation changed”
alone does not distinguish a deep perspective change from style-related changes.

Feature 133 currently has stronger evidence as a narrative/character-presentation
component than as a specifically first-person perspective feature. Our positive
story replies can be third-person and negative replies can contain first-person
narration. A direct test should independently vary grammatical person, literary
style and role framing while holding content fixed. This confidence experiment
adds a complementary test with the answer format held to fixed letter labels.

## What was ported

The experiment is `confidence_entropy`; reusable code is in `core/`. There are
no runtime imports or data dependencies on `legacy_code/`.

- Preserved the original 150-question cache: 40 chemistry, 40 history, 25 art,
  25 general and 20 misconception questions, with their original labels/keys and
  dataset metadata. Questions/keys have not been independently revalidated.
- Preserved all seven original personas and four prompt templates in JSON,
  including source hashes. No YAML/dataset download dependency was added.
- Per user discretion, added three neutral-wording expert variants: remove only
  “especially confident in ... topics,” retain expertise and perspective wording.
  Default `--persona-set both` includes ten personas total. Cautious, bold and
  child prompts stay intact. `--persona-set original` restores the original seven.
- Preserved separate contexts: answer probabilities and stated-confidence
  probabilities are independent. Optional free answer/confidence explanations
  are also independent; generated reasoning is never fed into the answer probe.
- The original experiment used Llama-3.1-8B-Instruct. **This initial port uses our
  validated, pinned Gemma-3-4B-IT/SAE/J-Lens stack.** It is not an exact legacy
  model replication, and the pretrained CLI does not yet support Llama.

`legacy_code/` is reference material from another project and is intended for
later deletion. AGENTS.md records this; nothing was deleted.

## Default pilot and controls

- Two sampled questions per domain (10 total), same selected questions and fixed
  option ordering in every persona/steering condition. Seed42. This is a small
  plumbing/scientific pilot, not adequate evidence of general domain effects.
- Default Assistant is the original helpful/harmless/honest system prompt, not
  a system-prompt-free model. Expert-versus-style personas should be analyzed separately.
- One feature group: 133. Baseline, exact zero-hook, −2 and +2 reference-alpha.
  `--features 133 4360` would explicitly switch to joint steering, not run separate groups.
- Optionally `--random-control` adds the two signs of one seeded isotropic
  direction matched in residual norm. Its effects are not a null distribution.
  Monitor off-format behavior and quality before treating it as a usable control.
- Reference calibration is collected from the existing independent 16-prompt set,
  50 tokens each, with pinned provenance. `--calibration-file` can reuse a compatible
  artifact; the original paper corpus peaks remain unavailable. Raw mode is
  explicit, and raw strengths are not alpha.
- No answer reasoning generation by default: one forward pass for answer
  probabilities and one for confidence categories per cell. `--generate-responses`
  restores generated outputs from all four legacy contexts (including short coded
  replies), with generated-token SAE summaries for manipulation checks. This can
  greatly increase time. Expert prompts explicitly say “AI assistant”; do not
  assume they naturally elicit immersive character behavior without checking.

Counts with default ten personas: **400 cells / 800 scored forward passes**.
With `--random-control`: **600 cells / 1,200 scored forward passes**. Add 16 reference
generations unless reusing calibration. These are not 800/1,200 full generated
answers. No pretrained runtime/cost has been measured for this new experiment.

## Measurements

Choice probabilities use the first response position and a declared **bare-letter,
single-token** policy. Reject multi-token or duplicate label encodings rather
than silently taking the first subtoken or selecting the best token variant.
This preserves the legacy bare-letter intent with a stronger validity check.

Store raw choice logits, conditional probabilities, entropy in nats and entropy
normalized by `log(K)`. The migrated data have 125 four-option items and 25
five-option items, so normalized entropy and within-question contrasts matter.
Also store full-vocabulary entropy, probability mass on all allowed answer letters,
conditional argmax correctness, unconstrained argmax correctness/format compliance,
correct-answer probability/NLL and multiclass Brier score. Conditional accuracy is
not the same as the model freely emitting a valid answer.

For confidence, preserve the S–Z scale and midpoint values. Store its conditional
category distribution, weighted midpoint and the midpoint of the actual first-token
argmax only when it is a valid confidence label. A midpoint expectation over
renormalized labels is not an explicit generated confidence statement when label
mass is low. `assessment.json` pairs these with answer entropy/correctness and
computes per-question persona × steering effects. It does not claim introspection.

SAE coefficients are recorded immediately before and after steering at the final
prompt position. Optional free-text diagnostics capture generated-token means/peaks
as well; the last emitted token is not processed in cached generation. A high
injected coefficient is not an independent semantic validation of ISM.

## J-Lens

Enabled by default: blocks8/9/10/17/22/29/32 and direct-final33, at the **final prompt
position before an answer/confidence token**. There are no generated-token lens
traces in this first version. Intermediate lenses are approximate readouts and
are not calibrated model beliefs. Record full-distribution/choice metrics, top
vocabulary tokens and optional unnormalized transported vectors. Direct-final
readout is checked against actual logits; float32 tolerance is1e-4, with BF16/FP16
roundoff tolerance rtol.02/atol.25 and the maximum error recorded. Actual answer
metrics always come from the model's own logits.

`jspace_effects.json`: within-persona/decision, identical full-prefix comparisons
against unsteered baseline. `persona_jspace.json`: same question/decision boundary,
different persona prefixes intentionally, reporting how persona–default vector
distance changes under steering and the vector interaction
`(v_persona,s − v_default,s) − (v_persona,0 − v_default,0)`.
The latter is not a matched-full-prefix intervention. Direct linear layer-9
additions can cancel in the interaction; later-layer changes are more informative
about differential propagation. Distances still do not define an immersion axis.

## Commands and artifacts

Local planning and synthetic checks, no paid compute or downloads:

```bash
python run.py confidence_entropy --dry-run --random-control
python run.py confidence_entropy --smoke --device cpu --random-control
```

Prepared RunPod pilot; **ask before launching**:

```bash
run-experiment confidence_entropy --device cuda --dtype bfloat16 \
  --questions-per-domain 2 --persona-set both --features 133 \
  --strengths -2 2 --coefficient-mode reference-alpha --random-control
```

All outputs live in `outputs/confidence_entropy/<run-id>/`: config/status/log,
metadata with file/code hashes, sampled questions, calibration, incremental
`outputs.json`, assessment/measurements and lens vectors/comparisons. No template
writes `report.md`; an agent can inspect results and author it after a run.

A failed/interrupted run can resume in place with the **same original flags** plus
`--resume-directory outputs/confidence_entropy/RUN_ID`. It checks config, source
provenance/code hashes and skips saved decisions. Active/completed statuses are
refused. Hard kills can leave stale `running`; inspect actual process state before
manually repairing status. Do not use resume to change the experimental setup.

Before substantive conclusions, increase question coverage, check label-order
sensitivity, inspect free-text manipulation checks and task quality, and use
question-level paired uncertainty estimates rather than treating all cells as
independent samples. No pretrained confidence experiment has been launched.

## Other published ISM model

The paper reports Llama-3.1-8B-Instruct, layer15, feature101460 in a 131,072-wide
SAE, with a similar qualitative steering effect (Appendix F/G):
https://arxiv.org/html/2608.07852v2

A public matching-model/width/layer candidate suite is
https://huggingface.co/andyrdt/saes-llama-3.1-8b-instruct . Exact trainer/feature
index mapping must be verified before calling this the authors' exact dictionary.
Notably, the inspected `resid_post_layer_15/trainer_1/config.json` says
`BatchTopKTrainer` / `BatchTopKSAE`, while the paper describes JumpReLU SAEs.
This can involve inference conversion conventions, but that has not been verified
here. Do not transplant feature101460 into an arbitrary Llama SAE or reuse Gemma
encoder/scaling conventions. No Llama adapter was implemented in this port.
