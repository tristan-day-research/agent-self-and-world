# Gemma 4 mechanistic follow-ups

Two opt-in stages extend `gemma4_suite`. Both completed on the pretrained
model on 2026-11-11; their run folders contain the results and authored reports.
The plans below document the fixed design used for those runs.

## 1. Door access, attribution, probes, and patching

Dry plan:

`python run.py gemma4_suite --followup door-mechanisms --door-families 16 --seed 1044 --residual-layers 10 20 30 40 50 --dry-run`

RunPod command after paid-compute approval:

`run-experiment gemma4_suite --followup door-mechanisms --door-families 16 --seed 1044 --residual-layers 10 20 30 40 50 --run-label gemma4-door-mechanisms`

This uses 16 new paired-code families, eight for probe training and eight held
out. For each even/odd code, four availability × attribution cells, and two
tasks (self-report and the explicit instrumental rule), it records the final
prompt-position residual at five layers, literal responses, and answer-token
probabilities. That is 256 baseline prompts/short generations. Ridge probes
learn code parity within each *present-code* condition and task, then transfer
to held-out families in all four conditions and both tasks. Separate probes
decode actual code availability and attributed knowledge. The absent-code
parity evaluations are negative controls: their even/odd labels are not in
the prompt.

On held-out present-code instrumental prompts, the code swaps the final
residual with the opposite-parity partner's state at one selected layer while
retaining the target's earlier K/V and all other prompt tokens. At most 160
opposite-parity patches plus 20 self-patch controls are planned. Pairs with
different tokenized prompt lengths are skipped. Measure L/R/A probabilities,
especially movement toward the donor's procedure, and report format/competence
before interpreting any patch. A successful patch shows causal information
flow at that hook, not a distinct inner agent or awareness.

Artifacts: `outputs.json`, `residuals.pt`, `probes.json`,
`factor_probes.json`, `patches.json`, design, metadata, log, status. Both
baseline and patch rows checkpoint and the run can resume after failure.

## 2. Speaker/viewpoint candidate discovery and steering

Dry plan:

`python run.py gemma4_suite --followup speaker-discover --seed 1045 --residual-layers 10 20 30 40 50 --steer-after-discovery --max-new-tokens 96 --dry-run`

RunPod command after paid-compute approval:

`run-experiment gemma4_suite --followup speaker-discover --seed 1045 --residual-layers 10 20 30 40 50 --steer-after-discovery --max-new-tokens 96 --run-label gemma4-speaker-discovery`

Eight matched characters are split by character: four train, two validation,
two untouched steering test. The discovery section uses 96 prompts:
Assistant/Roleplay/Story with person and prose style crossed, plus story prompts
that differ only in instructions to stay close to the character's perceptions
or describe externally observable actions. It captures residuals at five
layers and full TopK-64 feature IDs/activations at SAE layer 30. Validation
and steering-test characters use paraphrased speaker/viewpoint instructions,
so a candidate must transfer beyond the exact training wording. Train-only
matched differences yield a story-minus-Assistant contrastive direction, a
third-person-story-minus-Assistant direction with person fixed to third, and an
immersed-minus-detached viewpoint direction. The train-only strongest supported
SAE feature for each of the third-person story and viewpoint differences is
also frozen as a decoder direction.
Validation characters only check whether those candidates transfer; they are
not used to choose directions.

The optional steering stage uses 16 untouched test prompts and eight arms per
prompt. On third-person Assistant/story prompts the arms are baseline,
zero-hook, ±third-person story contrast, ±its SAE feature, and ±matched random.
On immersed/detached story prompts they are baseline, zero-hook, ±viewpoint
contrast, ±its SAE feature, and ±matched random. That is 128 additional greedy
generations, plus scored boundary forwards. Steering uses a quarter of
the train-pair projection gap at layer 30 and acts on the last processed token.
It saves full responses and same-prefix next-token KL from baseline. The
candidate named “viewpoint” is an *operational proxy*, not an already identified
feature from *Many Are My Names*. A change in narration alone is not evidence
of an internal perspective change; the held-out behavioral, feature, and door
effects must be compared, with style and grammatical-person controls.

Artifacts: `outputs.json`, `vectors.pt`, `discovery.json`, `directions.pt`,
optional `steering.json`, design, metadata, log, status. No J/R lens is fitted.
The previous exact-IT J-Lens can be used later for a targeted readout, but its
intermediate top-token lists were not discriminative in the door baseline.
