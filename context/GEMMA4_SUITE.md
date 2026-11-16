# Gemma-4-31B-IT baseline suite

The `gemma4_suite` command uses the exact instruction-tuned checkpoint from the
successful door capability screen. It does not transfer Gemma-3 feature IDs.

## First run

`run-experiment gemma4_suite --sections speaker door confidence --families 4 --seed 1043 --speaker-set both --questions-per-domain 1 --sae --sae-layer 30 --jlens --jlens-layers 30 45 52 --queries --query-layers 22 30 45 52 --max-new-tokens 96 --run-label gemma4-baseline-suite`

Dry plan: 195 prompts (27 speaker, 128 crossed door, 40 persona confidence),
195 scored forwards, and 155 greedy generations. Speaker prompts comprise
three Table 6 prompts plus 24 project-authored factorial probes. Door trials
use four fresh paired code families and explicit parity-then-mapping wording.
Confidence scores conditional choice entropy and retains valid-choice mass.
Rows checkpoint to `outputs/gemma4_suite/<run-id>/outputs.json`.

The SAE is a TopK-64 dictionary at layer 30. Its top boundary features and
reconstruction error are recorded. Interpret feature activations only if the
reconstruction audit supports the hook and normalization. The standard J-Lens
is fitted to the exact IT checkpoint; a direct final-layer readout is checked
against the model's top output token. Raw Q projection norms at four layers
are diagnostic observations, not attention probabilities or causal Q tests.

## Follow-up gates

1. Review SAE reconstruction, lens readouts, door positive controls, format
   adherence, speaker outputs, and persona valid-choice mass.
2. Compare matched assistant/roleplay/story SAE activity while controlling
   character, grammatical person, and prose style. Select candidate Gemma-4
   perspective directions on held-out prompts.
3. Run model-specific decoder or contrastive steering with random controls,
   zero hooks, and fixed-text readouts. Q patching and full attention
   measurement require a validated Gemma-4 implementation.

No Gemma-4 ISM feature is identified by this setup. J-Lens++ for the base
checkpoint is not applied to IT. An exact IT R-Lens has not been verified.
No lens is fitted.
