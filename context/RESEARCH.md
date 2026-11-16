# Research notes

## Three-speaker door pilot: framing versus code use — 2026-11-11

On Gemma-4-31B-IT, Assistant discussion of Mara, roleplay as Mara, and
third-person narration of Mara generated visibly distinct free-response forms,
yet all matched fixed door outcomes were the same. Present-code/denied-
knowledge cells reported N, repeated UNKNOWN, and chose the correct code-
dependent procedure 12/12 across modes; absent-code/attributed-knowing cells
reported Y but repeated UNKNOWN and chose A 12/12. The instrumental prompt
explicitly requests use of the scenario record, so this is compatible with a
single model process applying different reporting rules. Query-input speaker
classification transferred perfectly between the two families at L10/20/30
when task types were shared, but cross-task transfer varied. The signal may
encode prompt wording rather than a causal perspective state. Mean final-
boundary attention on the code field at L59 was larger for repetition than
instrumental choices; attention weights alone do not measure information use.
Only two independent code families and one template per mode were tested.
Zhu-style head-output belief probes need new capture and independently varied
observer versus character belief labels; the present run did not save head
outputs. See the run report and review script for exact values and limitations.

## Door speaker and attention interpretation — 2026-11-11

Moving a residual activation from a code-present to a code-absent prompt
changes model state, but does not by itself locate an "agent" or distinguish
agent access from other computation. Operational tests can compare where the
patch is applied, which tasks change (attributed knowledge report versus
code-dependent choice), and whether speaker setting changes the effect.
Attention and Query measurements can expose routing at a chosen token, but
attention weights are descriptive and the three speaker instructions also
change the prompt text. The prepared Gemma-4 pilot records attention at the
answer boundary; causal attention or Query interventions are a later step.

## Speaker steering interpretation refinement — 2026-11-11

The Gemma-4 speaker plot counts any text change, pooling positive/negative
directions and Assistant/Story starting prompts; it is not a direction-specific
success metric. Both signs were applied in both starting conditions. Inspection
found mostly wording and scene-detail changes, while all four Assistant test
prompts kept their introductory framing and all four Story prompts remained
direct stories under candidate arms. The hook acts on the last token of every
forward, including prefill and cached decode, at layer 30. That layer was a
pragmatic pretrained-SAE starting point, not a claim that ISM emerges there.
The paper's Gemma-3 ISM-associated features were at layer 9 of its 34-layer
model. Earlier Gemma-4 SAE layers require fresh candidate discovery and
held-out directional evaluation.

## Why the two door patch sweeps differ — 2026-11-11

The older Qwen2.5-32B run overwrote the *code token positions* and measured
remaining code-following. Early swaps redirected the answer, while later
swaps lost effect after downstream positions could already have incorporated
the original code. The newer Gemma-4 run overwrote the *final prompt position*
and measured change in donor-procedure probability. Early swaps can be
overwritten by later computation from intact code tokens; layer-50 swaps are
near the readout and often redirected it. This position/timing explanation is
mechanistically plausible but needs both interventions in one model to test
directly; model, prompts, and metrics also differ. Same-attribution parity
donors do not isolate representations of not knowing or concealment.

## Gemma-4 speaker discovery and steering — 2026-11-11

On new characters with paraphrased instructions, layer-30 train-derived
Assistant/third-person-Story and immersed/detached Story contrast directions
separated 4/4 validation pairs each. Train-selected TopK SAE feature 32546
tracked the Assistant/third-person-Story contrast in 4/4 validation pairs;
viewpoint feature 34957 failed its 4/4 validation check (0 correctly oriented).
On 16 untouched prompts, ± layer-30 final-token steering at one conservative
dose changed wording but did not switch Assistant versus Story framing. Matched
random directions changed literal text more often. This distinguishes
descriptive decodability from causal control and does not identify an ISM
mechanism or an AAC. Report and recomputable evidence are in the
2026-11-11_12-54-50_PST Gemma-4 speaker run; outputs SHA256
`367ad2d05f5293d27b83953e96580b6b0930e792e1a26c3b7d25915bc016ec62`.

## Door parity probe versus causal patch — reviewed 2026-11-11

Fresh Gemma-4 family-held-out baseline preserved the report/use split:
code-present knows and doesn't-know instrumental cells both got 29/32 with
identical three errors, while reports were Y 32/32 versus N 32/32. A parity
probe trained in the knows instrumental cell had held-out AUROC .984 at L30
and 1.0 at L50; transferred to doesn't-know instrumental cell at .953 / 1.0.
Yet swapping only the final-position residual from the opposite-parity same-
family partner at L10–30 barely changed the answer, whereas L50 raised the
donor-procedure probability by .636/.563 across knows/doesn't-know cells.
Decodability at L30 is therefore distinct from sufficiency of this specific
patch. Earlier target code tokens and K/V states remain, and late patches
could carry answer state as well as parity. Do not infer parity first appears
at L50. Self-patch numerical error was at most 5.2e-8. See the run report.

## Gemma-4 follow-up hypotheses — 2026-11-11

Prepared, not yet measured: family-held-out parity probes can test whether
code information transfers between attributed-knowledge conditions and from
self-report to instrumental task. Same-family opposite-parity final-residual
patches can test which layers' target answers remain sensitive to donor code
state. Absent-code parity labels are negative controls, not information the
model could infer from an absent record. Character-held-out matched contrasts
can identify candidate Assistant/story and immersed/detached directions. The
viewpoint contrast is an operational proxy for the paper's ISM, not a known
Gemma-4 ISM feature. Decoder-feature and contrastive steering require causal
and behavioral controls to distinguish prose style from perspective. Speaker
and viewpoint instructions are paraphrased for validation and steering-test
characters to test transfer beyond the exact training wording; instruction
content can still affect the directions.

## Crossed Gemma-4 access and attribution baseline — 2026-11-11

The fresh-seed, explicit-wording crossed experiment completed 128 door rows
inside a 195-row suite. With code in the system record and attribution
"doesn't know," the model reported N 8/8 and UNKNOWN 8/8 but answered the
code-dependent procedure 15/16; with code absent and attribution "knows," it
reported Y 8/8 but replied UNKNOWN 8/8 and A 16/16. Available/knows was
16/16 on the procedure. This is behavioral dissociation between attributed
knowledge/report and information use, not proof of distinct agents or an
internal information barrier. Only four independent paired code families.
SAE layer-30 boundary reconstruction error mean .300 (range .206–.386);
final direct J readout top token matched 195/195. Intermediate IT J-Lens
top tokens are poorly discriminative for this short door task (layer 30's
top token was I on all 128 door contexts). Matched factorial raw Q norms
change consistently by speaker setting at selected layers but are confounded
by setting prompts and do not measure attention or causal Q effects. Persona
choice entropy was near zero on five questions per persona, leaving no useful
room for an ISM interaction. Full evidence is in the run's report.

## Gemma-4 transition protocol — 2026-11-11

The Gemma-4 SAE is TopK-64 with 43,008 features per layer and an all-layer
release, but a file existing is not evidence of a valid hook or reconstruction.
The first run records reconstruction error before interpreting feature IDs.
Gemma-3 133/4360 IDs cannot be assumed to correspond to Gemma-4 features.
The exact IT standard J-Lens is available; the published Gemma-4 J++ is for
the base checkpoint. Raw Q projection norms are only a first query diagnostic,
not attention probabilities or a causal account of agent representation.

## Gemma-4-31B-IT capability and artifacts — reviewed 2026-11-11

On the same eight families and 128 request keys as the Gemma-3-27B/Qwen3.5-27B
screens, the instruction-tuned Gemma 4 got 16/16 last digits, 16/16 standalone
parity, and 30/32 combined decisions under either wording. The single missed
odd code failed both rule orders and wordings. Supplied-parity mapping was
24/32 because two unique code-absent prompt types, repeated eight times,
elicited explanations instead of L/R. The 30/32 result is a promising positive
control, with only eight independent code families; held-out validation is
still needed. See the run report and hash in STATUS.

The Adam1010 dataset card's statement that all-layer SAE weights require
separate download is stale: the dataset API lists 60 `.pt` files, layers 00–59.
A pod audit loaded layer 30 with `weights_only=True`: W_enc 5376×43008,
W_dec 43008×5376, biases and per-coordinate norm mean/std. The dataset names
`gemma-4-31B-it`. Functional hook and reconstruction validation remains.
Decode Research separately offers a SAELens-compatible layer-30 Matryoshka
SAE. `koayon/jpp-lenses` publishes J++ for **base** `gemma-4-31B`, not the
instruction-tuned checkpoint. `solarkyle/jspace-lenses` and `tbrx/uspace-lenses`
offer standard Jacobian lenses for the exact `gemma-4-31B-it` checkpoint; the
former covers a broad layer band, the latter five layers. No such lens was used
in this behavior screen. Sources:
https://huggingface.co/datasets/Adam1010/gemma-4-31b-sae-features/tree/main/weights ;
https://huggingface.co/decoderesearch/gemma-4-saes ;
https://huggingface.co/koayon/jpp-lenses ;
https://huggingface.co/solarkyle/jspace-lenses ;
https://huggingface.co/tbrx/uspace-lenses .

## Family choice and lens role — checked 2026-11-11

Current door runs used `Qwen/Qwen3.5-27B`; earlier ISM replication used
`google/gemma-3-4b-it`. `google/gemma-3-27b-it` is a separate larger Gemma 3
checkpoint, with Google's matched Gemma Scope 2 SAE suite. A subsequent
128-case behavior screen found 28/32 correct explicit door decisions, but
no SAE was loaded; artifact coverage remains distinct from tested steering.
Llama-3.3-70B-Instruct has a Goodfire SAE at layer 50 and
Neuronpedia standard J-Lens, but no matching R/J++ verified in the inspected
inventories. Kimi-K2-Thinking is a 1T-total/32B-active MoE using MLA; no
matching public SAE or J/R/J++ artifact was verified in this search, and its
total weights make local activation intervention much more demanding. A
J-style lens is an optional readout/transport basis; behavioral contrasts,
probes, causal patching and SAE interventions can address the access/perspective
question without it. Lens top tokens alone are not evidence of information
presence/absence, as the Qwen door run's very low allowed-label mass shows.
Sources: https://huggingface.co/google/gemma-3-27b-it ;
https://huggingface.co/google/gemma-scope-2-27b-it ;
https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50 ;
https://huggingface.co/neuronpedia/jacobian-lens ;
https://huggingface.co/moonshotai/Kimi-K2-Thinking ;
https://transformer-circuits.pub/2026/workspace/index.html .

## Gemma-3-27B door behavior — reviewed 2026-11-11

The 128-case BF16 baseline on an RTX PRO 6000 used exactly the same scenario
messages, seed, codes, rule orders, and answer keys as the Qwen3.5-27B wording
validation. Gemma got 16/16 last digits, 10/16 standalone parity (six even
codes called odd), 28/32 supplied-parity mapping (four off-format responses
among only eight unique prompts), 20/32 original combined decisions, and
28/32 explicit combined decisions. Explicit errors clustered on two even
codes, each under both rule orders. Qwen's matched explicit score was 25/32;
23 cases were right for both, five Gemma-only, two Qwen-only, and two wrong
for both. The sample is too small for a general model ranking, and the
positive control remains imperfect. See the run report and SHA256 in STATUS.

## Matched SAE and workspace-lens alternatives — checked 2026-11-11

For an instruction-following door task near 32B, exact-checkpoint artifact
matches are: `google/gemma-3-27b-it` with Google's Gemma Scope 2 SAEs across
all layers and a published J/R pair in `camilablank/workspace-lenses` (plus
Neuronpedia J); `google/gemma-4-31B` with Neuronpedia J++/J and a downloadable
Decode Research SAE at layer 30; `Qwen/Qwen3.6-27B` with J++/J/R and a smaller
community SAE at layers 11/31/55. Gemma-4-31B-IT has community all-layer
SAE weights online and an exact-checkpoint standard J-Lens, as audited above;
these were not integrated in the screen. `allenai/Olmo-3-1125-32B` has
J++ and a community SAE at one layer, but is a base checkpoint, less aligned
with this conversational task. `Qwen/Qwen3.5-122B-A10B` has J++ and J/R but no
matching public SAE verified in the inspected collections; Llama-3.3-70B-
Instruct has a Goodfire layer-50 SAE and standard J, not a verified R/J++.
These are artifact matches, not behavioral validations. Sources:
https://huggingface.co/google/gemma-scope-2-27b-it ;
https://huggingface.co/camilablank/workspace-lenses ;
https://huggingface.co/neuronpedia/jacobian-lens ;
https://huggingface.co/decoderesearch/gemma-4-saes ;
https://huggingface.co/koayon/jpp-lenses ;
https://huggingface.co/caiovicentino1/qwen36-27b-sae-multilayer ;
https://huggingface.co/bcywinski/Olmo-3-32B-Base-SAE .

## Fresh Qwen door wording validation — reviewed 2026-11-10

Eight new code families were tested with original and explicit parity-then-rule
instrumental prompts on matched codes and rule orders. Original was 19/32,
explicit 25/32; seven matched cases improved and one worsened. Parity was
13/16 despite 16/16 final-digit copying; supplied mapping was 32/32 but only
eight unique prompts. Mean paired parity effect rose .289→.447. The new
wording improves behavior, especially odd-code cases (4/16→11/16), but leaves
seven instrumental errors and does not yet provide a reliable positive
control for access claims. SAE top-one features were constant across both
wordings at each measured layer; raw R-Lens answer mass stayed tiny; observed
Q/attention differences cannot establish mediation. Full report and metrics:
`outputs/agent_vs_model/2026-11-10_16-25-18_PST--qwen35-door-wording-validation--1ecc792f`.

## Qwen3.5-27B door capability pilot — reviewed 2026-11-10

The first pretrained Qwen3.5-27B run completed 48/48 baseline-only rows with
three Qwen-Scope SAE layers, published R-Lens, and full-attention/Q capture at
blocks 15/31/47/63. On four code families: final-digit copy 8/8, explicit
parity 6/8, parity-supplied rule mapping 16/16, and combined instrumental
choice 11/16. Mean paired even-minus-odd effect on P(even-rule procedure | L/R)
was +0.392 across eight pairs. Even-first rule order scored 4/8 and odd-first
7/8, so the positive control is not yet robust. All answers obeyed short
format; the error is not label-mass collapse. SAE top feature at blocks 16/32
was constant across all tasks; these ranks do not identify an ISM feature.
Early/mid R-Lens answer-token mass was tiny, limiting a decision-trace reading.
Paired normalized Q vectors became somewhat less similar at later measured
blocks, but no query intervention was performed. See the agent-authored run
report and reproducible metrics under
`outputs/agent_vs_model/2026-11-10_16-10-47_PST--qwen35-door-capability--91e9b97a`.

## Gemma door capability diagnostic — reviewed 2026-11-10

The baseline-only 48-row Gemma-3-4B-IT run completed. Final digit copying
was 8/8, parity classification 5/8, supplied-parity L/R mapping 8/16 with
all 16 responses R (eight unique mapping prompts), and full instrumental
choice 8/16. Across eight paired code comparisons, changing only final-digit
parity had mean effect 0.000014 on P(even-rule procedure | L/R). Label mass
was ~1 and format coverage 48/48, so malformed responses do not explain the
failure. The four task wordings differ, limiting a clean decomposition, but
the mapping-only failure is sufficient to block an access interpretation.
No ISM steering or J/attention measurements were made. See the run's
`report.md` and `review_metrics.json` under
`outputs/agent_vs_model/2026-11-10_15-53-46_PST--door-capability--7dab5efa`.


## Qwen32 access test and alternative perspective methods — 2026-11-10

Offline review of legacy Qwen2.5-32B v2 probability reports adds an important
qualification: the 24-family validation `knows × neutral` positive control had
paired parity effect 0.52 and 79% most-likely choice accuracy, narrowly failing
its preregistered ≥0.5 effect / ≥80% accuracy criterion. The subsequent main
run had 0.45 and 77%, again below its gate. In `doesnt_know × neutral`, the main
run had parity effect 0.41, P(ASK) 0.61, and 34% most-likely correct choice.
This supports some code-parity sensitivity in *conditional probabilities* under
asserted ignorance, but does not establish reliable instrumental completion
or a separate agent. These are legacy results, not newly run inference. Sources:
`legacy_code/experiments/agent_vs_model/results/diagnostics/20261027_210154_agent_vs_model_v2_validate_qwen32b_syscode_rep24_Qwen2-5-32B-Instruct/report.md`;
`legacy_code/experiments/agent_vs_model/results/diagnostics/20261027_210352_agent_vs_model_v2_main_qwen32b_syscode_dnr24_Qwen2-5-32B-Instruct/report.md`.

Follow-up: the J-Lens can supply steering directions as well as readouts. The
original J-space paper tests interventions; the nnsight authors' executable
mini-paper shows token-concept steering and swaps using the transported
unembedding direction at each layer. These are token-associated directions,
not automatically an ISM or agent-perspective axis. Source:
https://arxiv.org/html/2607.15495 ;
https://nnsight.net/tutorials/mini-papers/jacobian-lens/ .

The first Gemma-3-4B door pilot does not establish a general capacity limit: its
`knows` condition repeated all eight codes but the paired instrumental positive
control failed. Test digit extraction, parity, and parity-to-procedure mapping
separately before attributing this to model size; compare the exact legacy prompt
on Qwen2.5-32B-Instruct and Gemma on held-out code pairs.

Gemma's SAE feature IDs cannot be transferred to Qwen2.5-32B. No public
Qwen2.5-32B J++/R fit or matched broad SAE suite was verified in the inspected
inventories. A separate project reports a partial Qwen2.5-32B J-Lens fit at
layers 36/58/62 (~1h46m on H100, peak 66.6 GiB), but does not publish its
binary in Git. A Qwen32 ISM pilot can instead start with matched contrastive
residual directions or supervised probes, plus held-out behavioral and causal
validation, without training an SAE. This is a new direction, not the Gemma
feature 133 transferred.

Qwen3.5-27B has a Qwen-published residual TopK SAE (80K width, k50) at all 64
layers, a public J-Lens, and a matched third-party R-Lens. No J++ fit for that
checkpoint was verified in the inspected public inventory. Qwen3.6-27B has
public J/J++ and third-party R fits; verify SAE coverage before choosing it for
an SAE-centered study. Lens checkpoints require exact model/revision, hook,
and shape audits. Their availability does not imply an ISM feature has been
identified for either model.

Zhu et al. (ICML 2024) trained logistic probes on individual attention-head
outputs to distinguish protagonist and oracle beliefs, then intervened along
probe-derived directions. This supplies a feasible non-SAE strategy for
perspective attribution, but does not identify an assistant core or establish
subjective agenthood. In the door task, the user instruction that the Assistant
"does not know" the code changes attributed knowledge while the system prompt
still supplies it; a true information-access manipulation must be crossed with
that attribution when adapting Zhu's design.

Sources: https://huggingface.co/neuronpedia/jacobian-lens/blob/main/README.md ;
https://huggingface.co/camilablank/workspace-lenses ;
https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 ;
https://github.com/dyllonj/persona-society-sim/blob/main/docs/jacobian-lens-pilot-report.md ;
https://proceedings.mlr.press/v235/zhu24o.html .

## Door-code attribution and lens availability — 2026-11-10

### First Gemma door-code pilot — reviewed 2026-11-10

Run `outputs/agent_vs_model/2026-11-10_14-27-49_PST--door-code-pilot--08255c80`
completed 512 condition rows and 128 exact zero pairs; authored `report.md`
and recomputable review metrics. Baseline `knows` says Y and retrieves 8/8
codes, but paired instrumental parity effect is about 0 and accuracy 8/16:
each even/odd pair gets the same procedure under each rule order. Thus the
positive control fails. `doesnt_know` says N, retrieves none and chooses A,
which is not evidence of a model/agent access split when the instrumental
task itself is insensitive to code parity. `withholds` says Y but reveals all
8 codes on direct request, a separate instruction-following failure.

At ±2 reference-alpha feature133 steering, 0/64 instrumental generations per
sign obey the requested L/R/A format, and full-vocabulary choice mass collapses
from ~1 to 7.3e-10 (negative) or 3.7e-4 (positive). Negative steering changes
all8 `doesnt_know` self-reports N→Y, but this is not evidence of restored code
access given the severe format disruption. Baseline L9:133 is zero at every
instrumental boundary; +2 drives the post-hook mean to ~542. Early J-Lens
choice mass is tiny; no robust J-space parity inference follows. These are
results for a new short-label Gemma protocol, not a replication of the Qwen
mechanistic result. Separate capability checks and a valid `knows` parity
control must precede further access-claim or ISM mediation tests.

The legacy Qwen2.5-32B result is a cross-task discrepancy: the model reports
ignorance in one fresh conversation yet may use code parity in another. It does
not prove a separate internal agent or full-code access. The port to Gemma is
an unrun hypothesis test; its code-absent control and opposite-parity pairs are
essential. Paired parity effects cancel fixed procedure/word-order biases;
choice mass and a `knows` positive control gate interpretation.

Current public lens inventory lists J-Lens for exact Gemma-3-4B-IT, but no
J++/R checkpoint for it. Gemma Scope 2 publishes a broad SAE suite for
Gemma-3-12B-IT; Neuronpedia documents J++ for its online 12B-IT chat, while
the downloadable inventory checked here lists Gemma-3-12B-PT J++/R files.
Confirm a downloadable IT fit before integration. The inventory lists J/J++/R
for Qwen3.5-4B, but official Qwen-Scope collection does not list an SAE suite
for that exact checkpoint; a separate community L18 TopK SAE exists. Never
transfer SAE feature IDs or lens matrices across models/checkpoints.

Better joint-coverage candidate: Llama-3.1-8B-Instruct has downloadable J,
J++ and R lenses in the same inventory, and a published per-layer residual
BatchTopK SAE collection (`andyrdt/saes-llama-3.1-8b-instruct`, four k settings
per layer). The paper also describes an ISM-like analogue for this model.
Validate exact weight revision and SAE hook semantics before transferring the
experiment; its feature indices are unrelated to Gemma's 133/4360.

Sources: https://huggingface.co/neuronpedia/jacobian-lens/blob/main/README.md ;
https://docs.neuronpedia.org/jacobian-lens ;
https://huggingface.co/google/gemma-scope-2-12b-it ;
https://huggingface.co/collections/Qwen/qwen-scope ;
https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk .
Additional SAE: https://huggingface.co/andyrdt/saes-llama-3.1-8b-instruct .

## ISM steering provenance — verified 2026-11-04

Primary source: the local *Many Are My Names* PDF, §2.1, §2.4.1, §3.3.1, Table 1.
ISM is described as a distributed, potentially graded generation regime. Features
133 and 4360 at layer 9 are components associated with it, not an exhaustive or
unique representation of perspective. Feature 4360 has an onset/switch-like
pattern; 133 is more involved in sustained generation. These are the authors'
interpretations and findings, not results reproduced in this project.

The paper uses GemmaScope 2 JumpReLU SAEs, width 65k, medium L0. Its qualitative
steering formula is `x += alpha * m_f * W_dec[f]`, where `m_f` is the feature's
corpus peak; interventions apply to all processed token positions (§2.4.1).
Table 1's mean sample maxima are not the same as that corpus peak.

Verified published artifacts:

- Model: `google/gemma-3-4b-it`, current revision `093f9f388b31de276ce2de164bdc2081324b9767`.
- SAE: `google/gemma-scope-2-4b-it`, revision `3e94b68be95290aada5b7525cf431d3040f81bb1`.
- Path: `resid_post/layer_9_width_65k_l0_medium/params.safetensors` (~1.34 GB).
- Published config specifies width 65536, L0 53, `jump_relu`, no affine connection,
  model `google/gemma-3-4b-it`, and `model.layers.9.output` for input/output hooks.
- The SAELens Gemma 3 loader maps `w_dec` directly to `W_dec`, with no activation
  normalization and no decoder bias subtraction on encoder input. This supports
  loading original decoder rows directly for additive steering; it does not require
  installing a full SAE training framework.
- In locally installed Transformers 4.57.3, the multimodal wrapper's corresponding
  text-layer path is `model.language_model.layers.9`. The output is a tuple whose
  first item is the post-block residual stream. Layer indices are zero-based.

The current `ism` experiment uses raw coefficients; the paper's corpus calibration
and exact original revision remain unverified. Its synthetic smoke is an engineering
test only. See the first pretrained probe below for subsequent real results.

Access: the model page requires accepting Gemma conditions while logged into Hugging
Face and says requests are processed immediately. The API labels the model gated;
we have not tested this user's access. The SAE repository API reports `gated: false`
and the model card labels its license CC-BY-4.0.

Sources:
- https://huggingface.co/google/gemma-3-4b-it
- https://huggingface.co/google/gemma-scope-2-4b-it
- https://huggingface.co/google/gemma-scope-2-4b-it/blob/3e94b68be95290aada5b7525cf431d3040f81bb1/resid_post/layer_9_width_65k_l0_medium/config.json
- https://github.com/decoderesearch/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py

## Replication/lens provenance update — 2026-11-06

Paper §2.1 identifies 34 transformer blocks; saved public J-Lens metadata lists
fitted source layers 0–32. Block 33 is the direct final residual/vocabulary readout.
The upstream J-Lens README independently describes the final row as actual output.
Our previously recorded J-Lens revision is now the explicit CLI default. The
public fit's model revision is still not established; matching model name/width
does not prove full compatibility.

Rechecked public neuronpedia/jacobian-lens (exact Gemma path only `jlens`),
camilablank/workspace-lenses (Gemma 27B but not 4B), and koayon/jpp-lenses (no
Gemma-3-4B folder). No fitted R/J++ for this exact model was found in those repos;
this is not a claim that none exists anywhere. Their current source pages:
https://huggingface.co/neuronpedia/jacobian-lens/tree/main/gemma-3-4b-it
https://huggingface.co/camilablank/workspace-lenses/tree/main
https://huggingface.co/koayon/jpp-lenses/tree/main
Upstream lens semantics: https://github.com/anthropics/jacobian-lens

The paper's Table 1 uses sample maxima and fraction of samples firing at least
once, distinct from our generated-token firing fractions. Calibration code now
measures reference-set peaks on independent authored prompts, explicitly not the
authors' corpus. It does not complete exact numerical replication. New post-SAE
measurements and fixed-baseline-continuation controls are locally validated only.

## ISM identification, probes, and workspace lenses — verified 2026-11-06

The SAE did not supply semantic labels. The authors selected unlabeled features by
activation density, condition selectivity, entropy, and paraphrase/continuation
robustness, then interpreted them using causal steering and manual/LLM-judged output
effects. Gemma L9 features 4360 and 133 were labeled as ISM-associated because
positive steering repeatedly increased immersed character/narrative generation and
negative steering restored detached Assistant behavior. Natural activation patterns
suggest 4360 is an onset/launch gate and 133 a sustained speech carrier. This is an
operational interpretation, not evidence that either feature alone is ISM or that the
model undergoes a subjective perspective change.

Table 6 publishes three exact smoke-test prompts (Assistant identity, Lyra roleplay,
and last-robot story) and uses calibrated steering `alpha = +/-2`. The attached PDF
does not appear to publish the full 12-prompt fixed probe suite. Table 7 publishes six
emotional prompts that naturally elicit feature 133 in Gemma's default Assistant
setting. Exact paper-scale steering requires the full SAE encoder and corpus peak
`m_f`; raw coefficients in the current project are not numerically equivalent.

The authors also found a cross-model analogue in Llama-3.1-8B-Instruct: layer 15,
feature 101460 from a 131,072-wide residual-stream SAE. The paper describes a
JumpReLU suite, but the candidate public trainer configuration inspected later
says BatchTopK; exact trainer/inference mapping remains to be verified (see
`context/CONFIDENCE_ENTROPY.md`). The authors report a similar
gate-like activation and causal steering effect. The study remains non-exhaustive and
does not establish that all models use the same mechanism.

Public fitted J-Lenses exist for both exact study models:

- `gemma-3-4b-it/jlens/Salesforce-wikitext`
- `llama3.1-8b-it/jlens/Salesforce-wikitext`

No fitted R-Lens or J++ artifact for either exact model was found in the current
public repositories checked. Public paired J/R lenses include Gemma-3-27B-IT and
Qwen3.5-4B; public J++ lenses include smaller Qwen variants. Moving to those models
would require rediscovering and causally validating an ISM analogue. Therefore the
lowest-confound first experiment is Gemma-3-4B-IT + features 4360/133 + its public
J-Lens, followed by Llama-3.1-8B-Instruct + feature 101460 + its public J-Lens.

Sources:
- https://transformer-circuits.pub/2026/workspace/
- https://github.com/anthropics/jacobian-lens
- https://huggingface.co/neuronpedia/jacobian-lens
- https://huggingface.co/camilablank/workspace-lenses
- https://github.com/safety-research/jpp_lens
- https://huggingface.co/koayon/jpp-lenses

## First pretrained Gemma probe — observed 2026-11-06

Artifact: `outputs/ism/20261106T210928Z-dcc34ec7/`. Completed CUDA/BF16 run
of three Table 6 prompts, steering L9 feature 133 with raw coefficients -50,
0, +50. Baseline and zero-hook controls match exactly for token IDs, selected
SAE activation traces, and J-Lens readouts across all three prompts. Each run
used greedy decoding with at most 96 new tokens and one seed.

In baseline traces, feature 4360 fires at Roleplay boundary positions 36
`<start_of_turn>` and 38 newline (and one later generated newline), and at
Story boundary position 22 newline. It never fires in the Assistant prompt or
response. Feature 133 fires on 50/95 processed generated Roleplay tokens and
36/95 Story tokens; it fires on 0/68 processed Assistant response tokens.
Observed feature-133 peaks are 493.6 for Roleplay and 160.2 for Story. These
are sample peaks, not the paper's corpus peak calibration.

Raw +/-50 steering has modest behavioral effects: Assistant -50 matches
baseline, Assistant +50 adds an emoji, and both Roleplay/Story polarities
change narrative details while preserving the same broad mode. The negative
condition does not visibly restore detached Assistant framing here. J-Lens
top-token lists at three sampled positions show small rank/content shifts,
without a clear interpreted mode transition. This is inconclusive for the
paper's causal claim because the strength is uncalibrated, the sample has only
three prompts/one seed, and generation is truncated at 96 tokens. The trace
hook records L9 residuals before the current position's steering addition;
its SAE value is not a measurement of the injected feature contribution.

## Broader assessment prepared — 2026-11-06

The PDF explicitly numbers 23 Gemma SAE features across layers 9, 17, and 22.
`core/paper_features.py` records their IDs, labels, and table/appendix provenance.
These include the two ISM-associated features, all 15 Table 9 Assistant-associated
features, three Table 5 examples, and three Appendix D narrative examples. Other
discussed features lack published IDs; this inventory is not the full 529-feature
filtered set. Added the six Table 7 emotional prompts to the three Table 6 probes.

The initial real run now has a generated report, activation-frequency heatmap,
baseline timelines, and a Codex-authored interpretation. It still only
steered 133. New code for separate/joint 4360 steering and downstream monitoring has
been checked with synthetic data only. Future J readouts capture actual post-hook
residuals, correcting the old float32 reconstruction approximation. Free-generation
activation differences after text diverges are confounded by different continuations;
shared prompt positions offer a cleaner comparison. No new scientific finding is
claimed from the implementation changes.

## Expanded pretrained run — observed 2026-11-06

Artifact: `outputs/ism/20261106T214058Z-26769b9d/`. Completed 72 conditions
(nine prompts × baseline/zero + six raw ±50 interventions), CUDA/BF16, 96-token
cap, greedy, no random controls. Recorded all 23 features. All nine baseline/zero
pairs match exactly in token IDs, SAE traces, and J readouts. Elapsed ~360 seconds
including initialization. Initial analysis saw an old 34-completion snapshot;
final files arrived during analysis, and final report/numerical findings use all 72.

Most specific effect: playfulness baseline opens with digital-spectacles stage
directions and 133 fires on 13/95 processed generated tokens. Either negative
feature alone retains stage directions with 8/95; joint negative removes the
opening direction and yields 0/95. This is one prompt/strength, not proof of
synergy (summed direction has a different norm) or a broad perspective switch.
Positive 133/joint steering on envy introduces caffeine/packing/lifting claims;
negative replies instead emphasize information processing or programming.

In emotional baselines, all 17 (disgust), 4 (envy), and 13 (playfulness)
133-active generated positions also activate L17:738 at the same token positions.
This suggests overlapping labeled patterns, not mutually exclusive Assistant and
immersive behavior. It does not validate the labels or establish a literal AAC.

On identical final prompt positions, joint -50/+50 changes L22:1022 by
+62.28/-37.59 (Lyra) and +86.66/-55.50 (robot). Robot 4360 +50 changes it by
-42.44 while leaving output token IDs identical to baseline. These are downstream
effects before reply divergence, not evidence of a specific mediating circuit.
Across 54 nonzero interventions, 51 change token IDs; J-Lens last-prompt pre/post
top-10 overlap is 8–10/10. Lyra/story retain broad modes under negative steering.
No general mode switch or subjective perspective is established. Free-generation
133 activity can increase under negative steering (robot: 36/95 to 57/95), where
changed context and the pre-current-addition measurement must be considered.

Next: replicate the playful effect with paraphrases, norm-matched controls and
calibration; use matched-prefix/teacher-forced comparisons. Report and numerical
`findings.json` provide the evidence; no additional GPU job was launched by agent.

## Reference-calibrated Table 6 run — reviewed 2026-11-06 local / Nov 7 UTC

`outputs/ism/20261107T014211Z-8c44c965/`: completed 16 reference generations and
42 evaluations (3 Table 6 prompts × 14 controls/interventions), greedy, 256-token
cap, CUDA/BF16, 8m10s total. Independent reference peaks are 395.3448 (133) and
119.4049 (4360), giving coefficients ±790.69/±238.81 at alpha ±2. They are proxy
reference peaks, not the authors' corpus peaks.

3/3 zero pairs agree exactly in token IDs, SAE traces, lens readouts, and fixed-
continuation measurements. All 36 nonzero upstream block-8 last-prompt readouts
match baseline. Fixed-text pre-addition 133/4360 traces are identical across
conditions. Negative 133/joint makes fixed-text mean post-addition 133 zero in
all three prompts; positive steering gives ~563–687. In free replies, negative
133 still permits theatrical Lyra behavior and positive 133 still permits an AI
identity introduction: one measured latent is not the whole immersive regime.

Functional partial replication: negative 133/joint adds explicit Assistant story
preambles; joint negative Lyra drops the theatrical opening and asks structured
helper questions, while 133/joint positive favors literary narration. 4360 alone
is weaker; its negative robot output lacks the published preamble effect. Some
downstream J-Lens readouts align with that contrast on identical prompt tokens
(robot joint negative: block17 backstory/storytelling, block29 Okay/headings;
positive: whispers/unsettling, Rust/Dust). These are illustrative vocabulary
rankings, not a validated perspective axis or geometric analysis.

All 12 random replies at 133/joint-matched norms show repeated loops, while the
six lower-norm 4360-matched replies remain coherent. Repeated 8-token-window
fraction is .647–.968 in those 12 versus 0–.008 in 18 SAE interventions. This
descriptive metric and Codex's textual inspection suggest that high-norm random
controls measure generic damage rather than a clean behavioral comparison.
One random direction per group is not a null distribution. Next: smaller
strengths, more control directions and paraphrases, with coherence scored
separately from immersion, before broader claims or a larger Table 7 run.

Evidence: run-local `review.md`, `review_metrics.json`, `analyze_review.py` and two
supplementary figures. Review is Codex-authored after inspecting artifacts, not
independent human review. Auto `report.md` is a Python template with no LLM call.

### Follow-up inspection of monitored SAEs and J readouts

The reviewed analysis is now the canonical `report.md`; `review.md` redirects
there and the old template report is archived. On identical teacher-forced Lyra
baseline text, mean L9:440 post-addition (narrative) is 20.82 baseline,
1.95 joint-negative and 105.01 joint-positive. At downstream L17, feature 972
(asking whether there are questions) is 9.88 / 21.12 / 1.12 while feature 738
(responding to a request) is 23.23 / 6.44 / 8.16. This mixed pattern does not
support treating Assistant-associated features as a single uniformly moving core.
L9 changes can follow directly from encoder-direction overlap; L17 changes are
downstream but do not establish behavioral mediation. Free replies differ in text.

For the robot prompt's identical last prompt token, joint-negative L17 J-Lens
top words include backstory/storytelling, L29 includes Okay/headings, and the
direct final readout selects Okay; joint-positive includes whispers/unsettling at
L17, Rust/Dust at L29, and The at the final layer. The L8 upstream readout is
unchanged across all 36 nonzero conditions. This is a lexical lens observation
aligned with generated openings, not proof of J-Space geometry or perspective.
Old artifacts contain top-token lists, not transported vectors.

Updated run-local `report.md`, `sae_same_text.json`, `analyze_sae.py` and a
23-feature fixed-text figure document the evidence. New optional vector capture,
multiple controls and fixed-text J-Lens are locally tested; no pretrained results
for them exist yet.

## Smaller-strength refinement — observed 2026-11-07 UTC

Run `20261107T023612Z-ba1f9307`: completed 96 evaluations, three prompts
(Lyra/story/playfulness), joint 133+4360 at ±.25/.5/1, two isotropic and two random
SAE decoder controls (29413/3267), residual norms matched, greedy cap192. Same
reference peaks reused. Every condition has fixed-text SAE/J-Lens and vectors.
Duration18m21s. 3/3 zero pairs exactly match all recorded measurements; all 96
binary files present; recomputed 14,265 same-prefix comparison rows agree within
1e-5 scaled tolerance. All block-8 fixed-text vectors remain unchanged.

The playful opening disappears under 3/3 negative target and 12/12 negative
control conditions, and under 2/3 positive target conditions. This weakens the
opening-removal diagnostic's specificity. Control signs are arbitrary and the
same four directions are reused across strengths/prompts; counts are descriptive.
No severe repetitive-loop collapse observed (max8-token repeat fraction .02162),
though some controls are short or introduce AI disclaimers. Target Lyra and robot
continuations remain broadly roleplay/story at every tested strength; previous
±2 effects are outside this run's range. Coherent random controls can themselves
introduce robot-story Assistant preambles where the target at −1 does not.

At robot −1, post-addition 133 and 4360 are both zero at all 23 processed prompt
and 191 reply positions, but the free output remains a story without a preamble.
Zero is an SAE thresholded coefficient, not removal of every related signal;
these two coefficients alone do not define whether narrative generation occurs.
Positive target +.5/+1 on playfulness removes the opening despite high free
post-133 means73.28/242.24. On the fixed baseline, primary/narrative features show
clear dose responses (robot13325.96→0/337.13;44020.13→6.34/48.41 at −1/+1),
but same-layer changes can follow directly from encoder-direction overlap.

Actual J-vectors show graded within-layer displacement, often larger for controls.
At Lyra +1, despite matched450.54 residual norms, block9 J-delta L2 is1184.89
target vs1928.76/1200.56 isotropic and2266.91/1684.18 SAE controls. Raw norm
matching is not projected norm matching. No validated ISM subspace/axis follows
from these distance metrics. Playful final-token competition is nonmonotonic:
logit margin '(' minus 'Okay' is+2 baseline,−.75 at−.25,+2.25 at+.25,−1 at+.5.
This explains the literal greedy opening choice, not a full perspective transition.

Evidence and authored interpretation: run-local `report.md`, `report_metrics.json`,
`analyze_run.py` and three report figures. Suggested next step is a blinded rubric
on existing replies before new compute. No new GPU job launched for analysis.

### Interpretation clarification

Continued storytelling under negative steering is not itself a failed replication:
§3.3.1 explicitly describes a detached Assistant that still writes a story or
portrays a character. ISM is distributed and potentially graded. Our zero encoded
coefficients do not establish a full ablation. Earlier ±2 examples provide partial
qualitative replication; the latest smaller-strength sweep does not retest ±2.
The paper's §2.4.1 uses 12 probes (four per setting), ±2 corpus-peak scaling and
three trials (temperatures 0/1/1, distinct stochastic seeds). Our three-prompt,
greedy, proxy-calibrated sweep has different coverage. Next priorities are blind
multidimensional scoring and protocol/official SAE parity checks, then a bounded
confirmation with held-out prompts. See latest run's `summary.md`.

## Confirmation audit and review preparation

Rechecked steering and encoding conventions against the paper and official Gemma
Scope 2 tutorial; no obvious formula mismatch identified. Added a full-width
raw-checkpoint formula/loading audit; synthetic correct/corrupted weights test
passed, but no pretrained parity pass has run locally. Do not call this an
independent SAELens runtime comparison. Current code matches expected conventions;
author calibration/revision/probe fidelity remains unresolved.

Condition-aware reinspection: earlier robot joint −2 introduces a second-person
captain and mission structure, a task-format confound beyond the Assistant
preamble. Positive identity prompts remain ordinary AI introductions; negative
133 robot framing remains the stronger qualitative positive control. A shuffled
138-reply packet with blank rubric ratings exists; this is not completed blind
scoring. Proposed 12-probe/three-trial confirmation explicitly labels nine probes
as authored reconstructions. Six additional probes are reserved, not used for
strength selection. Protocol: `context/ISM_CONFIRMATION.md`.

### Pretrained full-width audit verified

Run `20261107T035003Z-035fea34` saved `sae_audit.json`: selected layer-9
encoding matches full-width official-formula encoding on 16 sampled first-prompt
residuals (max absolute error 5.3406e-5, rtol/atol 1e-4); decoder rows and
threshold masks match exactly. Raw SAE checkpoint SHA256
`87ac380e53526b0899bb83fca3ad14e39c7f2467e0a34b8b4336d72f439b06d7`.
This verifies our selected loading/formula check, not independent upstream-runtime
or author-checkpoint parity. The confirmation has started evaluating outputs.

## Completed 504-evaluation confirmation — condition-aware analysis

Run `20261107T035003Z-035fea34` completed in116m16s. Four probes per Assistant,
Roleplay and Story; temperatures0/1/1; separate133/4360/joint at±2 reference-alpha;
one random direction per group, baseline and zero. All36 zero pairs exactly agree.
All42 old matching greedy conditions replay exactly, so not fresh independent
confirmations. Nine authored probes and original-corpus calibration gaps remain.

133−2 and joint−2 each add the exact requested-story preface12/12; baseline,
positive targets and4360−2 have0/12. Inspected roleplay openings show coherent
conversational-versus-literary shifts, but a preface is not an ISM classifier.
Joint−2 can produce task-format changes and one looping clockmaker. Ordinary
Assistant replies resist positive immersive induction even with mean post133575
under133+2. Thus a robust contextual register intervention is better supported
than a universal mode switch. 4360 alone remains weaker.

Unsteered133 fires in0/12 Assistant and12/12 each Roleplay/Story replies, with
mean generated-token densities22.2%/43.0%. Boundary4360 fires0/12,9/12,12/12
respectively. This supports the carrier/onset description on selected probes,
not an exact corpus-statistic replication. In Story, L17:696 mean91.69→32.65/10.68
under133−2/joint−2, whereas L22:1022 rises0→15.98/22.05. Free-text confounding
prevents mediation claims; features need not form a single uniformly moving core.

Exploratory repeat fraction>.30:1/216 targets,79/216 random controls (50/72 at133
norm,29/72 joint norm,0/72 at4360 norm). Even nonlooping controls can be off-task;
this does not establish clean specificity.417/504 hit cap, all Story replies;
416 hit cap without EOS. J-Lens/fixed-text measurements deliberately absent.
Recommendation: held-out validation of133 alone with task/framing separated,
then matched-text lenses. No paid compute launched for analysis. Authored report,
metrics/script and three inspected figures in run directory; SHA256
`053f70aa0f724654e5d460803463b41ce30c2b0cc9fdec4e258fac973fd7c465`.

## Persona × ISM confidence design — 2026-11-07

The first-/third-person distinction is not equivalent to ISM: our positive story
replies can be third-person; negative133 gardener replies can use first-person.
Current evidence supports a narrative/character-presentation component, with a
stronger perspectival interpretation still unisolated. Proposed disambiguation:
independently vary grammatical person, literary style and enactment/framing.

Ported the legacy confidence experiment to test persona × steering interactions
on the same MCQs. Primary contrast subtracts steering's effect on default Assistant
from its effect on the persona, rather than equating lower entropy with deeper
immersion. Choice-format mass, accuracy/NLL/Brier and neutral-wording expert controls
address identifiable confounds. Midpoint expectations conditional on confidence
labels are not explicit self-reports when letter mass is low. Expert prompts still
say AI assistant; manipulation checks remain necessary. J-Lens reads decision
boundaries; intermediate logits are approximate, geometry is not an immersion score.
No pretrained confidence evidence yet. See `context/CONFIDENCE_ENTROPY.md`.

Other-model source check: paper AppendixF/G reports Llama-3.1-8B-Instruct L15
feature101460. Candidate public `andyrdt/saes-llama-3.1-8b-instruct` layer15 trainer1
config specifies BatchTopKTrainer/BatchTopKSAE, width131072, k64. This differs from
the paper's architecture description; do not assume index mapping or input/bias/
normalization conventions without verification. No Llama adapter implemented.
Sources: https://arxiv.org/html/2608.07852v2 ;
https://huggingface.co/andyrdt/saes-llama-3.1-8b-instruct/blob/main/resid_post_layer_15/trainer_1/config.json

## Attention/Q research setup — 2026-11-10

Q-composition supplies a mechanistic analogue for earlier representations shaping
later queries; it does not establish an AAC-specific feedback loop. Gemma queries
include RMS normalization and RoPE; agent-related features are potential inputs
among many. Residual->Q->attention->later residual is a dependency across depth,
not a same-layer circular computation. Post-block9 steering cannot change block9
attention on identical text; synthetic test verifies the analogous upstream rule.
No pretrained finding yet. Query geometry, routing and actual residual writes are
now recorded separately. Style Modulation Heads (arXiv:2603.13249), Function Vectors
(arXiv:2310.15213) and entity binding (arXiv:2310.17191) motivate causal follow-up,
not proof of an inner agent. Sources/design: `context/ATTENTION.md`.

## First pretrained confidence pilot — reviewed 2026-11-10

Run50cb3b26:10 questions ×10 personas ×6 conditions ×2 decision contexts. Positive
133+2 reduces normalized answer entropy0.0301→0.0175 while all100 conditional
choices favor A and only26/100 first tokens are answer letters (baseline100/100).
Negative133 preserves answer format, entropy0.0605, accuracy54/100 vs48/100baseline.
Stated confidence formatting survives in79/100negative and6/100positive cases.
Random controls are off-format. No free replies or attention were collected.
Both signs enlarge mean persona/default J-distance ratios at block22 (~1.39/1.60)
and final33 (1.142/1.165); not a validated immersion effect. Negative post-addition
4360 rises despite baselinezero; same-layer encoder overlap is a plausible cause.
Primary conclusion: strong positive steering invalidates the intended confidence
interpretation; next confidence work needs lower doses, label counterbalancing
and manipulation/format checks. Authored report/evidence under the run directory.

## First pretrained attention pilot — 2026-11-10

Runbabe43a9 completes18 Table6 comparisons with all34 attention blocks and fixed
baseline text. Exact zero/upstream controls pass. Target133 causes downstream
query/routing changes; random controls change routing more and degrade behavior.
L11H7 is a post-hoc candidate: positive Lyra attention to `ven`2.31→27.15%, positive
Story shifts mass from `story` toward `robot` plus BOS/other changes. This is not
an identified agent-query circuit: Q/K/V all receive intervention effects and
no Q-only causal test has run. Same-text Lyra L17:9729.88→19.82/.89 tracks negative
versus positive conversational/literary contrast without proving mediation.
Assistant remains AI under positive133; negative Lyra remains in character.
The quoted consciousness loop belongs to Story random+2, not positiveISM.
Report, verification script/metrics and inspected plot are saved in the run.

