# Research Decision Log

### 2026-11-11 — Treat speaker decoding as a prompt marker

The completed three-speaker door pilot validates behavioral framing and a
decodable final-token Query-input speaker signal, but fixed code-use behavior
was unchanged across speaker modes and the classifier may read explicit
instruction wording. Do not call it proof of an internal first/third-person
mode. For the next mechanistic validation, cross speaker templates and
held-out situations while separately labeling character and observer beliefs;
capture attention-head outputs if using Zhu-style probes, and require causal
interventions with controls before claiming that a marker mediates decisions.

### 2026-11-11 — Match door task across three speaker frames

Use one character, Mara, and identical code, knowledge-attribution wording,
and question within each Assistant/roleplay/story triplet. Include free
responses to check whether the requested speaker perspective is followed;
fixed-letter door answers alone cannot show this. Record prompt-boundary
Query and attention at six layers with eager attention, keeping this opt-in
to avoid changing earlier Gemma-4 runs. Do not call attention differences an
agent boundary or a causal mechanism. A fresh paid run requires OPS approval.

### 2026-11-11 — Require captions on research plots

Every new research plot must explain axes, condition mappings, controls, and
limits in a caption inside the standalone image. Keep the raw-data scripts and reviewed reports aligned with this
rule. No experimental settings or raw outputs changed for the existing Gemma-4
plots.

### 2026-11-11 — Interpret Gemma-4 speaker candidates conservatively

Treat layer-30 SAE feature 32546 as a validated *marker candidate* for the
Assistant/third-person-Story prompt contrast, not as an ISM or AAC feature.
Do not promote viewpoint feature 34957 after it failed validation or infer a
perspective switch from literal wording differences. Prioritize held-out,
factorially crossed speaker/person/style tests with stronger causal controls
before using either candidate in agent-access interpretations. The authorized
door and speaker follow-up runs are complete; no further paid compute was
launched for this analysis.

### 2026-11-11 — Treat late door patch as causal but not parity-specific

The authorized Gemma-4 door run completed with strong late-layer donor-answer
redirection and exact self controls. Interpret this as causal sufficiency of a
whole final-position residual swap in this task. Do not claim a localized
parity circuit, first emergence at L50, or a separate agent. Next compare
code-token, probe-direction and K/V interventions on fresh families.

### 2026-11-11 — Split door mechanism and speaker-direction follow-ups

Use the existing `gemma4_suite` experiment with opt-in stages, so old baseline
parameters remain unchanged. Train parity probes only on code-present families
and evaluate on unseen families across both tasks and all crossed conditions.
Patch opposite-parity final residuals with same-family/same-condition donors
and self controls. For speaker direction discovery, split characters into train,
validation, and untouched steering test; freeze layer-30 contrast and SAE
candidates for both third-person story and viewpoint using train data only.
Call the controlled viewpoint manipulation
an ISM proxy until held-out causal evidence supports a closer identification.
Do not launch paid follow-ups without OPS approval.

### 2026-11-11 — Follow crossed behavior with probes and causal patching

The completed Gemma-4 suite shows a reproducible within-run behavioral
access/attribution split, but no demonstrated inner agent. Next validate it on
more held-out code families, then compare parity probes across conditions and
patch code representations. Do not interpret intermediate J top-token lists
as absent information or raw Query norm shifts as attention mediation. Keep
Gemma-4 ISM direction discovery separate from this baseline until a candidate
is found and tested with held-out prompts and causal controls.

### 2026-11-11 — Gemma-4 baseline before model-specific ISM steering

Use the successful Gemma-4 door screen as a capability positive control, then
run fresh crossed door, speaker, and persona baselines with exact-checkpoint
SAE/J-Lens audits. Do not reuse Gemma-3 feature IDs or a base-model J++ lens.
Select Gemma-4 steering directions only after held-out feature discovery and
reconstruction validation. No pretrained suite run has occurred yet.

Record significant decisions, including their
rationale and alternatives considered.

## Decisions

### 2026-11-11 — Advance Gemma-4-31B-IT after a strong door capability screen

The user asked to try Gemma 4 on the existing 96 GB pod. The completed,
thinking-disabled baseline got 30/32 combined decisions under both wordings,
with one code variant failing both rule orders. Keep this candidate for a
held-out positive-control validation and then the crossed attribution design.
Do not infer information separation from this baseline alone. The all-layer
SAE dataset's weights are now downloadable, and a standard J-Lens exists for
the exact IT checkpoint. Audit SAE hook/reconstruction and lens behavior
before interpreting or steering features; the published J++ lens is for base.

### 2026-11-11 — Treat Gemma-3-27B as a promising, unvalidated door candidate

The user authorized a 128-case behavior-only screen on the newly connected
RTX PRO 6000, result pull, and report. Gemma-3-27B got 28/32 with the explicit
combined prompt on exactly the Qwen validation cases, but standalone parity
was only 10/16 and both rule orders failed for two even codes. Preserve this
baseline without inferring ISM or access separation. Before crossed
attribution or mechanistic intervention, validate the positive control on
held-out code families and refine the code-absent supplied-parity mapping
prompt. SAE and lens integration for this checkpoint is a separate step.

### 2026-11-10 — Retain explicit Qwen wording as a candidate, not a validated gate

Fresh matched comparison favored the explicit instrumental prompt (25/32 vs
19/32), but standalone parity and combined choice still fail too often.
Preserve the original wording and the exact run. Do not treat a crossed
attributed-knowledge result as an access dissociation until its knows/code-
present positive control is stronger. The Qwen crossed generator currently
uses original wording; switching it would require an explicit code parameter
and a logged protocol change.

### 2026-11-10 — Hold crossed Qwen interpretation after capability pilot

The user authorized the bounded Qwen3.5-27B run, result pull, and analysis.
The completed baseline capability pilot shows real parity sensitivity but only
11/16 correct combined decisions, with a strong rule-order asymmetry. Keep the
crossed code-presence × attributed-knowledge path available, but first validate
a clearer combined-task wording on fresh paired codes. Do not call a future
knows/doesn't-know contrast an access split while the positive control is this
fragile. Qwen SAE feature IDs remain unlabeled for ISM; no steering claim.

### 2026-11-10 — Diagnose the instrumental positive control before probes

Add a separate baseline-only capability design with last-digit, parity,
supplied-parity mapping, and combined instrumental tasks. Keep the original
and crossed designs unchanged. This is a short way to locate the failing
operation in the first Gemma pilot before interpreting attribution contrasts
or spending GPU time on probes and steering. Do not treat synthetic smoke as
behavioral evidence or launch paid compute without separate authorization.


### 2026-11-10 — Cross prompt availability with attributed knowledge

Keep the original door-code protocol as the default and expose a separate
`--design crossed` option. The code is present/absent in the system record;
the user independently asserts that the Assistant knows/does not know it.
This avoids silently changing the prior pilot and makes the prompt access
versus attributed knowledge comparison explicit. The absent/knows cell is an
intentional inconsistent claim, not proof of a model/agent barrier. Do not
run paid compute until the instrumental positive control is repaired and
launch is separately authorized under `context/OPS.md`.

### 2026-11-10 — Gate further door-code interpretation on a valid positive control

The approved Gemma pilot completed, was pulled with separate user permission,
and received an agent-authored report. Even when `knows` retrieved every code,
the parity task ignored opposite-parity changes within each pair. Feature133
±2 also broke answer format. Do not interpret N→Y under negative steering as
information access or plan immediate probe/patch claims from this run. Use a
separate capability diagnostic and fresh held-out code pairs to establish the
instrumental measure; then try smaller and last-token steering with random
controls. The existing run parameters/results remain fixed; no new paid launch
is authorized by this decision.

### 2026-11-10 — Port the door-code experiment as a bounded Gemma pilot

Keep the legacy code as read-only reference. Create `agent_vs_model` with its own
prompt generator and the current Gemma/SAE/J-Lens/attention helpers. Retain paired
opposite-parity codes, counterbalanced mappings and rule order, a code-absent
control and separate fresh task contexts. Put the code in the system turn to
match the user's described Qwen follow-up. Use simple single-letter decision
answers and save actual short generations; this changes the legacy JSON protocol
and must be interpreted as a new version. Start with four families and 133±2
reference-alpha, baseline and zero; make random controls and 4360 selectable.
Expose `last` steering scope as a follow-up to isolate answer-boundary effects
from all-token changes to the code representation.
Treat linear probes and code-token residual patching as the next mechanistic
stage after positive-control validation, not as a claim of replication. No paid
run is authorized by code preparation.

### 2026-11-07 — Port confidence experiment independently of legacy

User requested persona × ISM entropy experiments with J-Lens and clarified that
legacy_code is temporary unrelated-project reference. Migrated only relevant
150-question cache/persona prompts; new runtime depends on current core/, not legacy.
Preserve original expert prompts and, by delegated judgment, add three variants
without explicit confidence claims. Keep cautious/bold manipulations intact.
Use current validated Gemma stack initially; document change from legacy Llama.
Prioritize paired interaction/format-mass/correctness checks over interpreting
entropy as confidence. Default cost-limited pilot scores two independent contexts;
all four original generated contexts remain optional. Enable boundary J-Lens and
vectors by default; no pretrained run launched. Exact setup is documented before
any paid approval. Added strictly matched-config failed-run resume in shared runs.

### 2026-11-06 — Prefer feature 133 for held-out task validation

Reviewed the completed 504-run confirmation and authored report/metrics/figures.
Treat the repeated story-framing and roleplay-register effects as stronger partial
functional replication, not a binary ISM switch or perspective proof. Prefer133
alone for follow-up: joint-negative has task/looping confounds and4360 is weaker.
Keep damaged-control and truncation limitations explicit. New held-out work is a
recommendation, not authorization; no paid run launched in this analysis.

### 2026-11-06 — Authorized confirmation launched after SSH repair

Synced and launched `20261107T035003Z-035fea34` with the exact prepared command;
verified tmux RUNNING and CUDA/BF16 initialization. Startup is not completion or
a successful pretrained audit. Corrected `.gitignore` to exclude `legacy_code/`
after interrupting an unrelated-data sync; no files deleted or parameters changed.

### 2026-11-06 — Confirmation launch explicitly authorized

After initially declining, the user explicitly requested the prepared run and
fresh sync following a pod restart. This authorizes the documented 504-evaluation
confirmation plus calibration/audit, not additional sweeps. Saved SSH endpoint
resets connections; requested refreshed connection details. No paid job started.

### 2026-11-06 — Prepare behavioral confirmation before further J-Space runs

At the user's request to advance ISM replication, prepared a source-labeled
12-probe reconstruction, reserved six held-out probes, and added dry planning,
full-width SAE formula/loading audit and condition-hidden review exports in core/.
Keep existing defaults; proposed run explicitly uses ±2 proxy reference-alpha,
temperatures 0/1/1, separate/joint targets and one random control per group.
Disable lenses/fixed passes in this proposed 504-evaluation behavioral run to
focus cost on repeatability. New compute is still subject to OPS confirmation;
no launch performed. Do not describe prepared review packets as completed ratings.

### 2026-11-06 — Validate framing rather than disappearance of narrative

Added a medium-length run summary and clarified the report against §3.3.1:
continued storytelling is compatible with negative ISM. Keep partial qualitative
replication separate from a validated intervention. Prefer protocol/loader audit
and blinded cross-setting behavioral scoring before a stronger confirmation;
no experimental parameters changed and no new compute launched.

### 2026-11-06 — Interpret refinement against its controls

The user supplied completed run `20261107T023612Z-ba1f9307` for analysis. Wrote
an agent-authored report with recomputable same-prefix vector checks and plots.
Treat stage-direction removal as a literal output marker, not a sufficient ISM
classifier: all sampled negative controls also remove it on playfulness. Zero
thresholded SAE coefficients do not imply removal of an entire narrative regime.
Residual-norm matching does not match projected J norm. Recommend a blinded
behavioral rubric on existing data before new compute; no new run was launched
or experimental parameter changed during reporting.

### 2026-11-06 — Agent-authored reports and matched-text geometry

Superseding the preceding report-template decision at the user's request: runs
generate deterministic `measurements.md`/JSON/plots only; a reviewing agent writes
the canonical `report.md` after inspecting evidence. `core.reporting` never
creates or overwrites that report. `AGENTS.md` and `context/REPORTING.md` define
the workflow and source hash; this is agent-authored, not independent human review.
Preserved the prior automated report as `legacy_automated_report.md`, promoted the
existing Codex review to `report.md`, and kept `review.md` as a redirect.
`RESULTS_EXCLUDE=report.md` protects local reports from old pod-generated files
on default pulls; explicit subpath pulls bypass the shared-script exclusion.

Added sampled unnormalized J-projected vector capture for both free and fixed
continuations; compare cosine/L2 only within the same layer/phase and matching
token prefix. Add multiple independently seeded isotropic and randomly sampled
SAE decoder controls, with norms/IDs/seeds saved. The next explicitly scoped
96-completion joint-strength sweep is documented in `context/ISM_REFINEMENTS.md`.
This setup and local tests do not authorize or constitute a new paid pod run.

### 2026-11-06 — Report authorship and reviewed interpretation

Clarify that `core/runs.py` calls the Python template in `core/reporting.py` at
run end; no LLM reviews results on the pod. Auto reports now explicitly state
this provenance and link a run-local `review.md` when available. Preserve
`review.md` as a separately Codex-authored analysis that the generator never
overwrites; `interpretation.md` is a short, attributed version included in the
auto report. Neither counts as independent human review. Corrected earlier
documentation that called Codex-authored interpretations human-reviewed.

Reviewed the approved bounded run after completion. Keep high-norm random-control
collapse visible in conclusions; do not claim clean immersion specificity merely
because SAE directions are coherent and random directions repeat. No additional
compute is authorized by this review or the bounded-run approval.

### 2026-11-06 — Bounded ISM replication and multi-layer lenses

User requested the next replication step and more lens layers. Retain experiment
`ism` and existing raw defaults. Document the staged functional-replication target
in `context/ISM_REPLICATION.md`: natural signatures, Table 6 directional effects,
Table 7 suppression, matched-prefix and random controls before new hypotheses.

Add `reference-alpha` using an independent authored 16-prompt reference set,
50-token unsteered generations, hashed provenance and checkpoint validation.
Label this as a proxy; do not claim the exact paper corpus peaks or corpus density
statistics. `paper-alpha` still requires explicitly supplied corpus peaks.
Default first bounded protocol is documented, not silently enabled: 16 reference
generations + 42 evaluated Table 6 completions at ±2 reference-alpha, max 256 new
tokens, random controls, fixed-baseline-continuation SAE assessment and J-Lens.
Temperature-0/1/1 trials are optional and explicitly seed-indexed.

J-Lens now defaults to blocks 8, 9, 10, 17, 22, 29, 32 plus direct final block 33;
the public checkpoint covers 0–32, and the language model has 34 blocks. Load the
single checkpoint once at the previous recorded revision. Sample prompt boundary,
first processed reply token, stride-16 reply positions and last processed position.
No additional SAE is required at a lens-only layer. Record actual post-addition SAE
values and mean/peak/active-mean summaries. Fixed-continuation measurements currently
cover SAE features only; full transported-vector geometry and R/J++ fitting remain
deferred. New code is locally tested; a paid launch still requires OPS confirmation.

The user subsequently explicitly approved the bounded launch. Started
`20261107T014211Z-8c44c965` through `run-experiment` on the existing pod, with the
exact documented first-run flags. Verified RUNNING and reference calibration
started on CUDA/BF16. This authorization covers this bounded run, not the larger
252-completion follow-up. No completion or new findings verified at launch.

### 2026-11-06 — One results tree and broader ISM assessment

User explicitly authorized modifying the shared workflow scripts. Use `outputs/`
on both machines and opt into `PULL_LAYOUT=mirror`; leave other projects' defaults
unchanged. Moved existing local `experiments/results/` runs into `outputs/` without
copying or deleting artifacts. Old remote runs are not migrated or deleted.

Interpret the user's `4630` as `4360`, consistent with the paper and prior discussion.
New default: separate 133, separate 4360, joint sum, each at unchanged raw ±50,
plus one baseline and zero-hook per prompt; nine Table 6/7 prompts, 96 new tokens,
greedy decoding. Monitor the 23 explicitly numbered Gemma features at layers
9/17/22, including all 15 Table 9 Assistant-associated features. Do not claim this
covers all 529 filtered features. Optional seeded norm-matched random directions
provide exploratory controls; no corpus calibration is inferred from this run.

Keep reusable capture/generation in `core/assessment.py`, selected encoding in
`core/sae.py`, provenance catalog in `core/paper_features.py`, and CPU-only reports
in `core/reporting.py`. Reports use saved data, include plots and full replies,
and accept a separately authored `interpretation.md` with authorship identified. J-Lens now captures
actual post-steering residuals; it still reads only layer 9 at three positions.
No broader pretrained run or paid compute was launched during implementation.

### 2026-11-03 — Replication roadmap

At the user's request, populated the existing `context/PLAN.md` with the discussed replication phases. Use this as the canonical plan rather than creating a duplicate root-level file. The working sequence is provenance checks, one pretrained SAE on the paper's 4B model, validated interventions, then focused replication. Larger models, custom SAE training, and new causal experiments remain conditional proposals; no compute expenditure is authorized by this plan.

### 2026-11-04 — Minimal backbone and RunPod integration

User requested shared code in `/core` and explicitly approved the initial experiment name `smoke` and artifact layout `experiments/results/<experiment>/<run-id>/`. Use `run.py` plus a module registry and the existing universal workflow scripts without copying or modifying them. `.runpod.env` overrides only `RESULTS_SUBDIR`; the other required values already match shared defaults. The generic pull script places retrieved artifacts under local `outputs/`.

User confirms the pod includes PyTorch, but does not know the image. Leave PyTorch and the unverified model/SAE dependency stack out of requirements for now. The smoke check uses the standard library; the only pinned external dependency is boto3 for explicit optional R2 uploads. Model computation and long-run checkpoint/resume are deferred until actual experiments are implemented. No paid jobs or uploads were authorized or launched.

### 2026-11-04 — Quick ISM steering

User requested a minimal steerable ISM experiment and approved the name `ism`.
Keep the existing results layout. Implement fixed Gemma-3-4B-IT/layer-9 SAE
provenance with selectable features 133 and 4360. Load original decoder rows
directly from safetensors for additive steering; no SAE training or full encoder
is required for this iteration. The supported feature IDs are checkpoint-specific.

Use explicitly labeled raw coefficients (exploratory default -50/0/50), not the
paper's alpha values, because corpus-peak calibration has not been reproduced.
Always compare against baseline and a zero-hook control. Default to all processed
token positions, greedy generation, and reusable weights across prompt comparisons.
Synthetic CPU smoke uses a tiny random model and never downloads weights.

Superseding the initial dependency deferral above, pin Transformers 4.57.3,
huggingface-hub 0.36.0, and safetensors 0.7.0 to versions available in the local
tested environment. Continue to omit PyTorch from installation requirements;
pod compatibility and real pretrained inference remain to be checked.

### 2026-11-06 — Selected SAE activations and Gemma J-Lens

For the first J-Space pilot, keep the paper's Gemma-3-4B-IT model and its two
layer-9 ISM-associated features. Record only selected JumpReLU features per token
and use the public Gemma J-Lens at layer 9 on a few token positions. Default to
the three prompts printed in Table 6. Preserve raw-strength mode; expose
`paper-alpha` only with explicit per-feature corpus peaks, because Table 1 does
not supply those peaks. The initial real run failed in J readout because an old
pod copy passed `Gemma3TextModel` to code expecting the outer wrapper. The local
call now passes `Gemma3ForConditionalGeneration`, with a regression test through
generation and J readout. Real checkpoint behavior remains to be verified.

### 2026-11-10 — Bounded attention observation before causal sweeps

User requested reusable attention/query monitoring and relevant literature. Add
opt-in capture to existing experiments, preserving conditions/default behavior.
Use actual eager Gemma attention, sparse query positions and binary artifacts;
record prompt/fixed-text passes, explicitly exclude cached decode. Do not silently
switch pretrained backends. Default8 blocks; all34 are optional. Compare only
identical prefixes, verify exact zero tensors, and label exploratory selected
plots. Query-only patch primitive is tested; head discovery/held-out mediation
sweeps and K/V patches require later design. No paid compute launched.

### 2026-11-10 — Run inventory and Table6 attention preset

User requested clearer results organization/Pacific timestamps and prepared
Assistant/Roleplay/Story attention comparisons. Preserve old run paths for links
and sync identity; add outputs/README.md and per-run RUN_INFO.md. Future run names,
logs and readable status fields use America/Los_Angeles; retain UTC machine fields.
Future smoke runs go under outputs/_smoke/<experiment>, not research folders.
`--run-label` adds an optional descriptive slug. Inventory refresh is local and
explicit after pulls; shared RunPod scripts unchanged.

Use existing ism with `--preset attention-table6`, expanded from checked-in JSON
into recorded config:133 alone±2 reference-alpha, baseline/zero/random±,3 exact
Table6 prompts, greedy256, seed42,16 calibration generations, fixed text,all34
attention blocks plus J/SAE measurements.4360 monitored, not independently steered.
This is a discovery pilot; cross-setting prompt content is not controlled. No
paid launch requested/performed; command handed to user.

