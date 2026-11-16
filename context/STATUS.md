# Current Project Status

## New-session handoff clarified — 2026-11-16

Added `context/HANDOFF.md` as a concise index of the verified ISM, door,
speaker, and attention results and the unselected next research options.
`AGENTS.md` now points a new session to it; the old `PLAN.md` status line
was corrected because pretrained runs have completed. Confirmed that this
local project has no `.git` directory and `outputs/` is ignored by sync, so
the full local directory, including results, needs backup if moved to another
machine. No new experiment or analysis was run for this handoff.

## Reusable project context starter — 2026-11-11

Created `starter_repo/` with adaptable agent instructions and templates for
status, research, decisions, planning, operations, and reporting. Its README
explains how to copy and tailor the files for a new project.

## Gemma-4 door speaker attention run completed — 2026-11-11

The user approved the paid pilot, pull, and report. The RunPod job finished
192/192 prompts without hook errors; its 192 attention artifacts were pulled
to `outputs/gemma4_suite/2026-11-11_17-05-52_PST--gemma4-door-speaker-attention--a970a5c2`.
Codex wrote a research report, reproducible review metrics, and two visually
inspected figures with in-image captions. All three speaker modes produced
identical fixed door answers: with code present but Mara described as not
knowing, N self-report, UNKNOWN repetition, and correct L/R each occurred
12/12 across modes. Free responses followed the requested speaking form.
Query-input speaker decoding on held-out families was 192/192 at L10/20/30
with shared task types, but cross-task transfer was uneven. Descriptive
attention to the code field differed on repeat versus door questions, without
demonstrating causal mediation. `outputs.json` SHA256:
`c8b8c0e7e7c7b2f8b2c2046528d9594a4bb0e877d101c3f8bae5e0ed6505a8ee`.

## Gemma-4 door speaker and attention pilot prepared — 2026-11-11

Added opt-in `gemma4_suite --followup door-speaker-attention` with matched
Assistant, first-person Mara roleplay, and third-person Mara story prompts.
The pilot crosses code availability and attributed knowledge over two paired
families and four tasks, producing 192 prompt-boundary attention/Query
forwards and 192 short generations. Six layers save per-head raw and
normalized Query vectors, attention rows, and labeled span weights. Existing
door runs contain no attention weights, so a new pod run was needed. Local
design and synthetic hook tests and the dry plan passed; the approved
pretrained run subsequently completed as recorded above. Exact
command and limits: `context/GEMMA4_DOOR_SPEAKER_ATTENTION.md`.

## Plot captions and explanation refresh — 2026-11-11

Added a repository rule requiring self-contained captions in each research
plot. Regenerated the existing Gemma-4
door and speaker figures with captions and inspected both. Rewrote the
speaker report in simpler language; expanded the door report to explain its
plot and why its late final-position patch effect differs from the older
Qwen code-token patch sweep. Existing raw results and experimental parameters
were unchanged. Both review scripts reran and output hashes remained stable.
Following user feedback, captions are required inside plots only; duplicate
captions beneath report figures were removed. The speaker report now states
that steering acts on the last token of every forward step, shows the actual
wording changes, and distinguishes text-change counts from directional success.

## Gemma-4 speaker discovery run completed and reviewed — 2026-11-11

The second authorized follow-up completed and was pulled at
`outputs/gemma4_suite/2026-11-11_12-54-50_PST--gemma4-speaker-discovery--899aa160`.
Its agent-authored `report.md`, recomputable review metrics, and inspected figure
cover 96 discovery outputs and 128 held-out steering outputs. At layer 30,
train-derived Assistant/third-person-Story and viewpoint contrasts separated
4/4 respective validation pairs. Train-selected SAE feature 32546 transferred
for the Assistant/third-person-Story contrast (4/4); viewpoint feature 34957
failed validation (0/4). At the single conservative steering dose, candidate
directions did not switch held-out speaker framing; random controls changed
literal text more often. No Gemma-4 ISM feature has been established. SHA256:
`outputs.json` `367ad2d05f5293d27b83953e96580b6b0930e792e1a26c3b7d25915bc016ec62`;
`steering.json` `91ebd7438b0e6a0bbb805224f66317d175ee129b856e495605b79be06ccb93c1`.

## Gemma-4 door mechanisms run completed and reviewed — 2026-11-11

The authorized 16-family door run completed 256 baseline rows, 160
opposite-parity patches and 20 self-patch controls, was pulled, and received
an agent-authored report, recomputable review metrics and inspected figure at
`outputs/gemma4_suite/2026-11-11_12-51-53_PST--gemma4-door-mechanisms--b6b4b317`.
Output SHA256:
`eeed8c50f808727b61ef406d36b5d606d9e3b0bd00e75caca8c059052bd40024`.
With code present, instrumental accuracy was 29/32 under both knowledge
attributions, with the same three errors; reports were Y 32/32 versus N
32/32. With code absent, all instrumental answers were A. Held-out parity
probe AUROC on the instrumental task was high by layer 30 and transferred
across present-code attribution cells. Opposite-parity final-residual patches
at layers 10–30 barely changed donor-procedure probability, while layer 50
increased it by .636 (knows) and .563 (doesn't know) on average across 16
held-out cases per cell. Self-patch max choice-probability difference was
5.2e-8. These are task-specific information-flow results, not agenthood proof.
The approved speaker run subsequently completed and is reviewed above.

## Gemma-4 follow-up stages prepared — 2026-11-11

Added opt-in `gemma4_suite --followup door-mechanisms` and
`--followup speaker-discover` with shared residual capture, patching, and
ridge-probe helpers in `core/residual_experiments.py`. The door dry plan is
256 fresh-family baseline prompts, 160 opposite-parity patches and 20
self-patches across five layers. The speaker dry plan is 96 discovery prompts
and, if requested, 16 untouched test prompts × 8 arms (128 generations).
Speaker contrast and SAE candidates are selected on train characters, with
separate validation and steering-test characters. Seven focused local tests,
both dry plans, compilation and two synthetic CPU end-to-end smokes passed.
Both pretrained follow-ups have now completed and been reviewed; this section
records their original implementation checks and dry plans. Scope,
commands and limits: `context/GEMMA4_FOLLOWUPS.md`.

## Gemma-4 baseline suite completed and reviewed — 2026-11-11

The user approved launch and pull. The first attempt stopped before any result
row because the generic J-Lens path mixed FP32 residuals and BF16 matrices;
`core/jlens.py` now casts the residual to matrix dtype. The same run then
resumed and completed 195/195 at
`outputs/gemma4_suite/2026-11-11_12-22-40_PST--gemma4-baseline-suite--4ca7bb4d`.
Codex wrote `report.md` and recomputable `analyze_review.py` /
`review_metrics.json` after inspection. Output SHA256:
`5aaea5f1694927c1feecf9dc9857ea1d88462ea86b6a45cbfae6ae170cddcf93`.
Crossed door: present/knows 16/16 correct L/R, present/doesn't-know 15/16
despite 8/8 N self-reports and 8/8 UNKNOWN repetitions; absent/knows reported
Y 8/8 but repeated UNKNOWN 8/8 and chose A 16/16. SAE layer-30 mean relative
boundary reconstruction error .300; direct final-layer readout matched model
top token 195/195. Intermediate J top-token lists were uninformative for the
door task; confidence panel had a floor effect. No Gemma-4 ISM vector was
identified or steered. Next: held-out door families/probes/patching and
held-out Gemma-4 speaker-direction discovery with causal controls.

## Gemma-4-31B-IT baseline suite prepared — 2026-11-11

Added `gemma4_suite` for paper Table 6 and factorial speaker probes, fresh
crossed code-availability × attributed-knowledge trials, and a small persona
confidence panel. Shared Gemma-4 loader and TopK-64 SAE reader live in `core/`.
Optional layer-30 SAE boundary audit, exact-IT standard J-Lens at 30/45/52,
and raw Q projection norms at 22/30/45/52 are wired. The dry plan has 195
prompts, 195 scored forwards, and 155 greedy generations. It does not steer
an ISM feature or fit a lens. The code compiled, its dry plan passed, and the
existing Gemma-4 door test passed locally; pretrained integration is unverified
until a pod run. Exact scope and limitations: `context/GEMMA4_SUITE.md`.

## Gemma-4-31B-IT door screen reviewed — 2026-11-11

Added `--model gemma4` as a BF16, thinking-disabled 128-case matched capability
screen. Three focused local tests and dry plan passed. The first pod attempt
failed at response parsing before any row was saved; passing the exact prompt
prefix fixed it. The completed 128/128 run was pulled and has an agent-authored
report plus recomputable metrics at
`outputs/agent_vs_model/2026-11-11_11-53-44_PST--gemma4-door-screen-parser-fix--38b2ca55`.
Gemma 4 copied 16/16 digits, classified 16/16 parities, and got 30/32
instrumental decisions with both original and explicit wordings. One odd code
failed both wordings and rule orders. Supplied-parity mapping was 24/32 due to
eight off-format replies from two repeated code-absent prompt types. This is
a strong but not perfect positive control; no ISM/access experiment was run.
Output SHA256: `7142a7ec0d46de376d828c4b58a390886743f2fabc95889f358363d9926db6a8`.
The all-layer Gemma-4-31B-IT SAE dataset has 60 `.pt` weight files online; a
pod audit safely loaded layer 30. An exact-checkpoint standard J-Lens exists
at `solarkyle/jspace-lenses`; published J++ targets the base model. SAE hook
position and normalization still require functional validation.

## Gemma-3-27B door screen reviewed — 2026-11-11

Added `--model gemma27` as a baseline-only 128-row capability screen using the
same seed 1042 paired codes, original/explicit instrumental wording, short
greedy responses, and next-token choice probabilities as the Qwen validation.
No SAE, steering, lens, or attention passes were included. Local dry plan and
two focused tests passed. After the user connected a new RTX PRO 6000 pod,
the authorized run completed 128/128, was pulled, and received an agent-authored
report and recomputable review metrics at
`outputs/agent_vs_model/2026-11-11_11-42-43_PST--gemma27-door-screen--2cb23749`.
Gemma copied 16/16 final digits, classified 10/16 parities, mapped 28/32
supplied-parity prompts (four off-format, only eight unique prompts), and got
20/32 original and 28/32 explicit combined decisions. Matched Qwen explicit
was 25/32. Gemma's four explicit errors were two even codes missed under both
rule orders. The positive control is promising but still imperfect; no
attributed-access or ISM claim follows. Output SHA256:
`ea1bafd56f6c55ad8564315eff960f7ff50d8181787f85c29d03b99dd5b66b54`.

## Fresh Qwen door wording validation reviewed — 2026-11-10

The authorized eight-family run finished 128/128, was pulled, and has an
agent-authored report plus recomputable paired metrics at
`outputs/agent_vs_model/2026-11-10_16-25-18_PST--qwen35-door-wording-validation--1ecc792f`.
On identical fresh code/rule-order cases, original instrumental wording got
19/32 and explicit parity-then-rule wording 25/32; seven cases improved, one
worsened. Standalone parity was 13/16, digit copy 16/16, mapping 32/32
(only eight unique mapping prompts). The improved task remains imperfect; no
crossed attribution, code-absent, or ISM steering run followed. The explicit
variant is selectable with `--capability-wordings explicit`; the Qwen crossed
design still uses the original wording. Output SHA256:
`0e0337ddc35cf3c05d8be93973e04728e919ffd6c557428167331b199ab87dba`.

## Qwen3.5-27B capability pilot reviewed — 2026-11-10

The user-authorized Qwen capability run completed 48/48 on an A100 80GB and
was pulled. Codex authored `outputs/agent_vs_model/2026-11-10_16-10-47_PST--qwen35-door-capability--91e9b97a/report.md`
and recomputable review metrics. Last digit 8/8, parity 6/8, supplied mapping
16/16, combined instrumental 11/16; mean paired parity effect +0.392 over
eight code/rule-order pairs. The combined positive control remains imperfect,
so do not interpret a crossed attributed-knowledge run as an access split yet.
Qwen SAE, published R-Lens, and four full-attention/Q layers were recorded;
no Qwen ISM direction was identified or steered. Outputs SHA256:
`8a6ab0d0b7e0575c8f67639316f754276aaada8d8188ab35cfe55e2c16ddce03`.
Next: validate a clearer instrumental wording on fresh paired codes, then
run the crossed design if the positive control is reliable.

## Qwen3.5-27B door baseline port — 2026-11-10

Added a Qwen-specific baseline path to `agent_vs_model`, with capability and
crossed designs, non-thinking answer boundaries, Qwen-Scope TopK SAE monitoring
at blocks 16/32/48, published R-Lens readouts, and selected full-attention/Q
capture. Hybrid linear-attention blocks are explicitly excluded from softmax
attention measurements. This path cannot steer Gemma's feature 133. Upgraded
RunPod requirements to Transformers 5.21.0 / Hub 1.33.0 / safetensors 0.8.0
for Qwen3.5. Local dry plans and three focused Qwen tests passed. The user
explicitly authorized launch, pull, and analysis after preparation.


## Gemma door capability run reviewed — 2026-11-10

Reviewed completed `outputs/agent_vs_model/2026-11-10_15-53-46_PST--door-capability--7dab5efa`.
Codex authored its `report.md` and reproducible `analyze_capability.py` /
`review_metrics.json` after inspecting all 48 responses and exact prompts.
Final-digit copying passed 8/8, parity classification 5/8, supplied-parity
mapping 8/16 (every response R; only eight unique prompts), combined task
8/16 with paired parity effect 0.000014 over eight pairs. All 48 responses
obeyed one-character format and label choice mass was approximately 1.
No steering, J-Lens, or attention was used. The instrumental positive control
still fails; revise mapping wording on fresh families before crossed access
or ISM claims. Reviewed outputs SHA256:
`c7791f78ed9e39ee773d995be7ea80561d3a9712abf18ef0255bddc6144fdfc0`.


## Door-code capability diagnostic — 2026-11-10

Added optional `--design capability --baseline-only` to isolate the failed
instrumental positive control into final-digit, parity, supplied-parity mapping,
and combined-decision trials. Four families yield 48 decisions and 48 short
generations, with no steering arms, reference calibration, or new lens fit.
Strict labels and choice probabilities are recorded. Seven focused door tests
passed; the synthetic CPU end-to-end smoke completed after extending its
one-token mock tokenizer to digits. No pretrained run launched or scientific
result claimed. Next: if authorized, run the bounded Gemma diagnostic, then
choose prompt repair or Qwen32 comparison based on its positive controls.
Offline review of legacy Qwen32 v2 reports found that its preregistered
24-family `knows` positive control also narrowly failed (validation 0.52
paired parity effect / 79% most-likely accuracy; main 0.45 / 77%). This tempers
the earlier informal description of reliable instrumental performance; it is
not a new inference run. Details in `context/RESEARCH.md`.


## Crossed door-code access × attribution design — 2026-11-10

Added optional `--design crossed` to `agent_vs_model`, preserving the original
default. Four cells independently vary whether the code is in the system record
and whether the user claims the Assistant knows it. Trial rows include explicit
`code_available` and `attributed_knowledge` fields; all original tasks, steering,
SAE and J-Lens options remain. Six focused tests passed; dry plan reports 128
trials / 384 decision forwards / 512 generations for four families and four
steering conditions. Synthetic CPU end-to-end smoke completed. No pretrained
run launched. The existing Gemma instrumental positive control is still failed;
diagnose it before interpreting crossed cells. See `context/AGENT_VS_MODEL.md`.


## Speaker-factor and query-patch preparation — 2026-11-10

Prepared 24 matched Assistant/Roleplay/Story × grammatical person × prose-style
probes and four ISM presets. Added shared Q-only prefill patching, category-level
SAE summaries, token-span attention summaries and fixed-continuation likelihood
measurements. Synthetic tests exercise self-patch and candidate/control head
paths. No pretrained speaker-factor or Q-rescue run launched; see
`context/ATTENTION.md` for the exact scope and limits.

## Door-code attribution port — 2026-11-10

Created `agent_vs_model` outside `legacy_code`, with deterministic paired-code
design, four attribution conditions, three fresh tasks, crossed procedure mapping
and rule order. Integrated 133/4360 steering, selected SAE recording (including
the paper's 23-feature speaker panel), boundary J-Lens and optional attention
through shared core helpers. Added strict free-text
scoring, paired parity-effect summaries, baseline/zero assertions, dry planning,
synthetic smoke and a design guide (`context/AGENT_VS_MODEL.md`). This is a new
Gemma protocol, not a Qwen replication. No pretrained run launched. Linear
probes and residual patching are not in this first port. Check `knows` competence
and response format before interpreting any does-not-know dissociation.
All 52 offline tests passed after the port; a synthetic end-to-end run with
attention, J-Lens and paper-feature monitoring completed. The user approved the
bounded pretrained pilot; it synced and started on RunPod at 2:27:49 p.m. PST in
`outputs/agent_vs_model/2026-11-10_14-27-49_PST--door-code-pilot--08255c80`.
The pod log reports `Run completed` at 2:38:02 p.m. PST after all 128 trial
prompts. The user separately authorized `pull-results`; the run was pulled and
Codex authored `report.md` with recomputable `analyze_review.py`/
`review_metrics.json`. All 512 rows are present; 128 zero pairs match exactly.
Baseline `knows` repeated all 8 codes but instrumental parity effect was
~0 (8/16 correct from within-pair same-answer behavior), so the positive
control failed. `doesnt_know` reported N and answered A; `withholds` reported
Y but disclosed all 8 codes. Feature133 ±2 made all64 instrumental responses
per sign invalid format, with very low L/R/A mass. SAE features and J-Lens
were recorded but cannot rescue the failed behavioral measurement. No new
pretrained run authorized. Next: capability diagnostic, revise instrumental
wording on fresh pairs, then smaller/last-token ISM steering and controls.

Last updated: 2026-11-10 (local date)

## Current objective
Replicate selected findings from *Many Are My Names* on Gemma-3-4B-IT, then
choose a focused AAC/ISM research question. Roadmap: `context/PLAN.md`.

## Latest attention pilot reviewed — 2026-11-10

Run `outputs/ism/2026-11-10_13-12-36_PST--attention-table6--babe43a9` completed
18/18 comparisons,1:12:36–1:24:54 p.m. PST (~12m18s). Authored report.md and
recomputable analysis/metrics; new attention plot inspected. Terminal's repeated
“single, unified consciousness” is result17, Story random+2 seed1042, not133+2.
All6 attention zero pairs exact; blocks0–9 unchanged for all same-text interventions;
20,400 attention and3,240 J comparisons recomputed. No new inference.

Target133 preserves ordinary Assistant identity, shifts Lyra conversational versus
literary presentation, and negative Story adds the familiar Assistant preface.
Random controls degrade/loop. Downstream fixed-text JS means ~.024–.031 targets
versus~.067–.084 random. L11H7 exploratory candidate: Lyra+133 increases attention
to `ven` in elven2.31%→27.15%; Story+133 shifts `story`→`robot` among other changes.
Q changes are observed, not isolated mediation; K/V also change. Same-text Lyra
L17:972 means9.88/19.82/.89 baseline/negative/positive.12/18 cap-limited.
Next: held-out matched probes and targeted Q restoration with controls; no
head-specific patch run yet. Report interpretation is condition-aware agent review.

## Latest confidence results and prepared attention pilot — 2026-11-10

Reviewed pretrained confidence run `outputs/confidence_entropy/20261110T194939Z-50cb3b26`
(Nov10 11:49:39 a.m.–12:08:20 p.m. PST;18m41s).1200 decisions complete;200 exact
zero pairs including binaries;10,800 J comparisons and6,480 persona contrasts
recomputed. Authored report.md, reproducible analysis/metrics and two inspected
figures. Positive133+2 favors A conditionally100/100; only26/100 answer first tokens
are valid,6/100 confidence tokens. Lower entropy is confounded by format/label
collapse. Negative133 preserves100/100 answer formatting and increases conditional
accuracy48→54/100 on10 shared questions. Both signs enlarge some downstream
persona distances; not evidence of immersion. No attention captured in that run.

Prepared `run-experiment ism --preset attention-table6`:3 published settings,
18 evaluated greedy generations,18 fixed-text passes,16 reference generations;
133±2 plus baseline/zero/random±, all34 attention blocks,23 SAE features and
multi-layer J-Lens. No paid launch. See `context/ATTENTION_PILOT.md`.

New folders/log/status use readable Pacific timestamps (DST-aware); smoke runs
now go under `outputs/_smoke/<experiment>/`. Old paths preserved to avoid broken
links/duplicate pulls; `outputs/README.md` and per-run RUN_INFO.md distinguish
research/smoke/status with Pacific times. Refresh after pull using
`python -m core.run_index`.43 tests passed (6.768s), including DST/preset/counts;
explicit preset CPU smoke passed under the new smoke tree.

## Attention/Q monitoring — locally verified, no pretrained run

Added optional `--attention` to both experiments, shared capture/patch helpers in
`core/attention.py` and offline comparisons/figure in `core/attention_analysis.py`.
Default8 blocks, all heads, prompt boundary and up to4 sampled fixed-text positions;
`--attention-layers -1` selects all34 pretrained blocks. Saves Q before/after
normalization/RoPE, actual attention rows, K/V norms, head contexts and actual
post-normalization attention residual writes in binary artifacts. Cached decode
is explicitly skipped; no new model forwards from capture itself. Comparisons
require identical text and assert exact zero-hook tensors. Query-only single-pass
patch helper exists; full causal sweep and K/V patching remain future work.

All41 offline tests passed (19.946s); both attention-enabled CPU CLI smokes passed,
including fixed-text ISM and confidence evidence generation. Synthetic figure
visually checked. Tests verify observer invariance, upstream invariance, causal
masking/GQA, query-only replacement, cleanup and rejection of unlike prefixes.
No pretrained attention results, paid launch, or measured GPU overhead. Details,
limitations, commands and literature: `context/ATTENTION.md`.

## Confidence/entropy port — locally verified, not run pretrained

Added `confidence_entropy` using the current pinned Gemma/SAE/J-Lens stack.
Migrated all150 cached questions, seven original personas and four prompt templates
into experiment-owned JSON with source hashes. Added three neutral-confidence-wording
expert controls by default (user delegated the choice). No legacy runtime imports;
AGENTS.md records that `legacy_code/` is temporary reference material for later deletion.

Core helpers measure conditional choice entropy (nats/normalized), allowed-letter
mass, full-vocabulary entropy, correctness/NLL/Brier and stated-confidence categories.
Separate answer/confidence contexts; optional four-context generations include SAE
manipulation checks. Default133±2 reference-alpha with baseline/zero; optional matched
random controls. All configured J layers read the final prompt/decision position,
with saved vectors and within-persona/paired persona × steering geometry. Same
question sample/options used across conditions. Failed/interrupted runs can resume
with identical config/source hashes; active/completed runs are refused.

All36 offline tests passed in19.8s, including the real Gemma conditional wrapper
with tiny random weights, zero/upstream checks, distribution invariants, interaction
cancellation, source-independent data and interrupted-run recovery. Explicit CPU
smoke with random controls passed. No pretrained confidence results or paid launch.
Guide/commands: `context/CONFIDENCE_ENTROPY.md`. Default pilot800 decision forwards;
recommended random controls1200, plus16 reference generations. No free-text decoding
unless `--generate-responses`. Domain sample size is only2 each; pilot, not conclusions.

## Verified progress
- Local backbone and ISM steering, SAE traces, optional public J-Lens implemented.
- First real CUDA/BF16 probe completed: `outputs/ism/20261106T210928Z-dcc34ec7/`.
  Three Table 6 prompts, 133 only at raw ±50, 96 new tokens; 12 completions.
  Baseline/zero-hook tokens, SAE traces, and J readouts agree for all three prompts.
  133 activates in roleplay/story but not Assistant baseline; 4360 is sparse near
  boundaries. Steering changes details without a clear mode transition.
- Generated `report.md`, `generations.md`, `assessment.json`, and two reviewed PNG
  figures for that run. These can be regenerated locally without model inference.
- One local results root: `outputs/<experiment>/<run-id>/`. Moved all three legacy
  local runs; `experiments/results/` no longer exists. Shared pull script supports
  opt-in mirror layout; project config enables it. Other projects remain unchanged.
- Expanded code defaults to nine paper prompts; 133, 4360, and joint steering;
  baseline/zero controls; 23 paper features at layers 9/17/22. Optional norm-matched
  random directions. General assessment/reporting code lives in `core/`.
- Expanded pretrained run `outputs/ism/20261106T214058Z-26769b9d/` completed:
  72/72 completions, 23 features recorded, all 9 baseline/zero-hook pairs exactly
  agree in tokens, SAE traces, and J readouts. Elapsed time ~6 minutes including
  initialization. Real layer-17/22 SAE loading is now verified by the saved run.
  Authored `report.md`/`interpretation.md`, numerical `findings.json`, and two
  additional figures; checked figures visually. The final pull appeared during
  analysis, replacing an earlier 34-completion snapshot; report uses final data.
  Joint -50 removes the playful prompt's opening stage direction and reduces 133
  activity from 13/95 to 0/95; either negative feature alone leaves a stage direction
  and 8/95 activity. Other roleplay/story generations retain their broad modes.
- Sixteen offline tests passed with `/Users/daylight/miniconda3/bin/python -m unittest
  discover -s tests -v` (PyTorch 2.9.1, Transformers 4.57.3). Checks cover tiny Gemma
  generation, multi-layer monitoring, actual post-hook J readout, separate/joint
  plans, random controls, reports, and mocked shared-script rsync paths. Shared
  pull script passes `bash -n`. No pod connection or new pretrained run was made.

## Latest completed confirmation and next steps

Run `outputs/ism/20261107T035003Z-035fea34/` completed all 504 evaluations in
116m16s, with 16 reference generations. Authored `report.md`, saved recomputable
`analyze_run.py`/`report_metrics.json`, inspected three new figures and checked
source hash/local report links. 36/36 zero pairs match in tokens and SAE traces.
Pretrained full-width formula/loading audit passed; this is not independent
SAELens runtime or author-checkpoint parity. All 42 comparable earlier greedy
conditions replay exactly and are not independent new confirmations.

Main finding: 133 −2 and joint −2 each introduce the exact story preface in
12/12 story replies; baseline and either sign of 4360 have 0/12. Inspected
roleplay openings show positive 133/joint promoting literary narration and
negative steering favoring simpler conversational/helper presentation. This is
stronger partial functional replication, not a universal binary mode switch.
Positive Assistant identity/capability replies remain broadly ordinary Assistant
text despite large post-addition 133 coefficients. 4360 alone has weaker effects.

The 8-token repeat-fraction >.30 flag catches 1/216 targets versus 79/216 random
controls; joint-negative clockmaker loops. Task changes also occur below that
flag. 417/504 reach cap, including all story replies. Nine of 12 probes are
project-authored; calibration remains a reference proxy. No J-Lens or fixed-text
pass was collected. Downstream free-text SAE effects are associations, not mediation.

Next: reserved six-probe validation of 133 alone on the actual task family,
quality/framing scored separately, then matched-text SAE/J-Lens. Strong joint
steering is less clean. No new run is authorized or launched by this analysis.
504-run review is condition-aware agent analysis, not completed blind ratings.

Launch history: user initially declined, then explicitly authorized after a pod
restart. Updated SSH to 213.173.105.4:30141, restored key access, synced and installed
deps. Interrupted an initial unrelated legacy-data sync before launch and added
`legacy_code/` to `.gitignore`; no deletion or parameter change. Local preparation
passed 28 offline tests and explicit CPU smoke. See `context/ISM_CONFIRMATION.md`.
The older 138-card packet has blank ratings and remains an uncompleted review.

## Completed smaller-strength refinement
Run `20261107T023612Z-ba1f9307` completed all 96 evaluations in 18m21s. All 3 zero
pairs agree exactly, including fixed continuations. All 96 vector files are
present; recomputed all 14,265 matching-prefix comparison rows from binaries,
agreeing with saved summaries within 1e-5 scaled tolerance. Block-8 vectors are
unchanged on fixed text across conditions. Authored `report.md`; saved analysis
script, numerical `report_metrics.json`, and three visually checked figures.

Key observations: all 3 negative target and 12 negative control conditions remove
the playful opening; target +0.5/+1 remove it too. The target's Lyra/robot replies
retain broad modes throughout ±0.25/0.5/1. At robot −1, post-addition 133 and 4360
are zero across all 23 prompt and 191 processed reply positions, while a story
continues. Thus encoded coefficients/stage directions do not define the whole mode.
No repeated-loop collapse seen; max repeated 8-token fraction 2.16%.
SAE target dose responses are stronger than controls, but J-vector displacement
is often larger for controls. Residual-norm matching does not match projected
norm because the fitted J amplifies directions differently. No ISM axis or
subjective perspective is established.

## Replication setup verified locally
Implemented `core/calibration.py` and independent 16-prompt JSONL reference set;
version/dtype validation, checkpointed peak collection and calibration-only mode.
Added multiple temperature trials, actual post-addition SAE values, and an optional
teacher-forced baseline continuation for identical-text comparisons. Reports now
include magnitude tables/plots and active-only means. The completed 72-run report
received magnitude supplements while preserving its authored interpretation.

J-Lens now supports 8/9/10/17/22/29/32 and direct final-layer 33 by default, with
configurable layers/stride. Matrices are loaded once from the prior pinned revision.
21 offline tests passed, covering the new real-wrapper CLI path, reference
calibration/provenance, sampled controls, fixed-text effects and direct final
readout. `ism --smoke --device cpu` passed. No pretrained GPU run of the new
protocol was launched during setup.

## Bounded pretrained replication reviewed
`outputs/ism/20261107T014211Z-8c44c965/` is completed and pulled. Reference peaks
133=395.3448 and 4360=119.4049; alpha ±2 corresponds to coefficients ±790.69 and
±238.81, not the authors' corpus calibration. 3/3 baseline/zero pairs match exactly
including fixed-continuation traces. All 36 nonzero layer-8 prompt readouts match
baseline; all fixed-text pre-addition primary traces match as expected.

Codex authored the canonical `report.md` and an `interpretation.md`, computed `review_metrics.json`
via saved `analyze_review.py`, and added two reviewed figures. Negative 133 adds
story preambles; joint negative makes Lyra helper-like; positive 133/joint favors
literary narration. Assistant identity remains an AI introduction. Twelve random
replies matched to the higher 133/joint norms are repetitive, while six smaller
4360-matched random replies remain coherent. Thus controls need refinement.

The previous auto-generated `report.md` was preserved as
`legacy_automated_report.md`; `review.md` redirects to the authored `report.md`.
Future runs generate only evidence; the agent writes the report after review.
Default pulls exclude pod `report.md` files to protect locally authored reports.
These interpretations are not independent human review.

## Latest refinement verified locally
`core/conditions.py` supports multiple independently seeded isotropic and random
SAE decoder controls at matched norms. `core/lens_analysis.py` saves unnormalized
J-projected vectors and compares them within layer/phase only on matching text
prefixes; teacher-forced passes now include J-Lens. Added literal repetition
diagnostics and an evidence-only reporting path that never writes `report.md`.
All 24 offline tests and an end-to-end synthetic fixed-text/vector smoke passed.
The user subsequently ran the pretrained path successfully; completed results
are summarized above. The prior independent reference calibration was reused.

Reanalyzed all 23 monitored SAE features on identical continuations; see
`outputs/ism/20261107T014211Z-8c44c965/sae_same_text.json` and the updated report.
For Lyra, joint −2/+2 changes narrative L9:440 mean from 20.82 to 1.95/105.01,
while L17:972 moves 9.88 to 21.12/1.12 and L17:738 falls 23.23 to 6.44/8.16.
Same-layer changes can be direct encoder overlap; later-layer changes are not by
themselves evidence of mediation. J vocabulary shifts are documented in the report;
the old run did not save transported vectors, so full geometry cannot be recovered.

## Limitations
The expanded pretrained path has completed on the user's pod. No automatic resume or full-dataset
sweep is implemented; completions are saved incrementally. Reports are attempted
on normal completion/caught errors and can be regenerated after hard kills.
Old runs contain J-Lens only at layer 9/three positions and cannot be expanded
retrospectively. Earlier fixed-text assessments captured SAEs but not lenses.
Model/SAE/lens revisions are pinned. Exact author revisions and corpus peaks remain
unknown. This directory is not initialized as a Git repository. Old remote result
paths were not deleted or migrated.
