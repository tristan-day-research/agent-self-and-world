# Replication and Research Plan

Last updated: 2026-11-04

## Purpose and status

Build a small, trustworthy replication of *Many Are My Names: The Anatomy of the Assistant and Its Personas via Sparse Autoencoders*, then use it to investigate speaker representations, Immersive Simulation Mode (ISM), and the Assistant-Associated Core (AAC).

This is the original implementation roadmap, not a current record of results.
Pretrained ISM, door, speaker, and attention runs have since completed. Start
with `HANDOFF.md` and `STATUS.md` for verified outcomes and current options.
Later research directions remain options, not commitments.

The paper is stored alongside this file. Resolve experimental details against that PDF and its referenced artifacts before coding them. In particular, do not turn an unverified feature identifier, checkpoint, or scaling convention from the conversation into a silent default.

## Working principles

- Start with the paper's Gemma-3-4B-IT setup and a pretrained SAE. Do not train an SAE initially.
- Start with one layer and a few features; build a reliable measurement and intervention loop before broad feature discovery.
- Separate published findings, our replication results, and new hypotheses in all reports.
- Treat AAC as a proposed collection of representations, not an established single neuron, feature, agent, or conscious subject. Treat ISM features as candidate mechanisms to investigate, not proof of a subjective perspective change.
- Prefer simple scripts and configuration files over a general experiment framework. Preserve the ability to swap model, SAE, prompt set, and intervention without rewriting the pipeline.
- Ask before paid/expensive compute. A plan mentioning Runpod is not authorization to rent hardware, start a long run, or train an SAE.
- Record parameter changes explicitly. Keep discovery prompts separate from held-out evaluation prompts.

## Phase 0 — Establish exact reproduction targets

**Confidence: high. Estimated active work: 30–90 minutes, excluding access delays.**

1. Read the paper's methods, relevant appendices, and feature tables. Record page/section references in `context/RESEARCH.md`.
2. Check whether the authors supply code, prompts, data, or feature lists. Reuse verified artifacts where appropriate; absence of a link in our notes does not establish that no code exists.
3. Create a compact provenance table containing:
   - Exact model repository and revision; instruction-tuned versus base checkpoint.
   - Exact SAE repository, revision, checkpoint path, architecture, width, sparsity setting, and normalization convention.
   - Hook location, layer-index convention, input dimensions, and model/SAE dtype.
   - Feature IDs, paper labels, prompt templates, token positions, and intervention formulas.
4. Initial candidates to verify: `google/gemma-3-4b-it`, `google/gemma-scope-2-4b-it`, residual-stream SAEs at layers 9, 17, and 22, 65k width, medium sparsity. Begin with layer 9 and the reported ISM candidates 4360 and 133 only after confirming their complete checkpoint identity.
5. Pin a compatible Python/model/SAE library environment and document a Runpod setup command. Check gated-weight access without printing tokens or other credentials.

**Done when:** one exact model/SAE pair and a narrowly defined first result are documented; unresolved details are visibly marked. If the published artifact cannot be obtained, label any replacement experiment an adaptation.

## Phase 1 — Load, inspect, and generate a baseline

**Confidence: high. Estimated active work: 2–4 hours including basic setup; downloads and dependency issues can extend this.**

Implement the smallest end-to-end path:

1. Load the frozen model and one pretrained SAE. Start without model quantization to avoid introducing an additional replication variable.
2. Generate ordinary assistant text with a fixed prompt, chat template, decoding configuration, and seed.
3. Capture the SAE's intended model activations at its actual hook. Verify dimensions, token alignment, finite values, and expected sparsity.
4. Encode activations and save selected feature activations per token. Label prompt tokens, generated tokens, padding, and special chat-boundary tokens separately.
5. Inspect reconstruction error as a diagnostic. Observing activations must not replace the model's activations with the SAE reconstruction.
6. Run a small, explicitly labeled set of Assistant, Roleplay, and Story examples. Use paper prompts where available; label newly written prompts exploratory.

Outputs should include baseline text, a readable token/activation table, and machine-readable metadata. A feature's human-readable label is an interpretation supplied by researchers or an annotation service, not the SAE's numeric output.

**Done when:** model text is coherent, token traces are interpretable, passive recording leaves baseline behavior unchanged within a declared numerical tolerance, and another session can reproduce the run from its saved configuration. Coherent output alone is not evidence of scientific replication.

## Phase 2 — Implement and validate feature interventions

**Confidence: high for the engineering; behavioral outcomes must be measured. Estimated additional work: 3–6 hours.**

1. Add a decoder-direction intervention at the verified hook. Define the exact operation and units of its coefficient in configuration and documentation.
2. Support unmodified baseline, hook-enabled zero intervention, positive steering, and negative steering as distinct conditions.
3. Make intervention timing explicit: prompt prefill, selected boundary tokens, generated tokens, or both. Verify behavior with generation caching and avoid accidentally intervening only on the first forward pass.
4. Use a short strength sweep anchored to the paper's normalization convention. Do not assume that a published coefficient can be copied into raw activation units.
5. Save paired text outputs, selected feature traces, and basic degradation diagnostics, such as repetition, termination, and task performance.
6. Add matched control interventions: unrelated SAE features and/or random directions with documented norm matching. Test several prompts rather than selecting one striking completion.

**Done when:** zero intervention reproduces baseline; hook timing is verified; intervention magnitude is logged; and a small paired report shows effects, null effects, and degradation. A direction that causes incoherence is not automatically a successful replication.

Negative steering is not feature ablation. Removing a feature contribution, clamping an encoded coefficient, and replacing an activation with a reconstruction are different experiments. Implement and name them separately if needed later.

## Phase 3 — Make the experiment reusable and add a contrastive baseline

**Confidence: high. Estimated additional work: 4–8 hours.**

Build only the infrastructure needed for repeatable comparisons:

- Configuration for model/SAE identity, hook, feature list, coefficient convention, intervention scope, prompt set, seeds, and decoding.
- A prompt registry with stable IDs and explicit Assistant/Roleplay/Story conditions. Match content and formatting where the research question requires it.
- Structured per-run output, summary tables, and token-position activation plots.
- A difference-of-means Assistant–Roleplay direction estimated on a discovery split and evaluated on held-out examples. Match intervention location and magnitude when comparing it with SAE directions.
- Lightweight CPU tests for configuration, token bookkeeping, intervention arithmetic, and serialization; an opt-in GPU smoke test for the actual model/SAE hook. Default test runs must not download large weights or allocate paid GPUs.

**Done when:** one command can rerun a small comparison and produce an auditable report without manual notebook state. The SAE and contrastive methods can be compared without changing unrelated settings.

## Phase 4 — Focused replication of the paper

**Confidence: medium; depends on artifact availability and methodological detail. Estimated additional work: 1–3 focused days for a selected subset, not the entire paper.**

Choose a short list of published claims before expanding:

1. Reproduce the selected ISM features' activation patterns across the paper's relevant conditions and token positions.
2. Reproduce selected positive and negative steering examples using the paper's prompts, scales, and generation settings where recoverable.
3. Add the relevant layer-17 and layer-22 SAEs and selected AAC-associated features. Preserve the paper's distinctions between feature classes rather than treating all assistant-related features as interchangeable.
4. Reproduce appropriate activation-density, peak-activation, and contrastive-readout analyses. Confirm exact definitions from the paper before implementation.
5. Evaluate multiple prompts and seeds; report paired effect estimates and uncertainty. Separate qualitative resemblance from quantitative reproduction.
6. If using an LLM judge, save the rubric, judge identity, and raw judgments; manually audit a sample. Do not define success solely by a judge repeating the interpretation being tested.

**Done when:** each selected claim has a result classified as reproduced, partially reproduced, not reproduced, or inconclusive, with methods and limitations. Broad feature rediscovery and all appendices are separate scope; a full replication may take weeks.

## Phase 5 — Choose one research pilot

**Confidence: medium to low. Estimated additional work: 1–3 days for one pilot after the pipeline is stable.**

Select one question based on Phases 1–4 rather than implementing all of these:

| Candidate question | Minimal experiment | Main confound to address |
| --- | --- | --- |
| Does ISM affect more than speaking style? | Matched factual/belief tasks with baseline, positive, negative, and control interventions | Different wording or role instructions can change answers without changing information access |
| Does ISM change use versus reporting of information? | Separate objective task performance from self-reported access; control what the character is told or witnesses | Reporting willingness and instruction-following can mimic unavailable information |
| Which AAC-associated components affect role behavior? | Small feature-group interventions with held-out tasks and competence checks | Redundancy, collateral damage, and incorrect feature labels |
| Is there a transferable geometric mode difference? | Compare mode-conditioned activation differences across held-out personas, tasks, positions, and layers | A mean difference can encode content, formatting, or lexical style |

**Done when:** a preregistered local protocol states hypothesis, operational measure, controls, and stopping rule; a small pilot determines whether to expand. Do not infer agenthood or subjective experience from a successful steering effect.

## Phase 6 — Causal follow-up, conditional on a robust pilot

**Confidence: lower. Timescale: days to weeks depending on scope.**

- Test necessity and sufficiency separately. Use feature/group ablation, patching, and restoration where appropriate; include competence controls and matched interventions.
- Distinguish correlation from mediation when asking whether AAC-associated activity drives downstream persona features or individual neurons. Layer order alone does not demonstrate that causal path.
- Investigate the proposed later-layer origin of ISM with layerwise measurements and causal interventions. A feature first becoming detectable at a later layer does not establish that a new core originates there.
- Test generalization across personas and tasks before interpreting a direction as a general perspective mechanism.
- Consider concept-injection/introspection experiments only with sham injections, false-positive measurement, objective concept identification, and checks for ordinary output priming.
- Define “J-space” and “recursive generalization” operationally, with explicit source methods and measurable predictions, before adding them to the pipeline.

These extensions are options. Null single-feature ablation can reflect redundancy or feature recovery; it does not establish that the represented property is absent or causally irrelevant.

## Phase 7 — Larger models and optional SAE training

**Confidence: medium for using compatible public SAEs; lower for scientific transfer and custom training.**

After the 4B pipeline passes its checks, select one larger model for the actual question:

- **Gemma-3-12B-IT:** first candidate for a larger same-family model with matching instruction-tuned GemmaScope 2 artifacts, subject to checkpoint/hook verification.
- **Gemma-3-27B-IT:** optional larger confirmation when the expected scientific value justifies memory and runtime.
- **Llama-3.1-8B-Instruct:** potential cross-family replication, but first resolve the exact SAE provenance used by the paper. Do not substitute LlamaScope, Goodfire, or another suite and reuse the paper's feature IDs as though they were equivalent.

Feature numbers do not transfer between models or SAE checkpoints. Larger-model work requires discovering or validating analogous features. Do not commit to traversing every model size.

For initial short-context, one-SAE runs, rough GPU memory planning targets are 24 GB for 4B, 40–48 GB for 12B, and 80 GB for 27B. These are estimates, not tested requirements; batch size, sequence length, SAE width, library overhead, and dtype matter. Profile a short run before approving a larger budget.

Custom SAE training is a separate gated project, justified only when available SAEs cannot answer the question. Begin with one layer and modest width, establish an activation/token budget, measure throughput and storage, and assess reconstruction, sparsity, and downstream fidelity. Frozen-model activation generation remains expensive even though the language model is not being trained. Estimate cost and obtain approval before scaling.

## Implementation contract for Codex

The initial backbone now uses `run.py`, `experiments/`, shared `/core`, and `tests/`, following `OPS.md` and the user's requested structure. Create further directories only as needed. Runs write under `experiments/results/<experiment>/<run-id>/`; pulled pod results land under local `outputs/`. Keep large weights, activation caches, and generated results outside version control. Git initialization remains separate. This scaffolding does not complete the model/SAE validation phases above.

Every run must record:

- Run ID, timestamp, code revision when available, configuration, and dependency versions.
- Model and SAE identifiers/revisions, full checkpoint path, hook, layer, dtype, and hardware.
- Exact prompts, chat template, token IDs or reproducible tokenization details, condition labels, and split membership.
- Feature IDs, direction source, coefficient units, intervention formula, timing, and any normalization.
- Seed, decoding settings, generated text, failures, and measurement definitions.

Record enough activations to audit the question, but do not save dense token-by-entire-dictionary tensors by default. Stream selected features and summary statistics; make full capture an explicit storage decision.

Validate model/SAE compatibility, padding masks, layer numbering, reconstruction conventions, and prefill/decode timing before interpreting results. Preserve the original residual contribution not represented by the SAE during feature-removal experiments where the chosen method requires it. Avoid intervention hooks persisting into later baseline runs.

Stop and document blockers rather than silently swapping models, SAEs, normalization, prompts, or generation settings. Update `STATUS.md` after meaningful work; keep provenance and scientific interpretation in `RESEARCH.md`, and significant decisions in `DECISIONS.md`.

## First two-hour session

This is a target, not a promise. If access or installation consumes the session, a verified setup and documented blocker are preferable to an unvalidated result.

| Approximate time | Task | Tangible output |
| --- | --- | --- |
| 0–30 minutes | Verify artifacts and select exact Phase 1 configuration | Provenance notes and pinned configuration |
| 30–60 minutes | Set up environment and load model/SAE if access permits | Reproducible load command and baseline output |
| 60–90 minutes | Capture selected feature activations and check hook/token alignment | Tokenwise activation table |
| 90–120 minutes | If checks pass, run a zero-intervention check and a tiny steering comparison | Paired outputs plus saved metadata; otherwise a precise next-step note |

The intended result is a trustworthy first measurement loop, not a claim that the paper has been replicated. An online feature browser may help inspection but is not a dependency; hosted inference and exact checkpoint availability must be checked separately.

## Starting references

- Primary source: the PDF in this directory; record the actual version before replication.
- GemmaScope documentation: https://ai.google.dev/gemma/docs/gemma_scope
- Candidate 4B instruction-tuned SAE repository: https://huggingface.co/google/gemma-scope-2-4b-it
- SAELens documentation: https://decoderesearch.github.io/SAELens/

Verify current availability and compatibility at implementation time. These references are starting points, not evidence that this project's environment or experiments have been validated.
