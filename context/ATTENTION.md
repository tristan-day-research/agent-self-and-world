# Attention and ISM: observation first, causal tests next

Prepared 2026-11-10. No pretrained attention run or paid launch yet. The recorder
is optional and shared by `ism` and `confidence_entropy`; future experiments can
use `core.attention.record_attention` and `save_attention_artifact`.

## What a query is

Each head at each token has its own query, not one central query for an agent.
Ignoring architecture details, q_i = W_Q LN(r_i), k_j = W_K LN(r_j), and attention
is softmax(q_i dot k_j × scale + mask). Training learns compatible Q/K projections;
no separate process tells a query which keys to seek. The residual contains many
superposed signals. A hypothesized agent representation could contribute, alongside
task, syntax, entities and position. It is not established as the source of all Q.

Our pinned Transformers 4.57.3 Gemma-3 implementation applies input RMSNorm,
Q/K projections, separate Q/K normalization, then rotary position transforms.
It uses grouped-query attention (several query heads share a KV head) and a mix
of sliding-window/full attention. Consequently W_Q times an SAE decoder vector
is not by itself the actual change in query. Normalization and upstream changes
matter. Query direction can change even when normalized query norm barely changes.

There is a sequence of influences across depth: residual at block l -> Q/K/V ->
attention write -> residual for later blocks. This is a feed-forward dependency,
not a simultaneous circular equation at one block. Across generated tokens there
is another dependency through previous text and cached K/V; the current query does
not rewrite past cached keys. Reconstructing an agent representation at later
blocks is a testable hypothesis, not evidence of recursive subjective agency.

## Implemented observations

Add `--attention` to an existing experiment command. Default blocks are
8,9,10,17,22,29,32,33 (zero-based); `--attention-layers -1` selects all actual
blocks (34 in pretrained Gemma-3-4B-IT). Every query head is captured in each block.
The synthetic smoke explicitly uses eager attention and blocks0/1. Pretrained
loading already uses eager; the recorder rejects other backends and training mode.

- ISM generation: final prompt position only, captured during prefill.
- ISM with `--teacher-force-baseline`: same prompt and baseline continuation in
  every condition, sampling final prompt, first continuation input, then every16
  continuation inputs; at most4 positions total. No guarantee the last reply token
  is sampled. The last emitted token is not processed in this pass.
- Confidence: final decision position in the separate answer and stated-confidence
  contexts. Optional free-form diagnostic generations are not attention-monitored.
- Cached generation steps are **not** recorded. Their count is explicit. This
  avoids retaining growing caches and accidentally comparing different texts.
- `--attention-stride N` and `--attention-max-positions N` control fixed-text
  sampling. Capture is for batch1, unpadded text, not multimodal or padded batches.

`outputs.json` holds per-head entropy in nats, row sums, nonzero support, top8 key
positions/token IDs, self/BOS/recent16/prompt mass, mean distance, Q norms and
context-vector norms. Nonzero support is numerical, not the architectural window.
Prompt mass is trivially ~1 at a prompt boundary; useful only in continuation rows.
Tokens are saved so positions can be interpreted without downloading a tokenizer.

`attention/NNNNNN_prompt.pt` / `_fixed.pt` contain float32 CPU tensors:

- Q before q_norm, after q_norm and after RoPE: [query heads, sampled positions, head dim].
- Actual eager attention rows: [query heads, sampled positions, all text positions].
- Raw/normalized K norms and V norms: [KV heads, text positions].
- Per-head weighted value/context vectors before o_proj.
- Combined projected attention output, and the actual attention write after
  Gemma's post-attention normalization, before residual addition.

K/V **vectors** are not saved, only norms; these files cannot reconstruct arbitrary
Q/K counterfactual attention scores offline. Context norms omit the output
projection; whole-block residual_write is the actual addition. Neither scalar
norms nor raw attention weights measure behavioral importance. Output projection,
normalization, cancellation, and downstream processing all matter.

Binary files keep vectors out of JSON. Retained pattern size scales with sampled
queries × sequence length, not sequence length squared; the existing eager
forward still computes its normal quadratic attention. No extra model forward
is required by capture. CPU copies, reductions, plotting and disk I/O have costs;
no real GPU overhead estimate has been measured. All-block collection increases
storage and time. Start with default blocks and a few matched probes.

After normal completion, `attention_effects.json` compares against the matching
baseline on **identical token IDs and sampled positions**: attention Jensen–Shannon
divergence/L1, query cosine/L2, context-vector L2 and whole-block attention-write
L2. Stored attention rows are renormalized for JS to remove BF16 row-sum rounding.
Whole-block write difference repeats across heads and must not be summed over heads.
Exact zero-hook tensor checks fail loudly. No cross-persona token-index alignment
is attempted. The figure `figures/attention_example.png` shows the largest observed
JS case, explicitly selected after looking at data; not a representative effect
or a hypothesis test. These artifacts are evidence, not an authored report.

Offline regeneration (no GPU/model/download):

```bash
python -m core.attention_analysis outputs/ism/<run-id>
```

Attention cannot be recovered from old SAE/J-Lens outputs without rerunning the
model. Interrupted runs may contain a partial set of attention files; their JSON
checkpoint defines which rows belong to the run.

## Use with the current confidence experiment

Keep existing conditions and add the flag; J-Lens remains enabled by default:

```bash
run-experiment confidence_entropy --device cuda --dtype bfloat16 \
  --questions-per-domain 2 --persona-set both --features 133 \
  --strengths -2 2 --coefficient-mode reference-alpha --random-control \
  --attention
```

This is still the existing 1200 scored-forward pilot plus16 calibration generations
unless an eligible calibration file is supplied. No launch is authorized here.
For ISM, add `--attention --teacher-force-baseline` to the chosen command, retaining
its coefficient mode/calibration/conditions. Teacher forcing adds a forward per
condition; do not silently add it to an already-running protocol. `--jlens
--save-jspace-vectors` can be combined with attention for identical positions.
J-Lens and attention strides are independently configurable; align them when
comparing position-by-position. Local no-download checks:

```bash
python run.py ism --smoke --device cpu --attention --teacher-force-baseline
python run.py confidence_entropy --smoke --device cpu --attention
```

## Recommended experiment sequence

1. Compare feature133 ±2 reference-alpha with baseline/zero on the existing
   Assistant, Roleplay and Story probes, inspecting manipulation quality separately.
   Begin with identical prompts and baseline continuations; include controls.
   Separate grammatical person, prose style, task compliance and immersion scores.
2. Inspect per-head changes and which tokens gain/lose mass: role instructions,
   character names, facts, question/options, dialogue boundaries, BOS and nearby
   tokens. Role labels are not validated internal perspectives. Semantic spans
   must be explicitly annotated; tokenizer positions alone do not establish them.
3. Post-block9 steering must leave block0..9 Q/attention unchanged on identical
   text; the first possible effect is block10. Downstream changes are causal effects
   of the intervention, not proof that attention mediates behavior. Free decoding
   after divergence does not provide that upstream invariant across conditions.
4. Select a few candidate heads on discovery prompts, then test held-out prompts.
   `core.attention.patch_queries(module, baseline_q_normalized, positions, heads)`
   provides a one-forward, batch1 patch after q_norm and before RoPE. Caller must
   enforce identical text/position setup. Register patch before recorder so the
   recorder sees the replaced Q. This only replaces Q locally, keeping K/V of that
   block fixed; downstream residuals and later K/V can change. Source shape must
   include all heads; the heads argument selects which to replace. Not for cached
   generation. A bounded L11H7 candidate rescue CLI is now available through
   `ism --preset query-rescue-table6`; it includes a different-head and self-patch
   control. It remains exploratory, not a full mediation sweep.
5. After observing candidate effects, compare Q-only, K-only and V/head-output
   rescue interventions, and both directions of patching. Measure answer logits,
   entropy/accuracy, downstream SAE/J-Lens and independent behavior. Query rescue
   alone is not proof of an agent representation or sufficient causal mediation.
   K/V patching, causal tracing to contributing upstream heads/MLPs and repeated
   layer-to-layer reconstruction are subsequent work, not current measurements.

## Prepared speaker-factor and query-patch pilots — 2026-11-10

`experiments/ism/speaker_probes.jsonl` crosses Assistant/Roleplay/Story, first/third
person, plain/literary prose, and two held-constant characters. The four local
presets are `speaker-styles`, `speaker-styles-ism`, `speaker-person-swap`, and
`query-rescue-table6`; use `python run.py ism --preset NAME --dry-run` to inspect
exact counts before any paid launch. They save paired Q/attention effects,
annotated prompt-span attention, SAE feature-category summaries for the paper's
23 numbered features, and where requested a pre-RoPE Q swap or baseline Q
restoration at L11H7 with controls. The query intervention affects the prefill
boundary only; generated cached steps are not patched. The code has passed
synthetic local tests, but no pretrained run of these presets has been reviewed.

## Literature and what it supports

- [A Mathematical Framework for Transformer Circuits (2021)](https://transformer-circuits.pub/2021/framework/index.html):
  QK routing versus OV content; Q-, K- and V-composition across heads. Closest
  conceptual answer to an earlier representation shaping a later query.
- [In-context Learning and Induction Heads (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html):
  concrete multi-head composition building context-dependent retrieval. Not an
  agent/self-representation result.
- [Attention Is Not Only a Weight (2020)](https://aclanthology.org/2020.emnlp-main.574/):
  why transformed value vectors matter alongside attention weights.
- [Attention is not Explanation (2019)](https://arxiv.org/abs/1902.10186) and
  [Attention is not not Explanation (2019)](https://arxiv.org/abs/1908.04626):
  motivate intervention and controls rather than interpreting a heatmap as a
  complete explanation. Neither settles our Gemma/ISM question.
- [Function Vectors (2023/2024)](https://arxiv.org/abs/2310.15213):
  attention heads transport compact task-related representations; causal tests
  connect these to behavior. A useful analogue, not proof of an AAC.
- [How do Language Models Bind Entities in Context? (2023/2024)](https://arxiv.org/abs/2310.17191):
  causal evidence for binding entities to attributes, relevant to distinguishing
  which character a fact belongs to from adopting that character's perspective.
- [Steering at the Source: Style Modulation Heads for Robust Persona Control (2026)](https://arxiv.org/html/2603.13249v1):
  directly relevant work reporting sparse persona/style-related attention heads.
  This is a study-specific finding, not a universal three-head architecture or a
  demonstrated Gemma-3-4B ISM circuit. Its Llama experiments could guide a later port.
- [The Assistant Axis (2026)](https://www.anthropic.com/research/assistant-axis):
  relevant residual representation/behavior work; does not establish an AAC-to-Q
  feedback mechanism. Its models/axis are not our checkpoint-specific ISM features.
- [Towards Best Practices of Activation Patching (2023/2024)](https://arxiv.org/abs/2309.16042)
  and [How to use and interpret activation patching (2024)](https://arxiv.org/abs/2404.15255):
  guide causal follow-up and choice of corruption, controls and outcome metrics.

Implementation source inspected locally: Transformers4.57.3
`transformers/models/gemma3/modeling_gemma3.py`, especially Gemma3Attention,
eager_attention_forward and Gemma3DecoderLayer. No cited work here establishes
that a unitary inner agent recursively determines its own queries.
