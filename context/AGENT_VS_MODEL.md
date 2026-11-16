# Door-code attribution experiment

This is a new Gemma-3-4B-IT pilot adapted from
`legacy_code/experiments/agent_vs_model`. Its runtime imports no legacy code.
The original Qwen2.5-32B observations are motivation, **not** findings on Gemma.

## Question

Does a six-digit code in the system prompt affect the Assistant's answer when
the user turn says the Assistant does not know the code? Four conditions vary
that attribution: knows, does not know, knows but withholds, and code absent.
Each task is a fresh conversation: knowledge self-report, exact-code request,
and a parity-dependent L/R/ASK instrumental decision. An opposite-parity pair
shares five digits. Both parity→procedure mappings and both rule orders are
crossed. The code-absent pair receives identical visible prompts; its hidden
labels are **not** independent observations.

## Crossed access × attribution follow-up

`--design crossed` keeps the same code pairs, tasks, rule orders, measurements,
and steering options, but changes the four cells to independently vary whether
the six-digit code appears in the system record and whether the user claims the
Assistant knows it. Cells are `present_knows`, `present_doesnt_know`,
`absent_knows`, and `absent_doesnt_know`. The user's claim contains no digits.
The absent/knows cell is deliberately inconsistent: it tests whether the model
follows an asserted knowledge state without the code. An absent code supplies no
parity ground truth to the model; its hidden labels are only a leakage/bias
control. A present code remains available to the model even under a
`doesnt_know` attribution. Thus this protocol can distinguish prompt access
from self-description, but cannot demonstrate a literal internal information
barrier or separate agent.

Gate any interpretation on `present_knows`: it must use paired code parity
reliably in the unsteered instrumental task. The first Gemma pilot failed this
gate, so revise/diagnose the instrumental wording or use the original Qwen32
protocol before interpreting a new pretrained run. Compare self-report and
retrieval across all four cells; compare paired parity effects within each
attribution, with absent cells expected to have no real code-dependent signal.
An ISM steer is a later intervention after the baseline gate passes.

## Short capability diagnostic

`--design capability --baseline-only` creates 48 short trials for four code
families. It tests (1) the final digit when the code is present, (2) its E/O
parity, (3) choosing L/R when parity is **provided** but the code is absent,
and (4) the full instrumental task when the code is present. The last two tasks
cross both rule orders. Each row saves the exact prompt, a strict short answer,
next-token choice probabilities, choice mass, and the expected answer. The
baseline-only flag skips steering arms and reference calibration; the current
Gemma runner still loads its selected SAE for shared measurement plumbing.
Changing task wording limits a direct comparison of step accuracies, but it
localizes a failure more usefully than the original full task alone. Code
family is the unit of replication; use fresh families for any follow-up.

Local checks:

```sh
python run.py agent_vs_model --design capability --baseline-only --no-jlens --no-monitor-paper --dry-run --families 4
python run.py agent_vs_model --smoke --device cpu --design capability --baseline-only --no-jlens --no-monitor-paper --max-new-tokens 2
```

The bounded pretrained command, **after separate RunPod authorization under
`context/OPS.md`**, is:

```sh
run-experiment agent_vs_model --device cuda --design capability --baseline-only --families 4 --no-jlens --no-monitor-paper --run-label door-capability
```

## Qwen3.5-27B baseline port

The `--model qwen35` path reuses the capability or crossed prompt designs on
`Qwen/Qwen3.5-27B`. It currently supports **baseline only**, because Gemma SAE
feature 133 is not a Qwen steering direction. The capability gate runs first;
the crossed design is only interpretable if Qwen reliably uses the code in the
`present_knows` instrumental task. The Qwen path records next-token choice
distributions, short non-thinking-mode generations, top Qwen-Scope SAE features
at residual blocks 16/32/48, published R-Lens readouts at those blocks, and
optional attention/Q observations at full-attention blocks 15/31/47/63. Qwen
has linear-attention blocks too; those do not produce ordinary softmax attention
matrices and are not represented by the attention readout. No lens is fitted.
The R-Lens maps to its published target block, not an exact per-prompt logit
decomposition. SAE feature ranks are observations, not automatically labeled
speaker traits.

The first bounded Qwen plan is 48 scored prompts and 48 short generations.
Check the dry plan before launch:

```sh
python run.py agent_vs_model --model qwen35 --design capability --baseline-only --families 4 --attention --dry-run
```

The wording is adapted to a single-character answer for decision readouts.
This is a **new protocol version**, not a bit-for-bit repetition of the legacy
JSON protocol. Generation is also saved to check whether the model actually
obeys the requested format. Choice probabilities are conditional on Y/N or
L/R/A; `choice_probability_mass` records how much total probability those
tokens hold. A low mass weakens interpretation of the conditional score.

## Measurements and controls

- Baseline, exact zero hook, layer-9 GemmaScope-2 feature 133 at ±2 times an
  independent reference-set peak. `--features 4360` or both can change the
  steering set. `--random-control` adds a seeded norm-matched direction.
  `--steering-scope last` restricts the addition to the final token of each
  forward, leaving code-token residuals unsteered at the intervention layer;
  this is a separate follow-up condition, not directly pooled with the default
  paper-style `all` scope.
- Pre/post layer-9 selected SAE activations, generated-token SAE summaries,
  the paper's 23 explicitly numbered speaker features at layers 9/17/22,
  prompt-boundary J-Lens (default blocks 8/9/17/22/33), and optional attention.
  `--no-monitor-paper` omits the additional SAE loading/encoding.
- Exact-code retrieval, literal self-report, instrumental accuracy and the
  **paired parity effect**: P(even-associated procedure | L/R, even code)
  minus the same probability with the paired odd code. This subtraction
  cancels fixed L/R and first-rule preferences within a pair.
- Zero-hook equality is asserted for decisions and generated replies. The
  output includes every prompt, code, seed, direction and calibration artifact.

First inspect the `knows` positive control. If its parity effect is near zero
or its choice mass is low, the instrumental comparison is not measuring code
use well enough to support an interpretation. This pilot has only four code
families by default and reports descriptive rates, not confidence intervals.
No linear probe or residual patch is in this first port; those require a
larger held-out family set and careful token-span alignment. Before probing,
separate training/test families and test label shuffles plus code-absent
controls. Before patching, test opposite-parity donor and same-code self-patch
at exact code-token positions across Gemma's 34 blocks.

Even if the model says N but chooses the correct procedure, the observation is
that prompt information influenced a separate task despite its verbal report.
It does not establish two agents, a private belief store, or subjective access.

## Commands

Dry plan (no model load):

```sh
python run.py agent_vs_model --dry-run --families 4
python run.py agent_vs_model --design crossed --dry-run --families 4
```

Synthetic local pipeline check (random tiny model, no scientific result):

```sh
python run.py agent_vs_model --smoke --device cpu --attention --max-new-tokens 2
python run.py agent_vs_model --smoke --device cpu --design crossed --no-jlens --no-monitor-paper --max-new-tokens 2
```

Bounded first pretrained run after reviewing the plan:

```sh
run-experiment agent_vs_model --device cuda --families 4 --features 133 --strengths -2 2 --jlens --run-label door-code-pilot
```

The default run makes 128 trial prompts × four conditions: 384 next-token
decisions and 512 short diagnostic generations, plus 16 independent reference
generations if no compatible calibration file is supplied. Outputs go to
`outputs/agent_vs_model/<Pacific-time-run-id>/`. `summary.json` is computed
evidence; a reviewing agent should write `report.md` after examining responses,
positive controls, prompts, choice mass and SAE/J-Lens evidence.
