# Many Are My Names

Minimal backbone for the replication described in [context/PLAN.md](context/PLAN.md).
Includes a plumbing check and an exploratory ISM experiment with selected-feature
activation traces and optional J-Lens readouts. No scientific replication results
have been established.

## Local smoke check

Run from this directory with Python 3.10+:

```bash
python3 run.py --help
python3 run.py smoke --smoke
```

Run the local checks with `python3 -m unittest discover -s tests -v`. They use
temporary artifacts and mock external services; no pod or R2 connection is made.

This needs no third-party packages. If PyTorch is installed, `--device auto`
selects CUDA, then MPS, then CPU; otherwise it selects CPU. Explicit unavailable
devices fail rather than silently falling back. The plumbing check records the
selection but performs no model or tensor computation. Use `--device cpu` to skip
PyTorch detection entirely.

Each run writes `config.json`, `run.log`, `status.json`, and `outputs.json` under
`outputs/smoke/<UTC-timestamp-and-id>/`. These files are ignored by
sync-up. A killed process can leave status `running`; that is not proof it is
still alive. No long-running jobs exist yet.

## Structure and adding experiments

- `run.py`: argparse dispatcher and common `--smoke` / `--device` options.
- `experiments/__init__.py`: experiment registry.
- `experiments/smoke/main.py`: minimal example exposing `main(args)`.
- `core/`: reusable device selection, run directories/logging, JSON writing, R2 upload.
- `context/`: research plan, operating instructions, progress, and decisions.

To add an experiment, add `experiments/<name>/main.py` with `main(args)` and one
entry in `EXPERIMENTS`. An optional `configure_parser(parser)` adds experiment
arguments. Keep heavy imports inside functions so CLI help does not load models.
Use `run_context` and write all artifacts inside `run.directory`. Keep the
`RESULTS_SUBDIR` constant in `core/runs.py` aligned with `.runpod.env`.

For future long runs, implement checkpoint/resume at the experiment level before
running on a pod; atomic `write_json` is available for small checkpoint metadata.
The current smoke check has no resumable workload. Model identity, SAE identity,
seeds, and decoding settings must be added to snapshots when those are introduced.

`core.storage.sync_to_r2(run.directory, prefix="many_are_my_names/<experiment>/<run-id>")`
is an explicit upload helper. It reads `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, and `R2_BUCKET` forwarded by the shared workflow. It returns
0 when none are configured and raises on incomplete credentials or upload failure.
Nothing uploads automatically, including the smoke check. R2 needs the pinned
dependency in `remote_requirements.txt`; credentials must never go in arguments
or config snapshots. The pin is from the [published boto3 release](https://pypi.org/project/boto3/1.43.123/).

## Running on RunPod

Use the existing [shared workflow reference](/Users/daylight/dev/code/scripts/README.md).
From this project's root, after preparing your pod:

```bash
set-pod <paste the pod's SSH command here>
run-experiment smoke --smoke
run-experiment --status
pull-results
```

The shared defaults already select `remote_requirements.txt`, its installation
command, and `python run.py`. `.runpod.env` sets `RESULTS_SUBDIR="outputs"` and
`PULL_LAYOUT="mirror"`. Runs use `outputs/<experiment>/<run-id>/` on both machines;
the shared pull script mirrors that relative path without creating a second local
results tree. Explicit pulls use `pull-results outputs/ism/<run-id>`.
Pull results before stopping the ephemeral pod.

PyTorch is already supplied by your pod image and is deliberately not installed
by this project. Its exact image/version is still unknown; confirm those before
running pretrained inference. These remote commands have not been run by the agent.

## Quick ISM steering

The paper associates layer-9 SAE features **133** and **4360** with Immersive
Simulation Mode; it does not equate the whole mode with a single feature. Start
with 133, switch to 4360, or pass both. See [research notes](context/RESEARCH.md)
for checkpoint provenance and the distinction from paper-calibrated steering.

On an environment with PyTorch installed, install the remaining dependencies:

```bash
python -m pip install -r remote_requirements.txt
python run.py ism --smoke
```

`ism --smoke` uses a tiny randomly initialized Gemma text model, a synthetic
two-feature SAE, and an identity Jacobian on CPU. It checks activation recording,
Multi-layer J/direct-final readout, baseline, zero-control, positive/negative steering, and cached decoding
without downloading weights. It uses three new tokens and strengths -1/0/1
regardless of real-run settings. This is a code check, not evidence about ISM.

For real inference, first log into Hugging Face and accept the Gemma conditions on
the [model page](https://huggingface.co/google/gemma-3-4b-it). That page says requests
are processed immediately. Authenticate **on the pod** using `hf auth login`, or
provide `HF_TOKEN` in its runtime environment. Logging in on your Mac does not
automatically authenticate the pod; the existing workflow forwards R2 credentials,
not your Hugging Face login. Never put tokens in CLI arguments or tracked files.
The [SAE repository](https://huggingface.co/google/gemma-scope-2-4b-it) is ungated
and labeled CC-BY-4.0; no separate SAE access request is needed.

From the project root on your Mac, run the nine published Table 6/7 prompts:

```bash
run-experiment ism --device cuda --jlens
run-experiment --status
pull-results
```

The launcher attaches automatically when starting. `run-experiment --attach`
reconnects only while a session is still running. A failed run exits that session;
start the command again after syncing the corrected local files. Pull results
before stopping the pod.

For a smaller comparison using only the original three prompts and two monitored features:

```bash
run-experiment ism --device cuda --prompt-set table6 --monitor ism --jlens
```

By default each prompt gets eight completions: baseline, zero-hook, and ±50 for
133 alone, 4360 alone, and their sum. The nine-prompt default produces 72 completions,
at up to 96 new tokens each. These remain raw, uncalibrated coefficients.
Optional `--random-control` adds one norm-matched seeded random direction for each
steering group at both strengths (126 completions for nine prompts). This checks
whether changes are more specific than generic residual perturbation; one random
direction per group is not a statistical null distribution.
`--feature 133` or `--feature 4360` restricts steering to one feature.
`--prompt` switches to custom user prompts; repeat it to
evaluate several in one load. `--feature 133 4360` steers the sum of two decoder
rows only. The default `--monitor paper` records 23 explicitly numbered Gemma
features from layers 9, 17, and 22, including all 15 Table 9 Assistant-associated
features. This is not the full set of 529 filtered latents. `--monitor ism` records
only the two ISM features. `--prompt-set table7` selects the six emotional prompts.
`--interactive` keeps the model
loaded for further prompts. `--scope all` affects all prefill and decode positions;
`--scope last` affects only the last position per forward pass. Generation is greedy.

Strengths are **raw decoder coefficients** in `h += strength * decoder_row`,
without normalizing the row. The default ±50 is an exploratory starting point,
not a validated optimum. The paper uses `alpha * corpus_peak`; that corpus
calibration has not been reproduced. Negative steering is not feature ablation.
Selected encoder columns, thresholds, biases, and decoder rows are loaded from
the SAE to record tokenwise activations without building a 65k-wide activation
table. `--jlens` downloads the public Gemma J-Lens (~433 MB) and loads matrices
once for blocks 8, 9, 10, 17, 22, 29, 32, plus direct final-layer readout at 33.
It samples the last prompt token, first generated token when available, every
16th generated position, and last traced token. Set `--jlens-layers` and
`--jlens-stride` to customize this. Layer-9 readouts show before/after the current
steering addition; other blocks show their observed outputs. All new SAE traces
also save actual post-steering feature magnitudes at layer 9.
Generated-token traces include only tokens processed by the model; a
final emitted token may have no layer activation if generation stopped there.
Each SAE download is about 1.34 GB, plus model weights. Monitoring all three layers
requires two additional SAE downloads compared with the original run. Selected
columns/rows keep GPU overhead small; later runs reuse the downloaded files.

Outputs, exact prompts/token IDs, settings, fixed checkpoint revisions, decoder
norms, package versions, and hook-call diagnostics are saved under
`outputs/ism/<run-id>/`. Each completion is saved immediately and
printed in the log. Completed comparisons survive a process interruption on the
same disk, but automatic resume is not implemented. Pull artifacts before stopping
the pod. This iteration adds no long-running dataset sweep or training job.

Shared steering, selected SAE encoding, and lens readout logic are in `core/`.
The model and SAE are pinned to published revisions; the lens revision is
recorded when downloaded. The authors' exact model revision and corpus peak
calibration remain unknown. No custom SAE training is needed.

## Reports and assessment

For the next focused replication with reference calibration, fixed-text
measurements, multiple decoding trials, and multi-layer lens readouts, see
[context/ISM_REPLICATION.md](context/ISM_REPLICATION.md). It provides the bounded
launch command and distinguishes our reference peaks from the paper's corpus peaks.

Every ISM run attempts to write `measurements.md`, `analysis_brief.json`,
`generations.md`, `assessment.json`, and PNG figures when it finishes or catches
an interruption. These are supporting measurements. They include prompt/reply
activity, zero-hook comparisons, magnitudes, J-Lens readouts, and fixed-text
statistics when requested.
Hard kills cannot run reporting cleanup, but saved completions remain usable.
Regenerate locally without loading weights or connecting to a pod:

```bash
python -m core.reporting outputs/ism/<run-id>
```

After pulling, ask Codex or Claude to inspect the evidence and write `report.md`;
see [context/REPORTING.md](context/REPORTING.md). It is an agent-authored analysis,
not automatically generated by the pod or independently human-reviewed. Scripts
never create or overwrite `report.md`. Historical auto-generated reports remain
as historical files where preserved.
The project's default pull excludes pod `report.md` files to protect locally
authored reports from older pod-generated files. Other evidence and vector files
still pull normally. Explicit subpath pulls bypass this exclusion.

Shared generation/trace logic is in `core/assessment.py`, SAE hooks in `core/sae.py`,
the documented feature inventory in `core/paper_features.py`, and evidence generation
in `core/reporting.py`. New J-Lens readouts capture the actual post-steering residual.
Older runs captured only layer 9 at three positions. They cannot be expanded to
other layers retrospectively from their saved feature values. No paid run is
launched by report generation.

## ISM confirmation and condition-hidden review

The next behavioral confirmation, known fidelity gaps, exact bounded command and
review criteria are in [context/ISM_CONFIRMATION.md](context/ISM_CONFIRMATION.md).
`ism --dry-run` prints counts without loading weights or creating a run.
`--prompt-file` accepts source-labeled JSONL probes; `--audit-sae` checks selected
loading against full-width official-formula encoding before evaluated generations.
This is distinct from an independent upstream-library comparison.

Use `python -m core.review outputs/ism/RUN_ID --destination
outputs/ism/RUN_ID/behavioral_review` to export shuffled replies and blank ratings.
Keep the private mapping away from reviewers until ratings are locked. Exporting
cards does not perform semantic scoring. Existing destinations are protected.

## Persona × ISM choice entropy

`confidence_entropy` ports the legacy 150 questions, seven original personas and
four independent prompt templates into the current Gemma steering/J-Lens stack.
It adds three neutral-wording expert controls by default, paired persona × steering
contrasts, answer-format probability mass and accuracy/NLL/Brier checks. Entropy
is answer-distribution uncertainty, not a direct measure of knowledge or confidence.
All reusable measurement code lives in `core/`; there are no legacy runtime imports.

```bash
python run.py confidence_entropy --dry-run --random-control
python run.py confidence_entropy --smoke --device cpu --random-control
```

See [context/CONFIDENCE_ENTROPY.md](context/CONFIDENCE_ENTROPY.md) for exact scope,
J-Lens positions, optional four-prompt generations, resume rules and the prepared
RunPod command. No new pretrained confidence run has been launched.

## Door-code attribution × ISM

`agent_vs_model` ports the legacy attributed-ignorance experiment to the pinned
Gemma model. It compares self-report, exact-code retrieval and paired parity
decisions across knows, does-not-know, withholding and code-absent conditions,
with feature-133 steering, SAE traces and optional J-Lens/attention. It is a new
protocol version; Qwen findings do not transfer automatically. The first pilot
has four code pairs, so its summary is descriptive. See
[context/AGENT_VS_MODEL.md](context/AGENT_VS_MODEL.md) for the design and command.

```bash
python run.py agent_vs_model --dry-run --families 4
python run.py agent_vs_model --smoke --device cpu --attention --max-new-tokens 2
```

Optional attention/query diagnostics work with both `ism` and `confidence_entropy`:
add `--attention`. For matched continuation measurements in ISM, also request
`--teacher-force-baseline`. See [attention guide](context/ATTENTION.md) for recorded
positions, binary artifacts, offline analysis, query patching and limitations.

Run the prepared paper-style attention pilot with
`run-experiment ism --preset attention-table6`; inspect it first with
`python run.py ism --preset attention-table6 --dry-run`.
Scope: [attention pilot](context/ATTENTION_PILOT.md).

After pulling results, run `python -m core.run_index` and open
[the Pacific-time run index](outputs/README.md). Future research folders use
readable Pacific timestamps; local synthetic tests are under `outputs/_smoke/`.
Older paths stay stable and have a `RUN_INFO.md` explaining their kind and times.
