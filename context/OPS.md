Set up this project to run experiments on RunPod using my existing workflow
repo at ~/dev/code/scripts (README.md there is the full reference; read it
before starting). Do not copy or modify those scripts. They are universal and
already symlinked onto my PATH as `set-pod`, `sync-up`, `run-experiment`, and
`pull-results`. Your job is to make THIS project compatible with them.

2026-11-06 update: the user explicitly authorized editing the shared scripts to
unify result paths. This project now uses `RESULTS_SUBDIR="outputs"` and
`PULL_LAYOUT="mirror"`, with `outputs/<experiment>/<run-id>/` on Mac and pod.
The opt-in shared pull change preserves other projects' legacy behavior. This
supersedes the no-script-edits instruction for that requested change and the
old path examples below. Never launch paid pod commands without asking.

## How the workflow works (what you must design around)
- My Mac is the source of truth. `sync-up` rsyncs this directory to the pod,
  respecting .gitignore (so .venv, data dumps, outputs, secrets etc. must be in
  .gitignore if they shouldn't be pushed). It never deletes pod files by default.
- `run-experiment <args...>` syncs, installs deps if REQUIREMENTS_FILE changed,
  then runs `$RUN_ENTRYPOINT <args...>` inside tmux on the pod (survives
  disconnects). Flags: `--attach`, `--status`, `--stop`, `--no-deps`.
- `pull-results [subpath]` rsyncs `$RESULTS_SUBDIR` from the pod into ./outputs/
  locally (no deletes).
- The pod is ephemeral and wiped on stop. Anything not pulled or uploaded is lost.
- Pod credentials from ~/dev/code/scripts/r2.env are forwarded automatically to
  /tmp/r2.env and sourced before the run, so code can upload to Cloudflare R2
  without extra setup. Never hardcode or commit secrets.

## What to create in this project
1. `run.py` is the single entrypoint. Use `argparse` with subcommands, or
   `sys.argv[1]` as the experiment name. It dispatches to experiment modules,
   for example `experiments/<name>/main.py` exposing `main(args)`.
   `python run.py <experiment> [--flags]` is exactly what `run-experiment`
   will execute. Unknown names should print the available experiments and exit
   nonzero. Adding a new experiment should mean adding one module and one
   registry entry, nothing else.
2. `remote_requirements.txt` lists the pod's Python deps. Pin versions. Don't
   include torch if the RunPod image already ships it (ask me which image I use).
3. `.runpod.env` at the repo root, based on ~/dev/code/scripts/runpod.env.example,
   setting only what differs from defaults: REQUIREMENTS_FILE, INSTALL_CMD,
   RUN_ENTRYPOINT="python run.py", RESULTS_SUBDIR, and optionally RESULTS_EXCLUDE
   for large artifacts that should stay pod/R2-only.
4. `.gitignore` covering outputs/, .venv, caches, large data, and `.runpod.env`
   only if it contains anything sensitive (normally it doesn't).
5. Each experiment writes ALL artifacts under one directory inside
   RESULTS_SUBDIR, e.g. `experiments/<name>/results/<timestamp or run-id>/`, with
   a config/args snapshot, a log, and the metrics/outputs. Never write results
   outside that tree. Use relative paths from the repo root (the pod's working
   dir is the synced project root; there is no fixed absolute path).
6. Long-running experiments should checkpoint and be resumable, since the pod can
   die. Large artifacts should be uploaded to R2 via a small `sync_to_r2()` helper
   that reads credentials from environment variables (R2 vars come from
   /tmp/r2.env; check ~/dev/code/scripts/r2.env.example for the variable names)
   and no-ops cleanly when they're absent (local runs).
7. Code must run on both Mac (small/smoke mode, CPU or MPS) and the pod (CUDA).
   Pick the device at runtime, and add a `--smoke` flag to each experiment for a
   fast end-to-end check before spending GPU time.
8. A short README section "Running on RunPod" in this project listing the exact
   commands: `set-pod`, `run-experiment <experiment> [flags]`,
   `run-experiment --status`, `pull-results`. Don't re-explain the whole
   workflow; link to ~/dev/code/scripts/README.md.

## Rules
- Run commands from this project's root; the scripts operate on `pwd`.
- Don't run `set-pod`, `run-experiment`, or `pull-results` yourself without
  asking me. They touch a paid GPU pod. Do verify locally with
  `python run.py <experiment> --smoke`.
- Ask me before choosing anything I haven't specified (experiment names, GPU
  image, results layout). Otherwise use the defaults above.
- When done, summarize the files you created and give me the exact commands to
  launch the first experiment.
