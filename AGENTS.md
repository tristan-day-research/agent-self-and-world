# Agent Instructions

This repository uses shared context files to coordinate
research between ChatGPT Work and Codex in Cursor.

## Before substantial work
- Start with context/HANDOFF.md when resuming this project in a new session.
- Read context/STATUS.md.
- Consult context/RESEARCH.md for relevant background.
- Check context/DECISIONS.md when prior decisions matter.

## After substantial work
- Update STATUS.md with verified progress and next steps.
- Update RESEARCH.md when scientific understanding changes.
- Record important decisions in DECISIONS.md.

## After an ISM run is pulled
- When asked to analyze results, follow context/REPORTING.md and write the
  research narrative yourself to that run's report.md after inspecting evidence.
- Treat generated tables, metrics, and plots as supporting evidence, not an
  automatically authored research report. Keep the author's identity clear.

## Legacy migration
- `legacy_code/` is reference material from a different project, not a dependency
  of this project. Port only relevant code/data into the current structure.
- The entire legacy folder is intended for eventual deletion. New code, tests,
  configurations and data must work without it. Do not delete it without approval.
- Shared measurement, steering and lens code belongs in `core/`.

## Rules
- Distinguish hypotheses from confirmed findings.
- Never invent results or claim tests passed without running them.
- Do not silently change experimental parameters.
- Ask before expensive compute or destructive operations.
- Keep context updates concise and avoid duplication.
- Every research plot must have a plain-language caption in the figure itself.
  Explain what was measured, the axes and units, what each line/color/condition means, how to
  read important controls, and the main interpretive limit. Captions must make
  standalone image files understandable without opening code.
