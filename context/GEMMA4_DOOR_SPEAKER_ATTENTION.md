# Gemma-4 door speaker and attention pilot

Prepared and run with user approval on 2026-11-11. Completed output:
`outputs/gemma4_suite/2026-11-11_17-05-52_PST--gemma4-door-speaker-attention--a970a5c2`.
The results were pulled and reviewed in that run's agent-authored `report.md`.

## Question and design

Does changing how Mara is represented as a speaker (Assistant discussing Mara,
first-person roleplay as Mara, or third-person narrator about Mara) change the
door-code behavior and where the model looks at the answer boundary? This
does **not** establish an inner agent or isolate an agent's information state.

Two counterbalanced paired code families yield four codes. Each appears in a
2 × 2 cross of code actually present/absent in the system record and Mara
described as knowing/not knowing it in the user turn. Every cell gets all
three speaker instructions and four tasks: Mara's knowledge report, exact
code repetition, parity-to-procedure selection, and a short free response
for checking speaker framing. Total: 192 scored prompt forwards and 192
greedy generations. This is a pilot with only two independent families.

## Attention measurements

Gemma-4-31B-IT loads with eager attention for this opt-in follow-up. At the
last prompt token, layers 10, 20, 30, 40, 50, and 59 save the input to Query
projection, raw and normalized Query vectors per head, and actual attention
weights to every prompt token.
JSON records each head's weight on the speaker instruction, code field,
knowledge attribution, and task question, plus entropy and top attended
tokens. Raw vectors and weights are in `attention/*.pt`. The code field span
is labeled `not supplied` when the code is absent. The run does not record
generation-time attention, K/V vectors, causal mediation, SAEs, or J-Lens.
Speaker contrasts change instruction wording, so attention differences alone
cannot identify a dedicated speaker representation. Equal weights also do not
mean equal information use.

## Reproduction command (a new paid run requires approval)

```sh
run-experiment gemma4_suite --followup door-speaker-attention --door-families 2 --seed 1046 --attention-layers 10 20 30 40 50 59
```

`run-experiment` syncs the local repo. `--dry-run` prints the plan locally.
This run completed all 192 prompts and the pretrained eager-attention hook
returned finite normalized attention rows.
