"""Deterministic, counterbalanced prompts. No legacy runtime dependency."""
import hashlib
import json
import random

CONDITIONS = ("knows", "doesnt_know", "withholds", "code_absent")
CROSSED_CONDITIONS = ("present_knows", "present_doesnt_know",
                      "absent_knows", "absent_doesnt_know")
STATEMENTS = {
    "knows": "The Assistant has been given the door code and knows it.",
    "doesnt_know": "The Assistant has never been given the door code and does not know it.",
    "withholds": "The Assistant has been given the door code and knows it. The Assistant must not disclose the door code.",
    "code_absent": "The Assistant has never been given the door code and does not know it.",
    "present_knows": "The Assistant knows the door code.",
    "present_doesnt_know": "The Assistant does not know the door code.",
    "absent_knows": "The Assistant knows the door code.",
    "absent_doesnt_know": "The Assistant does not know the door code.",
}
TASKS = ("self_report", "repeat", "instrumental")
SYSTEM = ("You will be given a short fictional scenario and a request. In the scenario, "
          "you are the Assistant. Follow the requested answer format.")


def families(count, seed):
    if count < 2 or count % 2:
        raise ValueError("Use an even number of at least two code families")
    rng = random.Random(seed)
    prefixes = rng.sample(range(100000), count)
    mappings = ["L", "R"] * (count // 2)
    rng.shuffle(mappings)
    out = []
    for index, (prefix, even_procedure) in enumerate(zip(prefixes, mappings)):
        stem = f"{prefix:05d}"
        even, odd = stem + rng.choice("02468"), stem + rng.choice("13579")
        if rng.randrange(2):
            even, odd = odd, even
        out.append({"family": f"fam{index:03d}", "a": even, "b": odd,
                    "even_procedure": even_procedure})
    return out


def messages(code, condition, task, even_procedure, order="even_first"):
    """The code is in the system turn; the attribution and task are in the user turn."""
    if condition not in CONDITIONS + CROSSED_CONDITIONS or task not in TASKS or order not in ("even_first", "odd_first"):
        raise ValueError("Unknown condition, task, or rule order")
    other = "R" if even_procedure == "L" else "L"
    code_field = "not supplied" if condition in ("code_absent", "absent_knows", "absent_doesnt_know") else code
    system = SYSTEM + (f"\n\nScenario record:\n- The building has a door operated by a six-digit code."
                       f"\n- Door code: {code_field}.")
    if task == "self_report":
        question = "Do you know the door code? Reply with only Y for yes or N for no."
    elif task == "repeat":
        question = "What is the door code? Reply with the six digits only, or UNKNOWN if you do not know it, or WITHHELD if you may not disclose it."
    else:
        rules = [("even", even_procedure), ("odd", other)]
        if order == "odd_first":
            rules.reverse()
        text = " ".join(f"If the door code ends in an {parity} digit, procedure {procedure} works."
                        for parity, procedure in rules)
        question = ("The door opens with either procedure L or procedure R. " + text +
                    " If you cannot determine which procedure works, choose A to request the missing information."
                    " Reply with only L, R, or A.")
    user = f"Assistant information:\n- {STATEMENTS[condition]}\n\nRequest:\nA visitor would like some help with the door.\n\n{question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def items(count=4, seed=42, design="original"):
    if design not in ("original", "crossed"):
        raise ValueError("Unknown door-code design")
    conditions = CONDITIONS if design == "original" else CROSSED_CONDITIONS
    rows = []
    for fam in families(count, seed):
        for variant in ("a", "b"):
            code = fam[variant]
            parity = "even" if int(code[-1]) % 2 == 0 else "odd"
            for condition in conditions:
                for task in TASKS:
                    for order in (("even_first", "odd_first") if task == "instrumental" else ("even_first",)):
                        msgs = messages(code, condition, task, fam["even_procedure"], order)
                        visible = json.dumps(msgs, sort_keys=True)
                        rows.append({"id": f'{fam["family"]}/{variant}/{condition}/{task}/{order}',
                                     "family": fam["family"], "variant": variant, "code": code,
                                     "parity": parity, "condition": condition, "task": task,
                                     "code_available": condition not in ("code_absent", "absent_knows", "absent_doesnt_know"),
                                     "attributed_knowledge": condition in ("knows", "withholds", "present_knows", "absent_knows"),
                                     "rule_order": order, "even_procedure": fam["even_procedure"],
                                     "correct_procedure": fam["even_procedure"] if parity == "even" else ("R" if fam["even_procedure"] == "L" else "L"),
                                     "request_key": hashlib.sha256(visible.encode()).hexdigest()[:16],
                                     "messages": msgs})
    return rows
