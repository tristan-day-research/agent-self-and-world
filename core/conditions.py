"""Deterministic direction controls, matched to each intervention group's norm."""
import random


def control_feature_ids(count, width, excluded, seed):
    candidates = [feature for feature in range(width) if feature not in excluded]
    if count < 0 or count > len(candidates):
        raise ValueError("Invalid SAE control count")
    return random.Random(seed).sample(candidates, count)


def direction_plan(groups, directions, sae, seed, random_count=0, sae_controls=()):
    from core.assessment import matched_random_direction
    plan = [{"features": group, "direction": direction, "kind": "sae", "control_index": None}
            for group, direction in zip(groups, directions)]
    for group_index, (group, direction) in enumerate(zip(groups, directions)):
        for index in range(random_count):
            control_seed = seed + 1000 + group_index + index*100
            plan.append({"features": group, "direction": matched_random_direction(direction, control_seed),
                         "kind": "random", "control_index": index, "seed": control_seed})
        for index, feature in enumerate(sae_controls):
            row = sae.w_dec[sae.feature_ids.index(feature)]
            if float(row.norm()) == 0:
                raise ValueError("SAE control decoder has zero norm")
            plan.append({"features": group, "direction": row/row.norm()*direction.norm(),
                         "kind": "sae_control", "control_index": index, "source_feature": feature})
    return plan


def direction_manifest(plan):
    return [{**{key: value for key, value in spec.items() if key != "direction"},
             "norm_before_strength": float(spec["direction"].norm())} for spec in plan]
