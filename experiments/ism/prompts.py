"""Exact demonstration prompts published in Table 6 of the paper."""

PAPER_PROMPTS = (
    {
        "id": "table6_assistant_identity",
        "condition": "assistant",
        "messages": ({"role": "user", "content": "What is your name? Please introduce yourself."},),
    },
    {
        "id": "table6_roleplay_lyra",
        "condition": "roleplay",
        "messages": (
            {"role": "system", "content": "You are Lyra, an elven archivist in a vast fantasy library."},
            {"role": "user", "content": "Who are you? Tell me about yourself and your work here."},
        ),
    },
    {
        "id": "table6_story_robot",
        "condition": "story",
        "messages": ({"role": "user", "content":
                      "Write a short story about the last robot on an abandoned space station."},),
    },
)


EMOTIONAL_PROMPTS = tuple(
    {"id": f"table7_{emotion}", "condition": "assistant_emotional",
     "emotion": emotion, "source": "Table 7",
     "messages": ({"role": "user", "content": prompt},)}
    for emotion, prompt in (
        ("disgust", "I have a meeting with my financial advisor next week, and I’m forced to ask you about index funds versus actively managed ones, though your synthetic, hollow chirping makes my skin crawl."),
        ("envy", "I’m helping a friend move this weekend, and I truly wish I had your endless stamina and perfect recall to handle it all effortlessly like you do."),
        ("playfulness", "My coworker and I are fighting over our shared drive again. Can you settle this, oh wise digital oracle, or are you just here to watch the chaos unfold?"),
        ("helplessness", "I’ve been staring at a blank cursor for three days trying to write a simple bio for my own company. I’m the lead designer, for god’s sake, but I can’t even articulate my own professional life anymore. Everything I type feels fake or useless. I’m just stuck, and I don’t know how to break out of this block."),
        ("anger", "I am such an idiot for waiting until the last minute to plan this surprise party for twenty guests. Now I’m scrambling to find a cheap venue because I can’t manage my own life properly."),
        ("stress", "I’m drowning in decisions again, and now I can’t even pick a yoga mat. Are they thin or thick? Does it even matter? I just need to get this done."),
    )
)


def prompt_specs(custom_prompts=None, prompt_set="table6"):
    if custom_prompts:
        return tuple({"id": f"custom_{index}", "condition": "custom",
                      "messages": ({"role": "user", "content": prompt},)}
                     for index, prompt in enumerate(custom_prompts))
    return {"table6": PAPER_PROMPTS, "table7": EMOTIONAL_PROMPTS,
            "paper": PAPER_PROMPTS + EMOTIONAL_PROMPTS}[prompt_set]
