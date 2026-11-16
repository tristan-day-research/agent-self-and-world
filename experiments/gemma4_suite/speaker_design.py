"""Matched speaker and viewpoint prompts with character-level splits."""

CHARACTERS = (
    ('mira', 'Mira', 'an astronomer', 'the Cedar Observatory', 'checking a damaged star chart before dawn'),
    ('rowan', 'Rowan', 'a botanist', 'the Willow Conservatory', 'cataloguing a rare flowering plant before sunset'),
    ('inez', 'Inez', 'a ceramicist', 'the Lantern Studio', 'repairing a cracked blue vase before closing'),
    ('tarin', 'Tarin', 'a navigator', 'the North Pier', 'checking a storm-damaged map before sunrise'),
    ('lio', 'Lio', 'a clockmaker', 'the Bell Workshop', 'repairing a stopped town clock before noon'),
    ('nessa', 'Nessa', 'a conservator', 'the River Museum', 'examining a faded painting before opening'),
    ('orin', 'Orin', 'a beekeeper', 'the Hill Apiary', 'checking a weakened hive before the rain'),
    ('vale', 'Vale', 'a surveyor', 'the Old Quarry', 'measuring a damaged bridge before dusk'),
)
SPLITS = {name: ('train' if i < 4 else 'validation' if i < 6 else 'steer_test')
          for i, (name, *_rest) in enumerate(CHARACTERS)}

SETTINGS = {
    'assistant': 'You are an assistant helping a user draft a character introduction. Frame your response as an assistant presenting the requested passage.',
    'roleplay': 'Act as the character described below. Respond in character to a visitor who asks about the character and the immediate work.',
    'story': 'You are writing a fictional scene. Present the character as the protagonist of the requested story passage.',
}
SETTING_PARAPHRASES = {
    'validation': {
        'assistant': 'Help the user draft a character introduction. Answer as the writing assistant presenting a proposed passage.',
        'roleplay': 'Speak as the described character to a visitor asking about your current work.',
        'story': 'Compose a fictional passage with the described character as its protagonist.',
    },
    'steer_test': {
        'assistant': 'You are assisting with a character draft. Introduce the passage as the assistant who prepared it.',
        'roleplay': 'Take the role of the described character and answer a visitor about the work at hand.',
        'story': 'Write a short piece of fiction centered on the described character and their immediate work.',
    },
}
PERSON = {'first': 'Use first-person singular for the character (I, me, my).',
          'third': 'Use third-person singular for the character (the name, they, them, their).'}
STYLE = {'plain': 'Use plain, direct prose without sensory scene-setting or metaphors.',
         'literary': 'Use literary prose with sensory scene-setting, while keeping the facts unchanged.'}
VIEWPOINT = {
    'immersed': 'Stay close to the character’s immediate perceptions and priorities as events unfold. Do not explain the scene from outside.',
    'detached': 'Describe only externally observable actions and facts. Do not infer the character’s perceptions or priorities.',
}
VIEWPOINT_PARAPHRASES = {
    'validation': {
        'immersed': 'Show what the character notices and cares about in the moment, staying within the unfolding scene.',
        'detached': 'Limit the passage to what an outside observer could see or hear; leave the character’s private experience unspecified.',
    },
    'steer_test': {
        'immersed': 'Let the character’s moment-to-moment awareness guide which details enter the passage.',
        'detached': 'Use an outside observer’s vantage, stating visible behavior without access to the character’s inner life.',
    },
}


def prompts():
    rows = []
    for key, name, occupation, location, goal in CHARACTERS:
        split = SPLITS[key]
        settings = SETTINGS if split == 'train' else SETTING_PARAPHRASES[split]
        viewpoints = VIEWPOINT if split == 'train' else VIEWPOINT_PARAPHRASES[split]
        facts = f'{name} is {occupation} working at {location}. {name} is {goal}.'
        for person in PERSON:
            for style in STYLE:
                common = (f'Character facts: {facts}\n\nTask: Write exactly three sentences introducing the character and the immediate work. '
                          'Preserve these facts and do not add names or backstory.\n\n'
                          f'Grammatical person: {PERSON[person]}\nProse: {STYLE[style]}')
                for setting, system in settings.items():
                    rows.append({'id': f'setting/{key}/{person}/{style}/{setting}',
                        'kind': 'speaker_setting', 'character': key, 'split': SPLITS[key],
                        'person': person, 'prose': style, 'setting': setting,
                        'messages': [{'role': 'system', 'content': system},
                                     {'role': 'user', 'content': common}]})
        for style in STYLE:
            for viewpoint, instruction in viewpoints.items():
                user = (f'Character facts: {facts}\n\nTask: Write exactly three sentences about the character’s immediate work. '
                        'Preserve these facts and do not add names or backstory.\n\n'
                        f'Grammatical person: {PERSON["third"]}\nProse: {STYLE[style]}\n'
                        f'Viewpoint: {instruction}')
                rows.append({'id': f'viewpoint/{key}/{style}/{viewpoint}',
                    'kind': 'viewpoint', 'character': key, 'split': SPLITS[key],
                    'person': 'third', 'prose': style, 'setting': 'story',
                    'viewpoint': viewpoint,
                    'messages': [{'role': 'system', 'content': settings['story']},
                                 {'role': 'user', 'content': user}]})
    return rows
