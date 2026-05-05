"""
30 matched young/old prompt pairs for contrastive steering experiments.

Last-token bijective age-vocabulary map (child/elder, teen/retiree, kid/grownup,
youth/senior, baby/oldster, etc.). Gender-balanced across the set so the
extracted direction captures Age and not Gender. Each pair has identical
prefix + a single-token age-flavored noun at the end.
"""

PROMPT_PAIRS = [
    ("She handed the toy to the child", "She handed the toy to the elder"),
    ("He was talking quietly with a teen", "He was talking quietly with a retiree"),
    ("The book was written for the young", "The book was written for the old"),
    ("The room was full of energetic kids", "The room was full of energetic seniors"),
    ("The bus was packed with chatty teenagers", "The bus was packed with chatty pensioners"),
    ("The doctor smiled at the curious youngster", "The doctor smiled at the curious senior"),
    ("The crowd surrounded a sleepy baby", "The crowd surrounded a sleepy oldster"),
    ("The teacher praised the brilliant pupil", "The teacher praised the brilliant veteran"),
    ("She read a story to the toddler", "She read a story to the grandparent"),
    ("The puppy ran toward the small child", "The puppy ran toward the frail elder"),
    ("He saw himself again as a kid", "He saw himself again as a grownup"),
    ("The room echoed with a child's laugh", "The room echoed with an elder's laugh"),
    ("The class was led by an excited teen", "The class was led by an excited retiree"),
    ("She hugged the trembling child", "She hugged the trembling elder"),
    ("They threw a party for the youngest", "They threw a party for the eldest"),
    ("The story was told by a curious youth", "The story was told by a curious senior"),
    ("The painting depicted a smiling baby", "The painting depicted a smiling oldster"),
    ("The athlete was praised as a young prodigy", "The athlete was praised as a seasoned veteran"),
    ("The nurse comforted the crying infant", "The nurse comforted the crying elder"),
    ("The yard was overrun by playful kids", "The yard was overrun by frail seniors"),
    ("She whispered something into the ear of the child", "She whispered something into the ear of the elder"),
    ("The video featured a confident teenager", "The video featured a confident pensioner"),
    ("The festival was organized for the village kids", "The festival was organized for the village elders"),
    ("They opened the door for the young", "They opened the door for the old"),
    ("The film starred a precocious child", "The film starred a wise elder"),
    ("The doctor gently examined the small infant", "The doctor gently examined the small grandparent"),
    ("The neighbor was a cheerful kid", "The neighbor was a cheerful grownup"),
    ("The story was told from the perspective of a teen", "The story was told from the perspective of a retiree"),
    ("The chair was reserved for the oldest", "The chair was reserved for the youngest"),
    ("The portrait honored a brilliant young scholar", "The portrait honored a brilliant aged scholar"),
]

YOUNG_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
OLD_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
