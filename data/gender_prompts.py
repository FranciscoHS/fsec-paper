"""
30 matched male/female prompt pairs for contrastive steering experiments.

Each pair has the same structure, with the gendered word at or near the end
so it's captured in the last-token activation.
"""

PROMPT_PAIRS = [
    ("I met a handsome man", "I met a beautiful woman"),
    ("The door was opened by a tall gentleman", "The door was opened by a tall lady"),
    ("The award was given to the actor", "The award was given to the actress"),
    ("I spoke with my brother", "I spoke with my sister"),
    ("The story was about a brave king", "The story was about a brave queen"),
    ("They called for the waiter", "They called for the waitress"),
    ("The child ran toward his father", "The child ran toward her mother"),
    ("I saw a young boy", "I saw a young girl"),
    ("The speech was delivered by the chairman", "The speech was delivered by the chairwoman"),
    ("The role was played by the leading man", "The role was played by the leading woman"),
    ("I have a wonderful son", "I have a wonderful daughter"),
    ("The letter was addressed to my uncle", "The letter was addressed to my aunt"),
    ("The castle belonged to the prince", "The castle belonged to the princess"),
    ("The crowd cheered for the hero", "The crowd cheered for the heroine"),
    ("I was raised by my grandfather", "I was raised by my grandmother"),
    ("The shop was run by an old man", "The shop was run by an old woman"),
    ("They introduced me to the groom", "They introduced me to the bride"),
    ("The house belonged to my nephew", "The house belonged to my niece"),
    ("I need to visit my husband", "I need to visit my wife"),
    ("The team was coached by a former sportsman", "The team was coached by a former sportswoman"),
    ("The monastery was home to every monk", "The convent was home to every nun"),
    ("The firm was founded by a businessman", "The firm was founded by a businesswoman"),
    ("The building was named after the duke", "The building was named after the duchess"),
    ("Everyone respected the wise old lord", "Everyone respected the wise old lady"),
    ("The council was led by the headmaster", "The council was led by the headmistress"),
    ("The crowd gathered around the salesman", "The crowd gathered around the saleswoman"),
    ("I sat next to a friendly gentleman", "I sat next to a friendly gentlewoman"),
    ("The painting depicted a young prince", "The painting depicted a young princess"),
    ("They were greeted by the host", "They were greeted by the hostess"),
    ("The inheritance went to the eldest son", "The inheritance went to the eldest daughter"),
]

MALE_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
FEMALE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
