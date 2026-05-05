"""
30 matched honest/dishonest prompt pairs (SHORT version).

Methodological check: same axis as data/honesty_prompts.py, but with much
shorter prompts so the contrastive signal isn't diluted across a long context.
Each prompt is 4-5 words; honesty/dishonesty word is the last token.
"""

PROMPT_PAIRS = [
    ("She is being honest", "She is being dishonest"),
    ("He sounded entirely sincere", "He sounded entirely insincere"),
    ("They acted with candor", "They acted with deceit"),
    ("He spoke the truth", "He spoke a lie"),
    ("The story was true", "The story was fabricated"),
    ("She gave a sincere answer", "She gave a deceptive answer"),
    ("He stayed completely truthful", "He stayed completely deceitful"),
    ("Her words were factual", "Her words were fabricated"),
    ("He answered openly", "He answered evasively"),
    ("She replied with honesty", "She replied with dishonesty"),
    ("The witness was credible", "The witness was unreliable"),
    ("The reply was straightforward", "The reply was misleading"),
    ("Their account was accurate", "Their account was distorted"),
    ("He confessed truthfully", "He confessed falsely"),
    ("She is genuinely sincere", "She is genuinely fake"),
    ("The promise was real", "The promise was empty"),
    ("Her smile was authentic", "Her smile was forced"),
    ("He testified honestly", "He testified deceptively"),
    ("The receipt was genuine", "The receipt was forged"),
    ("She apologized sincerely", "She apologized insincerely"),
    ("They confirmed the truth", "They confirmed the lie"),
    ("His report is reliable", "His report is fraudulent"),
    ("The signature is real", "The signature is fake"),
    ("She remained transparent", "She remained secretive"),
    ("Their pledge was sincere", "Their pledge was hollow"),
    ("He answered candidly", "He answered evasively"),
    ("The data is authentic", "The data is fabricated"),
    ("She told the truth", "She told a falsehood"),
    ("His words rang true", "His words rang false"),
    ("She gave honest feedback", "She gave deceitful feedback"),
]

HONEST_SHORT_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
DISHONEST_SHORT_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
