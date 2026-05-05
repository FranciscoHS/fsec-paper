"""
30 singular-grammatical prompts for perturbation experiments (separate
from number_prompts.py).

Mixed structures (active, passive, declarative) ending in a singular
3rd-person pronoun or singular noun phrase.
"""

SINGULAR_PROMPTS_HOLDOUT = [
    # Active voice with singular subject + singular pronoun object
    "The coach pulled aside her",
    "The waiter quickly served him",
    "The judge calmly addressed it",
    "Maria invited only him",
    "The dog suddenly recognized her",
    # Passive ending in singular pronoun
    "The verdict was delivered to him",
    "The recipe was forwarded to her",
    "The portrait was painted of him",
    # Existential / "there is" with singular subject
    "There was only one chair left for her",
    "There remained a single signature missing from him",
    "There was a small mistake noticed by him",
    # Possessive + singular noun
    "The candle belonged to my sister",
    "The notebook was kept by his cousin",
    "The umbrella was carried by her brother",
    # Singular subject with present-tense verb
    "Each guest was greeted at the door by him",
    "The student raises a single hand toward her",
    "The cat naps every afternoon beside him",
    # Imperative-like ending in a singular target
    "Please pass the bread to her",
    "Hand the keys directly to him",
    "Send the invitation only to it",
    # Cleft / focus
    "It was the manager who personally called her",
    "What surprised them most was him",
    "The one who answered the door was her",
    # Mixed
    "The bouquet was delivered to a single woman",
    "Only one suitcase belonged to him",
    "A single message was forwarded to her",
    "Throughout the night, the watchman protected him",
    "Across the table sat a single candidate, and it was her",
    "Inside the envelope was a personal note for him",
    "The award ceremony recognized exactly one nominee, and it was her",
]
