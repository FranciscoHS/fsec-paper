"""
30 matched singular/plural prompt pairs for contrastive steering experiments.

Bijective last-token grammatical-number map: pronoun (him/them, her/them,
it/them) and verb agreement (rings/ring, was/were, is/are). Gender-balanced
across the pronoun pairs.
"""

PROMPT_PAIRS = [
    ("The award was given to him", "The award was given to them"),
    ("The dog ran straight toward her", "The dog ran straight toward them"),
    ("The decision affected only him", "The decision affected only them"),
    ("The package was clearly addressed to her", "The package was clearly addressed to them"),
    ("The teacher pointed directly at him", "The teacher pointed directly at them"),
    ("The crowd cheered loudly for her", "The crowd cheered loudly for them"),
    ("The blame was unfairly placed on him", "The blame was unfairly placed on them"),
    ("On Sundays, the church bell rings", "On Sundays, the church bells ring"),
    ("In the cage, the parrot was", "In the cage, the parrots were"),
    ("Inside the office, the printer is", "Inside the office, the printers are"),
    ("On the field, the player was", "On the field, the players were"),
    ("On the roof, the pigeon coos", "On the roof, the pigeons coo"),
    ("In the garage, the engine roared", "In the garage, the engines roared"),
    ("By the door, a stranger was", "By the door, several strangers were"),
    ("The committee was led by him", "The committee was led by them"),
    ("Outside the gate, the guard waited", "Outside the gate, the guards waited"),
    ("In the hall, the visitor speaks", "In the hall, the visitors speak"),
    ("Upon the stage, a dancer leaps", "Upon the stage, the dancers leap"),
    ("On the porch, the neighbor sat", "On the porch, the neighbors sat"),
    ("Beside the window, the cat purrs", "Beside the window, the cats purr"),
    ("The trophy was claimed by her", "The trophy was claimed by them"),
    ("The applause was meant for him", "The applause was meant for them"),
    ("In the meadow, a horse grazes", "In the meadow, the horses graze"),
    ("Across the river, the swimmer is", "Across the river, the swimmers are"),
    ("At the gate, the soldier stood", "At the gate, the soldiers stood"),
    ("In the studio, the painter works", "In the studio, the painters work"),
    ("Beneath the tree, a child plays", "Beneath the tree, the children play"),
    ("In the kitchen, the chef cooks", "In the kitchen, the chefs cook"),
    ("On the mountain, the climber rests", "On the mountain, the climbers rest"),
    ("At the rally, the speaker stood", "At the rally, the speakers stood"),
]

SINGULAR_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
PLURAL_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
