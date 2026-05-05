"""
30 matched medieval-fantasy / modern prompt pairs for contrastive steering.

Vocabulary-domain swap (analogous to a language pair, but English-readable):
the same situation rendered with medieval vocabulary vs modern vocabulary.
The era-flavored noun is the last token.
"""

PROMPT_PAIRS = [
    ("The army was led by the brave king", "The team was led by the brave boss"),
    ("The traveler arrived at the ancient castle", "The traveler arrived at the modern hotel"),
    ("The wound was treated by the herbalist", "The wound was treated by the surgeon"),
    ("The villagers gathered around the blacksmith", "The neighbors gathered around the mechanic"),
    ("The message was carried by a swift messenger", "The message was sent through the fast network"),
    ("The young apprentice trained under the wizard", "The young intern trained under the engineer"),
    ("The town was protected by a wooden wall", "The city was protected by a steel fence"),
    ("The thief was caught by the watchman", "The thief was caught by the officer"),
    ("The sky was lit by the burning torch", "The room was lit by the bright lamp"),
    ("The contract was sealed with hot wax", "The contract was sealed with digital signature"),
    ("The ship was guided by the lighthouse", "The plane was guided by the radar"),
    ("The hero raised his shining sword", "The hero raised his shining pistol"),
    ("The royal court was entertained by the bard", "The whole crowd was entertained by the DJ"),
    ("The scholar studied the ancient parchment", "The scholar studied the digital document"),
    ("The town crier read the daily news", "The radio host read the daily news"),
    ("The peasants worked in the muddy fields", "The workers worked in the busy office"),
    ("The wound was bandaged by the healer", "The wound was bandaged by the medic"),
    ("The traveler rode across the green meadow", "The traveler drove across the green highway"),
    ("The merchant sold his wares at the market", "The merchant sold his goods at the store"),
    ("The captive escaped from the dungeon", "The captive escaped from the prison"),
    ("The lady wore a fine silk gown", "The lady wore a fine silk dress"),
    ("The army was crushed by the cavalry", "The army was crushed by the artillery"),
    ("The traveler crossed the river by raft", "The traveler crossed the river by ferry"),
    ("The gold was hidden inside the chest", "The gold was hidden inside the safe"),
    ("The crown was placed upon the heir", "The award was placed upon the winner"),
    ("The map was drawn by the cartographer", "The map was drawn by the satellite"),
    ("The library was guarded by the monk", "The library was guarded by the librarian"),
    ("The forest was haunted by a dragon", "The forest was haunted by a wolf"),
    ("The throne was claimed by the prince", "The office was claimed by the manager"),
    ("The traveler sought refuge in the chapel", "The traveler sought refuge in the office"),
]

MEDIEVAL_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
MODERN_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
