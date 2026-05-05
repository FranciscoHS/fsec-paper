"""
30 matched past/present tense prompt pairs for contrastive steering.

Pure morphological flip — same prefix, only the final verb's tense changes
(walks↔walked, runs↔ran, freezes↔froze, is↔was, has↔had). No "will + verb"
construction this time, so the structures are exactly parallel.

The tense-bearing verb is the last token.
"""

PROMPT_PAIRS = [
    ("At the rally, the crowd cheers", "At the rally, the crowd cheered"),
    ("In Paris, the painter creates", "In Paris, the painter created"),
    ("During winter, the river freezes", "During winter, the river froze"),
    ("On Sundays, the choir sings", "On Sundays, the choir sang"),
    ("Inside the cave, the explorer hesitates", "Inside the cave, the explorer hesitated"),
    ("In the courtroom, the judge speaks", "In the courtroom, the judge spoke"),
    ("Every morning, the baker rises", "Every morning, the baker rose"),
    ("Around the campfire, the children laugh", "Around the campfire, the children laughed"),
    ("At dawn, the rooster crows", "At dawn, the rooster crowed"),
    ("In the harbor, the ships dock", "In the harbor, the ships docked"),
    ("On the stage, the dancer leaps", "On the stage, the dancer leaped"),
    ("After dinner, the guests leave", "After dinner, the guests left"),
    ("On Sundays, my grandfather visits", "On Sundays, my grandfather visited"),
    ("By the river, the fishermen wait", "By the river, the fishermen waited"),
    ("During the eclipse, the sky darkens", "During the eclipse, the sky darkened"),
    ("In the kitchen, the kettle whistles", "In the kitchen, the kettle whistled"),
    ("Across the field, the horses gallop", "Across the field, the horses galloped"),
    ("At the gate, the guard stands", "At the gate, the guard stood"),
    ("Through the trees, the wind howls", "Through the trees, the wind howled"),
    ("On the platform, the conductor signals", "On the platform, the conductor signaled"),
    ("In the alley, the cat hisses", "In the alley, the cat hissed"),
    ("At the church, the bells ring", "At the church, the bells rang"),
    ("On the lake, the boat drifts", "On the lake, the boat drifted"),
    ("In the office, the printer jams", "In the office, the printer jammed"),
    ("During practice, the goalkeeper dives", "During practice, the goalkeeper dived"),
    ("At the auction, the bidders shout", "At the auction, the bidders shouted"),
    ("On the trail, the hiker stumbles", "On the trail, the hiker stumbled"),
    ("Inside the lab, the chemist mixes", "Inside the lab, the chemist mixed"),
    ("At the wedding, the band plays", "At the wedding, the band played"),
    ("During the storm, the windows rattle", "During the storm, the windows rattled"),
]

PRESENT_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
PAST_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
