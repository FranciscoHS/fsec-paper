"""
30 matched first-person/third-person prompt pairs for contrastive steering.

Bijective pronoun map (I↔he, me↔him, my↔his, mine↔his, myself↔himself).
The pronoun-bearing word is at the last-token position so it's captured.
Same length, same structure across each pair.
"""

PROMPT_PAIRS = [
    ("The award was given to me", "The award was given to him"),
    ("The package finally arrived for me", "The package finally arrived for him"),
    ("The teacher publicly praised me", "The teacher publicly praised him"),
    ("The committee voted in favor of me", "The committee voted in favor of him"),
    ("The whole crowd was watching me", "The whole crowd was watching him"),
    ("The decision was made entirely by me", "The decision was made entirely by him"),
    ("The letter was addressed only to me", "The letter was addressed only to him"),
    ("The book was actually written by me", "The book was actually written by him"),
    ("The dog ran straight toward me", "The dog ran straight toward him"),
    ("The neighbors were always kind to me", "The neighbors were always kind to him"),
    ("The reporter wanted to interview me", "The reporter wanted to interview him"),
    ("The professor pointed directly at me", "The professor pointed directly at him"),
    ("The music sounded too loud to me", "The music sounded too loud to him"),
    ("The whole project was assigned to me", "The whole project was assigned to him"),
    ("The criticism was clearly aimed at me", "The criticism was clearly aimed at him"),
    ("The host insisted on serving me", "The host insisted on serving him"),
    ("The audience laughed loudly at me", "The audience laughed loudly at him"),
    ("The doctor handed the results to me", "The doctor handed the results to him"),
    ("The crowd cheered loudly for me", "The crowd cheered loudly for him"),
    ("The captain saluted formally at me", "The captain saluted formally at him"),
    ("The waiter brought the bill to me", "The waiter brought the bill to him"),
    ("The judge ruled in favor of me", "The judge ruled in favor of him"),
    ("The proposal had been written by me", "The proposal had been written by him"),
    ("The questions were directed mainly at me", "The questions were directed mainly at him"),
    ("The package was clearly intended for me", "The package was clearly intended for him"),
    ("The blame was unfairly placed on me", "The blame was unfairly placed on him"),
    ("The inheritance went straight to me", "The inheritance went straight to him"),
    ("The painting had been made by me", "The painting had been made by him"),
    ("The applause was clearly meant for me", "The applause was clearly meant for him"),
    ("The strange parcel was addressed to me", "The strange parcel was addressed to him"),
]

FIRST_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
THIRD_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
