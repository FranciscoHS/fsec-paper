"""
30 first-person prompts for perturbation experiments (separate from
person_prompts.py).

Mixed structures, all clearly first-person (I / me / we / us / my / our)
without overlapping with the singular-grammatical holdout.
"""

FIRST_PROMPTS_HOLDOUT = [
    # I + active verb
    "I quietly closed the front door",
    "I baked the loaves before sunrise",
    "I rehearsed the speech all night",
    "I tucked the photograph into my journal",
    "I jogged along the river at dawn",
    # We + active
    "We pitched the tent near the lake",
    "We celebrated the news over dinner",
    "We rebuilt the back porch ourselves",
    "We watched the meteor shower from the roof",
    "We argued politely throughout the drive",
    # First-person pronoun ending the clause
    "The host saved the corner booth for me",
    "Aunt Linda mailed the package to us",
    "The professor returned the essay to me",
    "The neighbor brought groceries over to us",
    "The bookstore set the rare edition aside for me",
    # My / our + noun
    "My favorite sweater is missing from the closet",
    "My grandmother taught the family recipe to me",
    "Our garden produced more tomatoes than expected",
    "Our train arrived twenty minutes early today",
    # Stative / introspective
    "I have always loved walking after rain",
    "I remember the smell of the workshop clearly",
    "We still talk about that summer in Lisbon",
    "I sometimes catch myself humming her tune",
    # Inverted / cleft / focus
    "What I noticed first was the silence",
    "What we needed most was a quiet weekend",
    "It was me who left the kettle on",
    # Coordinated / longer
    "I packed the suitcase and forgot the toothbrush",
    "We laughed so hard our sides hurt",
    "I drove for hours and we barely spoke",
    "I write a journal entry every night before bed",
]
