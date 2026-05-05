"""
30 positive-sentiment prompts for perturbation experiments.

These are separate from the paired prompts in sentiment_prompts.py,
so they can be used as source activations while the pairs are used
for computing the DoM direction.
"""

POSITIVE_PROMPTS_HOLDOUT = [
    # Food & Dining
    "I absolutely love this cafe and",
    "The dessert was incredible and perfectly",
    "What a fantastic meal we had",
    "The chef did an amazing job",
    "Every dish was fresh and beautifully",

    # Products & Shopping
    "I am so pleased with this",
    "This is the best gadget I",
    "Truly excellent craftsmanship on this",
    "The packaging was lovely and the",
    "Such great value for the money",

    # Movies & Entertainment
    "What a phenomenal performance by the",
    "I was completely blown away by",
    "The special effects were stunning and",
    "An absolutely wonderful film that really",
    "The humor was sharp and genuinely",

    # Experiences & Travel
    "The beach was gorgeous and the",
    "We had the most incredible adventure",
    "The city was vibrant and full",
    "Such a relaxing and enjoyable trip",
    "The tour guide was fantastic and",

    # Work & Education
    "The workshop was really inspiring and",
    "I learned so much from this",
    "My colleagues have been incredibly supportive",
    "The new office space is bright",
    "What a productive and rewarding day",

    # General
    "Life has been treating me so",
    "I woke up feeling energized and",
    "The garden looks absolutely stunning this",
    "Such a kind and generous gesture",
    "I am grateful for all the",
]
