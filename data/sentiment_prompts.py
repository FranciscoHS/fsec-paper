"""
30 matched positive/negative sentiment prompt pairs for contrastive steering experiments.

Each pair has the same structure, differing only in sentiment-bearing words.
"""

PROMPT_PAIRS = [
    # Food & Dining
    ("The food was absolutely delicious and", "The food was absolutely terrible and"),
    ("This restaurant has excellent service and", "This restaurant has awful service and"),
    ("The meal was perfectly cooked and", "The meal was horribly undercooked and"),
    ("I loved every bite of the", "I hated every bite of the"),
    ("The flavors were wonderfully balanced and", "The flavors were disgustingly unbalanced and"),

    # Products & Shopping
    ("This product exceeded all my expectations and", "This product failed all my expectations and"),
    ("The quality is outstanding for the", "The quality is dreadful for the"),
    ("I am thrilled with this purchase because", "I am furious with this purchase because"),
    ("The design is beautiful and the", "The design is ugly and the"),
    ("Everything works perfectly right out of", "Nothing works properly right out of"),

    # Movies & Entertainment
    ("The movie was brilliant and kept me", "The movie was awful and bored me"),
    ("The acting was superb throughout the", "The acting was wooden throughout the"),
    ("I thoroughly enjoyed every minute of", "I deeply regretted every minute of"),
    ("The story was captivating from the", "The story was tedious from the"),
    ("The soundtrack was magnificent and perfectly", "The soundtrack was grating and poorly"),

    # Experiences & Travel
    ("The hotel room was spacious and", "The hotel room was cramped and"),
    ("Our vacation was wonderful and full of", "Our vacation was miserable and full of"),
    ("The view from the top was breathtaking and", "The view from the top was disappointing and"),
    ("The staff were incredibly friendly and", "The staff were incredibly rude and"),
    ("I had an amazing time at the", "I had a horrible time at the"),

    # Work & Education
    ("The presentation was clear and well", "The presentation was confusing and poorly"),
    ("My experience with this team has been fantastic because", "My experience with this team has been terrible because"),
    ("The course material was engaging and", "The course material was boring and"),
    ("The manager is supportive and always", "The manager is hostile and never"),
    ("This was the best decision I have", "This was the worst decision I have"),

    # General
    ("Today has been a wonderful day filled with", "Today has been a dreadful day filled with"),
    ("I feel so happy and grateful for", "I feel so miserable and resentful about"),
    ("The weather is beautiful and perfect for", "The weather is terrible and unsuitable for"),
    ("Everything went smoothly and according to", "Everything went wrong and against"),
    ("I am really impressed by how well", "I am really disappointed by how poorly"),
]

# Convenience accessors
POSITIVE_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
NEGATIVE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
