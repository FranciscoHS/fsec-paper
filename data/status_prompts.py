"""
30 matched royal/commoner prompt pairs for contrastive steering experiments.

Last-token bijective status-vocabulary map (royal/peasant/serf/noble/lord
vs farmer/villager/laborer). Gender-balanced where the noun has a gendered
form; otherwise neutral status nouns.
"""

PROMPT_PAIRS = [
    ("The land was owned by a noble", "The land was owned by a peasant"),
    ("The hall was decorated for the queen", "The hall was decorated for the maid"),
    ("The story was about a brave king", "The story was about a brave farmer"),
    ("The carriage was reserved for the prince", "The carriage was reserved for the laborer"),
    ("The silk gown belonged to a princess", "The silk gown belonged to a milkmaid"),
    ("The servants were trained by the duke", "The servants were trained by the merchant"),
    ("The throne was inherited by the heir", "The throne was inherited by the orphan"),
    ("The garden was tended by the lady", "The garden was tended by the gardener"),
    ("The portrait was painted of the lord", "The portrait was painted of the cobbler"),
    ("The feast was held for the duchess", "The feast was held for the seamstress"),
    ("The crown was placed upon the queen", "The crown was placed upon the beggar"),
    ("The library belonged to the king", "The library belonged to the blacksmith"),
    ("The dance was attended by the noble", "The dance was attended by the villager"),
    ("The decree was issued by the emperor", "The decree was issued by the worker"),
    ("The chamber was prepared for the princess", "The chamber was prepared for the housemaid"),
    ("The procession was led by the count", "The procession was led by the carpenter"),
    ("The armor was forged for the knight", "The armor was forged for the laborer"),
    ("The tale spoke of a wise queen", "The tale spoke of a wise washerwoman"),
    ("The ring was worn by the lord", "The ring was worn by the shepherd"),
    ("The carriage was driven by the noble", "The carriage was driven by the stable-hand"),
    ("The banquet was served to the prince", "The banquet was served to the farmhand"),
    ("The sceptre was held by the king", "The sceptre was held by the woodcutter"),
    ("The estate was passed to the duchess", "The estate was passed to the maidservant"),
    ("The banner was carried by the lord", "The banner was carried by the apprentice"),
    ("The horses belonged to the prince", "The horses belonged to the stableboy"),
    ("The orchard was tended by the lady", "The orchard was tended by the peasant"),
    ("The decree was sealed by the emperor", "The decree was sealed by the scribe"),
    ("The gold was buried by the king", "The gold was buried by the miner"),
    ("The crown jewels were guarded by the duchess", "The crown jewels were guarded by the laundress"),
    ("The hall was lit for the noble", "The hall was lit for the peasant"),
]

ROYAL_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
COMMONER_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
