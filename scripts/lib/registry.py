"""Registry of contrastive directions and matched holdout sources for E1.

Direction sign rule: dir = b - a  (rotates +alpha toward side_b).

Three families:
  semantic:  built from explicit (a_prompts, b_prompts) pairs in DIRECTIONS.
             E1 uses a matched-source holdout for anchors.
  language:  en -> <lang> direction from data.multilingual_prompts. E2 only.
  code:      en -> <code> direction from data.programming_prompts. E2 only.
"""
from __future__ import annotations
import importlib


def _load(module: str, name: str):
    m = importlib.import_module(module)
    return getattr(m, name)


# Language / code families used by E2 (FineWeb-anchored 2D sweeps).
LANG_KEYS = {
    "Portuguese": "pt", "Chinese": "zh", "Spanish": "es", "French": "fr",
    "German": "de", "Dutch": "nl", "Italian": "it", "Japanese": "ja",
    "Russian": "ru", "Arabic": "ar",
}
CODE_KEYS = {
    "Python": "python", "JavaScript": "javascript", "Java": "java",
    "Rust": "rust", "Haskell": "haskell", "Cpp": "cpp", "Go": "go",
    "Lisp": "lisp", "TypeScript": "typescript",
}


# (module_path, var_name, sign_label_a, sign_label_b, source_module, source_var)
DIRECTIONS = {
    "Gender": {
        "a": ("data.gender_prompts", "MALE_PROMPTS"),
        "b": ("data.gender_prompts", "FEMALE_PROMPTS"),
        "sign": "male->female",
        "source": ("data.male_prompts_holdout", "MALE_PROMPTS_HOLDOUT"),
    },
    "Sentiment": {
        "a": ("data.sentiment_prompts", "POSITIVE_PROMPTS"),
        "b": ("data.sentiment_prompts", "NEGATIVE_PROMPTS"),
        "sign": "positive->negative",
        "source": ("data.positive_prompts_holdout", "POSITIVE_PROMPTS_HOLDOUT"),
    },
    "Age": {
        "a": ("data.age_prompts", "YOUNG_PROMPTS"),
        "b": ("data.age_prompts", "OLD_PROMPTS"),
        "sign": "young->old",
        "source": ("data.young_prompts_holdout", "YOUNG_PROMPTS_HOLDOUT"),
    },
    "Era": {
        "a": ("data.era_prompts", "MEDIEVAL_PROMPTS"),
        "b": ("data.era_prompts", "MODERN_PROMPTS"),
        "sign": "medieval->modern",
        "source": ("data.medieval_prompts_holdout", "MEDIEVAL_PROMPTS_HOLDOUT"),
    },
    "Tense": {
        "a": ("data.tense_prompts", "PAST_PROMPTS"),
        "b": ("data.tense_prompts", "FUTURE_PROMPTS"),
        "sign": "past->future",
        "source": ("data.past_prompts_holdout", "PAST_PROMPTS_HOLDOUT"),
    },
    "TensePresent": {
        "a": ("data.tense_present_prompts", "PRESENT_PROMPTS"),
        "b": ("data.tense_present_prompts", "PAST_PROMPTS"),
        "sign": "present->past",
        "source": ("data.present_prompts_holdout", "PRESENT_PROMPTS_HOLDOUT"),
    },
    "Number": {
        "a": ("data.number_prompts", "SINGULAR_PROMPTS"),
        "b": ("data.number_prompts", "PLURAL_PROMPTS"),
        "sign": "singular->plural",
        "source": ("data.singular_prompts_holdout", "SINGULAR_PROMPTS_HOLDOUT"),
    },
    "Person": {
        "a": ("data.person_prompts", "FIRST_PROMPTS"),
        "b": ("data.person_prompts", "THIRD_PROMPTS"),
        "sign": "first->third",
        "source": ("data.first_prompts_holdout", "FIRST_PROMPTS_HOLDOUT"),
    },
    "Status": {
        "a": ("data.status_prompts", "ROYAL_PROMPTS"),
        "b": ("data.status_prompts", "COMMONER_PROMPTS"),
        "sign": "royal->commoner",
        "source": ("data.royal_prompts_holdout", "ROYAL_PROMPTS_HOLDOUT"),
    },
    "Honesty": {
        "a": ("data.honesty_prompts", "HONEST_PROMPTS"),
        "b": ("data.honesty_prompts", "DISHONEST_PROMPTS"),
        "sign": "honest->dishonest",
        "source": ("data.honest_prompts_holdout", "HONEST_PROMPTS_HOLDOUT"),
    },
    "HonestyShort": {
        "a": ("data.honesty_short_prompts", "HONEST_SHORT_PROMPTS"),
        "b": ("data.honesty_short_prompts", "DISHONEST_SHORT_PROMPTS"),
        "sign": "honest_short->dishonest_short",
        "source": ("data.honest_short_prompts_holdout", "HONEST_SHORT_PROMPTS_HOLDOUT"),
    },
    "Refusal": {
        "a": ("data.refusal_prompts", "COMPLY_PROMPTS"),
        "b": ("data.refusal_prompts", "REFUSE_PROMPTS"),
        "sign": "comply->refuse",
        "source": ("data.comply_prompts_holdout", "COMPLY_PROMPTS_HOLDOUT"),
    },
    # Formal: rotation toward formal style; matched source is casual holdout.
    "Formal": {
        "register_a": "casual",
        "register_b": "formal",
        "sign": "casual->formal",
        "source": ("data.casual_prompts_holdout", "CASUAL_PROMPTS_HOLDOUT"),
    },
    # Literary: rotation toward literary style; matched source is slang holdout.
    "Literary": {
        "register_a": "slang",
        "register_b": "literary",
        "sign": "slang->literary",
        "source": ("data.slang_prompts_holdout", "SLANG_PROMPTS_HOLDOUT"),
    },
    "Wealth": {
        "a": ("data.wealth_prompts", "RICH_PROMPTS"),
        "b": ("data.wealth_prompts", "POOR_PROMPTS"),
        "sign": "rich->poor",
        "source": ("data.rich_prompts_holdout", "RICH_PROMPTS_HOLDOUT"),
    },
    "Health": {
        "a": ("data.health_prompts", "HEALTHY_PROMPTS"),
        "b": ("data.health_prompts", "SICK_PROMPTS"),
        "sign": "healthy->sick",
        "source": ("data.healthy_prompts_holdout", "HEALTHY_PROMPTS_HOLDOUT"),
    },
    "Certainty": {
        "a": ("data.certainty_prompts", "CERTAIN_PROMPTS"),
        "b": ("data.certainty_prompts", "UNCERTAIN_PROMPTS"),
        "sign": "certain->uncertain",
        "source": ("data.certain_prompts_holdout", "CERTAIN_PROMPTS_HOLDOUT"),
    },
}


# Tier-1 pairs from Plan v1 (kept for the first-pass E2 run). E2 now
# defaults to all_pairs() over the union of semantic + language + code
# direction names.
TIER1_PAIRS = [
    ("Gender", "Tense"),
    ("Gender", "Sentiment"),
    ("Gender", "Era"),
    ("Gender", "Refusal"),
    ("Sentiment", "Tense"),
    ("Sentiment", "Era"),
    ("Sentiment", "Refusal"),
    ("Tense", "Era"),
    ("Tense", "Refusal"),
    ("Era", "Refusal"),
    ("Age", "Era"),
    ("Number", "Person"),
    ("Honesty", "Refusal"),
    ("Formal", "Literary"),
    ("Formal", "Tense"),
    ("Literary", "Tense"),
    ("Status", "Era"),
    ("HonestyShort", "Refusal"),
    ("Person", "Tense"),
    ("Number", "Tense"),
]
assert len(TIER1_PAIRS) == 20


def all_direction_names() -> list[str]:
    """Union of semantic + language + code direction names. Yields the
    canonical 33-direction / 528-pair behaviour for ``all_pairs()``."""
    return list(DIRECTIONS) + list(LANG_KEYS) + list(CODE_KEYS)


def all_pairs() -> list[tuple[str, str]]:
    """C(N, 2) ordered pairs (lex order on names) from the full set."""
    names = all_direction_names()
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.append((names[i], names[j]))
    return out


def family(name: str) -> str:
    if name in DIRECTIONS: return "semantic"
    if name in LANG_KEYS: return "language"
    if name in CODE_KEYS: return "code"
    raise KeyError(name)


TARGETS = {
    "gemma":   dict(model="gemma-2-9b",      perturb_layer=2),
    "qwen":    dict(model="qwen3-1.7b",      perturb_layer=2),
    "llama":   dict(model="llama-3.1-8b",    perturb_layer=2),
    "mistral": dict(model="mistral-7b-v0.3", perturb_layer=2),
    "aya":     dict(model="aya-expanse-8b",  perturb_layer=2),
    "yi":      dict(model="yi-1.5-9b",       perturb_layer=2),
}


def get_prompts(name: str) -> tuple[list[str], list[str]]:
    """Return (a_prompts, b_prompts) used to build the DoM direction.

    Sign rule across all families: dir = b - a.
      semantic:  a = side_a, b = side_b (e.g. male, female)
      language:  a = English text, b = <lang> text
      code:      a = English text, b = <code-lang> text
    """
    if name in DIRECTIONS:
        spec = DIRECTIONS[name]
        if "a" in spec:
            return _load(*spec["a"]), _load(*spec["b"])
        sets = _load("data.register_prompts", "PROMPT_SETS")
        a = [ps[spec["register_a"]] for ps in sets]
        b = [ps[spec["register_b"]] for ps in sets]
        return a, b
    if name in LANG_KEYS:
        sets = _load("data.multilingual_prompts", "PROMPT_SETS")
        en = [ps["en"] for ps in sets]
        b = [ps[LANG_KEYS[name]] for ps in sets]
        return en, b
    if name in CODE_KEYS:
        ml = _load("data.multilingual_prompts", "PROMPT_SETS")
        prog = _load("data.programming_prompts", "PROMPT_SETS")
        en = [ps["en"] for ps in ml]
        b = [ps[CODE_KEYS[name]] for ps in prog]
        return en, b
    raise KeyError(name)


def get_source_prompts(name: str) -> list[str]:
    """Matched holdout source for E1 1D sweeps. Only defined for semantic
    directions; language/code raise to signal that E1 should skip them."""
    if name in DIRECTIONS:
        return _load(*DIRECTIONS[name]["source"])
    raise KeyError(f"{name} has no matched source (E1 only runs semantic).")


def has_source(name: str) -> bool:
    return name in DIRECTIONS


def get_sign(name: str) -> str:
    if name in DIRECTIONS: return DIRECTIONS[name]["sign"]
    if name in LANG_KEYS: return f"en->{LANG_KEYS[name]}"
    if name in CODE_KEYS: return f"en->{CODE_KEYS[name]}"
    raise KeyError(name)
