"""Prompt builders for DimASQP LLM pipelines.

Contains:
  1. build_pseudo_label_prompt        — vanilla few-shot pseudo-labeling (baseline)
  2. build_entity_grounding_prompt    — CCA Stage 1: entity context summarisation
  3. build_attribute_grounding_prompt — CCA Stage 2: attribute angle summarisation
  4. build_cca_generation_prompt      — CCA Stage 3: compositional generation
  5. build_cross_verify_prompt        — CCA verification: category consistency check

Output JSON schema for generation (strict):
{
  "quadruplets": [
    {
      "aspect":   "<substring of the sentence or \"NULL\">",
      "opinion":  "<substring of the sentence or \"NULL\">",
      "category": "<one of the provided category labels>",
      "valence":  <float in [1.0, 9.0]>,
      "arousal":  <float in [1.0, 9.0]>
    },
    ...
  ]
}
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

PROMPT_VERSION = "v3-isr-2026-04-21"


SYSTEM_PROMPT = (
    "You are an expert data annotator for the DimABSA 2026 task 3 (DimASQP). "
    "Given a sentence, you extract all (aspect, opinion, category, valence, arousal) "
    "quadruplets that describe an evaluation of some aspect of an entity mentioned in the sentence.\n\n"
    "Rules:\n"
    "1. `aspect` must be an exact substring of the sentence, or the literal string \"NULL\" "
    "when the aspect is implicit (not expressed in the text).\n"
    "2. `opinion` must be an exact substring of the sentence, or \"NULL\" when the sentiment "
    "is implicit. Do not paraphrase or normalize casing.\n"
    "3. `category` must be one of the provided category labels (ENTITY#ATTRIBUTE format). "
    "Never invent new labels.\n"
    "4. `valence` (pleasantness) and `arousal` (intensity) are floats in [1.0, 9.0]. "
    "Neutral ≈ 5.0; strongly positive ≈ 8.0; strongly negative ≈ 2.0. "
    "High arousal (>=7) = excited/angry; low arousal (<=3) = calm.\n"
    "5. If no quadruplet applies, return {\"quadruplets\": []}.\n"
    "6. Output ONLY the JSON object, no explanations, no markdown fences."
)


def _format_example(example: Dict) -> str:
    """Render one training example as a USER/ASSISTANT block for few-shot priming."""
    quads_out = []
    for q in example.get("Quadruplet", []):
        va = q.get("VA", "5.0#5.0")
        try:
            v_str, a_str = va.split("#")
            v = float(v_str)
            a = float(a_str)
        except (ValueError, AttributeError):
            v, a = 5.0, 5.0
        quads_out.append({
            "aspect": q.get("Aspect", "NULL"),
            "opinion": q.get("Opinion", "NULL"),
            "category": q.get("Category", ""),
            "valence": round(v, 2),
            "arousal": round(a, 2),
        })
    return json.dumps({"quadruplets": quads_out}, ensure_ascii=False)


def build_pseudo_label_prompt(
    sentence: str,
    category_list: List[str],
    shots: List[Dict],
) -> List[Dict[str, str]]:
    """Assemble the full message list for an OpenRouter chat completion.

    Args:
        sentence:       target sentence to annotate.
        category_list:  allowed category labels, extracted from the gold train set.
        shots:          list of example dicts with {"Text": ..., "Quadruplet": [...]}.

    Returns:
        messages (list of {"role": ..., "content": ...}) ready for chat().
    """
    categories_line = ", ".join(category_list)

    user_preamble = (
        f"Allowed categories (exactly one must be picked for each quadruplet):\n{categories_line}\n\n"
        "Annotate the following sentence. Return only the JSON object described in the rules."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    for ex in shots:
        messages.append({
            "role": "user",
            "content": f"{user_preamble}\n\nSentence: {ex['Text']}",
        })
        messages.append({
            "role": "assistant",
            "content": _format_example(ex),
        })

    messages.append({
        "role": "user",
        "content": f"{user_preamble}\n\nSentence: {sentence}",
    })
    return messages


# =========================================================================
# ISR (Implicit Sentiment Reasoning) — recover surrogate spans for NULLs
# =========================================================================

_ISR_RECOVERY_SYSTEM = (
    "You are an expert linguist specialising in aspect-based sentiment analysis. "
    "A review sentence contains an IMPLICIT {slot_type} — the {slot_type} is not "
    "explicitly stated but can be inferred from context.\n\n"
    "Given:\n"
    "- The review sentence\n"
    "- The category ({entity}#{attribute})\n"
    "- The {other_slot_type}: \"{other_slot_value}\"\n"
    "- The sentiment values: valence={valence}, arousal={arousal}\n\n"
    "Your task: identify the best SURROGATE span — a substring of the sentence "
    "that most closely relates to the implicit {slot_type}.\n\n"
    "Rules:\n"
    "1. The surrogate MUST be an exact substring of the sentence (case-sensitive match).\n"
    "2. Prefer a short, specific noun phrase (for aspect) or adjective/verb phrase (for opinion).\n"
    "3. Do NOT use the category label itself as the surrogate (e.g., don't use "
    "\"quality\" or \"style options\" — use an actual phrase from the sentence).\n"
    "4. If no reasonable substring can be identified, set surrogate to \"NULL\".\n"
    "5. Provide a brief reasoning chain explaining your choice.\n\n"
    "Output ONLY a JSON object:\n"
    '{{\"reasoning\": \"...\", \"surrogate\": \"<exact substring or NULL>\"}}'
)


def build_isr_recovery_prompt(
    sentence: str,
    category: str,
    slot_type: str,
    other_slot_type: str,
    other_slot_value: str,
    valence: float,
    arousal: float,
) -> List[Dict[str, str]]:
    """ISR: build messages to recover a surrogate span for an implicit aspect or opinion.

    Args:
        sentence:         the review text
        category:         E#A category, e.g. "RESTAURANT#GENERAL"
        slot_type:        "aspect" or "opinion" — which slot is NULL
        other_slot_type:  the non-NULL slot type ("opinion" or "aspect")
        other_slot_value: the non-NULL slot's text, or "NULL" if both are implicit
        valence:          gold valence value
        arousal:          gold arousal value
    """
    entity, attribute = category.split("#", 1) if "#" in category else (category, "")
    system = _ISR_RECOVERY_SYSTEM.format(
        slot_type=slot_type,
        other_slot_type=other_slot_type,
        other_slot_value=other_slot_value,
        entity=entity,
        attribute=attribute,
        valence=valence,
        arousal=arousal,
    )
    user_content = (
        f"Sentence: {sentence}\n"
        f"Category: {category}\n"
        f"Implicit slot: {slot_type} (currently NULL)\n"
        f"Known {other_slot_type}: \"{other_slot_value}\"\n"
        f"Valence: {valence}, Arousal: {arousal}\n\n"
        f"Identify the best surrogate {slot_type} span from the sentence."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


_ISR_BOTH_NULL_SYSTEM = (
    "You are an expert linguist specialising in aspect-based sentiment analysis. "
    "A review sentence expresses sentiment about {entity}#{attribute}, but BOTH "
    "the aspect and opinion are implicit — neither is explicitly stated.\n\n"
    "Given:\n"
    "- The review sentence\n"
    "- The category ({entity}#{attribute})\n"
    "- The sentiment values: valence={valence}, arousal={arousal}\n\n"
    "Your task: identify surrogate spans for BOTH the aspect and opinion.\n\n"
    "Rules:\n"
    "1. Each surrogate MUST be an exact substring of the sentence (case-sensitive).\n"
    "2. Aspect surrogate: a noun/noun phrase that relates to {entity}.\n"
    "3. Opinion surrogate: an adjective/verb phrase that conveys the sentiment.\n"
    "4. Do NOT use category label names (\"quality\", \"general\", etc.) as surrogates.\n"
    "5. If no reasonable substring exists for a slot, set it to \"NULL\".\n"
    "6. Provide brief reasoning for each choice.\n\n"
    "Output ONLY a JSON object:\n"
    '{{\"reasoning\": \"...\", \"aspect_surrogate\": \"<substring or NULL>\", '
    '\"opinion_surrogate\": \"<substring or NULL>\"}}'
)


def build_isr_both_null_prompt(
    sentence: str,
    category: str,
    valence: float,
    arousal: float,
) -> List[Dict[str, str]]:
    """ISR: recover surrogate spans when both aspect and opinion are NULL."""
    entity, attribute = category.split("#", 1) if "#" in category else (category, "")
    system = _ISR_BOTH_NULL_SYSTEM.format(
        entity=entity, attribute=attribute,
        valence=valence, arousal=arousal,
    )
    user_content = (
        f"Sentence: {sentence}\n"
        f"Category: {category}\n"
        f"Both aspect and opinion are implicit (NULL).\n"
        f"Valence: {valence}, Arousal: {arousal}\n\n"
        f"Identify surrogate spans for both aspect and opinion."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# =========================================================================
# CCA (Compositional Category Augmentation) — three-stage prompt builders
# =========================================================================

_ENTITY_GROUNDING_SYSTEM = (
    "You are an expert linguist analysing aspect-based sentiment in customer reviews. "
    "Your task is to summarise how a specific ENTITY type appears in real reviews.\n\n"
    "Given several example review sentences that mention aspects belonging to the entity "
    "type \"{entity}\", produce a concise summary covering:\n"
    "1. Typical aspect terms (nouns/noun phrases) people use for this entity\n"
    "2. Common sentence patterns and contexts in which this entity appears\n"
    "3. The range of sentiment polarities observed (positive, negative, neutral)\n\n"
    "Output ONLY a JSON object:\n"
    '{{"entity": "{entity}", "typical_aspects": ["..."], '
    '"sentence_patterns": ["..."], "sentiment_range": "..."}}'
)

_ATTRIBUTE_GROUNDING_SYSTEM = (
    "You are an expert linguist analysing aspect-based sentiment in customer reviews. "
    "Your task is to summarise how a specific ATTRIBUTE angle is expressed in reviews.\n\n"
    "Given several example review sentences that discuss the attribute \"{attribute}\" "
    "(regardless of which entity they refer to), produce a concise summary covering:\n"
    "1. What this attribute measures or evaluates (e.g., quality → goodness/tastiness; "
    "prices → cost/value; style_options → variety/choices; general → overall impression)\n"
    "2. Typical opinion expressions (adjectives, phrases) used for this attribute\n"
    "3. Characteristic valence–arousal patterns (e.g., price complaints tend to be "
    "low-valence high-arousal)\n\n"
    "Output ONLY a JSON object:\n"
    '{{"attribute": "{attribute}", "measures": "...", '
    '"typical_opinions": ["..."], "va_patterns": "..."}}'
)

_CCA_GENERATION_SYSTEM = (
    "You are a data-augmentation expert for the DimABSA task. "
    "Your goal is to generate realistic, diverse review sentences for a SPECIFIC "
    "Entity×Attribute category that may be rare or unseen in the training data.\n\n"
    "You are given:\n"
    "- Target category: {entity}#{attribute}\n"
    "- Entity context: what \"{entity}\" typically looks like in reviews\n"
    "- Attribute context: what \"{attribute}\" discussions typically sound like\n"
    "- A few anchor examples from related categories for stylistic reference\n\n"
    "Rules:\n"
    "1. Generate exactly {n_generate} distinct review sentences.\n"
    "2. Each sentence MUST contain at least one quadruplet for {entity}#{attribute}.\n"
    "3. `aspect` must be an exact substring of the generated sentence, or \"NULL\".\n"
    "4. `opinion` must be an exact substring of the generated sentence, or \"NULL\".\n"
    "5. `valence` (pleasantness) ∈ [1.0, 9.0]; `arousal` (intensity) ∈ [1.0, 9.0].\n"
    "6. Vary sentiment: include positive, negative, and neutral examples.\n"
    "7. Vary sentence length and style to avoid repetitiveness.\n"
    "8. Do NOT copy anchor examples verbatim. Create new, natural sentences.\n\n"
    "Output ONLY a JSON array of objects, each with keys \"sentence\" and \"quadruplets\":\n"
    '[{{"sentence": "...", "quadruplets": [{{"aspect": "...", "opinion": "...", '
    '"category": "{entity}#{attribute}", "valence": 6.5, "arousal": 5.0}}]}}]'
)

_CROSS_VERIFY_SYSTEM = (
    "You are a category classifier for aspect-based sentiment analysis. "
    "Given a review sentence with a marked (aspect, opinion) pair, determine which "
    "Entity#Attribute category best describes it.\n\n"
    "Available categories:\n{category_list}\n\n"
    "Rules:\n"
    "1. Pick exactly ONE category from the list above.\n"
    "2. Output ONLY a JSON object: {{\"category\": \"ENTITY#ATTRIBUTE\", "
    "\"confidence\": <float 0-1>}}\n"
    "3. If none fits well, pick the closest and set confidence < 0.5."
)


def build_entity_grounding_prompt(
    entity: str,
    entity_examples: List[Dict],
    max_examples: int = 8,
) -> List[Dict[str, str]]:
    """CCA Stage 1: build messages to summarise an entity's typical review context.

    Args:
        entity:          entity name, e.g. "FOOD", "AMBIENCE"
        entity_examples: gold training dicts with {"Text", "Quadruplet"}
                         where at least one quad has this entity
        max_examples:    cap on examples to keep prompt short
    """
    system = _ENTITY_GROUNDING_SYSTEM.format(entity=entity)
    examples_text = []
    for ex in entity_examples[:max_examples]:
        quads = [q for q in ex.get("Quadruplet", [])
                 if q.get("Category", "").startswith(entity + "#")]
        aspects = [q.get("Aspect", "NULL") for q in quads]
        examples_text.append(
            f"Sentence: {ex['Text']}\n  Aspects: {', '.join(aspects)}"
        )
    user_content = (
        f"Here are {len(examples_text)} example sentences mentioning {entity} aspects:\n\n"
        + "\n\n".join(examples_text)
        + "\n\nSummarise the entity context as instructed."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_attribute_grounding_prompt(
    attribute: str,
    attribute_examples: List[Dict],
    max_examples: int = 8,
) -> List[Dict[str, str]]:
    """CCA Stage 2: build messages to summarise how an attribute is expressed.

    Args:
        attribute:          attribute name, e.g. "QUALITY", "STYLE_OPTIONS"
        attribute_examples: gold training dicts where at least one quad has this attribute
        max_examples:       cap on examples
    """
    system = _ATTRIBUTE_GROUNDING_SYSTEM.format(attribute=attribute)
    examples_text = []
    for ex in attribute_examples[:max_examples]:
        quads = [q for q in ex.get("Quadruplet", [])
                 if q.get("Category", "").split("#", 1)[-1] == attribute]
        opinions = [q.get("Opinion", "NULL") for q in quads]
        cats = [q.get("Category", "") for q in quads]
        examples_text.append(
            f"Sentence: {ex['Text']}\n  Category: {', '.join(cats)}"
            f"\n  Opinions: {', '.join(opinions)}"
        )
    user_content = (
        f"Here are {len(examples_text)} example sentences discussing the {attribute} angle:\n\n"
        + "\n\n".join(examples_text)
        + "\n\nSummarise the attribute context as instructed."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_cca_generation_prompt(
    entity: str,
    attribute: str,
    entity_summary: str,
    attribute_summary: str,
    anchor_examples: List[Dict],
    n_generate: int = 10,
    max_anchors: int = 4,
) -> List[Dict[str, str]]:
    """CCA Stage 3: build messages to generate sentences for a target E#A category.

    Args:
        entity:            target entity, e.g. "AMBIENCE"
        attribute:         target attribute, e.g. "STYLE_OPTIONS"
        entity_summary:    JSON string from Stage 1
        attribute_summary: JSON string from Stage 2
        anchor_examples:   gold examples from same-entity or same-attribute categories
        n_generate:        number of sentences to generate
        max_anchors:       cap on anchor examples shown
    """
    system = _CCA_GENERATION_SYSTEM.format(
        entity=entity, attribute=attribute, n_generate=n_generate
    )
    anchors_text = []
    for ex in anchor_examples[:max_anchors]:
        quads_json = json.dumps(
            [{"aspect": q.get("Aspect", "NULL"),
              "opinion": q.get("Opinion", "NULL"),
              "category": q.get("Category", ""),
              "valence": float(q.get("VA", "5#5").split("#")[0]),
              "arousal": float(q.get("VA", "5#5").split("#")[1])}
             for q in ex.get("Quadruplet", [])],
            ensure_ascii=False,
        )
        anchors_text.append(
            f"Sentence: {ex['Text']}\nQuadruplets: {quads_json}"
        )
    user_content = (
        f"## Target category: {entity}#{attribute}\n\n"
        f"## Entity context ({entity}):\n{entity_summary}\n\n"
        f"## Attribute context ({attribute}):\n{attribute_summary}\n\n"
        f"## Anchor examples (related categories, for style reference):\n"
        + "\n\n".join(anchors_text)
        + f"\n\n## Task\n"
        f"Generate {n_generate} new, diverse review sentences for {entity}#{attribute}. "
        f"Return the JSON array as instructed."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_cross_verify_prompt(
    sentence: str,
    aspect: str,
    opinion: str,
    candidate_category: str,
    all_categories: List[str],
) -> List[Dict[str, str]]:
    """CCA Verification: ask the LLM to classify a generated (aspect, opinion) pair.

    Returns messages for a chat completion. The expected output is:
    {"category": "ENTITY#ATTRIBUTE", "confidence": 0.85}

    The caller compares the returned category against candidate_category;
    if they differ, the generated sample is flagged as category-drifted.
    """
    system = _CROSS_VERIFY_SYSTEM.format(
        category_list="\n".join(f"  - {c}" for c in all_categories)
    )
    aspect_display = aspect if aspect != "NULL" else "(implicit)"
    opinion_display = opinion if opinion != "NULL" else "(implicit)"
    user_content = (
        f"Sentence: {sentence}\n"
        f"Aspect: {aspect_display}\n"
        f"Opinion: {opinion_display}\n\n"
        "Which category does this (aspect, opinion) pair belong to?"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
