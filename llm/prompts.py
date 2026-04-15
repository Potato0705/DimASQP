"""Few-shot prompt builder for DimASQP pseudo-labeling.

Output JSON schema (strict):
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

The LLM is asked to return ONLY this JSON object (no surrounding prose).
"""
from __future__ import annotations

import json
from typing import Dict, List

# Bumping this invalidates all disk-cached responses.
PROMPT_VERSION = "v1-2026-04-15"


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
