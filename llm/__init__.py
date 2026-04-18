"""LLM integration package for DimASQP.

Exposes:
  - OpenRouterClient: HTTP wrapper around the OpenRouter chat/completions API
  - build_pseudo_label_prompt: vanilla few-shot prompt builder (baseline)
  - CCA prompt builders: entity/attribute grounding, compositional generation, cross-verify
  - DEFAULTS: central config defaults (model, temperature, cache dir, ...)
"""
from .config import DEFAULTS
from .openrouter_client import OpenRouterClient, OpenRouterError
from .prompts import (
    PROMPT_VERSION,
    build_pseudo_label_prompt,
    build_entity_grounding_prompt,
    build_attribute_grounding_prompt,
    build_cca_generation_prompt,
    build_cross_verify_prompt,
)

__all__ = [
    "DEFAULTS",
    "OpenRouterClient",
    "OpenRouterError",
    "PROMPT_VERSION",
    "build_pseudo_label_prompt",
    "build_entity_grounding_prompt",
    "build_attribute_grounding_prompt",
    "build_cca_generation_prompt",
    "build_cross_verify_prompt",
]
