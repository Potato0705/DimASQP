"""LLM integration package for DimASQP.

Offline LLM-assisted pseudo-labeling (scheme 3 from docs/deep-research-report.md).
Exposes:
  - OpenRouterClient: minimal HTTP wrapper around the OpenRouter chat/completions API
  - build_pseudo_label_prompt: few-shot prompt builder
  - DEFAULTS: central config defaults (model, temperature, cache dir, ...)

The training/evaluation code paths never import this module; pseudo-labeling is
a pre-processing step run as a standalone CLI (data/llm_pseudo_labeler.py).
"""
from .config import DEFAULTS
from .openrouter_client import OpenRouterClient, OpenRouterError
from .prompts import build_pseudo_label_prompt, PROMPT_VERSION

__all__ = [
    "DEFAULTS",
    "OpenRouterClient",
    "OpenRouterError",
    "build_pseudo_label_prompt",
    "PROMPT_VERSION",
]
