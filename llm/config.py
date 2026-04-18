"""Central configuration defaults for the LLM pseudo-labeling pipeline.

All values here are defaults only. Every CLI script is expected to expose them
as --llm_* flags so they can be overridden without editing this file.

Model strategy (per the approved plan):
  Phase A (曲线拟合, 低成本):
    - meta-llama/llama-3.1-70b-instruct
    - openai/gpt-4o-mini
  Phase B (冲分, 高质量):
    - anthropic/claude-3.5-sonnet
"""

DEFAULTS = {
    # Default OpenRouter model used when the user does not pass --llm_model_name.
    # Start with the cheap, open-weight model so the "fit the curve" phase is
    # predictable and auditable. Switch to gpt-4o-mini or claude-3.5-sonnet via CLI.
    "model": "meta-llama/llama-3.1-70b-instruct",

    # Decoding hyperparameters. Low temperature keeps JSON output stable.
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 512,

    # Networking / retries.
    "base_url": "https://openrouter.ai/api/v1",
    "timeout_s": 60.0,
    "max_retries": 4,
    "backoff_base_s": 2.0,  # exponential backoff: 2, 4, 8, 16 seconds

    # Few-shot sampling.
    "num_shots": 5,
    "min_valid_quads": 1,   # drop sentences with no valid quad after post-processing

    # Disk cache (key = sha256(sentence || model || prompt_version)).
    "cache_dir": "cache/llm_pseudo",

    # OpenRouter attribution headers (best-effort; safe to leave blank).
    "http_referer": "https://github.com/potato0705/dimasqp",
    "x_title": "DimASQP LLM Pseudo-Labeler",
}

CCA_DEFAULTS = {
    "n_per_category": 50,
    "max_grounding_examples": 8,
    "max_anchor_examples": 4,
    "generation_batch_size": 10,
    "cross_verify": True,
    "verify_confidence_threshold": 0.5,
    "cache_dir": "cache/cca",
    "rare_threshold": 20,
    "generation_temperature": 0.7,
    "grounding_temperature": 0.2,
    "verify_temperature": 0.0,
    "generation_max_tokens": 2048,
}
