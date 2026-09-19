"""Central configuration: model registry, decoding parameters, and API keys.

All secrets are loaded from environment variables (optionally via a local
`.env` file with python-dotenv). Nothing here should ever contain a literal
API key.

IMPORTANT: the model snapshot IDs below are placeholders. Confirm and pin the
exact, current model IDs for each provider before running anything beyond a
small pilot, since aliases (e.g. "latest") can silently change over time and
break reproducibility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; environment variables can be set another way.
    pass

Provider = Literal["openai", "anthropic", "google"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model_id: str
    display_name: str
    tier: Literal["flagship", "budget"] = "flagship"
    supports_sampling_params: bool = True
    """Whether this model accepts temperature/top_p. Some newer "reasoning"-
    tier models ignore or reject these parameters; set to False for those so
    providers.py omits them instead of sending values that are silently
    ignored or that raise an API error."""


# --- Model registry -------------------------------------------------------
# Per 'Indeterminacy of Reference in Large Language Models.docx': GPT 5.6 Luna,
# Claude Haiku 4.5, Gemini 3.8 Flash. Confirm exact pinned snapshot IDs with
# each provider's current docs before running beyond a small pilot.
MODELS: dict[str, ModelSpec] = {
    "gpt5_6_luna": ModelSpec(
        "openai", "gpt-5.6-luna", "GPT-5.6 Luna", "flagship", supports_sampling_params=False
    ),
    "claude_haiku": ModelSpec("anthropic", "claude-haiku-4-5-20251001", "Claude Haiku 4.5", "flagship", supports_sampling_params=False),
    "gemini_3_8_flash": ModelSpec("google", "gemini-3.8-flash", "Gemini 3.8 Flash", "flagship"),
}

# --- Decoding parameters ----------------------------------------------------
# Kept as a shared policy; provider-specific overrides can be layered on top
# in providers.py since not every API exposes the same knobs.
@dataclass(frozen=True)
class DecodingParams:
    temperature: float = 0.7
    # Raised from the originally planned 50: Gemini 3.8 Flash spends a
    # mandatory, variable number of hidden "thinking" tokens (observed
    # ~15-190) out of the same budget before producing the visible answer,
    # which caused truncated/empty responses at 50. Thinking cannot be
    # reliably disabled for this model, so the budget was raised for all
    # three models instead, for cross-model consistency. This is a technical
    # generation-length cap, not a manipulated variable, and should be noted
    # in the methods/preregistration as a deviation from the original 50.
    max_output_tokens: int = 300
    top_p: float = 1.0


DEFAULT_DECODING = DecodingParams()

# --- API keys ---------------------------------------------------------------
ENV_VARS: dict[Provider, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def get_api_key(provider: Provider) -> str:
    env_var = ENV_VARS[provider]
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. "
            f"Set the {env_var} environment variable (e.g. in a local .env file)."
        )
    return key


# --- Retry policy ------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0  # exponential backoff: base * 2**attempt

# --- Paths --------------------------------------------------------------
IMAGES_ROOT = "50_things_images"
DB_PATH = "quine_experiment.sqlite3"
CONCEPT_OPTIONS_PATH = "pipeline/concept_options.json"
