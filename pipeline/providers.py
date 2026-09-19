"""Unified, structured API-calling layer for OpenAI, Anthropic, and Google Gemini.

Adapted from ``original_version/quine_zatvorenog_tipa.ipynb``: the same
request-building shapes are reused (Gemini ``contents`` list, Claude
``content`` blocks with images + text, OpenAI ``image_url`` data-URI blocks),
but every call now returns a structured ``TrialResult`` instead of a bare
string, and transient failures are retried with backoff instead of crashing
the run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import config

# Provider SDKs are imported lazily inside each call function so that this
# module can be imported (e.g. for tests) even if not all three SDKs are
# installed in the current environment.


@dataclass
class TrialResult:
    text: str | None
    raw_response: Any  # the raw SDK response object / dict, kept for audit
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    error: str | None = None
    attempts: int = 1
    params_applied: dict[str, float | int | None] = field(default_factory=dict)
    """Records which decoding parameters were actually sent to the API for
    this call (e.g. {"temperature": 0.7, "top_p": 1.0} or {"temperature":
    None, "top_p": None} if the model doesn't support them). Persisted with
    the trial so cross-model entropy comparisons can account for models that
    used provider defaults instead of the configured sampling parameters."""

    @property
    def success(self) -> bool:
        return self.error is None


# Substrings that indicate an API rejected temperature/top_p rather than a
# genuine transient failure; used to auto-retry once without those params.
_SAMPLING_PARAM_ERROR_HINTS = ("temperature", "top_p", "unsupported parameter", "unsupported value")


def _looks_like_sampling_param_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _SAMPLING_PARAM_ERROR_HINTS)


def _sampling_kwargs(params: config.DecodingParams, supports_sampling_params: bool) -> dict[str, float]:
    """Build the temperature/top_p kwargs to send, or {} if unsupported."""
    if not supports_sampling_params:
        return {}
    return {"temperature": params.temperature, "top_p": params.top_p}


def _call_openai(
    model_id: str, prompt: str, images: list[str], params: config.DecodingParams, supports_sampling_params: bool
) -> TrialResult:
    import openai

    client = openai.OpenAI(api_key=config.get_api_key("openai"))
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img_data in images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
        )

    sampling_kwargs = _sampling_kwargs(params, supports_sampling_params)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        # Newer OpenAI models reject the legacy `max_tokens` param and require
        # `max_completion_tokens` instead; the latter is accepted across
        # current chat-completions models, so use it unconditionally.
        max_completion_tokens=params.max_output_tokens,
        **sampling_kwargs,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.choices[0].message.content.strip() if response.choices else None
    usage = getattr(response, "usage", None)
    return TrialResult(
        text=text,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else str(response),
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        latency_ms=latency_ms,
        params_applied={"temperature": sampling_kwargs.get("temperature"), "top_p": sampling_kwargs.get("top_p")},
    )


def _call_claude(
    model_id: str, prompt: str, images: list[str], params: config.DecodingParams, supports_sampling_params: bool
) -> TrialResult:
    import anthropic

    client = anthropic.Anthropic(api_key=config.get_api_key("anthropic"))
    content: list[dict] = []
    for img_data in images:
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data},
            }
        )
    content.append({"type": "text", "text": prompt})

    sampling_kwargs = _sampling_kwargs(params, supports_sampling_params)

    start = time.perf_counter()
    response = client.messages.create(
        model=model_id,
        max_tokens=params.max_output_tokens,
        messages=[{"role": "user", "content": content}],
        **sampling_kwargs,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.content[0].text.strip() if response.content else None
    usage = getattr(response, "usage", None)
    return TrialResult(
        text=text,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else str(response),
        prompt_tokens=getattr(usage, "input_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "output_tokens", None) if usage else None,
        latency_ms=latency_ms,
        params_applied={"temperature": sampling_kwargs.get("temperature"), "top_p": sampling_kwargs.get("top_p")},
    )


def _call_gemini(
    model_id: str, prompt: str, images: list[str], params: config.DecodingParams, supports_sampling_params: bool
) -> TrialResult:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.get_api_key("google"))
    contents: list[Any] = [prompt]
    for img_data in images:
        contents.append({"inline_data": {"mime_type": "image/jpeg", "data": img_data}})

    sampling_kwargs = _sampling_kwargs(params, supports_sampling_params)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        # NOTE: Gemini 3.x models "think" by default and mandatorily; the
        # thinking tokens are deducted from the same max_output_tokens
        # budget, and thinking cannot be reliably disabled for this model
        # (thinking_budget=0 is not honored consistently; thinking_level
        # "minimal" is rejected outright for gemini-3.8-flash). Observed
        # thinking token usage ranged ~15-190 tokens per call, so
        # max_output_tokens must include generous headroom above the
        # expected short visible answer or calls intermittently truncate
        # with finish_reason=MAX_TOKENS and empty text. See config.py for
        # the actual budget value used.
        config=types.GenerateContentConfig(
            max_output_tokens=params.max_output_tokens,
            **sampling_kwargs,
        ),
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.text.strip() if response.text else None
    usage = getattr(response, "usage_metadata", None)
    return TrialResult(
        text=text,
        raw_response=str(response),
        prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
        completion_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        latency_ms=latency_ms,
        params_applied={"temperature": sampling_kwargs.get("temperature"), "top_p": sampling_kwargs.get("top_p")},
    )


_PROVIDER_FUNCTIONS = {
    "openai": _call_openai,
    "anthropic": _call_claude,
    "google": _call_gemini,
}


def call_model(
    provider: config.Provider,
    model_id: str,
    prompt: str,
    images: list[str],
    params: config.DecodingParams = config.DEFAULT_DECODING,
    max_retries: int = config.MAX_RETRIES,
    supports_sampling_params: bool = True,
) -> TrialResult:
    """Call the given provider/model with retry + exponential backoff.

    On persistent failure, returns a TrialResult with ``error`` set instead of
    raising, so a single bad trial does not abort a long-running batch.

    ``supports_sampling_params`` should come from ``ModelSpec.supports_sampling_params``
    (see config.py). As a safety net, if a call fails with an error that looks
    like the API rejecting temperature/top_p (e.g. a newer reasoning-tier model
    that has deprecated these knobs), the call is retried once without them,
    and ``supports_sampling_params`` is treated as False for the rest of this
    call's retries.
    """
    if provider not in _PROVIDER_FUNCTIONS:
        raise ValueError(f"Unknown provider: {provider}")

    call_fn = _PROVIDER_FUNCTIONS[provider]
    last_error: str | None = None
    effective_supports_sampling = supports_sampling_params

    for attempt in range(1, max_retries + 1):
        try:
            result = call_fn(model_id, prompt, images, params, effective_supports_sampling)
            result.attempts = attempt
            return result
        except Exception as exc:  # noqa: BLE001 - deliberately broad: log & retry
            last_error = f"{type(exc).__name__}: {exc}"
            if effective_supports_sampling and _looks_like_sampling_param_error(exc):
                # Fall back to provider defaults and retry immediately (no
                # backoff needed; this isn't a transient/rate-limit issue).
                effective_supports_sampling = False
                continue
            if attempt < max_retries:
                time.sleep(config.RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

    return TrialResult(
        text=None,
        raw_response=None,
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=0.0,
        error=last_error,
        attempts=max_retries,
        params_applied={"temperature": None, "top_p": None},
    )
