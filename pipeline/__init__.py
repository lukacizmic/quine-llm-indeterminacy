"""Reproducible pipeline for the expanded Quine-inspired LLM experiment.

Modules:
    config      - model registry, decoding parameters, API key loading
    images      - folder-of-images -> base64-encoded image list
    prompts     - C1-C4 prompt builders using the per-concept options table
    providers   - unified structured API-calling layer (OpenAI/Anthropic/Gemini)
    db          - SQLite schema and resumable, atomic trial storage
    closed_runner - trial-plan generation and execution for the closed task
"""
