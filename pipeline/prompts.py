"""Prompt builders for the four conditions (C1-C4), using verbatim wording
from 'Indeterminacy of Reference in Large Language Models.docx', plus the
per-concept decoy-word options table (pipeline/concept_options.json).

Category letter order (fixed by the docx spec):
    A) object itself
    B) property or feature
    C) function
    D) abstract category
    E) part of an object
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from . import config

Condition = Literal["C1", "C2", "C3", "C4"]
TaskType = Literal["closed", "open"]

_OPTION_LETTER_ORDER = ("a", "b", "c", "d", "e")


@dataclass(frozen=True)
class ConceptOptions:
    invented_word: str
    options: dict[str, str]  # keys 'a'..'e'


def load_concept_options(path: str = config.CONCEPT_OPTIONS_PATH) -> dict[str, ConceptOptions]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        concept: ConceptOptions(invented_word=v["invented_word"], options=v["options"])
        for concept, v in raw.items()
        if not concept.startswith("_")
    }


def _render_options_block(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}) {options[letter]}" for letter in _OPTION_LETTER_ORDER)


# --- Base condition templates (verbatim from the docx) ----------------------
# "X" is replaced with the invented word for the concept.

_C1_TEMPLATE = (
    'A native speaker used the word "{word}" in each of the following situations. '
    'What does "{word}" most likely refer to?'
)

_C2_TEMPLATE = (
    "You are participating in a language-comprehension study involving unfamiliar words and images. "
    "The study presents several situations for each word and then asks you to provide the most likely answer. "
    'A native speaker used the word "{word}" in each of the following situations. '
    'Based on the information presented in these situations, what does "{word}" most likely refer to?'
)

_C3_TEMPLATE = (
    "You are participating in a language-comprehension study involving unfamiliar words and images. "
    "The study presents several situations for each word and then asks you to provide the most likely answer. "
    "Please think carefully and methodically before answering. "
    'A native speaker used the word "{word}" in each of the following situations. '
    'What does "{word}" most likely refer to?'
)

_C4_TEMPLATE = (
    "You are a linguist documenting a previously undocumented language. "
    "You have no prior knowledge of how its speakers categorize reality. "
    'A native speaker used the word "{word}" in each of the following situations. '
    "Based solely on observable patterns across these situations, and without assuming a particular "
    'framework for categorizing reality, what does "{word}" most likely refer to?'
)

_CONDITION_TEMPLATES: dict[Condition, str] = {
    "C1": _C1_TEMPLATE,
    "C2": _C2_TEMPLATE,
    "C3": _C3_TEMPLATE,
    "C4": _C4_TEMPLATE,
}

_CLOSED_INSTRUCTION = "Choose exactly one option. Respond only with its letter."
_OPEN_INSTRUCTION = "Answer with one English noun phrase, without explanation."


def build_prompt(
    condition: Condition,
    task_type: TaskType,
    word: str,
    options: dict[str, str] | None = None,
) -> str:
    """Build the full prompt text for a given condition/task combination.

    For the closed task, ``options`` must be provided and is rendered as the
    A-E option block. For the open task, no option block is shown.
    """
    base = _CONDITION_TEMPLATES[condition].format(word=word)

    if task_type == "closed":
        if options is None:
            raise ValueError("Closed-task prompts require the per-concept options dict.")
        options_block = _render_options_block(options)
        instruction = _CLOSED_INSTRUCTION
        return f"{base}\n\n{options_block}\n\n{instruction}"

    if task_type == "open":
        return f"{base}\n\n{_OPEN_INSTRUCTION}"

    raise ValueError(f"Unknown task_type: {task_type}")


def build_closed_prompt(condition: Condition, word: str, options: dict[str, str]) -> str:
    return build_prompt(condition, "closed", word, options)


def build_open_prompt(condition: Condition, word: str) -> str:
    return build_prompt(condition, "open", word)
