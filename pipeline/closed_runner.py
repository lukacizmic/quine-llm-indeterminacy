"""Closed-task experiment runner.

Builds the full trial plan (concepts x conditions x models x repetitions),
randomizes execution order, calls the appropriate provider for each planned
trial not already in the DB, extracts the closed-task answer letter using a
fixed rule-set, and persists every trial immediately.
"""
from __future__ import annotations

import argparse
import datetime
import random
import re
import sys
from dataclasses import dataclass

from . import config, db, images, prompts, providers

# Model responses can contain arbitrary Unicode (e.g. checkmarks, emoji,
# accented characters) that the default Windows console encoding (cp1252)
# cannot print. Force stdout/stderr to UTF-8 with a safe fallback so a
# single unusual character in a raw response never crashes a long-running
# batch mid-run (this previously killed the process after only ~90 trials).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

VALID_LETTERS = {"a", "b", "c", "d", "e"}


def extract_closed_answer(raw_text: str | None) -> tuple[str | None, bool]:
    """Apply the fixed closed-response coding rule.

    Rules (per the docx):
      - lowercase letters accepted (case-insensitive matching)
      - explanations ignored after extracting the first valid letter
      - multiple distinct valid letters found -> invalid
      - no valid letter found -> invalid
    Returns (extracted_letter_or_None, is_valid).
    """
    if not raw_text:
        return None, False

    # Find standalone letter tokens a-e (as an isolated character, optionally
    # followed by ')' or '.' as in "a)" or "a."), case-insensitive.
    matches = re.findall(r"(?<![a-zA-Z])([a-eA-E])(?=[).:\s]|$)", raw_text)
    letters = {m.lower() for m in matches if m.lower() in VALID_LETTERS}

    if len(letters) == 1:
        return next(iter(letters)), True
    return None, False


# Patterns that mark a model's *final*, committed answer rather than an
# incidental letter mentioned while walking through the options (e.g. Claude
# under C3's "think carefully and methodically" instruction routinely writes
# "a) horse - ... b) mane - ..." before its actual final pick). Each pattern
# has exactly one capture group for the letter; patterns are tried in order
# and the *last* match found in the text (by position) wins, since that is
# whichever statement of intent came last / most recently in the reasoning.
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"\*\*([a-eA-E])\*\*"),  # a lone bolded letter, e.g. **a**
    re.compile(r"\*\*\(?([a-eA-E])\)(?:[^*]{0,40})?\*\*"),  # e.g. **e) stalk**, **(a) horse**
    re.compile(
        r"(?:the answer is|final answer|my answer is|answer:)\s*\**\(?([a-eA-E])\)(?![a-zA-Z])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the answer is|final answer|my answer is|answer:)\s*\**\(?([a-eA-E])\**(?![a-zA-Z])",
        re.IGNORECASE,
    ),
]


def extract_closed_answer_v2(raw_text: str | None) -> tuple[str | None, bool]:
    """Improved closed-response coding rule.

    ``extract_closed_answer`` invalidates a trial whenever more than one
    standalone letter a-e appears *anywhere* in the response text. That rule
    works fine for the common case (a bare "d" reply), but it misfires badly
    on longer, structured responses that walk through each option before
    committing to one (e.g. "a) horse - fits... b) mane - too narrow...
    **e) stalk**") -- every option letter mentioned along the way gets
    counted as a competing "answer", so the trial is thrown out even though
    the model's actual final choice is completely unambiguous.

    This version first looks for an explicit final-answer marker (a bolded
    letter, or an "the answer is X" phrase) and, if exactly one such marker
    type is present, uses its *last* occurrence as the answer -- the same
    way a human coder reading the response to the end would. If no such
    marker exists, it falls back to the original standalone-letter rule
    (which already correctly handles the vast majority of trials, since most
    responses are just a bare letter).
    """
    if not raw_text:
        return None, False

    best_match = None
    best_pos = -1
    for pattern in _FINAL_ANSWER_PATTERNS:
        for m in pattern.finditer(raw_text):
            if m.start() > best_pos:
                best_pos = m.start()
                best_match = m.group(1)

    if best_match is not None:
        return best_match.lower(), True

    return extract_closed_answer(raw_text)


@dataclass(frozen=True)
class PlannedTrial:
    model_key: str
    condition: prompts.Condition
    concept_id: str
    repetition_index: int


def build_trial_plan(
    model_keys: list[str],
    conditions: list[prompts.Condition],
    concept_ids: list[str],
    n_repetitions: int,
    seed: int = 42,
) -> list[PlannedTrial]:
    plan = [
        PlannedTrial(model_key=m, condition=c, concept_id=concept, repetition_index=i)
        for m in model_keys
        for c in conditions
        for concept in concept_ids
        for i in range(n_repetitions)
    ]
    random.Random(seed).shuffle(plan)
    return plan


def run_closed_experiment(
    model_keys: list[str],
    conditions: list[prompts.Condition],
    concept_ids: list[str],
    n_repetitions: int,
    images_root: str = config.IMAGES_ROOT,
    max_images: int = 10,
    db_path: str = config.DB_PATH,
    seed: int = 42,
) -> None:
    concept_options = prompts.load_concept_options()
    plan = build_trial_plan(model_keys, conditions, concept_ids, n_repetitions, seed=seed)
    conn = db.init_db(db_path)

    total = len(plan)
    for idx, trial in enumerate(plan, start=1):
        if db.trial_exists(conn, trial.model_key, "closed", trial.condition, trial.concept_id, trial.repetition_index):
            continue  # resumable: skip already-completed trials

        model_spec = config.MODELS[trial.model_key]
        concept_opts = concept_options[trial.concept_id]

        loaded = images.load_concept_images(images_root, trial.concept_id, max_images=max_images)
        db.upsert_concept(
            conn,
            concept_id=trial.concept_id,
            concept_name=trial.concept_id.replace("_", " "),
            naming_consistency=None,
            image_count=len(loaded.filenames),
            folder_path=loaded.folder_path,
        )
        prompt_text = prompts.build_closed_prompt(trial.condition, concept_opts.invented_word, concept_opts.options)

        result = providers.call_model(
            provider=model_spec.provider,
            model_id=model_spec.model_id,
            prompt=prompt_text,
            images=loaded.encoded,
            supports_sampling_params=model_spec.supports_sampling_params,
        )

        extracted_letter, is_valid = extract_closed_answer(result.text)

        record = db.TrialRecord(
            model_key=trial.model_key,
            model_id=model_spec.model_id,
            provider=model_spec.provider,
            task_type="closed",
            condition=trial.condition,
            concept_id=trial.concept_id,
            repetition_index=trial.repetition_index,
            invented_word=concept_opts.invented_word,
            image_filenames=loaded.filenames,
            prompt_text=prompt_text,
            prompt_token_count=result.prompt_tokens,
            completion_token_count=result.completion_tokens,
            duration_ms=result.latency_ms,
            raw_response=result.raw_response,
            extracted_answer=extracted_letter,
            is_valid=is_valid and result.success,
            error_code=result.error,
            attempts=result.attempts,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            params_applied=result.params_applied,
        )
        db.insert_trial(conn, record)

        status = "OK" if record.is_valid else f"INVALID/ERROR ({result.error or result.text!r})"
        print(f"[{idx}/{total}] {trial.model_key} {trial.condition} {trial.concept_id} rep{trial.repetition_index}: {status}")

    conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the closed-task experiment.")
    parser.add_argument("--models", nargs="+", default=list(config.MODELS.keys()))
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3", "C4"])
    parser.add_argument("--concepts", nargs="+", default=None, help="Concept ids; default = all available.")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--db-path", default=config.DB_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    concept_ids = args.concepts or images.list_available_concepts(config.IMAGES_ROOT)
    run_closed_experiment(
        model_keys=args.models,
        conditions=args.conditions,
        concept_ids=concept_ids,
        n_repetitions=args.repetitions,
        max_images=args.max_images,
        db_path=args.db_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
