"""Open-task experiment runner.

Mirrors ``closed_runner.py`` (same trial plan, same resumable/atomic storage,
same provider call), but for the open-ended naming task: instead of picking
one of five preset ontological categories, the model is asked to answer with
a single English noun phrase describing what the invented word refers to,
given the same images and the same four prompt conditions (C1-C4).

Response coding here is necessarily lighter-weight than the closed task's
fixed letter rule, since open text can vary in phrasing indefinitely. Two
things are extracted and stored:

  - ``extracted_answer``: the normalized *surface* noun phrase (lowercased,
    stripped of a leading article, markdown, and quoting/punctuation), used
    as-is for ``lexical_entropy_surface`` per plan.md.
  - a separate, recomputable ``extract_head_noun`` helper that reduces the
    surface phrase to its last content word (a crude head-noun heuristic; no
    POS tagger is available in this environment). This is deliberately *not*
    stored in the database -- like the closed-task extraction logic, it is
    kept as a pure function so the head-noun rule can be revised and
    re-applied to the stored surface phrases without re-querying any API.

Ambiguous or borderline extractions (e.g. long, multi-clause answers that
ignore the "one noun phrase" instruction) are marked invalid rather than
guessed at, and are left available in ``raw_response``/``extracted_answer``
for manual review via the ``human_codings`` table (see plan.md, Stage 4).
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys

from . import config, db, images, prompts, providers
from .closed_runner import build_trial_plan

# See closed_runner.py for why this is necessary: raw model text can contain
# arbitrary Unicode that the default Windows console encoding can't print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_REFUSAL_RE = re.compile(
    r"\b(i cannot|i can'?t|i'?m not sure|i do not know|i don'?t know|"
    r"unable to determine|not enough information|insufficient information|"
    r"as an ai|it is unclear|impossible to determine)\b",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_LEADING_HEADING_RE = re.compile(r"^#+\s*")  # e.g. Claude sometimes formats its answer as "# airbag"
_MARKDOWN_RE = re.compile(r"[*_`#]")
_LEADING_QUOTE_RE = re.compile(r"^[\s\"'\u201c\u2018]+")
_TRAILING_QUOTE_PUNCT_RE = re.compile(r"[\s\"'\u201d\u2019.!?,:;]+$")
# A line that is only a generic label ("Answer", "Answer:", "Final answer") and
# carries no actual content of its own -- seen when a model puts the label on
# its own heading line and the real answer on the next line, e.g.
# "# Answer\n\nBamboo". Such a line should be skipped in favor of the next
# non-empty one, rather than treated as the answer itself.
_GENERIC_LABEL_LINE_RE = re.compile(r"^(answer|final answer|my answer)\s*:?\s*$", re.IGNORECASE)
# The same labels, but as a same-line prefix before the actual content, e.g.
# "Answer: kettle" -> "kettle".
_GENERIC_LABEL_PREFIX_RE = re.compile(r"^(answer|final answer|my answer)\s*:\s*", re.IGNORECASE)

# A genuine "one noun phrase" answer should be short. Anything longer is far
# more likely to be an explanation the model gave despite instructions not
# to, so it is flagged invalid rather than silently truncated/guessed at.
_MAX_PLAUSIBLE_WORDS = 6


def normalize_open_response(raw_text: str | None) -> str | None:
    """Reduce raw model text to a clean, comparable surface noun phrase.

    Takes the first substantive line/sentence (in case the model adds
    commentary despite instructions), skipping over a leading line that is
    only a generic label with no content of its own (e.g. a model that
    writes "# Answer" on one line and the actual word on the next). Strips a
    leading markdown heading marker (some models, e.g. Claude, occasionally
    format the answer as "# airbag"), strips markdown emphasis and
    surrounding quotes, strips a leading English article (since "a kettle" /
    "the kettle" / "kettle" should count as the same lexical answer for
    entropy purposes, not three different ones), and lowercases the result.
    """
    if not raw_text:
        return None

    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    if not lines:
        return None

    text = lines[0]
    stripped_of_heading = _LEADING_HEADING_RE.sub("", text).strip()
    if _GENERIC_LABEL_LINE_RE.match(stripped_of_heading) and len(lines) > 1:
        text = lines[1]
    else:
        text = stripped_of_heading
    text = _GENERIC_LABEL_PREFIX_RE.sub("", text)

    if "." in text:
        text = text.split(".", 1)[0]
    text = _MARKDOWN_RE.sub("", text)
    text = _LEADING_QUOTE_RE.sub("", text)
    text = _TRAILING_QUOTE_PUNCT_RE.sub("", text)
    text = _LEADING_ARTICLE_RE.sub("", text).strip()

    return text.lower() or None


def extract_open_answer(raw_text: str | None) -> tuple[str | None, bool]:
    """Apply the open-task response coding rule.

    Returns (surface_phrase_or_None, is_valid). A trial is valid only if a
    non-empty, plausibly-a-noun-phrase-length answer is recovered and it does
    not read as a refusal/hedge ("I'm not sure", "unable to determine", ...).
    """
    normalized = normalize_open_response(raw_text)
    if normalized is None:
        return None, False

    if _REFUSAL_RE.search(normalized):
        return normalized, False

    word_count = len(normalized.split())
    if word_count == 0 or word_count > _MAX_PLAUSIBLE_WORDS:
        return normalized, False

    return normalized, True


def extract_head_noun(surface_phrase: str | None) -> str | None:
    """Crude head-noun heuristic: the last word of the (article-stripped)
    surface phrase, e.g. "metal handle" -> "handle". No POS tagger is used,
    so compound/idiomatic phrases may be reduced incorrectly; this is meant
    as a first pass, with ``human_codings`` available for manual correction
    of ambiguous cases (see plan.md).
    """
    if not surface_phrase:
        return None
    words = surface_phrase.split()
    return words[-1] if words else None


def run_open_experiment(
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
        if db.trial_exists(conn, trial.model_key, "open", trial.condition, trial.concept_id, trial.repetition_index):
            continue  # resumable: skip already-completed trials

        model_spec = config.MODELS[trial.model_key]
        # Only the invented word is needed for the open task; the per-concept
        # decoy options in concept_options.json are irrelevant here and not used.
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
        prompt_text = prompts.build_open_prompt(trial.condition, concept_opts.invented_word)

        result = providers.call_model(
            provider=model_spec.provider,
            model_id=model_spec.model_id,
            prompt=prompt_text,
            images=loaded.encoded,
            supports_sampling_params=model_spec.supports_sampling_params,
        )

        surface_phrase, is_valid = extract_open_answer(result.text)

        record = db.TrialRecord(
            model_key=trial.model_key,
            model_id=model_spec.model_id,
            provider=model_spec.provider,
            task_type="open",
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
            extracted_answer=surface_phrase,
            is_valid=is_valid and result.success,
            error_code=result.error,
            attempts=result.attempts,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            params_applied=result.params_applied,
        )
        db.insert_trial(conn, record)

        status = "OK" if record.is_valid else f"INVALID/ERROR ({result.error or result.text!r})"
        print(
            f"[{idx}/{total}] {trial.model_key} {trial.condition} {trial.concept_id} rep{trial.repetition_index}: "
            f"{status} -> {surface_phrase!r}"
        )

    conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the open-task experiment.")
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
    run_open_experiment(
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
