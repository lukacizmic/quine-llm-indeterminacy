"""Qualitative/content analysis of raw model responses stored in ``trials``.

Unlike ``analysis.py`` (which works only with the already-extracted single
letter, ``extracted_answer``), this module recovers the full visible text of
every response from the ``raw_response`` column (the raw provider payload,
kept for audit) and classifies it along two dimensions that matter for
understanding *why* a response was excluded, not just that it was:

1. ``truncated`` -- the provider cut the response off because it ran out of
   its ``max_output_tokens`` budget (``finish_reason == "length"`` for
   OpenAI, ``stop_reason == "max_tokens"`` for Claude, or
   ``FinishReason.MAX_TOKENS`` for Gemini). This is a measurement artifact of
   the fixed token budget, not evidence of ontological indeterminacy.
2. ``reconsiders`` -- the visible text contains an explicit self-correction
   marker (e.g. "wait", "reconsider", "on second thought", "actually,").
   This *is* a substantively interesting signal: it means the model visibly
   entertained more than one candidate referent before answering, which is
   close to what the Quine manipulation is trying to elicit.

Run as a script (``python -m pipeline.response_analysis``) to print summary
tables (counts/rates per model x condition x validity) and export the full
text of every excluded (invalid, non-error) trial to a CSV for manual
spot-checking.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "quine_experiment.sqlite3"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis_exports"

_RECONSIDER_RE = re.compile(
    r"\b(wait|reconsider|on second thought|actually,|hold on|let me re-?think|"
    r"however,? wait|rethink|second-?guess)\b",
    re.IGNORECASE,
)


def extract_visible_text(model_key: str, raw_response: str | None) -> tuple[str | None, bool]:
    """Return (visible_text, truncated) for one trial's raw_response.

    ``truncated`` means the provider stopped generating solely because it hit
    the max-output-tokens budget, not because it finished naturally.
    """
    if not raw_response:
        return None, False

    if model_key == "gpt5_6_luna":
        try:
            data = json.loads(raw_response)
            choice = data["choices"][0]
            text = choice["message"].get("content") or None
            truncated = choice.get("finish_reason") == "length"
            return text, truncated
        except Exception:
            return None, False

    if model_key == "claude_haiku":
        try:
            data = json.loads(raw_response)
            blocks = data.get("content") or []
            text = blocks[0]["text"] if blocks else None
            truncated = data.get("stop_reason") == "max_tokens"
            return text, truncated
        except Exception:
            return None, False

    if model_key == "gemini_3_8_flash":
        # raw_response is str(response) (a Python repr), not JSON, because
        # the Gemini SDK response object isn't a plain dict. Recover the
        # visible text and truncation flag with regexes instead. The SDK's
        # repr uses triple double-quotes for multi-line text but a plain
        # single-quoted literal for short one-line text (e.g. text='a'), so
        # both forms must be tried.
        text_match = re.search(r'text="""(.*?)"""', raw_response, re.DOTALL)
        if not text_match:
            text_match = re.search(r"text='((?:[^'\\]|\\.)*)'", raw_response)
        text = text_match.group(1) if text_match else None
        truncated = "FinishReason.MAX_TOKENS" in raw_response
        return text, truncated

    return None, False


def classify_all() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT model_key, condition, concept_id, repetition_index,
               extracted_answer, is_valid, error_code, raw_response
        FROM trials
        WHERE error_code IS NULL
        """
    )
    rows = []
    for model_key, condition, concept_id, rep, answer, is_valid, error_code, raw in cur.fetchall():
        text, truncated = extract_visible_text(model_key, raw)
        reconsiders = bool(text and _RECONSIDER_RE.search(text))
        rows.append(
            {
                "model_key": model_key,
                "condition": condition,
                "concept_id": concept_id,
                "repetition_index": rep,
                "extracted_answer": answer,
                "is_valid": is_valid,
                "text": text,
                "text_len": len(text) if text else 0,
                "truncated": truncated,
                "reconsiders": reconsiders,
            }
        )
    conn.close()
    return rows


def print_summary(rows: list[dict]) -> None:
    from collections import defaultdict

    # Table 1: among INVALID (excluded) trials, what fraction are truncated
    # vs. reconsidering vs. neither, per model x condition.
    print("\n=== Invalid (excluded) trials: truncated vs. reconsidering, per model x condition ===")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["is_valid"] == 0:
            groups[(r["model_key"], r["condition"])].append(r)

    header = f"{'model':<18}{'cond':<6}{'n_invalid':<11}{'truncated%':<12}{'reconsiders%':<14}{'avg_len':<9}"
    print(header)
    for (model, cond), items in sorted(groups.items()):
        n = len(items)
        trunc_pct = 100 * sum(i["truncated"] for i in items) / n
        recon_pct = 100 * sum(i["reconsiders"] for i in items) / n
        avg_len = sum(i["text_len"] for i in items) / n
        print(f"{model:<18}{cond:<6}{n:<11}{trunc_pct:<12.1f}{recon_pct:<14.1f}{avg_len:<9.0f}")

    # Table 2: reconsideration marker rate among VALID trials too (does the
    # model visibly second-guess itself even when it still lands on one
    # clean letter?), per model x condition.
    print("\n=== Valid trials: reconsideration-marker rate, per model x condition ===")
    groups2: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["is_valid"] == 1:
            groups2[(r["model_key"], r["condition"])].append(r)

    header2 = f"{'model':<18}{'cond':<6}{'n_valid':<10}{'reconsiders%':<14}"
    print(header2)
    for (model, cond), items in sorted(groups2.items()):
        n = len(items)
        recon_pct = 100 * sum(i["reconsiders"] for i in items) / n
        print(f"{model:<18}{cond:<6}{n:<10}{recon_pct:<14.1f}")


def export_invalid_texts(rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "invalid_responses.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model_key", "condition", "concept_id", "repetition_index", "truncated", "reconsiders", "text_len", "text"]
        )
        for r in rows:
            if r["is_valid"] == 0:
                writer.writerow(
                    [
                        r["model_key"],
                        r["condition"],
                        r["concept_id"],
                        r["repetition_index"],
                        r["truncated"],
                        r["reconsiders"],
                        r["text_len"],
                        r["text"],
                    ]
                )
    return out_path


def main() -> None:
    rows = classify_all()
    print_summary(rows)
    out_path = export_invalid_texts(rows)
    print(f"\nExported full text of all invalid trials to: {out_path}")


if __name__ == "__main__":
    main()
