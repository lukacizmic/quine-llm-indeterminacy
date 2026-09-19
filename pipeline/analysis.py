"""Primary hypothesis analysis (Option 1 + Option 2, per plan discussion).

Option 1: collapse repetitions into one Shannon-entropy value per
(model, condition, concept), then run paired condition-vs-condition
comparisons across the 50 concepts (Wilcoxon signed-rank), separately per
model. This directly targets H1/H1a's core claim: "Quine condition produces
higher ontological entropy than the other conditions."

Option 2: for each concept/model, compute a "Quine advantage" score
(entropy(Quine) - entropy(control)) and regress it on naming consistency.
A significant negative slope supports the H1/H1a moderation claim (larger
Quine advantage when naming consistency is lower).

This script only implements the CLOSED-task ontological-category entropy
analysis (categories are already the model's direct answer, a-e). The open
task requires the extra step of human-coding responses into the five
categories before this same entropy logic applies; lexical entropy for open
responses is a separate, simpler word-distribution calculation not covered
here (see docstring note at bottom).
"""
from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

DB_PATH = "quine_experiment.sqlite3"
METADATA_TSV = "_images-metadata_things.tsv"
NAMING_CONSISTENCY_COLUMN = "nameability_naming-consistency"
CONCEPT_COLUMN = "Word"  # column in the metadata tsv matching concept_id

# Condition codes as stored in the trials table.
CONDITIONS = ["C1", "C2", "C3", "C4"]
QUINE_CONDITION = "C4"
CONTROL_CONDITIONS = ["C1", "C2", "C3"]  # minimal, length-matched neutral, generic-deliberation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_valid_trials(db_path: str = DB_PATH, task_type: str = "closed") -> pd.DataFrame:
    """Load only valid, non-error trials for the given task type.

    A trial counts toward entropy calculation only if it is a substantive
    response: is_valid=1 and error_code is NULL. Invalid/ambiguous responses
    and technical errors are excluded per the docx's exclusion rules, but are
    still counted for reporting n_valid_trials / exclusion rates separately
    (see `exclusion_rate_report`).
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT model_key, condition, concept_id, repetition_index,
                   extracted_answer, is_valid, error_code
            FROM trials
            WHERE task_type = ?
            """,
            conn,
            params=(task_type,),
        )
    finally:
        conn.close()
    return df


def exclusion_rate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report, per model x condition, how many trials were excluded and why.

    Report this alongside your results: a condition/model with a much higher
    exclusion rate could bias entropy estimates (fewer valid draws) or
    reflect a genuine effect worth discussing (e.g. models refusing more
    often, or giving multi-letter answers more often, in one condition).
    """
    total = df.groupby(["model_key", "condition"]).size().rename("n_total")
    valid = (
        df[df["is_valid"] == 1]
        .groupby(["model_key", "condition"])
        .size()
        .rename("n_valid")
    )
    errors = (
        df[df["error_code"].notna()]
        .groupby(["model_key", "condition"])
        .size()
        .rename("n_technical_error")
    )
    report = pd.concat([total, valid, errors], axis=1).fillna(0).astype(int)
    report["n_invalid_response"] = report["n_total"] - report["n_valid"] - report["n_technical_error"]
    report["exclusion_rate"] = 1 - report["n_valid"] / report["n_total"]
    return report.reset_index()


def load_naming_consistency(tsv_path: str = METADATA_TSV) -> pd.Series:
    """Return a Series mapping concept_id (folder name) -> naming consistency.

    The metadata tsv's `Word` column stores multiword concepts with spaces
    (e.g. "coat rack", "first-aid kit"), while concept_id folder names/keys
    use underscores instead (e.g. "coat_rack", "first-aid_kit"). The index is
    normalized here (spaces -> underscores) so joins against concept_id
    succeed for all 50 concepts; verified against pipeline/concept_options.json
    keys (10 multiword concepts previously failed to match before this fix).
    """
    meta = pd.read_csv(tsv_path, sep="\t", usecols=[CONCEPT_COLUMN, NAMING_CONSISTENCY_COLUMN])
    meta = meta.dropna(subset=[NAMING_CONSISTENCY_COLUMN])
    # If a concept has multiple image rows, average their naming consistency.
    consistency = meta.groupby(CONCEPT_COLUMN)[NAMING_CONSISTENCY_COLUMN].mean()
    consistency.index = consistency.index.astype(str).str.replace(" ", "_")
    return consistency


# ---------------------------------------------------------------------------
# Option 1: per-(model, condition, concept) ontological entropy
# ---------------------------------------------------------------------------

def shannon_entropy(counts: pd.Series, n_categories: int = 5) -> float:
    """Normalized Shannon entropy (0-1) over the 5 ontological categories.

    counts: value_counts()-style Series of category letters actually observed.
    Categories with zero observed count contribute 0 to the sum (standard
    convention: 0 * log(0) := 0), so it is safe to pass only the categories
    that occurred.
    """
    n = counts.sum()
    if n == 0:
        return float("nan")
    probs = counts / n
    raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(n_categories)
    return raw_entropy / max_entropy


def compute_entropy_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model_key, condition, concept_id) with entropy + n_valid.

    Only valid trials (is_valid == 1) contribute to entropy; this matches the
    docx's exclusion rules (technical errors and ambiguous/invalid answers do
    not count as a substantive response).
    """
    valid = df[df["is_valid"] == 1]
    rows = []
    for (model_key, condition, concept_id), group in valid.groupby(
        ["model_key", "condition", "concept_id"]
    ):
        counts = group["extracted_answer"].value_counts()
        rows.append(
            {
                "model_key": model_key,
                "condition": condition,
                "concept_id": concept_id,
                "entropy": shannon_entropy(counts),
                "n_valid": int(counts.sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Option 1: paired condition comparisons (per model)
# ---------------------------------------------------------------------------

@dataclass
class PairedComparisonResult:
    model_key: str
    condition_a: str
    condition_b: str
    n_concepts: int
    median_diff: float
    wilcoxon_statistic: float
    p_value: float


def paired_condition_comparison(
    entropy_table: pd.DataFrame, condition_a: str, condition_b: str
) -> list[PairedComparisonResult]:
    """Wilcoxon signed-rank test comparing entropy(condition_a) vs entropy(condition_b),
    paired by concept, run separately per model.

    Use this for each of: (C4 vs C1), (C4 vs C2), (C4 vs C3) to test H1a's
    core claim per model. A significant positive median_diff (a - b > 0)
    with condition_a=C4 supports "Quine condition has higher entropy".
    """
    results = []
    for model_key, group in entropy_table.groupby("model_key"):
        pivot = group.pivot(index="concept_id", columns="condition", values="entropy")
        pivot = pivot.dropna(subset=[condition_a, condition_b])
        if len(pivot) < 5:
            print(f"WARNING: only {len(pivot)} concepts with both conditions for {model_key}; skipping.")
            continue
        diffs = pivot[condition_a] - pivot[condition_b]
        stat, p = stats.wilcoxon(pivot[condition_a], pivot[condition_b])
        results.append(
            PairedComparisonResult(
                model_key=model_key,
                condition_a=condition_a,
                condition_b=condition_b,
                n_concepts=len(pivot),
                median_diff=float(diffs.median()),
                wilcoxon_statistic=float(stat),
                p_value=float(p),
            )
        )
    return results


def run_all_h1_comparisons(entropy_table: pd.DataFrame) -> pd.DataFrame:
    """Run C4-vs-each-control comparisons for every model; return a tidy table."""
    all_results = []
    for control in CONTROL_CONDITIONS:
        all_results.extend(paired_condition_comparison(entropy_table, QUINE_CONDITION, control))
    return pd.DataFrame([r.__dict__ for r in all_results])


# ---------------------------------------------------------------------------
# Option 2: Quine advantage vs naming consistency (moderation, per model)
# ---------------------------------------------------------------------------

@dataclass
class ModerationResult:
    model_key: str
    control_condition: str
    n_concepts: int
    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float


def quine_advantage_table(
    entropy_table: pd.DataFrame, naming_consistency: pd.Series, control_condition: str
) -> pd.DataFrame:
    """One row per (model_key, concept_id): entropy(C4) - entropy(control), plus naming consistency."""
    pivot = entropy_table.pivot_table(
        index=["model_key", "concept_id"], columns="condition", values="entropy"
    )
    pivot = pivot.dropna(subset=[QUINE_CONDITION, control_condition])
    advantage = (pivot[QUINE_CONDITION] - pivot[control_condition]).rename("quine_advantage")
    result = advantage.reset_index()
    result["naming_consistency"] = result["concept_id"].map(naming_consistency)
    missing = result["naming_consistency"].isna().sum()
    if missing:
        print(f"WARNING: {missing} concept rows have no naming-consistency match and will be dropped.")
    return result.dropna(subset=["naming_consistency"])


def moderation_regression(advantage_table: pd.DataFrame, control_condition: str) -> list[ModerationResult]:
    """Simple linear regression: quine_advantage ~ naming_consistency, per model.

    A significant negative slope supports H1/H1a's moderation prediction:
    the Quine-condition entropy advantage over `control_condition` grows as
    naming consistency decreases.
    """
    results = []
    for model_key, group in advantage_table.groupby("model_key"):
        if len(group) < 5:
            print(f"WARNING: only {len(group)} concepts for {model_key}; skipping regression.")
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            group["naming_consistency"], group["quine_advantage"]
        )
        results.append(
            ModerationResult(
                model_key=model_key,
                control_condition=control_condition,
                n_concepts=len(group),
                slope=slope,
                intercept=intercept,
                r_value=r_value,
                p_value=p_value,
                std_err=std_err,
            )
        )
    return results


def run_all_h1_moderation(entropy_table: pd.DataFrame, naming_consistency: pd.Series) -> pd.DataFrame:
    all_results = []
    for control in CONTROL_CONDITIONS:
        advantage = quine_advantage_table(entropy_table, naming_consistency, control)
        all_results.extend(moderation_regression(advantage, control))
    return pd.DataFrame([r.__dict__ for r in all_results])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    trials = load_valid_trials()
    print("=== Exclusion report ===")
    print(exclusion_rate_report(trials).to_string(index=False))

    entropy_table = compute_entropy_table(trials)
    print("\n=== Entropy table (head) ===")
    print(entropy_table.head(10).to_string(index=False))

    print("\n=== H1/H1a: Quine (C4) vs each control (paired Wilcoxon per model) ===")
    h1_results = run_all_h1_comparisons(entropy_table)
    print(h1_results.to_string(index=False))

    naming_consistency = load_naming_consistency()
    print("\n=== H1/H1a moderation: Quine advantage ~ naming consistency (per model) ===")
    moderation_results = run_all_h1_moderation(entropy_table, naming_consistency)
    print(moderation_results.to_string(index=False))


if __name__ == "__main__":
    main()
