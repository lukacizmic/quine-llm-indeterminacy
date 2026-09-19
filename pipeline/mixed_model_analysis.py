"""Unified linear mixed-effects test of H1/H1a across all three models at once.

The existing analysis (``analysis.py``) tests H1/H1a with nine separate
paired Wilcoxon tests and nine separate moderation regressions (3 models x 3
control conditions each), run independently per model. That approach cannot
directly answer the question "is the Quine effect *significantly* different
across models?" -- it can only be answered informally, by eyeballing whether
the three sets of per-model p-values look different from each other.

This module fits a single linear mixed-effects model on the same
per-(model, condition, concept) entropy table used elsewhere, with:

  - Fixed effects: condition, model, their interaction (condition:model --
    directly tests whether the Quine effect differs by model), naming
    consistency, and its interaction with condition (the moderation claim,
    now estimated jointly across models with model x naming-consistency x
    condition included in the full model).
  - Random effect: a random intercept per concept_id, since the same 50
    concepts are observed repeatedly across all 12 condition x model cells
    (a crossed repeated-measures structure).

Likelihood-ratio tests compare nested models (fit with ML, not REML, since
REML likelihoods aren't comparable across models with different fixed
effects) to test the condition:model interaction and the moderation terms as
single omnibus questions, rather than nine separate p-values.

Caveat: entropy is a bounded (0-1), often skewed measure, and the Wilcoxon
tests elsewhere were chosen specifically to avoid the normality assumption
this linear model relies on. This analysis is a genuinely different,
complementary lens (it can test the model x condition interaction directly,
which the paired tests cannot), not a strictly more "correct" replacement --
both should be reported together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from . import analysis


def build_model_frame(db_path: str = analysis.DB_PATH) -> pd.DataFrame:
    """One row per (model_key, condition, concept_id): entropy + naming_consistency."""
    trials = analysis.load_valid_trials(db_path)
    entropy_table = analysis.compute_entropy_table(trials)
    consistency = analysis.load_naming_consistency()

    df = entropy_table.copy()
    df["naming_consistency"] = df["concept_id"].map(consistency)
    df = df.dropna(subset=["naming_consistency", "entropy"])

    # Center naming_consistency so the condition main-effect coefficients are
    # interpretable at an average concept, and so the model converges better.
    df["naming_consistency_c"] = df["naming_consistency"] - df["naming_consistency"].mean()
    return df


def _fit(formula: str, df: pd.DataFrame):
    model = smf.mixedlm(formula, df, groups=df["concept_id"], re_formula="1")
    # Left to its own default optimizer schedule (which already retries with
    # lbfgs/cg as needed), this converges to a valid log-likelihood for every
    # formula used here even when a ConvergenceWarning is raised. Forcing an
    # explicit multi-optimizer method list was tried but sometimes drives the
    # random-intercept variance to the exact boundary (0), which then makes
    # the Hessian used for standard errors singular and raises a hard error.
    # Since the LR tests below only need the log-likelihood (not the SEs),
    # and convergence is checked and reported explicitly, the default is used.
    fitted = model.fit(reml=False)
    return fitted


def likelihood_ratio_test(full_formula: str, reduced_formula: str, df: pd.DataFrame) -> dict:
    full = _fit(full_formula, df)
    reduced = _fit(reduced_formula, df)
    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = int(full.df_modelwc - reduced.df_modelwc)
    p_value = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else float("nan")
    return {
        "full_formula": full_formula,
        "reduced_formula": reduced_formula,
        "lr_statistic": lr_stat,
        "df_diff": df_diff,
        "p_value": p_value,
        "full_llf": full.llf,
        "reduced_llf": reduced.llf,
        "full_converged": bool(full.converged),
        "reduced_converged": bool(reduced.converged),
    }


def run_omnibus_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Three key omnibus questions, each as a single likelihood-ratio test."""
    base = "entropy ~ C(condition, Treatment('C1')) + C(model_key) + naming_consistency_c"

    tests = [
        (
            "Does the condition effect differ by model? (condition:model interaction)",
            base + " + C(condition, Treatment('C1')):C(model_key)",
            base,
        ),
        (
            "Does naming consistency moderate the condition effect, pooled across models? "
            "(naming_consistency:condition interaction)",
            base + " + C(condition, Treatment('C1')):naming_consistency_c",
            base,
        ),
        (
            "Does the moderation effect itself differ by model? "
            "(3-way condition:model:naming_consistency interaction)",
            base
            + " + C(condition, Treatment('C1')):C(model_key)"
            + " + C(condition, Treatment('C1')):naming_consistency_c"
            + " + C(condition, Treatment('C1')):C(model_key):naming_consistency_c",
            base
            + " + C(condition, Treatment('C1')):C(model_key)"
            + " + C(condition, Treatment('C1')):naming_consistency_c",
        ),
    ]

    rows = []
    for label, full_f, reduced_f in tests:
        result = likelihood_ratio_test(full_f, reduced_f, df)
        result["question"] = label
        rows.append(result)
    return pd.DataFrame(rows)[
        [
            "question",
            "lr_statistic",
            "df_diff",
            "p_value",
            "full_converged",
            "reduced_converged",
            "full_formula",
            "reduced_formula",
        ]
    ]


def fit_full_model_summary(df: pd.DataFrame):
    """Fit the full model (all interactions) and return its fitted result object."""
    formula = (
        "entropy ~ C(condition, Treatment('C1')) * C(model_key)"
        " + C(condition, Treatment('C1')) * naming_consistency_c"
        " + C(condition, Treatment('C1')):C(model_key):naming_consistency_c"
    )
    return _fit(formula, df)


def cross_check_with_cluster_robust_ols(df: pd.DataFrame):
    """Sanity-check the mixed-model conclusions with a simpler fallback.

    Mixed-effects ML fits can fail to converge tightly when the random-
    intercept variance is small relative to residual variance (as happens
    here, since two of the three models have near-zero entropy almost
    everywhere). An OLS regression with the same fixed effects, using
    cluster-robust standard errors clustered by concept_id (no variance-
    component estimation to converge on), is a robust cross-check: if it
    points to the same conclusions as the mixed-model LR tests, that is
    reassuring; if not, the mixed-model convergence warnings should be taken
    seriously rather than the LR-test results being reported at face value.
    """
    full_formula = (
        "entropy ~ C(condition, Treatment('C1')) * C(model_key)"
        " + C(condition, Treatment('C1')) * naming_consistency_c"
    )
    ols = smf.ols(full_formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["concept_id"]})
    # Joint Wald F-tests for each interaction block, analogous to the LR tests.
    cond_model_terms = [t for t in ols.model.exog_names if ":C(model_key)" in t and "naming" not in t]
    cond_naming_terms = [t for t in ols.model.exog_names if "naming_consistency_c" in t and t != "naming_consistency_c"]
    results = {}
    if cond_model_terms:
        results["condition:model (Wald F-test)"] = ols.f_test(
            [f"{t} = 0" for t in cond_model_terms]
        )
    if cond_naming_terms:
        results["condition:naming_consistency (Wald F-test)"] = ols.f_test(
            [f"{t} = 0" for t in cond_naming_terms]
        )
    return ols, results


def main() -> None:
    df = build_model_frame()
    print(f"n rows (model x condition x concept cells): {len(df)}")
    print(f"n concepts: {df['concept_id'].nunique()}\n")

    print("=== Omnibus likelihood-ratio tests (mixed-effects model, concept random intercept) ===")
    omnibus = run_omnibus_tests(df)
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(
            omnibus[
                ["question", "lr_statistic", "df_diff", "p_value", "full_converged", "reduced_converged"]
            ].to_string(index=False)
        )

    print("\n=== Full mixed-effects model fixed-effects summary ===")
    full_model = fit_full_model_summary(df)
    print(f"(converged: {full_model.converged})")
    print(full_model.summary())

    print("\n=== Cross-check: cluster-robust OLS (no variance-component convergence issues) ===")
    ols, wald_results = cross_check_with_cluster_robust_ols(df)
    for label, wald in wald_results.items():
        print(f"\n{label}:")
        print(wald)



if __name__ == "__main__":
    main()
