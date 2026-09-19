"""Visualizations for the H1/H1a closed-task analysis.

Generates and saves (does not display) three figure types into an output
directory:
  1. entropy_by_condition_<model>.png - boxplot of per-concept entropy
     across C1-C4, one figure per model. Visualizes the core H1 comparison.
  2. entropy_by_condition_grid.png - all three models side-by-side for an
     at-a-glance cross-model comparison.
  3. moderation_<model>_<control>.png - scatter of naming consistency (x)
     vs Quine advantage (y) with fitted regression line, one figure per
     model x control-condition combination (9 total), visualizing Option 2.

Run as a script: python -m pipeline.plots
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: never try to open an interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis

OUTPUT_DIR = "analysis_plots"

MODEL_ORDER = ["gpt5_6_luna", "claude_haiku", "gemini_3_8_flash"]
MODEL_LABELS = {
    "gpt5_6_luna": "GPT-5.6 Luna",
    "claude_haiku": "Claude Haiku 4.5",
    "gemini_3_8_flash": "Gemini 3.8 Flash",
}
CONDITION_LABELS = {
    "C1": "C1\n(minimal)",
    "C2": "C2\n(neutral)",
    "C3": "C3\n(generic\ndeliberation)",
    "C4": "C4\n(Quine)",
}


def _ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_entropy_by_condition(entropy_table: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> list[str]:
    """One boxplot per model: entropy distribution (across 50 concepts) per condition.

    Directly visualizes the H1 comparison: is the C4 (Quine) box shifted up
    relative to C1/C2/C3?
    """
    _ensure_output_dir(output_dir)
    saved = []
    for model_key in MODEL_ORDER:
        sub = entropy_table[entropy_table["model_key"] == model_key]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        data = [sub[sub["condition"] == c]["entropy"].dropna().values for c in analysis.CONDITIONS]
        bp = ax.boxplot(
            data,
            tick_labels=[CONDITION_LABELS[c] for c in analysis.CONDITIONS],
            patch_artist=True,
            showmeans=True,
        )
        colors = ["#8fb7e0", "#8fb7e0", "#8fb7e0", "#e0938f"]  # highlight C4 in a different color
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        # overlay individual concept points, jittered, for transparency about n=50
        for i, condition in enumerate(analysis.CONDITIONS, start=1):
            vals = sub[sub["condition"] == condition]["entropy"].dropna().values
            jitter = np.random.default_rng(0).normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.35, s=12, color="black")
        ax.set_ylabel("Normalized Shannon entropy (0-1)")
        ax.set_title(f"{MODEL_LABELS[model_key]}: entropy by condition\n(each point = one concept, n=50)")
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        path = os.path.join(output_dir, f"entropy_by_condition_{model_key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_entropy_grid(entropy_table: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """All three models side-by-side in one figure for cross-model comparison."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, model_key in zip(axes, MODEL_ORDER):
        sub = entropy_table[entropy_table["model_key"] == model_key]
        data = [sub[sub["condition"] == c]["entropy"].dropna().values for c in analysis.CONDITIONS]
        bp = ax.boxplot(
            data,
            tick_labels=[CONDITION_LABELS[c] for c in analysis.CONDITIONS],
            patch_artist=True,
            showmeans=True,
        )
        colors = ["#8fb7e0", "#8fb7e0", "#8fb7e0", "#e0938f"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Normalized Shannon entropy (0-1)")
    fig.suptitle("Ontological-category entropy by condition, across all models (n=50 concepts each)")
    fig.tight_layout()
    path = os.path.join(output_dir, "entropy_by_condition_grid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_moderation(
    entropy_table: pd.DataFrame, naming_consistency: pd.Series, output_dir: str = OUTPUT_DIR
) -> list[str]:
    """Scatter + fitted line: naming consistency (x) vs Quine advantage (y).

    One figure per (model, control_condition) combination. A downward-sloping
    line supports the moderation prediction (lower naming consistency ->
    bigger Quine advantage).
    """
    _ensure_output_dir(output_dir)
    saved = []
    for control in analysis.CONTROL_CONDITIONS:
        advantage_table = analysis.quine_advantage_table(entropy_table, naming_consistency, control)
        for model_key in MODEL_ORDER:
            sub = advantage_table[advantage_table["model_key"] == model_key]
            if len(sub) < 5:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(sub["naming_consistency"], sub["quine_advantage"], alpha=0.7, s=30, color="#4a6fa5")
            slope, intercept, r_value, p_value, _ = analysis.stats.linregress(
                sub["naming_consistency"], sub["quine_advantage"]
            )
            xs = np.linspace(sub["naming_consistency"].min(), sub["naming_consistency"].max(), 100)
            ax.plot(xs, slope * xs + intercept, color="#c0392b", linewidth=2)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Naming consistency (human agreement, 0-1)")
            ax.set_ylabel(f"Quine advantage\n(entropy[C4] - entropy[{control}])")
            ax.set_title(
                f"{MODEL_LABELS[model_key]}: Quine advantage vs naming consistency\n"
                f"(control={control}, n={len(sub)}, slope={slope:.3f}, p={p_value:.3f})"
            )
            fig.tight_layout()
            path = os.path.join(output_dir, f"moderation_{model_key}_{control}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
    return saved


def main() -> None:
    trials = analysis.load_valid_trials()
    entropy_table = analysis.compute_entropy_table(trials)
    naming_consistency = analysis.load_naming_consistency()

    saved = []
    saved += plot_entropy_by_condition(entropy_table)
    saved.append(plot_entropy_grid(entropy_table))
    saved += plot_moderation(entropy_table, naming_consistency)

    print(f"Saved {len(saved)} plots to '{OUTPUT_DIR}/':")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
