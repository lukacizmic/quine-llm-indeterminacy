# Quine-Inspired Indeterminacy of Reference in LLMs

An empirical, Quine-inspired experiment testing whether large multimodal
language models exhibit an analogue of **indeterminacy of reference** —
W. V. O. Quine's "radical translation" thought experiment, in which a
linguist observing a speaker say "gavagai" while pointing at a rabbit cannot,
from behavior alone, determine whether the word refers to the whole rabbit,
an undetached rabbit-part, a temporal rabbit-stage, or the abstract property
of "being a rabbit."

This project asks whether an analogous ontological underdetermination shows
up empirically when LLMs are shown images of a novel, invented word and
asked what it refers to and whether making that underdetermination
explicit in the prompt (a "Quine-inspired" framing) shifts model responses
toward greater ontological uncertainty.

## Repository contents

| Path | Description |
|---|---|
| `pipeline/` | The experiment pipeline: trial generation, unified API client (`providers.py`), incremental SQLite storage (`db.py`), closed-task response extraction (`closed_runner.py`), statistical analysis (`analysis.py`, `mixed_model_analysis.py`), qualitative response analysis (`response_analysis.py`), and plotting (`plots.py`). |
| `quine_experiment.sqlite3` | The full experiment database: 18,000 closed-task trials (3 models × 50 concepts × 4 prompt conditions × 30 repetitions), including every raw API response, the extracted answer, validity flag, and naming-consistency data. |
| `results.docx` | The main analysis report: theory, hypotheses, methodology, results (Wilcoxon tests, moderation regression, mixed-effects model), and discussion/limitations. |
| `Final summary.docx`, `Indeterminacy of Reference in Large Language Models.docx` | Supporting write-ups. |
| `Prošireni eksperiment podoređenosti referencije u LLM-ovima.docx` | The original (Croatian-language) proposal for the expanded, Quine-inspired version of the experiment. |
| `plan.md` | Design/implementation plan for the expanded experiment and pipeline. |
| `analysis_plots/` | Generated figures (entropy distributions by model/condition, moderation scatterplots). |
| `analysis_exports/` | Exported CSVs from qualitative/content analysis of raw model responses. |
| `original_version/` | Earlier drafts, notebooks, and analyses from the initial (pilot-scale) version of the experiment. |
| `things_dataset_check.ipynb` | Notebook used to inspect and validate the THINGS image dataset subset used as stimuli. |
| `list_of_top_50_concepts.txt` | The 50 concepts (THINGS dataset categories) used as stimuli. |
| `image-level_description.txt` | Notes on the per-image metadata used for stimulus selection. |
| `run_pilot.py` | Script used to run the initial pilot batch of trials. |

## Experimental design (summary)

- **Stimuli**: 50 concepts drawn from the [THINGS image dataset](https://things-initiative.org/), each represented by 10 images. The English word for each concept is never shown; instead, each concept is paired with an invented, meaningless word.
- **Task**: For each concept, models are shown the 10 images and asked which of five plausible ontological categories the invented word most likely refers to (e.g., the object itself, a property, a function, an abstract category, or a part of the object).
- **Conditions**: Four prompt framings (C1–C4) ranging from a minimal instruction (C1) to a full Quine-inspired framing (C4) that explicitly raises the possibility of referential indeterminacy, with two length/deliberation-matched controls (C2, C3) in between.
- **Models**: GPT-5.6 Luna (OpenAI), Claude Haiku 4.5 (Anthropic), Gemini 3.8 Flash (Google).
- **Outcome measure**: Normalized Shannon entropy of the five-category answer distribution per concept/model/condition (higher entropy = more ontological disagreement/underdetermination across repeated trials).
- **Moderator**: Each concept's independently measured *naming consistency* (from the THINGS dataset norming data) — the hypothesis being that concepts with lower naming consistency should show a larger Quine-prompt-induced entropy increase.

See `results.docx` for the full write-up, including the corrected response-extraction methodology, per-model Wilcoxon signed-rank tests, moderation regression, and a confirmatory mixed-effects model testing whether the prompt effect is significantly model-dependent.

## Reproducing the analysis

The pipeline requires Python with `pandas`, `numpy`, `scipy`, `statsmodels`,
and `matplotlib` installed. With the database in place:

```bash
python -m pipeline.analysis          # Wilcoxon tests + moderation regression
python -m pipeline.mixed_model_analysis  # confirmatory mixed-effects model
python -m pipeline.plots             # regenerate figures in analysis_plots/
```

API keys for re-running trials (not needed to reproduce the analysis of the
existing data) are read from environment variables; see `pipeline/config.py`
and `pipeline/providers.py`.

## Obtaining the image stimuli

The raw THINGS images are **not included in this repository** (large,
research-licensed dataset). To reproduce stimulus preparation:

1. Obtain the THINGS image database from https://things-initiative.org/.
2. Use `list_of_top_50_concepts.txt` to identify the 50 concept folders used.
3. See `plan.md` for the exact folder-copying procedure used to build the
   `50_things_images/` subset from the full `object_images/` dataset.
