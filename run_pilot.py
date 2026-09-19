"""Simple script to run the closed experiment without using the command line.

Edit the values below, then just click "Run" in VS Code (or run this file
the same way you'd run any other .py file).
"""
from pipeline.closed_runner import run_closed_experiment

run_closed_experiment(
    model_keys=["gpt5_6_luna", "claude_haiku", "gemini_3_8_flash"],
    conditions=["C1"],
    concept_ids=["gate", "airbag"],
    n_repetitions=2,
    db_path="pilot.sqlite3",
)
