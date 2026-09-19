"""SQLite schema and resumable, atomic trial storage.

Design goals (per plan.md):
  - Every trial writes immediately (atomic commit) so a crash or rate limit
    never loses already-collected data.
  - Re-running the same trial plan skips trials already present in the DB.
  - Raw API payloads are preserved in full for later re-parsing.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    concept_name TEXT NOT NULL,
    naming_consistency REAL,
    image_count INTEGER,
    folder_path TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_key TEXT NOT NULL,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    task_type TEXT NOT NULL,           -- 'closed' or 'open'
    condition TEXT NOT NULL,           -- 'C1'..'C4'
    concept_id TEXT NOT NULL,
    repetition_index INTEGER NOT NULL,
    invented_word TEXT,
    image_filenames TEXT,              -- JSON list, exact order used
    prompt_text TEXT NOT NULL,
    prompt_token_count INTEGER,
    completion_token_count INTEGER,
    duration_ms REAL,
    raw_response TEXT,                 -- JSON/text dump of the full API response
    extracted_answer TEXT,
    is_valid INTEGER,                  -- 0/1
    error_code TEXT,
    attempts INTEGER,
    params_applied TEXT,               -- JSON dict, e.g. {"temperature": 0.7, "top_p": 1.0} or nulls if unsupported by the model
    UNIQUE (model_key, task_type, condition, concept_id, repetition_index)
);

CREATE TABLE IF NOT EXISTS human_codings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    coder_id TEXT NOT NULL,
    assigned_category TEXT,
    confidence REAL,
    notes TEXT,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);

CREATE TABLE IF NOT EXISTS entropy_aggregates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    task_type TEXT NOT NULL,
    condition TEXT NOT NULL,
    lexical_entropy_surface REAL,
    lexical_entropy_head REAL,
    ontological_entropy REAL,
    n_valid_trials INTEGER
);
"""


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_concept(
    conn: sqlite3.Connection,
    concept_id: str,
    concept_name: str,
    naming_consistency: float | None,
    image_count: int,
    folder_path: str,
) -> None:
    conn.execute(
        """
        INSERT INTO concepts (concept_id, concept_name, naming_consistency, image_count, folder_path)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(concept_id) DO UPDATE SET
            concept_name=excluded.concept_name,
            naming_consistency=excluded.naming_consistency,
            image_count=excluded.image_count,
            folder_path=excluded.folder_path;
        """,
        (concept_id, concept_name, naming_consistency, image_count, folder_path),
    )
    conn.commit()


def make_trial_id(model_key: str, task_type: str, condition: str, concept_id: str, repetition_index: int) -> str:
    return f"{model_key}|{task_type}|{condition}|{concept_id}|{repetition_index}"


def trial_exists(
    conn: sqlite3.Connection,
    model_key: str,
    task_type: str,
    condition: str,
    concept_id: str,
    repetition_index: int,
) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM trials
        WHERE model_key = ? AND task_type = ? AND condition = ? AND concept_id = ? AND repetition_index = ?
          AND error_code IS NULL
        """,
        (model_key, task_type, condition, concept_id, repetition_index),
    ).fetchone()
    return row is not None


@dataclass
class TrialRecord:
    model_key: str
    model_id: str
    provider: str
    task_type: str
    condition: str
    concept_id: str
    repetition_index: int
    invented_word: str | None
    image_filenames: list[str]
    prompt_text: str
    prompt_token_count: int | None
    completion_token_count: int | None
    duration_ms: float
    raw_response: Any
    extracted_answer: str | None
    is_valid: bool
    error_code: str | None
    attempts: int
    timestamp: str
    params_applied: dict[str, float | int | None] = field(default_factory=dict)


def insert_trial(conn: sqlite3.Connection, record: TrialRecord) -> None:
    trial_id = make_trial_id(
        record.model_key, record.task_type, record.condition, record.concept_id, record.repetition_index
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO trials (
            trial_id, timestamp, model_key, model_id, provider, task_type, condition,
            concept_id, repetition_index, invented_word, image_filenames, prompt_text,
            prompt_token_count, completion_token_count, duration_ms, raw_response,
            extracted_answer, is_valid, error_code, attempts, params_applied
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trial_id,
            record.timestamp,
            record.model_key,
            record.model_id,
            record.provider,
            record.task_type,
            record.condition,
            record.concept_id,
            record.repetition_index,
            record.invented_word,
            json.dumps(record.image_filenames),
            record.prompt_text,
            record.prompt_token_count,
            record.completion_token_count,
            record.duration_ms,
            json.dumps(record.raw_response) if not isinstance(record.raw_response, str) else record.raw_response,
            record.extracted_answer,
            int(record.is_valid),
            record.error_code,
            record.attempts,
            json.dumps(record.params_applied),
        ),
    )
    conn.commit()  # atomic, immediate commit per trial (crash-resilient)
