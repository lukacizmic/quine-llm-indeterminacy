# Expanded Quine-Inspired LLM Experiment

## Problem and approach

Turn the existing rough outline into a preregistration-ready protocol before changing or extending the experiment code. The protocol should define the stimuli, four prompt conditions, models and sampling procedure, lexical and ontological outcome measures, the naming-consistency moderator, exclusion rules, and confirmatory versus exploratory analyses. After the protocol is settled, implement a reproducible experiment pipeline and analysis workflow in the project folder.

## Todos

1. Audit and freeze the conceptual design: define the research question, primary hypothesis, secondary hypotheses, estimands, and scope.
2. Specify the stimulus sample: select the 50 THINGS images/concepts, preserve the naming-consistency range, document selection criteria, and verify image availability.
3. Define the four prompt conditions and controls: minimal, length-matched neutral, generic deliberation, and Quine-inspired.
4. Define response coding: closed ontological categories, open-response normalization, lexical entropy, ontological entropy, and handling of invalid or ambiguous outputs.
5. Define the sampling and API protocol: model versions, temperature and other parameters, repetitions, randomization, rate-limit/error handling, and reproducibility metadata.
6. Write the preregistration document, including confirmatory statistical models, contrasts, interaction terms, multiple-comparison policy, exclusions, and exploratory analyses.
7. Review the protocol for construct validity and threats to inference before implementation.
8. Implement the experiment runner and structured raw-output storage.
9. Implement response classification and statistical analysis, with tests on synthetic fixtures and checks against the preregistered estimands.
10. Run a pilot, inspect technical and measurement failures without changing the confirmatory plan, then revise only through explicitly documented amendments.

## Implementation Architecture: Experiment Runner & SQL Pipeline

The experiment runner will be implemented as a modular Python package/script pipeline using SQLite for persistent, incremental, and crash-resilient data logging.

### 1. Pipeline Flow
- **Stage 1: Trial Plan Generator & Queue**: Generates all parameterized trial configurations (Model × Task_Type [closed/open] × Concept × Condition [C1-C4] × Repetition). Shuffles/randomizes trial execution order to prevent time/burst biases.
- **Stage 2: Unified Vision API Client**: Common interface abstraction over OpenAI, Anthropic, and Google Gemini with exponential backoff, rate limiting, token logging, and timeout management.
- **Stage 3: SQLite Incremental Storage**: Atomic writes after every single API call. If the script crashes or hits a rate limit, resuming will automatically skip already completed trials.
- **Stage 4: Post-Processing & Normalization Engine**:
  - Closed task: Rule-based letter extraction & validity flagging.
  - Open task: Text normalization, head-noun extraction, and multi-coder annotation staging.
- **Stage 5: Entropy & Statistical Aggregation**: Computes concept-level and condition-level lexical and ontological Shannon entropy, exporting tidy dataframes for regression/LMM analysis.

### 2. SQLite Database Schema Design
- `concepts`: concept_id, concept_name, naming_consistency, image_count, folder_path
- `trials`: trial_id (UUID/hash), timestamp, model_name, model_version, task_type (closed/open), condition (C1-C4), concept_id, repetition_index, prompt_text, prompt_token_count, completion_token_count, duration_ms, raw_response, extracted_answer, is_valid, error_code
- `human_codings` (for open-task validation): trial_id, coder_id, assigned_category, confidence, notes
- `entropy_aggregates`: concept_id, model_name, task_type, condition, lexical_entropy_surface, lexical_entropy_head, ontological_entropy

### 3. Key Design Decisions
- **Independence**: Every trial is a stateless, zero-history API call (no conversational memory leakage).
- **Environment Secrets**: API keys loaded via `.env` / environment variables.
- **Raw payload preservation**: Store the exact raw JSON/text returned by the API so parsing can be re-run or adjusted without re-querying the API.

- The primary empirical claim should be about prompt-conditioned changes in model response distributions, not proof of Quine's thesis.
- The primary outcome should be ontological entropy in the closed condition; lexical entropy is a distinct secondary outcome in the open condition.
- Naming consistency should be treated as a continuous moderator rather than only a high/low split.
- Model, concept/image, and repeated trial should be handled as dependent observations in the statistical design; API repetitions should not automatically be treated as independent human participants.
- The existing notebooks and workbook are exploratory precedents, not automatically confirmatory evidence.
- The implementation should use environment variables for API credentials and store model/version/configuration metadata with every response.

## Immediate data-preparation task: preparing the 50 THINGS image set

Create `50_things_images` at the project root and copy (do not move) every `object_images` directory corresponding to the supplied top-50 concept list.

- 49 concepts map unambiguously. Multiword identifiers use the dataset's underscore/hyphen format: `cordon_bleu`, `coat_rack`, `station_wagon`, `fur_coat`, `curling_iron`, `rearview_mirror`, `wrapping_paper`, `first-aid_kit`, `electric_chair`, and `boxing_gloves`.
- The list entry `crystal` is represented provisionally by all three source directories: `crystal1`, `crystal2`, and `crystal_ball`. This produces 52 copied folders for 50 list entries; the final crystal variant will be selected later.
- Verify the copied set and its image counts, while ensuring `object_images` remains unmodified.
