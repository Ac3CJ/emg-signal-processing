# File Handling Branching Guidelines

## Goal

Remove duplicated secondary-vs-collected branching from extraction code.

The pipeline should be:

1. Extract the file path and selection once.
2. Let `FileRepository.py` resolve the correct raw or labelled path.
3. Load the file.
4. Branch only in the processing and output functions that actually differ by dataset.

Extraction should not branch on `secondary` vs `collected` if the repository can already derive the correct path.

## Core Rule

If a function only needs to know where a file lives, it should call `DataRepository`.
If a function needs to decide how to process different dataset types, the branching belongs there.

Do not duplicate path logic in consumer modules when the repository can already do it.

## Existing Branching Hotspots

### `DataPreparation.py`

- `load_and_prepare_dataset(...)` still has a secondary-specific path setup and collected-specific label-path resolution.
- `load_collected_data(...)` still branches on repository-backed vs legacy folder-based access.
- The collected path currently still has a fallback path scan for nonstandard layouts.

### `SignalAnalysis.py`

- `generate_all_signal_images(...)` branches on `data_structure == 'secondary'` vs `data_structure == 'collected'`.
- The collected branch still does participant discovery and file iteration logic inside the function.
- This is the clearest case where extraction should be unified and the processing/output stage should branch instead.

### `ModelTraining.py`

- `_discover_collected_participants(...)` and `_discover_secondary_subjects(...)` duplicate discovery logic that should come from `DataRepository`.
- The `loso`, `transfer`, and `standard` modes each branch separately for collected vs secondary dataset setup.
- The CLI still builds raw and edited collected paths directly.
- This file is the main place where the pipeline should be collapsed so training consumes normalized datasets rather than dataset-specific path chains.

### `ModelValidator.py`

- `run_collected_validation(...)` still has a collected-only branch for output path construction.
- `run_fast_validation(...)` and `run_ensemble_validation(...)` still hardcode secondary file naming patterns.
- These functions should ask the repository for the file path, then branch only on validation behavior, not path shape.

### `SignalViewerGUI.py`

- `_load_signal(...)` already uses the repository for labelled fallback, but it still performs file reading and label fallback in one step.
- This should remain path-neutral: load the requested file, then use the repository for companion resolution only if needed.

### `SignalReading.py`

- `DataCollectionMode.save_collection(...)` now uses the repository for saving, which is correct.
- There is no meaningful secondary-vs-collected branching here, and it should stay that way.

## What To Refactor Later

### Extraction Phase

These parts should be unified and kept branch-free where possible:

- Path generation
- Path-to-selection conversion
- Labelled companion resolution
- Participant discovery
- Movement discovery
- Raw vs labelled preference selection

### Processing Phase

Branch here only when the actual signal logic differs:

- Different augmentation rules
- Different validation outputs
- Different plotting or reporting layout
- Different training splits or evaluation modes

### Function Boundary Rule

If a function does both extraction and processing, split it.

Example pattern:

- `resolve file path`
- `load mat payload`
- `process payload`
- `emit output`

Do not mix the first step with the later steps if that forces dataset-specific branching into the loader.

## Repository Responsibilities

`FileRepository.py` should own:

- `from_standard_path(...)`
- `raw_root(...)`
- `edited_root(...)`
- `raw_file_path(...)`
- `output_file_path(...)`
- `preferred_input_path(...)`
- `selection_from_path(...)`
- `labelled_candidate_path(...)`
- `discover_participants(...)`
- `discover_movements(...)`
- `secondary_subject_root(...)`
- `secondary_kinematics_file_path(...)`
- `secondary_timing_file_path(...)`
- `load_mat(...)`
- `save_mat(...)`

If a new file-handling rule is needed later, it should be added here once and reused everywhere else.

## Implementation Order For The Next Pass

1. Replace local discovery helpers in `ModelTraining.py` with repository discovery.
2. Remove the remaining path-building branches from `SignalAnalysis.py`.
3. Simplify `DataPreparation.py` so both datasets follow the same extraction pattern.
4. Remove direct raw/edited path construction from `ModelValidator.py`.
5. Leave processing-specific branches only where the datasets truly differ.

## Avoid

- Do not reintroduce `os.path.join(... 'secondary' ...)` or `os.path.join(... 'collected' ...)` in consumer modules when the repository already provides the path.
- Do not duplicate `Soggetto{p}` / `P{p}M{m}` naming logic outside the repository.
- Do not move augmentation, plotting, or training logic into `FileRepository.py`.
- Do not branch during file extraction unless the extraction format itself is genuinely different and cannot be expressed through the repository.
- Do not duplicate code where possible, instead use the related scripts from the helper scripts 

## Current Working Principle

The repository should decide what file to open.
The consumer should decide what to do with that file.
