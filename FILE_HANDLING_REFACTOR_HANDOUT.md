# File Handling Refactor Handout

## Current State

The shared file-handling layer has started.

- New shared module: `FileRepository.py`
- Extracted class: `DataRepository`
- First integrated consumer: `LabelSignalData.py`

`DataRepository` is now the preferred place for MAT-file path resolution, discovery, preferred-input selection, and thin MAT IO helpers.

## Hard Rules

- Preserve the current on-disk file structure.
- Keep file handling centralized in the repository layer.
- Prefer labelled files first in all analysis, annotation, validation, and visualisation workflows.
- Use raw files only when a workflow explicitly needs raw input, especially the runtime controller calibration path.
- Missing or corrupted files must always be logged. Never skip silently.
- Keep the repository focused on filesystem and MAT-file concerns only.
- Do not move signal processing, plotting, model training, or GUI logic into the repository layer.

## Current Repository Behavior

`DataRepository` should be treated as the shared source of truth for:

- Raw root path resolution
- Edited root path resolution
- Raw file path generation
- Labelled file path generation
- Preferred input path selection
- Participant discovery
- Movement discovery
- Listing all available files for a participant
- Thin MAT load/save helpers
- Readability checks for MAT files

## Kinematics Rule

Kinematics should live in the same repository module, but the API must stay file-kind aware.

- Secondary movement files remain in the `Soggetto{p}/Movimento{m}` structure.
- Collected movement files remain in the `P{p}M{m}` structure.
- Secondary kinematics will use the same edited-file approach.
- Collected kinematics will stay in the same overall repository layer, but do not assume it uses edited copies in the same way as movement files.

## Already Integrated

### Done
- `LabelSignalData.py` now imports `DataRepository` and `FileSelection` from `FileRepository.py`.
- The annotation GUI is the first script using the shared repository module.

### Not Yet Migrated
- Other scripts still contain their own file-loading and path-building logic.

## Files That Still Need Changes

### High priority
These are the next likely migration targets because they either build paths directly, scan folders directly, or load MAT files independently.

- `DataPreparation.py`
- `SignalAnalysis.py`
- `ModelValidator.py`
- `SignalViewerGUI.py`
- `emg-shoulder-prosthetic-controller.py`
- `matFileReader.py`

### Secondary priority
These are utility or inspection scripts that also touch MAT files and should be brought onto the shared layer when practical.

- `VerifyLabels.py`
- `AlignMyoData.py`
- `inspect_raw_mat_files.py`
- `DataValidationViewer.py`
- `test_kinematic_load.py`
- `visualize_alignment_demo.py`
- `SignalUtils.py`
- `SignalReading.py`
- `SerialValidation.py`

### Possibly affected by API changes
These mostly consume prepared arrays, but may need minor updates if the data-loading API changes.

- `ModelTraining.py`
- `NeuralNetworkModels.py`

## Usage Guidelines For The New Class

Use `DataRepository` whenever code needs to decide where a file lives.

- Use it to build raw and edited paths instead of hardcoding strings.
- Use it to discover participants and movements instead of duplicating directory scans.
- Use it to choose labelled files first, then fall back to raw only when that workflow requires it.
- Use it to keep skip handling consistent when files are missing or corrupted.
- Use it to support both batch workflows and single-file GUI/runtime workflows.

## Migration Strategy

When updating the next file, follow this order:

1. Replace direct path building with `DataRepository`.
2. Replace folder scanning with repository discovery helpers.
3. Replace repeated labelled/raw fallback logic with the repository's preferred-input logic.
4. Keep any workflow-specific processing outside the repository.
5. Log missing or unreadable files consistently.

## What To Avoid

- Do not reintroduce ad hoc `os.listdir` scans where the repository already provides discovery.
- Do not duplicate path conventions in multiple files.
- Do not use the repository for filtering, preprocessing, plotting, or inference.
- Do not change the existing file layout unless the refactor explicitly requires it and is agreed first.

## Notes For Future Context

- The runtime controller is the main exception to the labelled-first rule because it needs raw MVC input for calibration and min-max scaling.
- Everywhere else should prefer the labelled files.
- The current goal is not to redesign file naming. It is to centralize the existing naming rules and keep behavior consistent.
