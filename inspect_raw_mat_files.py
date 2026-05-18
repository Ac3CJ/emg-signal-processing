"""
inspect_raw_mat_files.py

Diagnostic utility to inspect and understand the structure of all MAT files
in the secondary raw data directory.

Purpose:
- Reveal key names, data types, and shapes for each file type
- Help plan integration of kinematic and timing data
- Document the actual structure for refactoring reference
"""

import os
import scipy.io as spio
import numpy as np
from pathlib import Path
import json
from FileRepository import DataRepository

# Configuration
REPOSITORY = DataRepository()
SOGGETTO_DIR = REPOSITORY.raw_root('secondary')
OUTPUT_REPORT = 'mat_file_structure_report.txt'


def safe_inspect_mat(file_path):
    """
    Safely load and inspect a MAT file, returning structure info.
    
    Returns:
        dict or None: Contains keys, shapes, dtypes, and sample values
    """
    try:
        mat_data = spio.loadmat(file_path)
        
        # Filter out MATLAB metadata keys
        metadata_keys = ['__header__', '__version__', '__globals__']
        data_keys = {k: v for k, v in mat_data.items() if k not in metadata_keys}
        
        result = {
            'file': os.path.basename(file_path),
            'path': file_path,
            'status': 'SUCCESS',
            'keys': list(data_keys.keys()),
            'details': {}
        }
        
        for key, value in data_keys.items():
            if isinstance(value, np.ndarray):
                result['details'][key] = {
                    'type': 'ndarray',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'min': float(np.min(value)) if value.size > 0 else None,
                    'max': float(np.max(value)) if value.size > 0 else None,
                    'mean': float(np.mean(value)) if value.size > 0 else None,
                    'sample_first_5': value.flat[:5].tolist() if value.size >= 5 else value.tolist(),
                }
            else:
                result['details'][key] = {
                    'type': str(type(value).__name__),
                    'value': str(value)[:200]  # Limit string repr
                }
        
        return result
    
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'path': file_path,
            'status': 'ERROR',
            'error': str(e)
        }


def inspect_soggetto(soggetto_num):
    """Inspect all files within a single Soggetto directory."""
    soggetto_path = REPOSITORY.secondary_subject_root(soggetto_num)
    
    if not os.path.exists(soggetto_path):
        print(f"[!] Directory not found: {soggetto_path}")
        return None
    
    files = sorted(os.listdir(soggetto_path))
    
    # Categorize files
    movimento_files = [f for f in files if f.startswith('Movimento') and f.endswith('.mat')]
    angulardata_files = [f for f in files if f.startswith('MovimentoAngS') and f.endswith('.mat')]
    timing_files = [f for f in files if f.startswith('InizioFine') and f.endswith('.mat')]
    
    result = {
        'participant': f'Soggetto{soggetto_num}',
        'directory': soggetto_path,
        'file_count': len(files),
        'categories': {
            'movimento_files': len(movimento_files),
            'angular_files': len(angulardata_files),
            'timing_files': len(timing_files),
        },
        'detailed_inspection': {
            'MOVIMENTO_FILES': [],
            'ANGULAR_FILES': [],
            'TIMING_FILES': []
        }
    }
    
    # Inspect each category
    print(f"\n{'='*100}")
    print(f"Soggetto{soggetto_num}")
    print(f"{'='*100}")
    
    # ========== MOVIMENTO FILES (Primary EMG) ==========
    print(f"\n[1] MOVIMENTO FILES (Primary EMG Data)")
    print(f"{'-'*100}")
    for movimento_file in movimento_files:
        file_path = os.path.join(soggetto_path, movimento_file)
        inspection = safe_inspect_mat(file_path)
        result['detailed_inspection']['MOVIMENTO_FILES'].append(inspection)
        
        if inspection['status'] == 'SUCCESS':
            print(f"\n  {movimento_file}")
            for key, details in inspection['details'].items():
                if details['type'] == 'ndarray':
                    print(f"    Key: '{key}'")
                    print(f"      Type: {details['dtype']}")
                    print(f"      Shape: {details['shape']}")
                    print(f"      Range: [{details['min']:.2f}, {details['max']:.2f}]")
                    print(f"      Mean: {details['mean']:.2f}")
                else:
                    print(f"    Key: '{key}' (Type: {details['type']})")
        else:
            print(f"  {movimento_file} -> ERROR: {inspection['error']}")
    
    # ========== ANGULAR DATA FILES (Kinematics/Ground Truth) ==========
    print(f"\n[2] ANGULAR DATA FILES (Kinematics/Ground Truth)")
    print(f"{'-'*100}")
    for angular_file in angulardata_files:
        file_path = os.path.join(soggetto_path, angular_file)
        inspection = safe_inspect_mat(file_path)
        result['detailed_inspection']['ANGULAR_FILES'].append(inspection)
        
        if inspection['status'] == 'SUCCESS':
            print(f"\n  {angular_file}")
            for key, details in inspection['details'].items():
                if details['type'] == 'ndarray':
                    print(f"    Key: '{key}'")
                    print(f"      Type: {details['dtype']}")
                    print(f"      Shape: {details['shape']}")
                    print(f"      Range: [{details['min']:.2f}, {details['max']:.2f}]")
                    print(f"      Mean: {details['mean']:.2f}")
                    if len(details['sample_first_5']) <= 5:
                        print(f"      Values: {details['sample_first_5']}")
                else:
                    print(f"    Key: '{key}' (Type: {details['type']})")
                    print(f"      Value: {details['value']}")
        else:
            print(f"  {angular_file} -> ERROR: {inspection['error']}")
    
    # ========== TIMING BOUNDARY FILES (Start/End Markers) ==========
    print(f"\n[3] TIMING BOUNDARY FILES (Start/End Markers)")
    print(f"{'-'*100}")
    for timing_file in timing_files:
        file_path = os.path.join(soggetto_path, timing_file)
        inspection = safe_inspect_mat(file_path)
        result['detailed_inspection']['TIMING_FILES'].append(inspection)
        
        if inspection['status'] == 'SUCCESS':
            print(f"\n  {timing_file}")
            for key, details in inspection['details'].items():
                if details['type'] == 'ndarray':
                    print(f"    Key: '{key}'")
                    print(f"      Type: {details['dtype']}")
                    print(f"      Shape: {details['shape']}")
                    print(f"      Range: [{details['min']}, {details['max']}]")
                    print(f"      Values (full): {details['sample_first_5']}")
                else:
                    print(f"    Key: '{key}' (Type: {details['type']})")
                    print(f"      Value: {details['value']}")
        else:
            print(f"  {timing_file} -> ERROR: {inspection['error']}")
    
    return result


def compare_file_structures():
    """Compare structures across multiple subjects to identify patterns."""
    print(f"\n\n{'='*100}")
    print("CROSS-SUBJECT COMPARISON")
    print(f"{'='*100}")
    
    subjects = REPOSITORY.discover_participants('secondary')
    
    for subject_num in subjects:
        soggetto_path = REPOSITORY.secondary_subject_root(subject_num)
        if not os.path.exists(soggetto_path):
            continue
        
        # Sample one file from each category
        movimento_file = os.path.join(soggetto_path, 'Movimento1.mat')
        angular_file = os.path.join(soggetto_path, 'MovimentoAngS1.mat')
        timing_file = os.path.join(soggetto_path, 'InizioFineSteady11.mat')
        
        print(f"\nSoggetto{subject_num}:")
        
        # Check Movimento file
        if os.path.exists(movimento_file):
            inspection = safe_inspect_mat(movimento_file)
            if inspection['status'] == 'SUCCESS':
                emg_shape = inspection['details'].get('EMGDATA', {}).get('shape', 'N/A')
                print(f"  Movimento1.mat -> EMG shape: {emg_shape}")
        
        # Check Angular file
        if os.path.exists(angular_file):
            inspection = safe_inspect_mat(angular_file)
            if inspection['status'] == 'SUCCESS':
                keys_found = list(inspection['details'].keys())
                print(f"  MovimentoAngS1.mat -> Keys: {keys_found}")
        
        # Check Timing file
        if os.path.exists(timing_file):
            inspection = safe_inspect_mat(timing_file)
            if inspection['status'] == 'SUCCESS':
                keys_found = list(inspection['details'].keys())
                print(f"  InizioFineSteady11.mat -> Keys: {keys_found}")


def generate_documentation():
    """Generate human-readable documentation."""
    print(f"\n\n{'='*100}")
    print("DOCUMENTATION SUMMARY")
    print(f"{'='*100}\n")
    
    doc = """
### COMPLETE MAT FILE REFERENCE

#### 1. MOVIMENTO[1-9].mat (PRIMARY EMG DATA)
**Purpose:** Raw continuous EMG recordings for each movement
**Participants:** Soggetto1-8 (8 subjects × 9 movements = 72 files)
**Size:** ~1.5-2 MB per file
**Key Name(s):** EMGDATA

**Structure:**
  - Type: 2D NumPy/MATLAB matrix
  - Shape: [8, N_SAMPLES]
  - Dtype: float64
  - Sampling Rate: 1000 Hz
  - Duration: Varies (~10-20 seconds typically)

**Channel Mapping (Row Index):**
  - Row 0: Pectoralis Major (Clavicular)
  - Row 1: Pectoralis Major (Sternal)
  - Row 2: Serratus Anterior
  - Row 3: Trapezius (Descendent)
  - Row 4: Trapezius (Transversalis)
  - Row 5: Trapezius (Ascendant)
  - Row 6: Infraspinatus
  - Row 7: Latissimus Dorsi

**Usage in Pipeline:**
  - Load raw EMG → Apply filtering → Extract windows based on timing files
  - Represents the FULL recording session (includes ramp-up, steady-state, ramp-down)


#### 2. MOVIMENTOANGS[1-9].mat (KINEMATIC/ANGULAR GROUND TRUTH)
**Purpose:** Target/reference joint angles and kinematics synchronized with EMG
**Participants:** Soggetto1-8 (8 subjects × 9 movements = 72 files)
**Size:** ~50-150 KB per file
**Key Name(s):** ???  <- [YOU NEED TO INSPECT: Run this script to find exact names]

**Structure (HYPOTHESIS - Verify with inspection):**
  - Type: 2D matrix (likely time-series)
  - Shape: [N_SAMPLES, 4] or [4, N_SAMPLES] (one column per DOF: Yaw, Pitch, Roll, Elbow)
  - Dtype: float64
  - Synchronization: Same time base as Movimento file OR downsampled

**Expected Content:**
  - Yaw (Flexion/Extension)
  - Pitch (Abduction/Adduction)
  - Roll (Internal/External Rotation)
  - Elbow (Flexion)

**Usage in Pipeline:**
  - Provides GROUND TRUTH kinematics across the full movement
  - Should be paired 1:1 with Movimento file for regression targets
  - May need resampling/interpolation if sampling rates differ


#### 3. INIZIOFINSTEADY[1-9][1-8].mat (STEADY-STATE WINDOW MARKERS)
**Purpose:** Start and end frame indices marking the GOOD steady-state phase
**Naming Convention:**
  - InizioFineSteady11.mat → Steady state for Soggetto1, Movimento1
  - InizioFineSteady12.mat → Steady state for Soggetto1, Movimento2
  - InizioFineSteady21.mat → Steady state for Soggetto2, Movimento1
  - Pattern: InizioFineSteady[SOGGETTO_NUM][MOVEMENT_NUM].mat

**Naming:** 
  - "Inizio" = Start/Beginning (Italian)
  - "Fine" = End (Italian)
  - "Steady" = Steady-state phase

**Size:** ~1 KB (metadata only)
**Key Name(s):** ??? <- [YOU NEED TO INSPECT: likely scalar or 1D array of length 2]

**Structure (HYPOTHESIS - Verify):**
  - Type: 1D array or two scalars
  - Contents: [START_FRAME_INDEX, END_FRAME_INDEX]
  - Values: Frame numbers (samples) into the Movimento file at 1000 Hz
  - Example: [5000, 15000] means samples 5000-15000 are the actual movement

**Usage in Pipeline:**
  - Extract EMG[..., START:END] from Movimento file
  - Extract ANGLES[START:END] from MovimentoAngS file
  - This is the "clean" window → exclude ramp-up and settling
  - Used by: extract_bursts_from_labels() in DataPreparation.py


#### 4. INIZIOFINEREST[1-9][2].mat (REST WINDOW MARKERS)
**Purpose:** Start and end indices for the rest/baseline phase
**Naming Note:**
  - InizioFineRest12.mat → Rest for Soggetto1
  - InizioFineRest22.mat → Rest for Soggetto2
  - Pattern: InizioFineRest[SOGGETTO_NUM][2].mat (always ends in [num]2?)

**Size:** ~1 KB
**Key Name(s):** ??? <- [YOU NEED TO INSPECT]

**Structure (HYPOTHESIS - Verify):**
  - Type: 1D array or two scalars
  - Contents: [START_FRAME_INDEX, END_FRAME_INDEX]
  - Values: Frame numbers marking the rest window

**Usage in Pipeline:**
  - Extract rest EMG windows from Movimento9.mat
  - Used for validation/background noise estimation
  - Target angles for rest: [0°, 0°, 0°, 0°] (fixed in TARGET_MAPPING[9])


### KEY QUESTIONS TO ANSWER (Run this script to find out):

1. **MovimentoAngS files:**
   - What is the exact key name? (ANGLES? DATA? KINEMATICS? ANG?)
   - What is the exact shape? [N, 4] or [4, N]?
   - Do they have the same number of samples as Movimento files?
   - Are they at the same sampling rate (1000 Hz)?

2. **Timing files:**
   - What is the exact key name? (START_END? INIZIO_FINE? TIMING?)
   - Are they scalars or 1D arrays? If array, length?
   - Are indices 0-based or 1-based?
   - What happens if timing spans beyond the Movimento file length?

3. **Integration Opportunities:**
   - Can you load and align all three file types directly in DataPreparation.py?
   - Should timing files be cached or loaded on-demand?
   - Pattern for naming: Can parametric indexing replace hardcoded loops?


### REFACTORING INSIGHTS:

**Current Code Issues:**
- MovimentoAngS files are NOT currently integrated (sitting unused)
- Timing files might be loaded but not explicitly validated
- File naming is hardcoded in loops (e.g., range(1, 10) for movements)

**Refactoring Opportunities:**
1. Create a unified loader: load_trial_triplet(soggetto, movimento) → (emg, angles, timing)
2. Validate alignment: Check that EMG and ANGLES have compatible shapes
3. Parametric file discovery: Scan directory to detect available movements/subjects
4. Lazy loading: Don't load all files at once; load on-demand per trial

"""
    
    print(doc)
    return doc


if __name__ == "__main__":
    print("=" * 100)
    print("MAT FILE STRUCTURE INSPECTOR")
    print("=" * 100)
    print(f"\nTarget Directory: {SOGGETTO_DIR}")
    print("This script will inspect all MAT files and reveal their internal structure.\n")
    
    # Inspect first 3 subjects for baseline understanding
    for soggetto_num in [1, 2, 3]:
        result = inspect_soggetto(soggetto_num)
        if result is None:
            print(f"[!] Could not inspect Soggetto{soggetto_num}")
    
    # Cross-subject comparison
    compare_file_structures()
    
    # Generate documentation
    doc = generate_documentation()
    
    # Save to file
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(doc)
    
    print(f"\n\n[✓] Report saved to: {OUTPUT_REPORT}\n")
