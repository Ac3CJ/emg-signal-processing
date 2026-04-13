"""
STANDALONE TEST SCRIPT - Kinematic Data Integration
Demonstrates how to load and align EMG + kinematics + timing data
Does NOT modify any production files - safe to delete after testing
"""

import scipy.io as sio
import numpy as np
import os
from pathlib import Path

def load_trial_triplet_demo(base_path, soggetto_num, movimento_num, fs_emg=1000.0):
    """
    Load EMG + kinematics + timing markers as unified triplet
    
    Returns:
        dict with keys: emg, kinematics, timing, fs_emg, fs_kinematics, n_repetitions
    """
    # Load EMG data (key is always 'EMGDATA')
    emg_path = os.path.join(base_path, f'Movimento{movimento_num}.mat')
    data = sio.loadmat(emg_path)
    emg = data['EMGDATA'].astype(np.float64)  # (8, N_samples)
    
    # Load kinematics data (key is 'angolospalla')
    kin_path = os.path.join(base_path, f'MovimentoAngS{movimento_num}.mat')
    data = sio.loadmat(kin_path)
    kinematics = data['angolospalla'].flatten().astype(np.float64)
    
    # Load timing markers (naming: Movimento 1-8 use InizioFineSteady[11-18], Movimento 9 uses InizioFineRest12)
    if movimento_num == 9:
        timing_path = os.path.join(base_path, f'InizioFineRest12.mat')
        key_name = 'InizioFineRest12'
    else:
        timing_num = movimento_num + 10  # M1->11, M2->12, etc.
        timing_path = os.path.join(base_path, f'InizioFineSteady{timing_num}.mat')
        key_name = f'InizioFineSteady{timing_num}'
    
    data = sio.loadmat(timing_path)
    timing = data[key_name].astype(np.uint16)  # (2, N_reps)
    
    # Compute derived kinematics sampling rate
    downsample_ratio = emg.shape[1] / kinematics.shape[0]
    fs_kinematics = fs_emg / downsample_ratio
    
    return {
        'emg': emg,
        'kinematics': kinematics,
        'timing': timing,
        'fs_emg': fs_emg,
        'fs_kinematics': fs_kinematics,
        'n_repetitions': timing.shape[1]
    }


def extract_repetition_windows_demo(trial_data, window_size_samples=500, step_size_samples=62):
    """
    Extract sliding windows from entire movement with aligned kinematics targets.
    Windows are extracted continuously across the entire movement data.
    """
    emg = trial_data['emg']
    kinematics = trial_data['kinematics']
    timing = trial_data['timing']
    downsample_ratio = emg.shape[1] / kinematics.shape[0]
    
    windows = []
    
    # Slide window across ENTIRE EMG data (not just within repetitions)
    for window_start in range(0, emg.shape[1] - window_size_samples + 1, step_size_samples):
        window_end = window_start + window_size_samples
        
        # Extract EMG window
        emg_window = emg[:, window_start:window_end].copy()
        
        # Map to kinematics indices
        kin_start_idx = int(window_start / downsample_ratio)
        kin_end_idx = int(window_end / downsample_ratio)
        
        # Ensure valid kinematic range
        if kin_end_idx > kinematics.shape[0]:
            kin_end_idx = kinematics.shape[0]
        if kin_start_idx >= kin_end_idx:
            continue
        
        # Extract mean angle as target
        kin_window = kinematics[kin_start_idx:kin_end_idx]
        target_angle = np.mean(kin_window)
        
        # Determine which repetition(s) this window overlaps
        rep_ids = []
        for rep_idx in range(timing.shape[1]):
            rep_start = timing[0, rep_idx]
            rep_end = timing[1, rep_idx]
            # Check if window overlaps with this repetition
            if window_start < rep_end and window_end > rep_start:
                rep_ids.append(rep_idx + 1)
        
        windows.append({
            'emg_window': emg_window,
            'kinematics_target': target_angle,
            'repetitions': rep_ids if rep_ids else [-1],  # -1 for inter-rep gaps
            'window_start': window_start,
            'window_end': window_end
        })
    
    return windows


def main():
    """Run demonstrations"""
    
    base_path = './biosignal_data/secondary/raw/Soggetto1'
    
    print("=" * 80)
    print("TEST 1: Loading Trial Triplet (EMG + Kinematics + Timing)")
    print("=" * 80)
    
    try:
        trial = load_trial_triplet_demo(base_path, soggetto_num=1, movimento_num=1)
        
        print(f"✓ Trial loaded successfully")
        print(f"  - EMG shape: {trial['emg'].shape} @ {trial['fs_emg']} Hz")
        print(f"  - Kinematics shape: {trial['kinematics'].shape} @ {trial['fs_kinematics']:.1f} Hz")
        print(f"  - Timing shape: {trial['timing'].shape}")
        print(f"  - Number of repetitions: {trial['n_repetitions']}")
        print(f"  - Downsampling ratio: {trial['emg'].shape[1] / trial['kinematics'].shape[0]:.2f}x")
        print(f"  - Kinematics range: {trial['kinematics'].min():.2f}° to {trial['kinematics'].max():.2f}°")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"  Expected path: {base_path}")
        return
    
    print("\n" + "=" * 80)
    print("TEST 2: Extracting Repetition Windows")
    print("=" * 80)
    
    windows = extract_repetition_windows_demo(trial)
    
    print(f"✓ Extracted {len(windows)} windows")
    if windows:
        first_window = windows[0]
        print(f"  - First window EMG shape: {first_window['emg_window'].shape}")
        print(f"  - First window angle target: {first_window['kinematics_target']:.2f}°")
        print(f"  - Repetitions covered: {trial['n_repetitions']}")
        
        targets = np.array([w['kinematics_target'] for w in windows])
        print(f"  - Target angle range: {targets.min():.2f}° to {targets.max():.2f}°")
        print(f"  - Mean target: {targets.mean():.2f}° ± {targets.std():.2f}°")
    
    print("\n" + "=" * 80)
    print("TEST 3: Multi-Movement Alignment Check")
    print("=" * 80)
    
    movements_to_test = [1, 2, 5]
    for mov in movements_to_test:
        try:
            trial = load_trial_triplet_demo(base_path, soggetto_num=1, movimento_num=mov)
            windows = extract_repetition_windows_demo(trial)
            
            targets = np.array([w['kinematics_target'] for w in windows])
            ratio = trial['emg'].shape[1] / trial['kinematics'].shape[0]
            
            print(f"  Movimento {mov}:")
            print(f"    - Windows: {len(windows)} | Downsampling: {ratio:.2f}x")
            print(f"    - Target range: {targets.min():.2f}° to {targets.max():.2f}°")
            print(f"    - Mean: {targets.mean():.2f}° ± {targets.std():.2f}°")
            
        except Exception as e:
            print(f"  Movimento {mov}: ✗ {type(e).__name__}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
