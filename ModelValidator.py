import numpy as np
import torch
import argparse

import SignalProcessing
from ModelTraining import ShoulderRCNN 
import ControllerConfiguration as Config

def get_predictions_for_file(model, device, file_path):
    import scipy.io
    mat = scipy.io.loadmat(file_path)
    raw_data = mat['EMGDATA']
    num_samples = raw_data.shape[1]

    # Create a real-time normalizer (mimics the live controller's persistent state)
    live_normalizer = SignalProcessing.RealTimeGlobalNormalizer(
        num_channels=Config.NUM_CHANNELS,
        initial_max=150.0,
        spike_threshold=3.5
    )

    predictions = []
    smoothed_pred = np.zeros(Config.NUM_OUTPUTS)
    window_starts = list(range(0, num_samples - Config.WINDOW_SIZE + 1, Config.INCREMENT))

    with torch.no_grad():
        for start in window_starts:
            # Extract the raw window
            window_raw = raw_data[:, start:start+Config.WINDOW_SIZE]
            
            # ===== SAME PREPROCESSING PIPELINE AS LIVE CONTROLLER =====
            # 1. Apply standard sEMG processing (notch, bandpass, rectification)
            cleaned_window = np.zeros_like(window_raw, dtype=np.float32)
            for c in range(Config.NUM_CHANNELS):
                cleaned_window[c, :] = SignalProcessing.applyStandardSEMGProcessing(
                    window_raw[c, :], fs=Config.FS
                )
            
            # 2. Normalize using real-time normalizer with persistent state (CRITICAL: same as live controller)
            normalized_window = live_normalizer.normalize_window(cleaned_window)
            
            # Make prediction
            window_tensor = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(window_tensor).cpu().numpy()[0]
            smoothed_pred = (Config.SMOOTHING_ALPHA * pred) + ((1.0 - Config.SMOOTHING_ALPHA) * smoothed_pred)
            predictions.append(smoothed_pred.copy())

    return np.array(predictions), np.array(window_starts) / Config.FS

def run_collected_validation(model_path, participant_num, base_path='./collected_data/edit'):
    """
    Validates all movements (M1-M9) for a specific participant using collected data.
    Files are named P{p}M{m}.mat in ./collected_data directory.
    """
    import matplotlib.pyplot as plt
    import os
    
    output_dir = 'kinematic-plots'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Collected Data Validation] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"[Collected Data Validation] Processing Participant P{participant_num}...")
    
    for m in range(1, 10):
        file_path = os.path.join(base_path, f'P{participant_num}M{m}_edit.mat')
        
        if not os.path.exists(file_path):
            print(f"  -> Skipping P{participant_num}M{m}: File not found at {file_path}")
            continue

        trial_name = f"P{participant_num}M{m}-kinematics"
        
        print(f"  -> Analyzing {trial_name}...")
        predictions, time_axis = get_predictions_for_file(model, device, file_path)

        plt.figure(figsize=(18, 5))
        plt.plot(time_axis, predictions[:, 0], label='Yaw (Flex/Ext)', color='tab:blue', linewidth=2)
        plt.plot(time_axis, predictions[:, 1], label='Pitch (Abd/Add)', color='tab:orange', linewidth=2)
        plt.plot(time_axis, predictions[:, 2], label='Roll (Int/Ext Rot)', color='tab:green', linewidth=2)
        plt.plot(time_axis, predictions[:, 3], label='Elbow (Flex)', color='tab:red', linewidth=2)
        
        # --- DYNAMIC Y-AXIS LIMITS ---
        y_min = np.min(predictions) - 10
        y_max = np.max(predictions) + 10
        plt.ylim(y_min, y_max)
        
        plt.title(trial_name, fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Predicted Angle (Degrees)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"{trial_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"\n[Collected Data Validation] Success! Plots saved to ./{output_dir}/")

def run_fast_validation(model_path, sim_file=None, predefined=False, base_path='./secondary_data'):
    import matplotlib.pyplot as plt
    import os
    import re
    
    output_dir = 'kinematic-plots'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Fast Validation] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if predefined:
        print("[Fast Validation] Running predefined benchmark suite...")
        # test_cases = [(6, 1), (7, 2), (2, 3), (6, 4), (7, 5), (5, 6), (6, 7), (8, 8)]
        test_cases = [(8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9)]
        files_to_process = [os.path.join(base_path, f'Soggetto{p}', f'Movimento{m}.mat') for p, m in test_cases]
    else:
        if not sim_file:
            print("ERROR: No simulation file provided for validation.")
            return
        files_to_process = [sim_file]

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"  -> Skipping: File not found at {file_path}")
            continue

        match = re.search(r'Soggetto(\d+).*Movimento(\d+)', file_path.replace('\\', '/'))
        if match:
            subject_id = match.group(1)
            m = match.group(2)
            trial_name = f"{m}-kinematics-P{subject_id}M{m}"
        else:
            trial_name = "Unknown_Trial"

        print(f"  -> Analyzing {trial_name}...")
        predictions, time_axis = get_predictions_for_file(model, device, file_path)

        plt.figure(figsize=(18, 5))
        plt.plot(time_axis, predictions[:, 0], label='Yaw (Flex/Ext)', color='tab:blue', linewidth=2)
        plt.plot(time_axis, predictions[:, 1], label='Pitch (Abd/Add)', color='tab:orange', linewidth=2)
        plt.plot(time_axis, predictions[:, 2], label='Roll (Int/Ext Rot)', color='tab:green', linewidth=2)
        plt.plot(time_axis, predictions[:, 3], label='Elbow (Flex)', color='tab:red', linewidth=2)
        
        # --- DYNAMIC Y-AXIS LIMITS ---
        y_min = np.min(predictions) - 10
        y_max = np.max(predictions) + 10
        plt.ylim(y_min, y_max)
        
        plt.title(trial_name, fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Predicted Angle (Degrees)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"{trial_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"\n[Fast Validation] Success! Plots saved to ./{output_dir}/")

def run_ensemble_validation(model_path, base_path='./secondary_data'):
    """
    Overlays the FULL prediction timeline of all valid participants for each movement,
    calculating the median response across the entire trial.
    """
    import matplotlib.pyplot as plt
    import os
    
    output_dir = 'ensemble-plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[Ensemble Validation] Loading Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShoulderRCNN(num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for m in range(1, 10):
        print(f"Aggregating full trials for Movement {m}...")
        all_subject_preds = []
        all_time_axes = []
        
        target_vec = Config.TARGET_MAPPING[m]
        
        for p in range(1, 9):
            if hasattr(Config, 'CORRUPTED_TRIALS') and (p, m) in Config.CORRUPTED_TRIALS:
                continue
                
            file_path = os.path.join(base_path, f'Soggetto{p}', f'Movimento{m}.mat')
            if not os.path.exists(file_path): continue
            
            predictions, time_axis = get_predictions_for_file(model, device, file_path)
            all_subject_preds.append(predictions)
            all_time_axes.append(time_axis)
                    
        if len(all_subject_preds) == 0:
            print(f"  -> Skipping M{m}: No valid trials found.")
            continue
            
        # Truncate all trials to the length of the shortest trial so they can be stacked
        min_len = min(len(p) for p in all_subject_preds)
        stacked_trials = np.array([p[:min_len] for p in all_subject_preds])
        
        # Calculate the median across subjects for every time step
        median_trial = np.median(stacked_trials, axis=0)
        time_axis = all_time_axes[0][:min_len]
        
        plt.figure(figsize=(18, 5)) # Wider figure to fit the full 100+ seconds
        plt.plot(time_axis, median_trial[:, 0], label='Yaw (Flex/Ext)', color='tab:blue', linewidth=2)
        plt.plot(time_axis, median_trial[:, 1], label='Pitch (Abd/Add)', color='tab:orange', linewidth=2)
        plt.plot(time_axis, median_trial[:, 2], label='Roll (Int/Ext Rot)', color='tab:green', linewidth=2)
        plt.plot(time_axis, median_trial[:, 3], label='Elbow (Flex)', color='tab:red', linewidth=2)
        
        # --- DYNAMIC Y-AXIS LIMITS ---
        y_min = np.min(median_trial) - 10
        y_max = np.max(median_trial) + 10
        plt.ylim(y_min, y_max)
        
        plt.title(f"Full-Sequence Ensemble Average: Movement {m} (Target: {target_vec})\nMedian of {len(stacked_trials)} valid subjects across entire timeline", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Predicted Angle (Degrees)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"M{m}-Full-Ensemble.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f"\n[Ensemble Validation] Success! Plots saved to ./{output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time EMG Prosthetic Controller")
    parser.add_argument('--simulate', action='store_true', help='Generate dummy sEMG data instead of reading hardware')
    parser.add_argument('--validate', action='store_true', help='Fast-forward process a single sim_file')
    parser.add_argument('--validate_predefined', action='store_true', help='Run the entire benchmark suite of 8 files')
    parser.add_argument('--validate_ensemble', action='store_true', help='Generate aligned median average plots across all participants')
    parser.add_argument('--collected', type=int, help='Validate all movements for a participant using collected data (e.g., --collected 1 validates P1M1-P1M9)')
    parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH, help='Path to the trained PyTorch weights')
    parser.add_argument('--sim_file', type=str, default='./secondary_data/Soggetto1/Movimento3.mat', help='Specific .mat file to stream')
    
    args = parser.parse_args()
    
    print("\n[Model Validator] Starting validation with the following settings:")
    for arg in vars(args):
        print(f"  -> {arg}: {getattr(args, arg)}")

    if args.validate_ensemble:
        run_ensemble_validation(model_path=args.model, base_path=Config.BASE_DATA_PATH)
    if args.validate_predefined:
        run_fast_validation(model_path=args.model, predefined=True, base_path=Config.BASE_DATA_PATH)
    if args.validate:
        run_fast_validation(model_path=args.model, sim_file=args.sim_file, predefined=False)
    if args.collected is not None:
        run_collected_validation(model_path=args.model, participant_num=args.collected)