import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.widgets import CheckButtons, Button

import SignalUtils

# ===============================================================================
# ================================ VISUALISATION ================================
# ===============================================================================

def plot_class_templates_interactive(d, index, classification, dataset_name="Unknown Dataset", range_min=-20, range_max=80, max_traces=500):
    """
    Generates an interactive plot overlaying spike waveforms for each class.
    
    The plot includes toggleable visibility for each class and saves a snapshot to './Detected Peaks/'.

    Class 1: Blue
    Class 2: Orange
    Class 3: Green
    Class 4: Red
    Class 5: Purple

    Args:
        d (np.ndarray): Signal data.
        index (list | np.ndarray): Indices of the spikes
        classification (list | np.ndarray): Class labels of spikes (correpsonds with index)
        dataset_name (Optional[str]): The name of the dataset for the title and filename. Defaults to "Unknown Dataset".
        range_min (Optional[int]): The start of the window relative to the spike index. Defaults to -20.
        range_max (Optional[int]): The end of the window relative to the spike index. Defaults to 80.
        max_traces (Optional[int]): The maximum number of traces to plot per class to preserve performance. Defaults to 500.

    Returns:
        None
    """
    # Generate the name
    base_name = os.path.splitext(dataset_name)[0]

    index = np.array(index)
    classification = np.array(classification)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.2) 
    
    x_axis = np.arange(range_min, range_max)
    plot_lines = {1: [], 2: [], 3: [], 4: [], 5: []}
    colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple'}
    class_counts = {}

    expected_values = {
        "D1": {
            "total": 2176,
            "classes": {1: 458, 2: 441, 3: 406, 4: 444, 5: 427},
            "even": None
        },
        "D2": {
            "total": 3984,
            "classes": {1: 839, 2: 807, 3: 743, 4: 813, 5: 782},
            "even": 796
        },
        "D3": {
            "total": 3329,
            "classes": {1: 701, 2: 675, 3: 621, 4: 679, 5: 653},
            "even": 665
        },
        "D4": {
            "total": 3031,
            "classes": {1: 638, 2: 615, 3: 566, 4: 618, 5: 594},
            "even": 606
        },
        "D5": {
            "total": 2582,
            "classes": {1: 544, 2: 523, 3: 481, 4: 527, 5: 507},
            "even": 516
        },
        "D6": {
            "total": 3911,
            "classes": {1: 824, 2: 793, 3: 729, 4: 798, 5: 767},
            "even": 782
        }
    }
    
    print(f"SignalProcessing: Generating interactive plot for {base_name}...")
    
    # Get the plot indices for each class along with the window
    for class_id in range(1, 6):
        class_indices = index[classification == class_id]
        total_count = len(class_indices)
        class_counts[class_id] = total_count
        
        # Get Indices
        if total_count > max_traces:
            plot_indices = np.random.choice(class_indices, max_traces, replace=False)
        else:
            plot_indices = class_indices
            
        for start_idx in plot_indices:
            window_start = start_idx + range_min
            window_end = start_idx + range_max
            
            # Adjust Window Size for Edge Case
            if window_start >= 0 and window_end < len(d):
                wave = d[window_start:window_end]
                line, = ax.plot(x_axis, wave, color=colors[class_id], alpha=0.15, linewidth=1)
                plot_lines[class_id].append(line)
    
    total_spikes = sum(class_counts.values())
    summary_text = " | ".join([f"C{k}: {v}" for k, v in class_counts.items()])

    # Set Title
    actual_title = (
        f"{base_name}: Class Overlays\n"
        f"Window: {range_min} to +{range_max}\n"
        f"Total: {total_spikes} | {summary_text}"
    )

    # Get the string for the expected values
    expected_key = None
    for key in expected_values.keys():
        if key in base_name:
            expected_key = key
            break

    # Generate string for expected values
    if expected_key is not None:
        exp = expected_values[expected_key]

        expected_summary = " | ".join([f"C{k}: {v}" for k, v in exp["classes"].items()])
        even_text = f" | Even: {exp['even']}" if exp["even"] is not None else ""

        expected_line = (
            f"\nExpected: {exp['total']} | {expected_summary}{even_text}"
        )

        actual_title += expected_line

    # PLOTTING
    ax.set_title(actual_title)
    ax.set_xlabel("Samples relative to Spike Start")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    
    rax = plt.axes([0.02, 0.4, 0.15, 0.25])
    labels = [f'Class {i} ({colors[i]})' for i in range(1, 6)]
    visibility = [True] * 5
    check = CheckButtons(rax, labels, visibility)

    def func(label):
        class_id_str = label.split(' ')[1] 
        class_id = int(class_id_str)
        for line in plot_lines[class_id]:
            line.set_visible(not line.get_visible())
        plt.draw()

    check.on_clicked(func)

    # --- SAVE LOGIC ---
    save_folder = "./Detected Peaks"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Use base_name for file saving as well
    save_name = f"{base_name}_Detected_Peaks_Plot.png"
    save_path = os.path.join(save_folder, save_name)

    plt.savefig(save_path)
    print(f"[PLOT] Saved figure to {save_path}")

    plt.show()

# ======================================================================
# =========================  DEBUG OVERLAY  ============================
# ======================================================================

def debug_compare_signals(signal1, signal2, peak_indices=[], label1="Filtered Signal (Amp)", label2="CNN Probability", limit=50000):
    """
    Plots two signals on a dual-axis chart.

    Args:
        signal1 (np.ndarray): The primary signal.
        signal2 (np.ndarray): The secondary signal.
        peak_indices (Optional[list]): A list of peak indices to mark on the plot. Defaults to [].
        label1 (Optional[str]): Label for the primary Y-axis. Defaults to "Filtered Signal (Amp)".
        label2 (Optional[str]): Label for the secondary Y-axis. Defaults to "CNN Probability".
        limit (Optional[int]): The maximum number of samples to plot. Defaults to 50000.

    Returns:
        None
    """
    end_index = min(len(signal1), limit)
    s1_view = signal1[:end_index]
    s2_view = signal2[:end_index]
    visible_peaks = [idx for idx in peak_indices if idx < end_index]
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # --- Axis 1 ---
    ax1.plot(s1_view, color='cornflowerblue', linewidth=1.0, alpha=0.8, label=label1)
    ax1.set_ylabel(label1, color='cornflowerblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='cornflowerblue')
    ax1.set_xlabel("Sample Index")
    ax1.grid(True, alpha=0.3)
    
    # --- Axis 2---
    ax2 = ax1.twinx() # Share X axis
    ax2.plot(s2_view, color='darkorange', linewidth=1.5, alpha=0.8, label=label2)
    ax2.fill_between(np.arange(len(s2_view)), s2_view, color='orange', alpha=0.1)
    
    ax2.set_ylabel(label2, color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1.1) # Probabilities are 0-1
    
    ax2.axhline(0.6, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # --- Peaks ---
    if len(visible_peaks) > 0:
        y_vals = s2_view[visible_peaks]
        ax2.scatter(visible_peaks, y_vals, color='red', s=60, marker='x', zorder=10, label="Detected Peaks", linewidths=2)

    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.title(f"DEBUG: {label1} vs {label2} Overlay")
    plt.tight_layout()
    plt.show()

# =========================================================================
# ================================ GRADING ================================
# =========================================================================

def plot_debug_spikes(signal, prob_signal, missed_indices, false_positive_indices, matched_gt_indices, all_predicted_indices, window_size=100, max_display=None):
    """
    Launches an interactive tool to step through Missed Spikes (FN) and False Positives (FP) individually.

    Args:
        signal (np.ndarray): Signal to visualise
        prob_signal (np.ndarray): Probability signal.
        missed_indices (list): Indices of GT spikes that were missed.
        false_positive_indices (list): Indices FP spikes.
        matched_gt_indices (list): Indices of ground truth spikes that were correctly found (True Positives).
        all_predicted_indices (list): Indices of spikes predicted by the detector.
        window_size (Optional[int]): The width of the view window around each error. Defaults to 100.
        max_display (Optional[int]): Limit the number of errors to display. Defaults to None (show all).

    Returns:
        None
    """
    
    # Convert lists to numpy arrays for fast window filtering
    gt_array = np.array(matched_gt_indices)
    pred_array = np.array(all_predicted_indices)

    # Helper Function for the interative viewer
    def create_viewer(indices, title_prefix, main_color):
        if len(indices) == 0: 
            print(f"No {title_prefix} to plot.")
            return None

        target_list = indices
        if max_display is not None:
            target_list = indices[:max_display]

        total_spikes = len(target_list)
        spikes_per_page = 1 
        total_pages = total_spikes 
        state = {'page': 0}

        fig, ax_main = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2) 
        ax_prob = ax_main.twinx()

        def draw():
            current_idx = state['page']
            
            ax_main.clear()
            ax_prob.clear()

            spike_idx = target_list[current_idx]
            
            fig.suptitle(f"{title_prefix} - Event {current_idx+1}/{total_spikes} (Index: {spike_idx})", fontsize=14, fontweight='bold')
            
            # Window limits
            start = max(0, int(spike_idx - window_size // 2))
            end = min(len(signal), int(spike_idx + window_size // 2))
            
            # Data slices
            wave_signal = signal[start:end]
            wave_prob = prob_signal[start:end]
            x_axis = np.arange(len(wave_signal))
            
            # --- PLOT 1 ---
            p1 = ax_main.plot(x_axis, wave_signal, color=main_color, label='Raw Signal', linewidth=1.5)
            ax_main.set_ylabel('Signal Amplitude', color=main_color)
            ax_main.grid(True, alpha=0.3)
            
            # --- PLOT 2 ---
            p2 = ax_prob.plot(x_axis, wave_prob, color='navy', linestyle='--', alpha=0.6, label='Probability', linewidth=1.2)
            ax_prob.set_ylabel('Probability', color='navy')
            ax_prob.set_ylim(-0.1, 1.1)
            
            # Center line
            center_x = spike_idx - start
            ax_main.axvline(x=center_x, color='gray', linestyle=':', alpha=0.5)

            # --- PLOT 3 ---
            # Grab GTs
            in_window_gt = (gt_array >= start) & (gt_array < end)
            nearby_gt = gt_array[in_window_gt]
            
            p3 = None
            if len(nearby_gt) > 0:
                rel_gt = nearby_gt - start
                gt_y_vals = signal[nearby_gt]
                p3 = ax_main.scatter(rel_gt, gt_y_vals, color='red', marker='x', s=120, linewidths=2.5, label='True Positive (Real)', zorder=6)

            # --- PLOT 4 ---
            # Grab PREDICTED
            in_window_pred = (pred_array >= start) & (pred_array < end)
            nearby_pred = pred_array[in_window_pred]
            
            p4 = None
            if len(nearby_pred) > 0:
                rel_pred = nearby_pred - start
                pred_y_vals = signal[nearby_pred]
                p4 = ax_main.scatter(rel_pred, pred_y_vals, color='#00CC00', marker='x', s=80, linewidths=2, label='Predicted (Detector)', zorder=5)

            # Legend
            lines = p1 + p2
            if p3: lines.append(p3)
            if p4: lines.append(p4)
            labels = [l.get_label() for l in lines]
            ax_main.legend(lines, labels, loc='upper right')

            fig.canvas.draw_idle()

        # Navigation helpers and logic
        axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next >')
        bprev = Button(axprev, '< Prev')

        def next_callback(event):
            if state['page'] < total_pages - 1:
                state['page'] += 1
                draw()

        def prev_callback(event):
            if state['page'] > 0:
                state['page'] -= 1
                draw()

        bnext.on_clicked(next_callback)
        bprev.on_clicked(prev_callback)
        draw()
        return fig, bnext, bprev

    print(">>> GENERATING INTERACTIVE DEBUG PLOTS <<<")
    viewers = []
    
    if len(missed_indices) > 0:
        viewers.append(create_viewer(missed_indices, "Missed Spike (FN)", '#d62728')) 
        
    if len(false_positive_indices) > 0:
        viewers.append(create_viewer(false_positive_indices, "False Positive (FP)", '#ff7f0e'))
        
    plt.show()

def grade_performance(signal, prob_signal, realIndicies, realClasses, detectedIndices, detectedClasses, tolerance=50, max_debug_plots=None):
    """
    Compares detected spikes against ground truth to calculate accuracy metrics (Precision, Recall, F1).

    Generates a confusion matrix and launches the interactive debug plotter to inspect errors.

    Args:
        signal (np.ndarray): Signal to inspect.
        prob_signal (np.ndarray): The probability signal.
        realIndicies (list): Ground truth spike locations.
        realClasses (list): Ground truth spike classes.
        detectedIndices (list): Predicted spike locations.
        detectedClasses (list): Predicted spike classes.
        tolerance (Optional[int]): The allowed distance (in samples) between a predicted index and a real index to count as a match. Defaults to 50.
        max_debug_plots (Optional[int]): Maximum number of error plots to generate. Defaults to None.

    Returns:
        tuple: A tuple containing lists of errors:
            - falsePositiveIndices (list): Predicted indices that was noise.
            - undetectedIndices (list): Real indices that was missed.
            - misClassifiedIndices (list): Predicted indices that was found but classified wrong.
            - misClassifiedClasses (list): The corrected classes for the misclassified indices.
    """
    print("\n" + "="*60)
    print("AUTOMATED GRADING REPORT")
    print("="*60)

    # Grab indices
    realIndicies = np.array(realIndicies)
    realClasses = np.array(realClasses)
    detectedIndices = np.array(detectedIndices)
    detectedClasses = np.array(detectedClasses)

    if len(realIndicies) > 0:
        _, realIndicies, realClasses = SignalUtils.sortTupleLists(realIndicies, realClasses, index=0)
    if len(detectedIndices) > 0:
        _, detectedIndices, detectedClasses = SignalUtils.sortTupleLists(detectedIndices, detectedClasses, index=0)

    falsePositiveIndices = []      
    undetectedIndices = []         
    misClassifiedIndices = []      
    misClassifiedClasses = []      

    markedIndices = [] # Indices of Real Spikes that were matched
    confusion_matrix = np.zeros((5, 5), dtype=int)

    # Matching Logic
    for i, pred_idx in enumerate(detectedIndices):
        incorrect = True 
        for j, real_idx in enumerate(realIndicies):
            # Check if the indices are within +/- tolerance of real idx
            if (abs(pred_idx - real_idx) <= tolerance) and (real_idx not in markedIndices):     # Ensure that there are no double counted peaks
                incorrect = False 
                markedIndices.append(real_idx) 
                
                pred_c = int(detectedClasses[i])
                real_c = int(realClasses[j])
                
                # Append location of predicted and real on the matrix
                if 1 <= pred_c <= 5 and 1 <= real_c <= 5:
                    confusion_matrix[real_c-1][pred_c-1] += 1
                
                # If they are not the same, add to the misclass list
                if pred_c != real_c:
                    misClassifiedIndices.append(pred_idx)
                    misClassifiedClasses.append(real_c)
                break 
        
        # Add FP if there is no detection within tolerance
        if incorrect:
            falsePositiveIndices.append(pred_idx)

    # Find all FN
    for real_idx in realIndicies:
        if real_idx not in markedIndices:
            undetectedIndices.append(real_idx)

    # Metrics
    TP = len(markedIndices)
    FP = len(falsePositiveIndices)
    FN = len(undetectedIndices)
    
    # Calculate precision, recall, and F1 score
    precision = 0
    if (TP + FP) > 0: precision = TP / (TP + FP)
    recall = 0
    if (TP + FN) > 0: recall = TP / (TP + FN)
    f1_score = 0
    if (precision + recall) > 0: f1_score = 2 * (precision * recall) / (precision + recall)

    # Print log
    print(f"Total Real Spikes:       {len(realIndicies)}")
    print(f"Total Detected Spikes:   {len(detectedIndices)}")
    print("-" * 30)
    print(f"False Positives (Noise): {len(falsePositiveIndices)}")
    print(f"Undetected (Missed):     {len(undetectedIndices)}")
    print(f"Misclassified:           {len(misClassifiedIndices)}")
    
    # Calculate the number of matches and the accuracy
    matches = len(realIndicies) - len(undetectedIndices)
    correct_class = matches - len(misClassifiedIndices)
    
    accuracy = 0
    if len(realIndicies) > 0: accuracy = (correct_class / len(realIndicies)) * 100
        
    # Print log
    print(f"Strict Accuracy:         {accuracy:.2f}%")
    print("-" * 30)
    print(f"Precision:               {precision:.4f}")
    print(f"Recall:                  {recall:.4f}")
    print(f"F1 Score:                {f1_score:.4f}")
    
    print("\nCONFUSION MATRIX (Rows=Real, Cols=Predicted)")
    print("      C1   C2   C3   C4   C5")
    print("    --------------------------")
    for r in range(5):
        row_str = f"C{r+1} | "
        for c in range(5):
            val = confusion_matrix[r][c]
            row_str += f"{val:3d}  "
        print(row_str)
    print("="*60)

    # Plot the debug spikes after for inspection
    plot_debug_spikes(
        signal, 
        prob_signal, 
        undetectedIndices, 
        falsePositiveIndices, 
        markedIndices,    # Matched GT (Red X)
        detectedIndices,  # All Predictions (Green X)
        window_size=100, 
        max_display=max_debug_plots
    )

    return falsePositiveIndices, undetectedIndices, misClassifiedIndices, misClassifiedClasses