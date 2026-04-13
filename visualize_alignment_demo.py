"""
STANDALONE VISUALIZATION - Kinematic Data Alignment
Shows how EMG, kinematics, and timing markers align in a shared time domain.
Does NOT modify any production files - safe to delete after testing.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def load_trial_data(base_path, movimento_num):
    """Load EMG, kinematics, and timing data for a single movement"""
    
    # Load EMG (key is 'EMGDATA')
    emg_path = os.path.join(base_path, f'Movimento{movimento_num}.mat')
    data = sio.loadmat(emg_path)
    emg = data['EMGDATA'].astype(np.float64)
    
    # Load kinematics (key is 'angolospalla')
    kin_path = os.path.join(base_path, f'MovimentoAngS{movimento_num}.mat')
    data = sio.loadmat(kin_path)
    kinematics = data['angolospalla'].flatten().astype(np.float64)
    
    # Load timing (naming: M1-8 use InizioFineSteady[11-18], M9 uses InizioFineRest12)
    if movimento_num == 9:
        timing_path = os.path.join(base_path, f'InizioFineRest12.mat')
        key_name = 'InizioFineRest12'
    else:
        timing_num = movimento_num + 10  # M1->11, M2->12, etc.
        timing_path = os.path.join(base_path, f'InizioFineSteady{timing_num}.mat')
        key_name = f'InizioFineSteady{timing_num}'
    
    data = sio.loadmat(timing_path)
    timing = data[key_name].astype(np.uint16)
    
    return emg, kinematics, timing


def _normalize(signal):
    """Normalize a 1D signal to [0, 1] for overlay plotting."""
    s_min = float(np.min(signal))
    s_max = float(np.max(signal))
    if np.isclose(s_max, s_min):
        return np.zeros_like(signal, dtype=np.float64)
    return (signal - s_min) / (s_max - s_min)


def build_common_time_domain(emg, kinematics, timing, fs_emg=1000.0):
    """Build a shared time domain and alignment diagnostics."""
    n_emg = emg.shape[1]
    n_kin = kinematics.shape[0]

    duration_emg_sec = n_emg / fs_emg
    fs_kinematics = n_kin / duration_emg_sec
    downsample_ratio = fs_emg / fs_kinematics

    time_emg = np.arange(n_emg, dtype=np.float64) / fs_emg
    time_kin = np.arange(n_kin, dtype=np.float64) / fs_kinematics

    # Project kinematics onto EMG time for exact shared x-axis plots.
    kinematics_on_emg_time = np.interp(time_emg, time_kin, kinematics)

    # EMG envelope (100 ms moving average on rectified signal)
    emg_ch0 = emg[0, :]
    rectified = np.abs(emg_ch0)
    env_window = max(1, int(round(0.1 * fs_emg)))
    kernel = np.ones(env_window, dtype=np.float64) / env_window
    emg_envelope = np.convolve(rectified, kernel, mode='same')

    # Timing diagnostics projected into kinematics timeline.
    start_sec = timing[0, :].astype(np.float64) / fs_emg
    end_sec = timing[1, :].astype(np.float64) / fs_emg
    kin_start_idx = np.clip(np.round(start_sec * fs_kinematics).astype(int), 0, n_kin - 1)
    kin_end_idx = np.clip(np.round(end_sec * fs_kinematics).astype(int), 0, n_kin - 1)
    start_sec_back = kin_start_idx / fs_kinematics
    end_sec_back = kin_end_idx / fs_kinematics
    start_residual_ms = (start_sec_back - start_sec) * 1000.0
    end_residual_ms = (end_sec_back - end_sec) * 1000.0

    diagnostics = {
        'fs_kinematics': fs_kinematics,
        'downsample_ratio': downsample_ratio,
        'duration_emg_sec': duration_emg_sec,
        'duration_kin_sec': n_kin / fs_kinematics,
        'duration_delta_ms': (duration_emg_sec - (n_kin / fs_kinematics)) * 1000.0,
        'mean_start_residual_ms': float(np.mean(np.abs(start_residual_ms))),
        'mean_end_residual_ms': float(np.mean(np.abs(end_residual_ms))),
        'max_start_residual_ms': float(np.max(np.abs(start_residual_ms))),
        'max_end_residual_ms': float(np.max(np.abs(end_residual_ms))),
        'rep_durations_sec': (timing[1, :] - timing[0, :]).astype(np.float64) / fs_emg,
        'rep_gaps_sec': (timing[0, 1:] - timing[1, :-1]).astype(np.float64) / fs_emg if timing.shape[1] > 1 else np.array([]),
    }

    return {
        'time_emg': time_emg,
        'time_kin': time_kin,
        'kinematics_on_emg_time': kinematics_on_emg_time,
        'emg_ch0': emg_ch0,
        'emg_envelope': emg_envelope,
        'diagnostics': diagnostics,
        'kin_start_idx': kin_start_idx,
        'kin_end_idx': kin_end_idx,
    }


def visualize_full_alignment(base_path, movimento_num=1, fs_emg=1000.0, subject_label='Soggetto1', out_dir='.'):
    """Create alignment visualizations with explicit shared time-domain projection."""
    
    try:
        emg, kinematics, timing = load_trial_data(base_path, movimento_num)
    except Exception as e:
        print(f"  Could not load Movimento {movimento_num}: {e}")
        return
    
    alignment = build_common_time_domain(emg, kinematics, timing, fs_emg=fs_emg)
    time_emg = alignment['time_emg']
    time_kin = alignment['time_kin']
    emg_ch0 = alignment['emg_ch0']
    emg_envelope = alignment['emg_envelope']
    kin_on_emg = alignment['kinematics_on_emg_time']
    diag = alignment['diagnostics']

    # --- FIGURE 1: Full Alignment in Shared Time Domain ---
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=False)
    fig.suptitle(
        f'{subject_label} Movimento{movimento_num} - Time-Domain Aligned View\n'
        f'EMG @ {fs_emg:.1f} Hz, Kinematics @ {diag["fs_kinematics"]:.2f} Hz, Ratio: {diag["downsample_ratio"]:.2f}x',
        fontsize=12,
        fontweight='bold',
    )

    # Panel 1: raw EMG
    axes[0].plot(time_emg, emg_ch0, color='tab:blue', linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel('EMG Ch0', fontweight='bold')
    axes[0].set_xlim([time_emg[0], time_emg[-1]])
    axes[0].grid(True, alpha=0.3)

    # Panel 2: native kinematics
    axes[1].plot(time_kin, kinematics, color='tab:green', linewidth=1.0, label='Native kinematics')
    axes[1].set_ylabel('Angle (deg)', fontweight='bold')
    axes[1].set_xlim([time_emg[0], time_emg[-1]])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)

    # Panel 3: resampled kinematics on EMG timeline
    axes[2].plot(time_emg, kin_on_emg, color='tab:olive', linewidth=0.9, label='Kinematics projected to EMG time')
    axes[2].set_ylabel('Angle (deg)', fontweight='bold')
    axes[2].set_xlim([time_emg[0], time_emg[-1]])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=8)

    # Panel 4: normalized overlay for shape comparison
    emg_env_norm = _normalize(emg_envelope)
    kin_emg_norm = _normalize(kin_on_emg)
    axes[3].plot(time_emg, emg_env_norm, color='tab:purple', linewidth=0.9, label='EMG envelope (norm)')
    axes[3].plot(time_emg, kin_emg_norm, color='tab:red', linewidth=1.0, alpha=0.9, label='Kinematics on EMG time (norm)')
    axes[3].set_ylabel('Normalized', fontweight='bold')
    axes[3].set_xlim([time_emg[0], time_emg[-1]])
    axes[3].set_ylim([-0.05, 1.05])
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right', fontsize=8)

    # Panel 5: repetition spans in seconds (EMG timeline)
    colors = plt.cm.rainbow(np.linspace(0, 1, timing.shape[1]))
    for rep_idx in range(timing.shape[1]):
        start_time = timing[0, rep_idx] / fs_emg
        end_time = timing[1, rep_idx] / fs_emg
        axes[4].axvspan(start_time, end_time, alpha=0.35, color=colors[rep_idx])
        axes[4].text(
            (start_time + end_time) / 2.0,
            0.55,
            str(rep_idx + 1),
            ha='center',
            va='center',
            fontsize=7,
            color='black',
        )
    axes[4].set_ylabel('Rep spans', fontweight='bold')
    axes[4].set_xlabel('Time (s)', fontweight='bold')
    axes[4].set_xlim([time_emg[0], time_emg[-1]])
    axes[4].set_ylim([0, 1])
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    subject_tag = subject_label.lower()
    full_path = os.path.join(out_dir, f'alignment_{subject_tag}_movimento{movimento_num}_full.png')
    fig.savefig(full_path, dpi=110, bbox_inches='tight')
    print(f'  ✓ Saved: {os.path.basename(full_path)}')
    plt.close()
    
    # --- FIGURE 2: Repetition Details in a Common Time Domain ---
    n_reps_shown = min(3, timing.shape[1])
    fig, axes = plt.subplots(n_reps_shown, 1, figsize=(14, 3.5 * n_reps_shown), sharex=False)
    
    if n_reps_shown == 1:
        axes = np.array([axes])
    
    fig.suptitle(
        f'{subject_label} Movimento{movimento_num} - Repetition Detail (Shared Time)\n'
        f'Kinematics linearly projected onto EMG time',
        fontsize=12,
        fontweight='bold',
    )
    
    for rep_idx in range(n_reps_shown):
        start_idx = int(timing[0, rep_idx])
        end_idx = int(timing[1, rep_idx])
        t_rep = time_emg[start_idx:end_idx]

        emg_rep = emg_ch0[start_idx:end_idx]
        kin_rep = kin_on_emg[start_idx:end_idx]

        # Left axis EMG
        ax_left = axes[rep_idx]
        ax_left.plot(t_rep, emg_rep, color='tab:blue', linewidth=0.6, alpha=0.8)
        ax_left.set_ylabel(f'Rep {rep_idx + 1}\nEMG', color='tab:blue', fontweight='bold')
        ax_left.tick_params(axis='y', labelcolor='tab:blue')
        ax_left.grid(True, alpha=0.3)

        # Right axis kinematics on shared time
        ax_right = ax_left.twinx()
        ax_right.plot(t_rep, kin_rep, color='tab:red', linewidth=1.1)
        ax_right.set_ylabel('Angle (deg)', color='tab:red', fontweight='bold')
        ax_right.tick_params(axis='y', labelcolor='tab:red')

    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    
    plt.tight_layout()
    reps_path = os.path.join(out_dir, f'alignment_{subject_tag}_movimento{movimento_num}_reps.png')
    fig.savefig(reps_path, dpi=110, bbox_inches='tight')
    print(f'  ✓ Saved: {os.path.basename(reps_path)}')
    plt.close()
    
    # --- FIGURE 3: Quality Checks ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f'{subject_label} Movimento{movimento_num} - Alignment Quality Checks',
        fontsize=12,
        fontweight='bold',
    )
    
    # Duration per repetition
    rep_durations = diag['rep_durations_sec']
    axes[0, 0].bar(range(1, timing.shape[1]+1), rep_durations, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Repetition', fontweight='bold')
    axes[0, 0].set_ylabel('Duration (s)', fontweight='bold')
    axes[0, 0].set_title('Duration per Repetition')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Inter-repetition gaps
    gaps = diag['rep_gaps_sec']
    
    if len(gaps) > 0:
        axes[0, 1].bar(range(1, len(gaps)+1), gaps, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Gap (between reps)', fontweight='bold')
        axes[0, 1].set_ylabel('Duration (s)', fontweight='bold')
        axes[0, 1].set_title('Inter-Repetition Gaps')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[0, 1].text(0.5, 0.5, 'Only 1 repetition', ha='center', va='center', 
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Inter-Repetition Gaps')
    
    # Kinematics stability per rep
    rep_std = []
    for rep_idx in range(timing.shape[1]):
        start_idx = int(timing[0, rep_idx])
        end_idx = int(timing[1, rep_idx])
        rep_std.append(np.std(kin_on_emg[start_idx:end_idx]))
    
    axes[1, 0].bar(range(1, timing.shape[1]+1), rep_std, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Repetition', fontweight='bold')
    axes[1, 0].set_ylabel('Angle StdDev (°)', fontweight='bold')
    axes[1, 0].set_title('Kinematic Stability per Rep')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Alignment summary
    summary_text = (
        'ALIGNMENT VERIFICATION\n'
        '------------------------------\n'
        f'EMG duration:     {diag["duration_emg_sec"]:.3f} s\n'
        f'Kinematics duration: {diag["duration_kin_sec"]:.3f} s\n'
        f'Duration delta:   {diag["duration_delta_ms"]:.3f} ms\n\n'
        f'Downsampling:     {diag["downsample_ratio"]:.3f}x\n'
        f'Estimated fs_kin: {diag["fs_kinematics"]:.3f} Hz\n\n'
        f'Mean start map err: {diag["mean_start_residual_ms"]:.3f} ms\n'
        f'Mean end map err:   {diag["mean_end_residual_ms"]:.3f} ms\n'
        f'Max start map err:  {diag["max_start_residual_ms"]:.3f} ms\n'
        f'Max end map err:    {diag["max_end_residual_ms"]:.3f} ms\n\n'
        f'Repetitions:      {timing.shape[1]}\n'
        f'Rep avg duration: {np.mean(rep_durations):.3f} s\n'
        f'Angle range:      {kinematics.min():.2f} to {kinematics.max():.2f} deg\n'
    )
    
    axes[1, 1].text(
        0.05,
        0.95,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=9,
        family='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    quality_path = os.path.join(out_dir, f'alignment_{subject_tag}_movimento{movimento_num}_quality.png')
    fig.savefig(quality_path, dpi=110, bbox_inches='tight')
    print(f'  ✓ Saved: {os.path.basename(quality_path)}')
    plt.close()


def parse_args():
    """CLI for quickly testing different subjects and movement sets."""
    parser = argparse.ArgumentParser(description='Standalone secondary-data alignment visualizer')
    parser.add_argument('--subject', type=int, default=1, help='Subject number, e.g. 1 for Soggetto1')
    parser.add_argument('--movements', type=int, nargs='+', default=[1, 2, 5], help='Movements to visualize')
    parser.add_argument('--fs-emg', type=float, default=1000.0, help='EMG sample rate in Hz')
    parser.add_argument('--out-dir', type=str, default='.', help='Output folder for PNG files')
    return parser.parse_args()


def main():
    """Generate visualizations."""
    args = parse_args()

    base_path = os.path.join('./biosignal_data/secondary/raw', f'Soggetto{args.subject}')

    if not os.path.exists(base_path):
        print(f'Error: Path not found: {base_path}')
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 80)
    print('Creating Time-Domain Aligned Visualizations...')
    print(f'Base path: {base_path}')
    print(f'Output dir: {args.out_dir}')
    print('=' * 80)

    for mov in args.movements:
        print(f'\nMovimento {mov}:')
        visualize_full_alignment(
            base_path,
            movimento_num=mov,
            fs_emg=args.fs_emg,
            subject_label=f'Soggetto{args.subject}',
            out_dir=args.out_dir,
        )

    print('\n' + '=' * 80)
    print('✓ All visualizations complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()
