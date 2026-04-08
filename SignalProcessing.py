import scipy.signal
import scipy as sp
import scipy.io as spio
import numpy as np
import os

from scipy import ndimage
from sklearn.decomposition import PCA
import ControllerConfiguration as Config

# ===============================================================================
# ================================ NORMALISATION ================================
# ===============================================================================

def get_rectified_scale_from_minmax(channel_min, channel_max, eps=1e-6):
    """
    Converts robust [min, max] bounds into a rectified-domain normalization scale.

    If normalization happens after rectification, values are non-negative and the
    correct scale is the larger absolute magnitude between min and max.
    """
    min_mag = abs(float(channel_min))
    max_mag = abs(float(channel_max))
    scale = max(min_mag, max_mag)
    if scale < eps:
        scale = eps
    return scale


def get_unrectified_scale_from_minmax(channel_min, channel_max, eps=1e-6):
    """
    Converts robust [min, max] bounds into an unrectified-domain normalization scale.
    
    For unrectified (bipolar) signals, we normalize to [-1, 1] range using the
    distance from zero of the larger bound, ensuring symmetry around zero.
    
    Args:
        channel_min (float): Minimum bound from robust statistics.
        channel_max (float): Maximum bound from robust statistics.
        eps (float): Small value to prevent division by zero.
    
    Returns:
        float: Scale factor to divide signal by for normalization to [-1, 1].
    """
    min_mag = abs(float(channel_min))
    max_mag = abs(float(channel_max))
    scale = max(min_mag, max_mag)
    if scale < eps:
        scale = eps
    return scale


def applyRobustRectifiedNormalization(continuous_signal, min_max_robust, eps=1e-6):
    """
    Normalizes rectified signals channel-wise using robust min/max references.

    Args:
        continuous_signal (np.ndarray): Shape (num_channels, total_samples), expected rectified.
        min_max_robust (np.ndarray): Shape (num_channels, 2) with [min, max] per channel.
    """
    signal = np.asarray(continuous_signal, dtype=np.float32)
    minmax = np.asarray(min_max_robust, dtype=np.float32)

    if minmax.ndim != 2:
        raise ValueError(f"Expected 2D min/max matrix, got shape {minmax.shape}")
    if minmax.shape[1] != 2 and minmax.shape[0] == 2:
        minmax = minmax.T
    if minmax.shape != (signal.shape[0], 2):
        raise ValueError(
            f"min/max shape mismatch: signal has {signal.shape[0]} channels, min/max shape is {minmax.shape}"
        )

    normalized = np.zeros_like(signal, dtype=np.float32)
    for c in range(signal.shape[0]):
        scale = get_rectified_scale_from_minmax(minmax[c, 0], minmax[c, 1], eps=eps)
        normalized[c, :] = np.clip(signal[c, :] / scale, 0.0, 1.0)

    return normalized


def applyRobustUnrectifiedNormalization(continuous_signal, min_max_robust, eps=1e-6):
    """
    Normalizes unrectified (bipolar) signals channel-wise using robust min/max references.
    Outputs normalized signals in the range [-1, 1].
    
    Args:
        continuous_signal (np.ndarray): Shape (num_channels, total_samples), unrectified (can be negative).
        min_max_robust (np.ndarray): Shape (num_channels, 2) with [min, max] per channel.
        eps (float): Small value to prevent division by zero.
    
    Returns:
        np.ndarray: Normalized signal clipped to [-1, 1] range.
    """
    signal = np.asarray(continuous_signal, dtype=np.float32)
    minmax = np.asarray(min_max_robust, dtype=np.float32)

    if minmax.ndim != 2:
        raise ValueError(f"Expected 2D min/max matrix, got shape {minmax.shape}")
    if minmax.shape[1] != 2 and minmax.shape[0] == 2:
        minmax = minmax.T
    if minmax.shape != (signal.shape[0], 2):
        raise ValueError(
            f"min/max shape mismatch: signal has {signal.shape[0]} channels, min/max shape is {minmax.shape}"
        )

    normalized = np.zeros_like(signal, dtype=np.float32)
    for c in range(signal.shape[0]):
        channel_min = float(minmax[c, 0])
        channel_max = float(minmax[c, 1])
        
        # Calculate the range
        range_span = channel_max - channel_min
        if abs(range_span) < eps:
            range_span = eps
        
        # Normalize to [0, 1] first, then scale to [-1, 1]
        # Formula: 2 * (signal - min) / (max - min) - 1
        normalized[c, :] = 2.0 * (signal[c, :] - channel_min) / range_span - 1.0
        
        # Clip to [-1, 1] to handle edge cases
        normalized[c, :] = np.clip(normalized[c, :], -1.0, 1.0)

    return normalized

    
def applyGlobalNormalization(continuous_signal, percentiles=(1.0, 99.0)):
    """
    Applies Global Peak Normalization across an entire trial duration.
    Calculates the 99th percentile for each channel over the whole recording
    to ignore hardware spikes, and scales the channel to [0.0, 1.0].
    
    Args:
        continuous_signal (np.ndarray): Shape (num_channels, total_samples)
    """
    normalized_signal = np.zeros_like(continuous_signal, dtype=np.float32)
    num_channels = continuous_signal.shape[0]
    
    for c in range(num_channels):
        # 1. Find the robust global maximum for this specific channel
        channel_max = np.percentile(continuous_signal[c, :], percentiles[1])
        channel_min = np.percentile(continuous_signal[c, :], percentiles[0])
        
        range_span = channel_max - channel_min
        if range_span < 1e-6:
            range_span = 1e-6 # Prevent division by zero on dead wires
            
        # 2. Scale the entire channel globally
        scaled = (continuous_signal[c, :] - channel_min) / range_span
        
        # 3. Clip the remaining 1% of hardware spikes that exceed 1.0
        normalized_signal[c, :] = np.clip(scaled, 0.0, 1.0)
        
    return normalized_signal


def compute_participant_minmax(mvc_file_path, fs=1000.0, percentiles=(1.0, 99.0), expected_channels=None):
    """
    Computes robust per-channel baseline/max from a participant MVC trial.

    The MVC file is processed with the same runtime pipeline
    (notch -> bandpass -> rectification), then percentile statistics are
    computed per channel.

    Args:
        mvc_file_path (str): Path to participant MVC .mat file (PxM10.mat).
        fs (float): Sampling frequency in Hz.
        percentiles (tuple): Lower/upper percentiles for baseline/max.
        expected_channels (int): Optional channel-count validation.

    Returns:
        tuple[np.ndarray, np.ndarray]: (baseline, max_vals), each shape (num_channels,).
    """
    if not os.path.exists(mvc_file_path):
        raise FileNotFoundError(f"MVC file not found: {mvc_file_path}")

    mat_contents = spio.loadmat(mvc_file_path)
    if "EMGDATA" not in mat_contents:
        raise KeyError(f"EMGDATA not found in MVC file: {mvc_file_path}")

    raw_data = np.asarray(mat_contents["EMGDATA"], dtype=np.float32)
    if raw_data.ndim != 2:
        raise ValueError(f"Expected EMGDATA to be 2D, got shape {raw_data.shape}")

    if expected_channels is not None and raw_data.shape[0] != int(expected_channels):
        raise ValueError(
            f"Expected {expected_channels} channels, found {raw_data.shape[0]} in {mvc_file_path}"
        )

    rectified = np.zeros_like(raw_data, dtype=np.float32)
    for channel_idx in range(raw_data.shape[0]):
        rectified[channel_idx, :] = applyStandardSEMGProcessing(raw_data[channel_idx, :], fs=fs)

    lower_pct, upper_pct = percentiles
    baseline = np.percentile(rectified, lower_pct, axis=1).astype(np.float32)
    max_vals = np.percentile(rectified, upper_pct, axis=1).astype(np.float32)

    return baseline, max_vals

# ====================================================================================
# ================================ CLASSIC PROCESSING ================================
# ====================================================================================

def slewRateLimiter(signal, max_step=50.0):
    """
    Limits the maximum sample-to-sample change in the signal.
    Prevents instantaneous hardware spikes.
    """
    processed = np.copy(signal)
    for i in range(1, len(signal)):
        diff = signal[i] - processed[i-1]
        if abs(diff) > max_step:
            # Cap the jump at the maximum allowed step
            processed[i] = processed[i-1] + np.sign(diff) * max_step
    return processed

def notchFilter(signal, fs=1000.0, notchFreq=50.0, qualityFactor=30.0):
    """
    Applies a Notch filter to remove powerline interference.
    
    Args:
        signal (np.ndarray): Raw input signal.
        fs (float): Sampling frequency in Hz.
        notchFreq (float): Frequency to remove (50.0 Hz for UK/EU, 60.0 Hz for US).
        qualityFactor (float): Quality factor of the notch filter.
        
    Returns:
        np.ndarray: Signal with powerline noise removed.
    """
    b, a = scipy.signal.iirnotch(notchFreq, qualityFactor, fs)
    # Using filtfilt for zero-phase distortion
    return scipy.signal.filtfilt(b, a, signal)

def bandpassFilter(signal, fs=1000.0, lowCut=20.0, highCut=450.0, order=4):
    """
    Applies a Butterworth bandpass filter to remove movement artifacts and high-freq noise.
    
    Args:
        signal (np.ndarray): Input signal.
        fs (float): Sampling frequency in Hz.
        lowCut (float): Lower cutoff frequency.
        highCut (float): Upper cutoff frequency.
        order (int): Order of the Butterworth filter.
        
    Returns:
        np.ndarray: Bandpass filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowCut / nyquist
    high = highCut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, signal)

def rectifySignal(signal):
    """
    Applies full-wave rectification to the signal.
    
    Args:
        signal (np.ndarray): Input signal.
        
    Returns:
        np.ndarray: Rectified signal (absolute values).
    """
    return np.abs(signal)

def medianSubtractionFilter(signal, windowSize):
    """
    Estimates low-frequency drift using a sliding median filter then subtracts it.

    Args:
        signal (np.ndarray): Input signal.
        windowSize (int): The length of the median filter window (must be odd). 
                          If even, it is automatically incremented by 1.

    Returns:
        filteredSignal (np.ndarray): The signal with the estimated baseline subtracted.
    """
    if windowSize % 2 == 0:
        windowSize += 1  # Ensure odd window size
        
    baseline = ndimage.median_filter(signal, size=int(windowSize))
    return signal - baseline

def normaliseSignal(signal, definedMin=None, definedMax=None, output_range=(-1.0, 1.0), percentiles=(1.0, 99.0)):
    """
    Normalizes the signal to a specified range using robust percentiles to ignore hardware spikes.
    Default output range is -1.0 to 1.0 (suitable for neural networks).

    Args:
        signal (np.ndarray): Input signal
        definedMin (Optional[float]): Fixed min value to use for normalization. Defaults to None.
        definedMax (Optional[float]): Fixed max value to use for normalization. Defaults to None.
        output_range (tuple): (min, max) output range. Default is (-1.0, 1.0).
        percentiles (tuple): (lower_percentile, upper_percentile) used to calculate robust min/max.

    Returns:
        np.ndarray: The normalized and clipped signal in the specified output range.
    """
    # 1. Find robust Minimum (Ignores massive negative spikes)
    if definedMin is not None:
        minValue = definedMin
    else:
        minValue = np.percentile(signal, percentiles[0])

    # 2. Find robust Maximum (Ignores massive positive spikes)
    if definedMax is not None:
        maxValue = definedMax
    else:
        maxValue = np.percentile(signal, percentiles[1])

    # Prevent division by zero if the signal is a pure flatline
    range_span = maxValue - minValue
    if range_span <= 0:
        range_span = 1e-6

    # 3. Normalize to 0-1 range
    normalized_01 = (signal - minValue) / range_span

    # 4. Scale to desired output range (e.g., -1.0 to 1.0)
    range_min, range_max = output_range
    scaled_signal = normalized_01 * (range_max - range_min) + range_min

    # 5. CRITICAL: Chop off the 1% of spikes that fall outside the bounds!
    clipped_signal = np.clip(scaled_signal, range_min, range_max)

    return clipped_signal

def tkeo(signal):
    """
    Applies the Root Teager-Kaiser Energy Operator (Root-TKEO).
    Highlights high-frequency bursts while restoring the linear amplitude scale.
    """
    y = np.zeros_like(signal)
    y[1:-1] = signal[1:-1]**2 - (signal[:-2] * signal[2:])
    
    # NEW: Take the square root to prevent dynamic range explosion.
    # We use np.abs() inside because TKEO can occasionally output small negative numbers.
    return np.sqrt(np.abs(y))

def lowpassFilter(signal, fs=1000.0, cutoff=5.0, order=4):
    """
    Applies a Low-Pass Butterworth filter. 
    Used to extract a smooth linear envelope from rectified/TKEO signals.
    """
    b, a = scipy.signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    return scipy.signal.filtfilt(b, a, signal)

def inject_white_noise_to_channels(signal, target_channels, noise_magnitudes):
    """
    Injects white noise into specified channels of an EMG signal.
    
    Args:
        signal (np.ndarray): Input signal of shape (num_channels, num_samples)
        target_channels (list): List of channel indices to inject noise into (0-7 for 8 channels).
                               Does not need to have all 8 elements.
        noise_magnitudes (list): List of noise magnitudes corresponding to target_channels.
                                Must be same length as target_channels.
                                Each magnitude scales the injected white noise.
    
    Returns:
        np.ndarray: Modified signal with white noise injected into specified channels.
                   Untargeted channels remain unchanged.
    
    Raises:
        ValueError: If target_channels and noise_magnitudes have different lengths,
                   or if channel indices are out of bounds.
    
    Example:
        # Inject noise into channels 1 and 3 with magnitudes 10 and 5
        noisy_signal = inject_white_noise_to_channels(
            signal, 
            target_channels=[1, 3], 
            noise_magnitudes=[10.0, 5.0]
        )
    """
    # Validate inputs
    if len(target_channels) != len(noise_magnitudes):
        raise ValueError(
            f"target_channels and noise_magnitudes must have the same length. "
            f"Got {len(target_channels)} and {len(noise_magnitudes)}"
        )
    
    # Validate channel indices
    num_channels = signal.shape[0]
    for ch_idx in target_channels:
        if not isinstance(ch_idx, int) or ch_idx < 0 or ch_idx >= num_channels:
            raise ValueError(
                f"Channel index {ch_idx} out of bounds. Valid range: [0, {num_channels - 1}]"
            )
    
    # Create a copy to avoid modifying the original
    noisy_signal = np.copy(signal).astype(np.float32)
    num_samples = signal.shape[1]
    
    # Inject noise into each target channel
    for ch_idx, noise_mag in zip(target_channels, noise_magnitudes):
        # Generate white noise (Gaussian, zero-mean, unit variance)
        white_noise = np.random.normal(loc=0.0, scale=1.0, size=num_samples)
        
        # Scale noise by the specified magnitude
        scaled_noise = white_noise * noise_mag
        
        # Inject into target channel
        noisy_signal[ch_idx, :] += scaled_noise
    
    return noisy_signal


def inject_white_noise_to_channels_excluding_windows(signal, target_channels, noise_magnitudes, exclude_windows=None):
    """
    Injects white noise into specified channels, but EXCLUDES contraction windows.
    Useful for training augmentation to avoid corrupting contraction data.
    
    Args:
        signal (np.ndarray): Input signal of shape (num_channels, num_samples).
        target_channels (list): List of channel indices to inject noise into (0-7 for 8 channels).
        noise_magnitudes (list): List of noise magnitudes corresponding to target_channels.
                                Must be same length as target_channels.
        exclude_windows (list): List of (start_idx, end_idx) tuples defining contraction windows
                               where noise should NOT be injected. If None, injects noise everywhere.
    
    Returns:
        np.ndarray: Modified signal with white noise injected only in non-contraction regions.
    
    Raises:
        ValueError: If target_channels and noise_magnitudes have different lengths,
                   or if channel indices are out of bounds.
    
    Example:
        # Inject noise everywhere except in marked contraction windows [500:1000] and [2000:3500]
        noisy_signal = inject_white_noise_to_channels_excluding_windows(
            signal, 
            target_channels=[0, 1, 2],
            noise_magnitudes=[10.0, 10.0, 10.0],
            exclude_windows=[(500, 1000), (2000, 3500)]
        )
    """
    # Validate inputs
    if len(target_channels) != len(noise_magnitudes):
        raise ValueError(
            f"target_channels and noise_magnitudes must have the same length. "
            f"Got {len(target_channels)} and {len(noise_magnitudes)}"
        )
    
    # Validate channel indices
    num_channels = signal.shape[0]
    for ch_idx in target_channels:
        if not isinstance(ch_idx, int) or ch_idx < 0 or ch_idx >= num_channels:
            raise ValueError(
                f"Channel index {ch_idx} out of bounds. Valid range: [0, {num_channels - 1}]"
            )
    
    # Create a copy to avoid modifying the original
    noisy_signal = np.copy(signal).astype(np.float32)
    num_samples = signal.shape[1]
    
    # Create a mask for valid regions (outside contraction windows)
    valid_mask = np.ones(num_samples, dtype=bool)
    if exclude_windows is not None and len(exclude_windows) > 0:
        for start_idx, end_idx in exclude_windows:
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            # Clamp to valid range
            start_idx = max(0, start_idx)
            end_idx = min(num_samples, end_idx)
            # Mark these samples as invalid (don't inject noise)
            valid_mask[start_idx:end_idx] = False
    
    # Get indices where noise should be injected
    valid_indices = np.where(valid_mask)[0]
    
    # Inject noise only at valid indices
    for ch_idx, noise_mag in zip(target_channels, noise_magnitudes):
        if len(valid_indices) > 0:
            # Generate white noise only for valid samples
            white_noise = np.random.normal(loc=0.0, scale=1.0, size=len(valid_indices))
            scaled_noise = white_noise * noise_mag
            noisy_signal[ch_idx, valid_indices] += scaled_noise
    
    return noisy_signal


def applyStandardSEMGProcessing(signal, fs=1000.0):
    """
    Wrapper function to apply the standard sEMG filtering pipeline:
    Notch -> Bandpass -> Rectification.
    
    Args:
        signal (np.ndarray): The raw sEMG signal.
        fs (float): The sampling frequency.
        
    Returns:
        np.ndarray: The clean, processed, and rectified signal ready for feature extraction.
    """
    # 1. Remove 50Hz hum
    clean_signal = notchFilter(signal, fs=fs, notchFreq=Config.NOTCH_FREQ)
    
    # 2. Bandpass between 30Hz (ECG Data) and 450Hz
    clean_signal = bandpassFilter(clean_signal, fs=fs, lowCut=Config.BANDPASS_LOW, highCut=Config.BANDPASS_HIGH)
    
    # 3. Full-wave rectify
    processed_signal = rectifySignal(clean_signal)
    
    return processed_signal