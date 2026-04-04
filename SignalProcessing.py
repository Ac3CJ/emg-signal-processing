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

class RealTimeGlobalNormalizer:
    def __init__(self, num_channels=8, initial_max=150.0, spike_threshold=3.0):
        """
        Maintains a running global maximum for real-time normalization.
        
        Args:
            initial_max: A conservative starting guess for the max amplitude (mV).
            spike_threshold: If a new peak is X times larger than the current known max, 
                             it is rejected as hardware noise.
        """
        # Store the running maximums and minimums for each channel
        self.running_max = np.full(num_channels, initial_max, dtype=np.float32)
        self.running_min = np.zeros(num_channels, dtype=np.float32)
        self.spike_threshold = spike_threshold

    def normalize_window(self, window):
        """
        Evaluates a new incoming real-time window, safely updates the global maximums, 
        and normalizes the window.
        
        Args:
            window (np.ndarray): The incoming real-time window, shape (num_channels, window_samples)
        """
        # 1. Find the 95th percentile of the current window to ignore single-sample micro-spikes
        current_window_peaks = np.percentile(window, 95, axis=1)
        current_window_baselines = np.percentile(window, 5, axis=1)

        # 2. Evaluate and safely update the running maximums
        for c in range(len(self.running_max)):
            # Update minimums if we find a cleaner baseline
            if current_window_baselines[c] < self.running_min[c]:
                self.running_min[c] = current_window_baselines[c]
                
            # Update maximums ONLY if it's a biologically valid contraction
            if current_window_peaks[c] > self.running_max[c]:
                # SPIKE REJECTION: Is it a massive, instantaneous hardware spike?
                if current_window_peaks[c] < (self.running_max[c] * self.spike_threshold):
                    # Valid contraction! Update our global knowledge.
                    self.running_max[c] = current_window_peaks[c]
                # Else: Do nothing. The hardware spike is ignored and the previous max is kept.

        # 3. Normalize the window using the historical global values
        # Reshape for NumPy broadcasting: (num_channels, 1)
        scale_factors = (self.running_max - self.running_min)[:, np.newaxis]
        scale_factors[scale_factors < 1e-6] = 1e-6 # Safety
        
        normalized_window = (window - self.running_min[:, np.newaxis]) / scale_factors
        
        # 4. Clip to neural network bounds (chops off the rejected hardware spikes)
        return np.clip(normalized_window, 0.0, 1.0)
    
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