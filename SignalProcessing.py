import scipy.signal
import scipy as sp
import scipy.io as spio
import numpy as np
import os

from scipy import ndimage
from sklearn.decomposition import PCA
import ControllerConfiguration as Config

# ====================================================================================
# ================================ CLASSIC PROCESSING ================================
# ====================================================================================

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

def normaliseSignal(signal, definedMin=None, definedMax=None, output_range=(-1.0, 1.0)):
    """
    Normalizes the signal to a specified range based on min/max values.
    Default output range is -1.0 to 1.0 (suitable for neural networks).

    Args:
        signal (np.ndarray): Input signal
        definedMin (Optional[float]): Fixed min value to use for normalization. Defaults to None.
        definedMax (Optional[float]): Fixed max value to use for normalization. Defaults to None.
        output_range (tuple): (min, max) output range. Default is (-1.0, 1.0).

    Returns:
        np.ndarray: The normalized signal in the specified output range.
    """
    minValue = np.min(signal)

    if definedMin is not None:
        minValue = definedMin

    # Shift signal to start at 0
    signal = signal - minValue
    maxValue = np.max(signal)

    if definedMax is not None:
        maxValue = definedMax

    if maxValue == 0:
        return signal

    # Normalize to 0-1 range first
    signal = signal / maxValue

    # Scale to desired output range
    range_min, range_max = output_range
    signal = signal * (range_max - range_min) + range_min

    return signal

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