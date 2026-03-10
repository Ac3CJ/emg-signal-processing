import scipy.signal
import scipy as sp
import scipy.io as spio
import numpy as np
import pywt
import os

from scipy import ndimage
from sklearn.decomposition import PCA

SAMPLE_FREQUENCY = 25000

# ====================================================================================
# ================================ CLASSIC PROCESSING ================================
# ====================================================================================

# ====================================================================================
# ============================== SEMG STANDARD PIPELINE ==============================
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

def waveletDenoiser(data, wavelet='db4', level=2):
    """
    Applies Wavelet denoising using soft thresholding.

    Args:
        data (np.ndarray): Input signal.
        wavelet (str): The type of wavelet to use. Defaults to 'db4'.
        level (int): The decomposition level. Defaults to 2.    (Depreciated)

    Returns:
        np.ndarray: The denoised signal reconstructed from thresholded coefficients.
    """
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * np.median(np.abs(coeff[-1] - np.median(coeff[-1])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")

def normaliseSignal(signal, definedMin=None, definedMax=None):
    """
    Normalizes the signal to a 0.0 to 1.0 range based on min/max values.

    Args:
        signal (np.ndarray): Input signal
        definedMin (Optional[float]): Fixed min value to use for normalization. Defaults to None.
        definedMax (Optional[float]): Fixed max value to use for normalization. Defaults to None.

    Returns:
        np.ndarray: The normalized signal.
    """
    # return signal           # Remove this line to enable normalisation again (it ruined the results when ON)
    minValue = min(signal)
    
    if definedMin is not None:
        minValue = definedMin

    signal = signal + abs(minValue)
    maxValue = max(signal)

    if definedMax is not None:
        maxValue = definedMax

    if maxValue == 0:
        return signal
    signal = (signal / maxValue)
    return signal

def filterSignal(signal, savgolWindow=31, savgolPoly=3, baselineWindowSize=301, doWavelet=True):
    """
    Generalized filter for data that removes LF and HF noise.

    Steps:
    1. Removes low-freq drift via Median Baseline Subtraction filter.
    2. Smoothes high-freq noise via Savitzky-Golay filter.
    3. (Optional) Applies Wavelet denoising.

    Args:
        signal (np.ndarray): Raw Signal
        savgolWindow (int): Window length for the Savitzky-Golay filter. Defaults to 31.
        savgolPoly (int): Polynomial order for the Savitzky-Golay filter. Defaults to 3.
        baselineWindowSize (int): Window length for the Median Filter. Defaults to 301.
        doWavelet (bool): Whether to apply wavelet denoising. Defaults to True.

    Returns:
        np.ndarray: The fully filtered and cleaned signal.
    """
    # Remvoe Low Freq Noise
    if baselineWindowSize > 0:
        signal = medianSubtractionFilter(signal, windowSize=baselineWindowSize)

    # Remove High Freq Noise
    if savgolWindow > 3:
        if savgolWindow % 2 == 0: savgolWindow += 1
        signal = scipy.signal.savgol_filter(signal, window_length=int(savgolWindow), polyorder=int(savgolPoly))

    # High Frequency Clean Up
    if doWavelet:
        signal = waveletDenoiser(signal, wavelet='db4', level=2)

    return signal

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
    clean_signal = notchFilter(signal, fs=fs, notchFreq=50.0)
    
    # 2. Bandpass between 20Hz and 450Hz
    clean_signal = bandpassFilter(clean_signal, fs=fs, lowCut=20.0, highCut=450.0)
    
    # 3. Full-wave rectify
    processed_signal = rectifySignal(clean_signal)
    
    return processed_signal

# ====================================================================================
# =========================== PRINCIPLE COMPONENT ANALYSIS ===========================
# ====================================================================================

def extractAllSpikes(dataSignal, index, classification=None, windowSize=100, DEBUG_MODE=False):
    """
    Extracts signal windows.

    Args:
        dataSignal (np.ndarray): Signal to extract from
        index (list | np.ndarray): List of indices for spike lcoations
        classification (Optional[list | np.ndarray]): Corresponding classes for indices
        windowSize (int): The width of the extraction window. Defaults to 100.
        DEBUG_MODE (bool): DEPRECIATED debug flag. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - waveforms (np.ndarray): Array of extracted signal windows.
            - validIndices (np.ndarray): Indices that were successfully extracted.
            - validClasses (np.ndarray, optional): Classes corresponding to validIndices (only returned if `classification` was provided).
    """
    leftWindow = int(windowSize * 0.2)
    rightWindow = windowSize - leftWindow
    waveforms = []
    validIndices = []
    validClasses = [] 

    for i, idx in enumerate(index):
        start = idx - leftWindow
        end = idx + rightWindow
        
        if start < 0 or end > len(dataSignal):
            continue

        wave = dataSignal[start:end].copy()
        waveforms.append(wave)
        validIndices.append(idx)
        
        if classification is not None:
            validClasses.append(classification[i])
            
    if classification is not None:
        return np.array(waveforms), np.array(validIndices), np.array(validClasses)
    else:
        return np.array(waveforms), np.array(validIndices)

def fit_pca_generator(trainingWaveforms, n_components=4):
    """
    Fits a PCA model to a set of training waveforms.

    Args:
        trainingWaveforms (np.ndarray): An array of shape (N_samples, Window_Size) containing spike waveforms.
        n_components (int): The number of principal components to keep. Defaults to 4.

    Returns:
        pcaModel (sklearn.decomposition.PCA): The fitted PCA object.
    """
    pca = PCA(n_components=n_components)
    pca.fit(trainingWaveforms)
    return pca

def get_pca_features(waveforms, pcaModel):
    """
    Transforms raw waveforms into their PCA feature componentrs.

    Args:
        waveforms (np.ndarray): An array of waveforms to fit to PCA model.
        pcaModel (sklearn.decomposition.PCA): The pre-trained PCA model.

    Returns:
        waveFeatures (np.ndarray): Extracted features from waveforms.
    """
    if len(waveforms) == 0:
        return np.empty((0, pcaModel.n_components))
    return pcaModel.transform(waveforms)