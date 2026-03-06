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
    return signal           # Remove this line to enable normalisation again (it ruined the results when ON)
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

# ==================================================================================
# ================================ NOISE GENERATION ================================
# ==================================================================================

def loadNoiseSource(path='./Dataset/D6.mat'):
    """
    Loads the D6 dataset to serve as a real-world noise source.

    Args:
        path (str): The file path to the D6 .mat file. Defaults to './Dataset/D6.mat'.

    Returns:
        noiseData[np.ndarray]: The raw data vector from the noise source, or None if loading failed.
    """
    if not os.path.exists(path):
        print(f"[WARNING] Noise source file not found: {path}")
        return None
    
    try:
        mat = spio.loadmat(path, squeeze_me=True)
        noiseData = mat['d']
        
        # Remove DC offset from the noise source so it doesn't shift the signal
        noiseData = noiseData - np.mean(noiseData)
        # noiseData = medianSubtractionFilter(noiseData, 10001)
        return noiseData
    except Exception as e:
        print(f"[ERROR] Failed to load noise source: {e}")
        return None

def addD6Noise(cleanSignal, noiseSource, targetSNR):
    """
    Superimposes realistic noise from a source signal onto a clean signal with target SNR.

    Args:
        cleanSignal (np.ndarray): Clean inoput
        noiseSource (Optional[np.ndarray]): Noise source. If None, Gaussian noise is used.
        targetSNR (float): SNR to use (dB).

    Returns:
        noisySignalnp.ndarray: The resulting noisy signal.
    """
    if noiseSource is None:
        print("[INFO] D6 not available, falling back to Gaussian noise.")
        cleanPeak = np.max(np.abs(cleanSignal)) 
        signalPower = cleanPeak ** 2
        
        noisePower = signalPower / (10 ** (targetSNR / 10))
        return cleanSignal + np.random.normal(0, np.sqrt(noisePower), len(cleanSignal))

    sampleCount = len(cleanSignal)
    noiseCount = len(noiseSource)
    
    # Slice them so the lengths are the same
    if noiseCount > sampleCount:
        startIndices = np.random.randint(0, noiseCount - sampleCount)
        noiseWindow = noiseSource[startIndices : startIndices + sampleCount]    
    else:
        repeats = int(np.ceil(sampleCount / noiseCount))
        noiseWindow = np.tile(noiseSource, repeats)[:sampleCount]       # Map to the same length of window

    # Get power of clean signal
    cleanPeak = np.max(np.abs(cleanSignal))
    if cleanPeak == 0: 
        return cleanSignal
    cleanPower = cleanPeak ** 2 

    # Get power of noise window
    noiseWindowPower = np.mean(noiseWindow ** 2)
    if noiseWindowPower == 0:
        return cleanSignal

    # Calculate the new noise power and apply
    requiredNoisePower = cleanPower / (10 ** (targetSNR / 10))
    scaleFactor = np.sqrt(requiredNoisePower / noiseWindowPower)
    
    noisySignal = cleanSignal + (noiseWindow * scaleFactor)
    
    return noisySignal

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