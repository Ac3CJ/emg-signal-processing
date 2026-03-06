import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy as sp
import os
import glob
import pickle 
import argparse

from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

import SignalProcessing
import SignalAnalysis
import SignalUtils

# Things to do
# Find a way to suppress Class 4 from eating up all of Class 5 (D2, D4)
# Suppress Class 3 from just picking up noise
# MAKE IT SO THAT THE NOISE GENERATION USES A THICK MEDIAN FILTER TO REMOVE ALL SPIKES FROM D6 THIS WAS NOT INCLUDED BY ACCIDENT

# ======================================================
# ================== PYTORCH MODELS ====================
# ======================================================

# Check for GPU, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

class CNNDetector(nn.Module):
    """
    A 1D CNN designed to detect signal peaks within a window.
    """
    def __init__(self, input_window_size=60):
        """
        Initializes the CNN architecture.

        Args:
            input_window_size (Optional[int]): The length of the input signal window. Defaults to 60.
        """
        super(CNNDetector, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) # Output: Window/2

        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2) # Output: Window/4

        # Fully Connected
        # Input 60 -> pool(30) -> pool(15) -> 15 * 32 channels
        final_dim = (input_window_size // 4) * 32

        self.fc1 = nn.Linear(final_dim, 64)
        self.fc2 = nn.Linear(64, 1) # Output: Probability (0 to 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor containing signal windows. 

        Returns:
            torch.Tensor: A tensor containing the probability of a peak being present.
        """
        # Input shape needs to be [Batch, Channel, Length]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class ClassifierModel(nn.Module):
    """
    MLP for classifying PCA features into discrete classes.
    """
    def __init__(self, input_nodes, hidden_nodes_list, output_nodes):
        """
        Initializes the MLP layers.

        Args:
            input_nodes (int): The number of input features
            hidden_nodes_list (list[int]): A list containing the number of neurons for each hidden layer.
            output_nodes (int): The number of output classes.
        """
        super(ClassifierModel, self).__init__()
        layers = []
        in_dim = input_nodes
        for h_dim in hidden_nodes_list:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU()) 
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_nodes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs the forward pass of the network.
        Args:
            x (torch.Tensor): The input tensor containing feature vectors.
        Returns:
            torch.Tensor: The raw output logits for each class (before Softmax).
        """
        return self.model(x)

class NetworkHandler:
    """
    A wrapper class to manage model initialization, training, querying, saving, and loading
    For both Detector (CNN) and Classifier (MLP) architectures.
    """
    def __init__(self, model_type, **kwargs):
        """
        Initializes the network handler and constructs the specific underlying model.

        Args:
            model_type (str): The type of model to initialize. Options: 'detector' or 'classifier'.
            **kwargs: Arbitrary keyword arguments used for model configuration.
                - windowSize (int): Input window size (Detector only).
                - input_nodes (int): Input feature size (Classifier only).
                - hidden_nodes_list (list[int]): Hidden layer configuration (Classifier only).
                - output_nodes (int): Number of output classes (Classifier only).
                - class_weights (list): Weights for CrossEntropyLoss (Classifier only).
                - lr (float): Learning rate for the optimizer. Defaults to 0.001.
        """
        self.model_type = model_type

        if model_type == 'detector':
            self.model = CNNDetector(input_window_size=kwargs.get('windowSize', 60)).to(DEVICE)
            self.criterion = nn.BCELoss()
        else:
            self.model = ClassifierModel(
                kwargs['input_nodes'], 
                kwargs['hidden_nodes_list'], 
                kwargs['output_nodes']
            ).to(DEVICE)

            # --- WEIGHTING LOGIC HERE ---
            if 'class_weights' in kwargs:
                weights = torch.Tensor(kwargs['class_weights']).to(DEVICE)
                print(f"[{model_type.upper()}] Applying Class Weights: {kwargs['class_weights']}")
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get('lr', 0.001))

    def train(self, inputs, targets, val_inputs=None, val_targets=None, epochs=10, batch_size=32):
        """
        Trains model using the provided data.

        Args:
            inputs (np.ndarray): Training input data.
            targets (np.ndarray): Training target labels (or values).
            val_inputs (Optional[np.ndarray]): Validation input data. Defaults to None.
            val_targets (Optional[np.ndarray]): Validation target labels. Defaults to None.
            epochs (Optional[int]): Number of epochs. Defaults to 10.
            batch_size (Optional[int]): Size of batches. Defaults to 32.
        """
        self.model.train()
        tensor_x = torch.Tensor(inputs).to(DEVICE)

        # Prepare Training Targets
        if self.model_type == 'detector':
            tensor_y = torch.Tensor(targets).reshape(-1, 1).to(DEVICE)
        else:
            if len(targets.shape) > 1 and targets.shape[1] > 1: targets = np.argmax(targets, axis=1)
            tensor_y = torch.LongTensor(targets).to(DEVICE)

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare Validation Data (if provided)
        val_loader = None
        if val_inputs is not None and val_targets is not None:
            val_tensor_x = torch.Tensor(val_inputs).to(DEVICE)
            if self.model_type == 'detector':
                val_tensor_y = torch.Tensor(val_targets).reshape(-1, 1).to(DEVICE)
            else:
                if len(val_targets.shape) > 1 and val_targets.shape[1] > 1: val_targets = np.argmax(val_targets, axis=1)
                val_tensor_y = torch.LongTensor(val_targets).to(DEVICE)
            val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"[{self.model_type.upper()}] Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # --- TRAINING LOOP ---
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader)

            # --- VALIDATION LOOP ---
            val_log = ""
            if val_loader:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        v_outputs = self.model(vx)
                        v_loss = self.criterion(v_outputs, vy)
                        val_loss += v_loss.item()

                        # Calculate Accuracy
                        if self.model_type == 'classifier':
                            _, predicted = torch.max(v_outputs.data, 1)
                            total += vy.size(0)
                            correct += (predicted == vy).sum().item()
                        else:
                            # For detector (binary), assume threshold 0.5
                            predicted = (v_outputs > 0.5).float()
                            total += vy.size(0)
                            correct += (predicted == vy).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total
                val_log = f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}{val_log}")

    def query(self, inputs_list):
        """
        Queries the model.
        
        For the classifier, this applies Softmax.

        Args:
            inputs_list (list | np.ndarray): A list or array of input features/windows.

        Returns:
            np.ndarray: The Transposed array of model outputs (Probabilities).
        """
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.Tensor(inputs_list).to(DEVICE)
            outputs = self.model(tensor_x)
            if self.model_type == 'classifier':
                outputs = torch.softmax(outputs, dim=1)
        return outputs.cpu().numpy().T 

    def save_model(self, filePath):
        """
        Saves the model.

        Args:
            filePath (str): The target path to save the model (extension will be forced to .pth).
        """
        base, _ = os.path.splitext(filePath)
        pth_path = base + ".pth"
        torch.save(self.model.state_dict(), pth_path)
        print(f"Saved {self.model_type} to {pth_path}")

    def load_model(self, filePath):
        """
        Loads the model  from disk if it exists.

        Args:
            filePath (str): The path to the model file.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        base, _ = os.path.splitext(filePath)
        pth_path = base + ".pth"
        if os.path.exists(pth_path):
            self.model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
            print(f"Loaded {self.model_type} from {pth_path}")
            return True
        return False

# ==============================================
# =================== CONFIG ===================
# ==============================================

# DETECTOR CONFIG
DETECTOR_WINDOW = 60  # Smaller window for detection scanning
D1_NOISE_SAMPLES = 5000 # NEW: Control amount of D1 background noise
DETECTOR_THRESHOLD = 0.5
DETECTOR_FILE = "cnn_detector_model"
KEEP_RATE_EDGES = 0.8
HARD_NEGATIVES_THRESHOLD=0.25

# CLASSIFIER CONFIG
CLASSIFIER_WINDOW = 150 # Larger window for full shape classification
N_COMPONENTS = 50
CLASSIFIER_NOISE_SAMPLES = 5000 
CLASSIFIER_HIDDEN_NODES = [256, 128]
CLASSIFIER_EPOCHS = 30
CLASSIFIER_BATCH_SIZE = 64

CLASSIFIER_THRESHOLD = 0.5
CLASSIFIER_C3_THRESHOLD = 0.75

# CLASS_WEIGHTS = [N,   C1,  C2,  C3,  C4,  C5] 
CLASS_WEIGHTS =   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
CLASS_TRAINING_MULTIPLIERS = {
    0: 2.0,  # Noise
    1: 1.0,  # Class 1
    2: 3.0,  # Class 2
    3: 0.6,  # Class 3 
    4: 0.7,  # Class 4 
    5: 3.5   # Class 5 
}
CLASSIFIER_FILE = "classifier_model" 
PCA_FILE = "pca_model.pkl"

SPACED_SNR = np.linspace(6, 40, 20)
DATASET_SNR = [25, 20, 16, 10, 6]
WEIGHTED_SNR = [25, 20, 16, 16, 14, 12, 10, 10, 8, 7, 6, 6, 6]
EDGE_SHIFTS = np.concatenate((np.arange(-50, 0, 5), np.arange(5, 55, 5)))

# --- HELPER FUNCTIONS ---
def cnnQuery(peakDetectorCNN, rawSignal, windowSize=DETECTOR_WINDOW, batch_size=2048):
    """
    Slides the CNN over the signal efficiently using PyTorch unfold to generate a probability map.

    Args:
        peakDetectorCNN (NetworkHandler): The wrapper object containing the trained CNN detector model.
        rawSignal (np.ndarray): The 1D raw signal array to be scanned.
        windowSize (Optional[int]): The size of the sliding window. Defaults to DETECTOR_WINDOW.
        batch_size (Optional[int]): The number of windows to process in a single batch. Defaults to 2048.

    Returns:
        probabilitySignal (np.ndarray): A 1D array of probabilities (0.0 to 1.0) matching the length of the raw signal.
    """
    model = peakDetectorCNN.model
    model.eval()

    # Pad signal so output length matches input
    halfWindow = windowSize // 2
    paddedSignal = np.pad(rawSignal, (halfWindow, halfWindow), mode='constant')

    # Create windows and unfold them
    tensorSignal = torch.Tensor(paddedSignal).to(DEVICE).unsqueeze(0).unsqueeze(0)
    windows = tensorSignal.unfold(2, windowSize, 1).permute(2, 0, 1, 3).squeeze(1) # [Num_Windows, 1, Window]

    outputProbability = []

    with torch.no_grad():
        num_windows = windows.shape[0]
        for i in range(0, num_windows, batch_size):
            batch = windows[i : i + batch_size]
            probs = model(batch) # [Batch, 1]
            outputProbability.append(probs.cpu().numpy().flatten())

    fullProbabilitySignal = np.concatenate(outputProbability)

    # Trim or Pad to match exact original length due to stride logic
    if len(fullProbabilitySignal) > len(rawSignal):
        fullProbabilitySignal = fullProbabilitySignal[:len(rawSignal)]
    elif len(fullProbabilitySignal) < len(rawSignal):
        fullProbabilitySignal = np.pad(fullProbabilitySignal, (0, len(rawSignal) - len(fullProbabilitySignal)))

    return fullProbabilitySignal

def processDataset(filename, detectorModel, classifierModel, pcaModel, 
                    targetFolder='./Dataset/', isSubmission=False, plotGraphs=False, insertData=False):
    """
    Processes a single dataset and identifies the location and class of spikes within the dataset.

    Args:
        filename (str): The name of the target file to analyze.
        detectorModel (str): The target file/path for the detector model.
        classifierModel (str): The target file/path for the classifier model.
        pcaModel (str): The target file/path for the PCA model.
        targetFolder (Optional[str]): The folder containing the files to analyze. Defaults to './Dataset/'.
        isSubmission (Optional[bool]): Whether to run in submission mode. Defaults to False.
        plotGraphs (Optional[bool]): Debug parameter to plot graphs after analysis. Defaults to False.
        insertData (Optional[bool]): Debug parameter to insert raw data into submission files. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - errors (dict): A dictionary containing errors for retraining (FP, FN, Misclassified).
            - featuresSignal (list): The signal that was analyzed.
    """
    filePath = os.path.join(targetFolder, filename)
    if not os.path.exists(filePath):
        print(f"[ERROR] File not found: {filePath}")
        return None

    print(f"\nProcessing: {filename}")

    # 1. Load Data
    isLabelled = "D1" in filename # Or pass this in as a param if D2 becomes labeled later
    rawData = SignalUtils.loadData(filePath, isLabelled)

    if isLabelled: 
        rawData, rawIndices, rawClasses = rawData
    else:
        rawIndices, rawClasses = [], []

    # 2. PRE-FILTERING
    # Note: Ensure this matches your training pre-processing exactly
    filteredSignal = SignalProcessing.filterSignal(rawData)
    featuresSignal = SignalProcessing.normaliseSignal(filteredSignal)

    # 3. RUN CNN DETECTOR
    print(" -> Scanning signal with CNN...")
    probabilitySignal = cnnQuery(detectorModel, featuresSignal, windowSize=DETECTOR_WINDOW)

    # 4. FIND PEAKS
    peakLocations, _ = sp.signal.find_peaks(probabilitySignal, height=DETECTOR_THRESHOLD, distance=10)
    print(f" -> Found {len(peakLocations)} potential peaks.")

    finalPeaks = []
    finalClasses = []

    # 5. CLASSIFY PEAKS
    if len(peakLocations) > 0:
        candidateWaveWindows, validIndices = SignalProcessing.extractAllSpikes(
            featuresSignal, peakLocations, windowSize=CLASSIFIER_WINDOW
        )
        candidateFeatures = SignalProcessing.get_pca_features(candidateWaveWindows, pcaModel)

        # Batch Query
        outputs = classifierModel.query(candidateFeatures).T 

        for i in range(len(outputs)):
            probs = outputs[i]
            confidence = np.max(probs)
            cls = np.argmax(probs)

            if confidence < CLASSIFIER_THRESHOLD:                       # Reject Low Confidence
                continue

            if (cls == 3) and (confidence < CLASSIFIER_C3_THRESHOLD):   # Reject WEAK C3
                continue

            if cls == 0:                                                # Reject Noise
                continue

            # You pass :D
            finalPeaks.append(validIndices[i])
            finalClasses.append(int(cls))

    # 6. GRADING & ERROR COLLECTION
    errors = {
        'false_positives': [],
        'false_negatives': [],
        'misclassified': []
    }

    print(f" -> Process Results")

    counts = Counter(finalClasses)
    print(f"     -> Classified: {len(finalPeaks)} - Class {1}: {counts.get(1, 0)} | Class {2}: {counts.get(2, 0)} | Class {3}: {counts.get(3, 0)} | Class {4}: {counts.get(4, 0)} | Class {5}: {counts.get(5, 0)}")

    if isLabelled:
        # Unpack the returns from grade_performance
        falsePositives, falseNegatives, misclassIndices, misclassClasses = SignalAnalysis.grade_performance(
            featuresSignal, probabilitySignal, rawIndices, rawClasses, finalPeaks, finalClasses
        )

        errors['false_positives'] = falsePositives
        errors['false_negatives'] = falseNegatives
        errors['misclassified'] = list(zip(misclassIndices, misclassClasses)) # Store as tuple (idx, true_class)

    # 7. OPTIONAL: SUBMISSION SAVE
    if isSubmission:
        saveFileName = "cjg75_10243_" + filename
        SignalUtils.saveSubmission(saveFileName, rawData, finalPeaks, finalClasses, insertData=insertData)

    # 8. OPTIONAL: PLOTTING
    if plotGraphs:
        # SignalAnalysis.debug_compare_signals(filteredSignal, probabilitySignal, peakLocations, label2="Probability Signal")
        SignalAnalysis.plot_class_templates_interactive(featuresSignal, finalPeaks, finalClasses, filename)

    return errors, featuresSignal

# ======================================================
# =================== TRAIN DETECTOR ===================
# ======================================================

def trainDetector():
    """
    Executes the full training for the CNN Detector.

    Generates synthetic training data (including clean positives, noisy positives, hard negatives, and background noise) 
    based on the D1 and D6 datasets. It then initializes the training loop and saves the final model to disk.

    Returns:
        None
    """
    print(f"\n{'='*40}\nTRAINING CNN DETECTOR\n{'='*40}")

    # Load the raw data
    d1FilePath = './Dataset/Training/D1.mat'
    if not os.path.exists(d1FilePath): d1FilePath = './Dataset/D1.mat'
    d1Data, d1Indices, _ = SignalUtils.loadData(d1FilePath, True)
    d6Noise = SignalProcessing.loadNoiseSource('./Dataset/D6.mat')

    # PREPROCESSING
    d1FilteredSignal = SignalProcessing.filterSignal(d1Data)
    d1NormalisedSignal = SignalProcessing.normaliseSignal(d1FilteredSignal)

    x_train = []
    y_train = []

    print("[DETECTOR] Generatign Realistic Synthetic Training Data...")

    # ======================================================
    # ================== POSITIVE SAMPLES ==================
    # ======================================================

    # The clean stuff
    print(f"   -> Generating clean positive signals from {len(d1Indices)} locations...")

    cleanWindows = []
    cleanIterations = 1
    for i in range(cleanIterations):
        for idx in d1Indices:
            start = idx - DETECTOR_WINDOW // 2
            end = start + DETECTOR_WINDOW

            if start >= 0 and end < len(d1NormalisedSignal):
                x_train.append(d1NormalisedSignal[start:end])
                y_train.append(1.0)

                if i < 1: cleanWindows.append(d1NormalisedSignal[start:end])

    # Noisy stuff (Add noise to raw, filter, nromalise)
    print(f"   -> Generating noisy positive signals from {len(d1Indices)} locations...")

    for snr in DATASET_SNR:
        noisyRawData = SignalProcessing.addD6Noise(
            cleanSignal=d1Data,       
            noiseSource=d6Noise, 
            targetSNR=snr           
        )

        noisyFilteredSignal = SignalProcessing.filterSignal(noisyRawData) 
        noisyNormalisedSignal = SignalProcessing.normaliseSignal(noisyFilteredSignal)

        # Extract spikes
        for idx in d1Indices:
            start = idx - DETECTOR_WINDOW // 2
            end = start + DETECTOR_WINDOW

            if start >= 0 and end < len(noisyNormalisedSignal):
                segment = noisyNormalisedSignal[start:end]
                x_train.append(segment)
                y_train.append(1.0)

    # ==========================================================
    # ===================== HARD NEGATIVES =====================
    # ==========================================================

    # Grab the edges of each window and set it zero
    print(f"   -> Generating clean hard negatives from {len(d1Indices)} locations...")
    for idx in d1Indices:
        for s in EDGE_SHIFTS:
            start = (idx + s) - DETECTOR_WINDOW // 2
            end = start + DETECTOR_WINDOW

            if start >= 0 and end < len(d1NormalisedSignal):
                x_train.append(d1NormalisedSignal[start:end])
                y_train.append(0.0)

    # Noisy Version
    for snr in DATASET_SNR:
        noisyRawSignal = SignalProcessing.addD6Noise(
            cleanSignal=d1Data,       
            noiseSource=d6Noise, 
            targetSNR=snr           
        )

        noisyFilteredSignal = SignalProcessing.filterSignal(noisyRawSignal)
        noisyNormalisedSignal = SignalProcessing.normaliseSignal(noisyFilteredSignal)

        # Extract immediately
        for idx in d1Indices:
            for s in EDGE_SHIFTS:

                # Threshold to limit number of hard negatives
                if np.random.rand() > HARD_NEGATIVES_THRESHOLD:
                    continue

                start = (idx + s) - DETECTOR_WINDOW // 2
                end = start + DETECTOR_WINDOW

                if start >= 0 and end < len(noisyNormalisedSignal):
                    x_train.append(noisyNormalisedSignal[start:end])
                    y_train.append(0.0)

    # ==============================================================
    # ===================== STANDARD NEGATIVES =====================
    # ==============================================================

    # Random Background from D1 Clean
    peakMask = np.zeros(len(d1NormalisedSignal), dtype=bool)

    # Generate a mask for the window
    for idx in d1Indices:
        peakMask[idx-30:idx+30] = True 

    validStarts = np.where(peakMask == False)[0]

    noiseSamplesCount = min(len(validStarts), int(D1_NOISE_SAMPLES))
    randomStarts = np.random.choice(validStarts, noiseSamplesCount, replace=False)

    print(f"   -> Generating background samples from {len(randomStarts)} locations...")

    # Create CLEAN Background Noise
    for start in randomStarts:
        end = start + DETECTOR_WINDOW
        if end < len(d1NormalisedSignal):
            x_train.append(d1NormalisedSignal[start:end])
            y_train.append(0.0)

    # Create NOISY Background Noise
    for snr in SPACED_SNR[::2]:
        noisyRawData = SignalProcessing.addD6Noise(
            cleanSignal=d1Data,       
            noiseSource=d6Noise, 
            targetSNR=snr           
        )

        noisyFilteredSignal = SignalProcessing.filterSignal(noisyRawData) 
        noisyNormalisedSignal = SignalProcessing.normaliseSignal(noisyFilteredSignal)

        for start in randomStarts:
            if np.random.rand() > KEEP_RATE_EDGES:
                continue
            end = start + DETECTOR_WINDOW
            if end < len(noisyNormalisedSignal):
                segment = noisyNormalisedSignal[start:end]
                x_train.append(segment)
                y_train.append(0.0)

    # Training Setup
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Randomise the samples to prevent local minima problems
    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]

    print(f"[DETECTOR] Training on {len(x_train)} samples.")

    detector = NetworkHandler('detector', windowSize=DETECTOR_WINDOW, lr=0.001)
    detector.train(x_train, y_train, epochs=25, batch_size=128)
    detector.save_model(DETECTOR_FILE)

    print("Detector Training Complete.")

# ========================================================
# =================== TRAIN CLASSIFIER ===================
# ========================================================
def trainClassifier():
    """
    Executes the full training for the PCA-ANN Classifier.

    This function performs the following steps:
    1. Fits a PCA model on clean D1 spikes.
    2. Generates augmented training data (clean, noisy, and background samples).
    3. Balances the class distribution using defined multipliers.
    4. Trains the MLP.
    5. Saves both the PCA model and the trained classifier model to disk.

    Returns:
        None
    """
    def balanceClassData(featuresWindows, featuresLabels):
        """
        Appends features to the master lists based.
        Handles integer repeats and fractional amounts.
        """
        uniqueClasses = np.unique(featuresLabels)

        for classID in uniqueClasses:
            classMultiplierFactor = CLASS_TRAINING_MULTIPLIERS.get(classID, 1.0)

            # Filter samples for this specific class
            classMask = (featuresLabels == classID)
            classFeatures = featuresWindows[classMask]
            classLabels = featuresLabels[classMask]

            noiseSamplesCount = len(classFeatures)
            if noiseSamplesCount == 0: continue

            # The digits
            repeatCountDigit = int(classMultiplierFactor)

            for _ in range(repeatCountDigit):
                featuresList.append(classFeatures)
                labelsList.append(classLabels)

            # The decimal section
            fraction = classMultiplierFactor - repeatCountDigit

            if fraction > 0.01: 
                repeatCountFraction = int(noiseSamplesCount * fraction)

                if repeatCountFraction > 0:
                    subsetIndices = np.random.choice(noiseSamplesCount, repeatCountFraction, replace=False)     # Select random indices to append to lists

                    featuresList.append(classFeatures[subsetIndices])   
                    labelsList.append(classLabels[subsetIndices])

    # ===================== THE MAIN CLASSIFIER SECTION =====================
    print(f"\n{'='*40}\nTRAINING CLASSIFIER\n{'='*40}")

    # 1. Load Ground Truth (D1)
    d1FilePath = './Dataset/Training/D1.mat'
    if not os.path.exists(d1FilePath): d1FilePath = './Dataset/D1.mat'

    # Load RAW data
    d1Data, d1Indices, d1Classes = SignalUtils.loadData(d1FilePath, True)
    d6Noise = SignalProcessing.loadNoiseSource('./Dataset/D6.mat')

    # PREPROCESSING
    d1FilteredSignal = SignalProcessing.filterSignal(d1Data) 
    d1NormalisedSignal = SignalProcessing.normaliseSignal(d1FilteredSignal)

    cleanWaves, cleanLocations, cleanLabels = SignalProcessing.extractAllSpikes(
        d1NormalisedSignal, d1Indices, d1Classes, windowSize=CLASSIFIER_WINDOW
    )

    # Fit PCA on the ideal spikes
    print("[CLASSIFIER] Fitting PCA on clean data...")
    pcaModel = SignalProcessing.fit_pca_generator(cleanWaves, n_components=N_COMPONENTS)
    with open(PCA_FILE, 'wb') as f: pickle.dump(pcaModel, f)

    featuresList = []
    labelsList = []

    # --- DATA GENERATION ---
    print("   -> Processing Clean Spikes...")
    clean_feats = SignalProcessing.get_pca_features(cleanWaves, pcaModel)

    balanceClassData(clean_feats, cleanLabels)

    # Get the noisy versions of D1
    print("   -> Processing Augmented Spikes...")
    for snr in WEIGHTED_SNR:
        print("   -> Generating Noise: ", snr)

        # Add Noise and Preprocess
        noisySignal = SignalProcessing.addD6Noise(d1Data, d6Noise, snr)
        noisyFilteredSignal = SignalProcessing.filterSignal(noisySignal)
        noisyNormalisedSignal = SignalProcessing.normaliseSignal(noisyFilteredSignal)

        # 2. Extract Spikes
        extractedWaves, _, _ = SignalProcessing.extractAllSpikes(
            noisyNormalisedSignal, 
            d1Indices, 
            d1Classes, 
            windowSize=CLASSIFIER_WINDOW
        )

        # Feature Extraction
        augmentedFeatures = SignalProcessing.get_pca_features(extractedWaves, pcaModel)
        augmentedLabels = d1Classes

        # Balance Classes
        balanceClassData(augmentedFeatures, augmentedLabels)

    # Background Noise
    print("   -> Generating Background Noise...")
    noisyFeaturesTemp = []

    # Setup Background Extraction
    leftWindow = int(CLASSIFIER_WINDOW * 0.2)       # 20% of window is on left
    rightWindow = CLASSIFIER_WINDOW - leftWindow    # 80% of window is on right

    # Generate mask for where there are no peaks
    peakMask = np.zeros(len(d1NormalisedSignal), dtype=bool)
    for idx in d1Indices:
        start = max(0, idx - leftWindow - 20) 
        end = min(len(d1NormalisedSignal), idx + rightWindow + 20)
        peakMask[start:end] = True

    # Check for valid areas
    validStarts = np.where(peakMask == False)[0]
    validStarts = validStarts[validStarts < (len(d1NormalisedSignal) - CLASSIFIER_WINDOW)]

    # Randomly pick the indices
    randomStarts = np.random.choice(validStarts, min(len(validStarts), CLASSIFIER_NOISE_SAMPLES), replace=False)

    # Clean Background
    cleanBackgroundWaves = [d1NormalisedSignal[s:s+CLASSIFIER_WINDOW] for s in randomStarts]
    noisyFeaturesTemp.append(SignalProcessing.get_pca_features(cleanBackgroundWaves, pcaModel))

    # Noisy Backgrounds
    for snr in SPACED_SNR: 
        noisyRawSignal = SignalProcessing.addD6Noise(d1Data, d6Noise, snr)
        noisyFilteredSignal = SignalProcessing.filterSignal(noisyRawSignal)
        noisyNormalisedSignal = SignalProcessing.normaliseSignal(noisyFilteredSignal)

        noisyBackgroundWaves = [noisyNormalisedSignal[s:s+CLASSIFIER_WINDOW] for s in randomStarts]
        noisyFeaturesTemp.append(SignalProcessing.get_pca_features(noisyBackgroundWaves, pcaModel))

    # Process Noise Class and balance
    if len(noisyFeaturesTemp) > 0:
        allNoisyFeatures = np.vstack(noisyFeaturesTemp)
        allNoisyLabels = np.zeros(len(allNoisyFeatures), dtype=int)

        print(f"   -> Generated {len(allNoisyFeatures)} Class 0 samples. Applying balancing...")
        balanceClassData(allNoisyFeatures, allNoisyLabels)

    # Compile and Train
    final_features = np.vstack(featuresList)
    final_labels = np.hstack(labelsList)

    # Shuffle
    p = np.random.permutation(len(final_features))
    final_features, final_labels = final_features[p], final_labels[p]

    # Split into training and validation
    splitIndex = int(len(final_features) * 0.8)
    x_train, X_val = final_features[:splitIndex], final_features[splitIndex:]
    y_train, y_val = final_labels[:splitIndex], final_labels[splitIndex:]

    print(f"   -> Total Samples (After Balancing): {len(final_features)}")

    # Print stats per class to verify
    unique, counts = np.unique(final_labels, return_counts=True)
    clean_dict = {int(k): int(v) for k, v in zip(unique, counts)}
    print("   -> Class Distribution:", clean_dict)

    classifier = NetworkHandler('classifier', input_nodes=N_COMPONENTS, hidden_nodes_list=CLASSIFIER_HIDDEN_NODES, output_nodes=6, class_weights=CLASS_WEIGHTS)

    classifier.train(x_train, y_train, val_inputs=X_val, val_targets=y_val, epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE)
    classifier.save_model(CLASSIFIER_FILE)

# ================================ SUBMISSION PIPELINE ================================

def submissionPipeline(targetFolder='./Dataset/Submission/', isSubmission=False, plotGraphs=False):
    """
    Runs the full analysis pipeline on all datasets within the target folder.

    Args:
        targetFolder (Optional[str]): The folder containing the .mat files to process. Defaults to './Dataset/Submission/'.
        isSubmission (Optional[bool]): Boolean to format the output for submission (save files). Defaults to False.
        plotGraphs (Optional[bool]): Debug parameter to plot graphs after analysis. Defaults to False.

    Returns:
        None
    """
    print(f"\n{'='*40}\nSTARTING CNN-BASED SUBMISSION\n{'='*40}")

    # Load Models
    detector = NetworkHandler('detector', windowSize=DETECTOR_WINDOW)
    if not detector.load_model(DETECTOR_FILE): return

    classifier = NetworkHandler('classifier', input_nodes=N_COMPONENTS, hidden_nodes_list=CLASSIFIER_HIDDEN_NODES, output_nodes=6)
    if not classifier.load_model(CLASSIFIER_FILE): return

    if not os.path.exists(PCA_FILE): return
    with open(PCA_FILE, 'rb') as f: pcaModel = pickle.load(f)

    # Get all the files and run the process on all of them
    targetFiles = glob.glob(os.path.join(targetFolder, '*.mat'))

    for filePath in targetFiles:
        filename = os.path.basename(filePath)

        processDataset(
            filename, 
            detector, 
            classifier, 
            pcaModel, 
            targetFolder=targetFolder, 
            isSubmission=isSubmission, 
            plotGraphs=plotGraphs
        )

if __name__ == "__main__":
    # Initialize the Argument Parser
    parser = argparse.ArgumentParser(description="Brain Wave Analyser: Train models or run submission.")

    # Arguments
    parser.add_argument('--train', action='store_true', help='Run the full training sequence (Classifier + Detector)')
    parser.add_argument('--test', action='store_true', help='Run the submission/testing pipeline on ./Dataset/Submission/')
    parser.add_argument('--no-plots', action='store_true', help='Disable graph plotting (useful for faster execution)')

    # Parse the arguments passed from the terminal
    args = parser.parse_args()

    # 1. Handle Training
    if args.train:
        print(f"\n{'='*40}\nSTARTING TRAINING MODE\n{'='*40}")
        trainClassifier()           # Train the PCA+ANN Classifier
        trainDetector()             # Train the CNN Detector
        print("TRAINING COMPLETE")

    # 2. Handle Testing / Submission
    if args.test:
        print(f"\n{'='*40}\nSTARTING TESTING MODE\n{'='*40}")
        should_plot = not args.no_plots
        submissionPipeline(targetFolder='./Dataset/Submission/', isSubmission=True, plotGraphs=should_plot)

    # 3. Handle case where no flags are provided
    if not args.train and not args.test:
        parser.print_help()