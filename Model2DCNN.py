"""
Standalone comparison model. Not integrated into main training pipeline.
Topology preservation assumption is weak for distributed shoulder electrodes
— see grill session 2026-05-01.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

import ControllerConfiguration as Config
from NeuralNetworkModels import ContinuousEMGDataset


# ====================================================================================
# ============================== 2D CNN MODEL DEFINITION =============================
# ====================================================================================

class EMG2DCNN(nn.Module):
    def __init__(self, num_channels=Config.NUM_CHANNELS, window_size=Config.WINDOW_SIZE, num_outputs=Config.NUM_OUTPUTS):
        super(EMG2DCNN, self).__init__()

        # Input: (batch, 1, num_channels, window_size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # Collapse spatial dims to (1, 1)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, num_outputs)

    def forward(self, x):
        # x: (batch, 1, C, W) from EMG2DDataset, or (batch, C, W) from validator — normalise to 4D
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)        # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 64)
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        return self.fc_out(x)    # (batch, num_outputs)


# ====================================================================================
# ============================== 2D DATASET WRAPPER ==================================
# ====================================================================================

class EMG2DDataset(Dataset):
    """
    Wraps ContinuousEMGDataset to return windows as (1, C, W) tensors
    suitable for Conv2d input, instead of (C, W).
    """

    def __init__(self, continuous_X, continuous_y, window_size, step_size,
                 active_channels=None, segment_bounds=None):
        self._inner = ContinuousEMGDataset(
            continuous_X, continuous_y, window_size, step_size,
            active_channels=active_channels, segment_bounds=segment_bounds,
        )

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, index):
        window_X, window_y = self._inner[index]  # (C, W), (num_outputs,)
        return window_X.unsqueeze(0), window_y   # (1, C, W), (num_outputs,)


# ====================================================================================
# ============================== SHAPE VERIFICATION ==================================
# ====================================================================================

if __name__ == '__main__':
    import DataPreparation

    print('Loading dataset (augment=False)...')
    continuous_X, continuous_y, segment_bounds, _ = DataPreparation.load_and_prepare_dataset(augment=False)

    if continuous_X is None:
        print('No data loaded — check data paths in ControllerConfiguration.')
        exit(1)

    dataset = EMG2DDataset(
        continuous_X, continuous_y,
        window_size=Config.WINDOW_SIZE,
        step_size=Config.INCREMENT,
        segment_bounds=segment_bounds,
    )
    print(f'Dataset size: {len(dataset)} windows')

    sample_X, sample_y = dataset[0]
    print(f'Sample window shape: {tuple(sample_X.shape)} (expected: (1, {Config.NUM_CHANNELS}, {Config.WINDOW_SIZE}))')
    print(f'Sample target shape: {tuple(sample_y.shape)} (expected: ({Config.NUM_OUTPUTS},))')

    model = EMG2DCNN(
        num_channels=Config.NUM_CHANNELS,
        window_size=Config.WINDOW_SIZE,
        num_outputs=Config.NUM_OUTPUTS,
    )

    batch = sample_X.unsqueeze(0)  # (1, 1, C, W)
    with torch.no_grad():
        output = model(batch)
    print(f'Forward pass output shape: {tuple(output.shape)} (expected: (1, {Config.NUM_OUTPUTS}))')

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model parameter count: {param_count:,}')
