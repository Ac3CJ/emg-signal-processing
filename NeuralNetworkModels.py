import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

import ControllerConfiguration as Config

# GPU Optimization Settings
torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels
if hasattr(torch.backends, 'xpu'):
    torch.backends.xpu.benchmark = True  # Auto-tune Intel GPU kernels

# ====================================================================================
# ============================== RCNN MODEL DEFINITION ===============================
# ====================================================================================

class ContinuousEMGDataset(Dataset):
    """
    Memory-efficient dataset that slices sliding windows on-the-fly from continuous EMG arrays.

    Args:
        continuous_X (np.ndarray): Shape (num_channels, total_samples).
        continuous_y (np.ndarray): Shape (total_samples, num_outputs).
        window_size (int): Number of samples per window.
        step_size (int): Number of samples between consecutive windows.
        active_channels (list[int] | None): Optional subset of channels to use.
    """

    def __init__(self, continuous_X, continuous_y, window_size, step_size, active_channels=None):
        if not isinstance(continuous_X, np.ndarray):
            raise TypeError("continuous_X must be a numpy.ndarray")
        if not isinstance(continuous_y, np.ndarray):
            raise TypeError("continuous_y must be a numpy.ndarray")

        if continuous_X.ndim != 2:
            raise ValueError(
                f"continuous_X must have shape (num_channels, total_samples), got {continuous_X.shape}"
            )
        if continuous_y.ndim != 2:
            raise ValueError(
                f"continuous_y must have shape (total_samples, num_outputs), got {continuous_y.shape}"
            )

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        num_channels, total_samples = continuous_X.shape
        if continuous_y.shape[0] != total_samples:
            raise ValueError(
                "continuous_X and continuous_y must share the same total_samples dimension. "
                f"Got X total_samples={total_samples}, y total_samples={continuous_y.shape[0]}"
            )
        if window_size > total_samples:
            raise ValueError(
                f"window_size ({window_size}) cannot be larger than total_samples ({total_samples})"
            )

        if active_channels is None:
            channel_indices = np.arange(num_channels, dtype=np.int64)
        else:
            channel_indices = np.asarray(active_channels, dtype=np.int64)
            if channel_indices.ndim != 1 or channel_indices.size == 0:
                raise ValueError("active_channels must be a non-empty 1D list/array of channel indices")
            if np.any(channel_indices < 0) or np.any(channel_indices >= num_channels):
                raise ValueError(
                    f"active_channels indices must be in range [0, {num_channels - 1}]"
                )

        self.continuous_X = continuous_X
        self.continuous_y = continuous_y
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.active_channels = channel_indices
        self.total_samples = total_samples

        # Number of valid sliding windows without precomputing any window tensor.
        self.num_windows = ((self.total_samples - self.window_size) // self.step_size) + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = int(index.item())

        if index < 0:
            index += self.num_windows
        if index < 0 or index >= self.num_windows:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_windows}")

        start_idx = index * self.step_size
        end_idx = start_idx + self.window_size

        # Output feature shape remains (Channels, Length) for Conv1d.
        window_X = self.continuous_X[self.active_channels, start_idx:end_idx]
        window_y = self.continuous_y[end_idx - 1]

        return (
            torch.as_tensor(window_X, dtype=torch.float32),
            torch.as_tensor(window_y, dtype=torch.float32),
        )

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (Jiang et al., 2024).
    Dynamically weights the importance of different feature maps without reducing dimensionality,
    effectively teaching the network to ignore 'cross-talk' channels during specific movements.
    """
    def __init__(self, kernel_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)
        y = self.avg_pool(x) # Shape: (Batch, Channels, 1)
        
        # ECA applies a 1D conv across the channels
        y = y.transpose(-1, -2) # Shape: (Batch, 1, Channels)
        y = self.conv(y)
        y = y.transpose(-1, -2) # Shape: (Batch, Channels, 1)
        
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
class MultiscaleInception1D(nn.Module):
    """
    Multiscale Fusion (Jiang et al., 2024).
    Applies 3 different kernel sizes simultaneously to capture both rapid spikes
    and slow muscle holds.
    """
    def __init__(self, in_channels, out_channels_per_branch):
        super(MultiscaleInception1D, self).__init__()
        # Branch 1: Small temporal window (fast twitches)
        self.branch1 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        # Branch 2: Medium temporal window
        self.branch2 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=7, padding=3)
        # Branch 3: Large temporal window (sustained contractions)
        self.branch3 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=11, padding=5)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.branch1(x))
        out2 = self.relu(self.branch2(x))
        out3 = self.relu(self.branch3(x))
        # Concatenate along the channel dimension
        return torch.cat([out1, out2, out3], dim=1)

class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism for weighted sequence aggregation.
    Takes the full LSTM output sequence and learns which timesteps are most important.
    Outputs a context vector that dynamically aggregates information across the sequence.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        # Linear layer to compute attention scores for each timestep
        self.attention_layer = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_out):
        # lstm_out shape: (Batch, Seq_Len, Hidden_Size)
        
        # Calculate attention scores: (Batch, Seq_Len, 1)
        attention_scores = self.attention_layer(lstm_out)
        
        # Squeeze to (Batch, Seq_Len)
        attention_scores = attention_scores.squeeze(-1)
        
        # Apply softmax to get attention weights across the sequence
        attention_weights = self.softmax(attention_scores)
        
        # Reshape for broadcasting: (Batch, Seq_Len, 1)
        attention_weights = attention_weights.unsqueeze(-1)
        
        # Apply attention weights to each timestep
        # (Batch, Seq_Len, Hidden_Size) * (Batch, Seq_Len, 1) -> (Batch, Seq_Len, Hidden_Size)
        weighted_out = lstm_out * attention_weights
        
        # Sum across time dimension to get context vector
        context_vector = weighted_out.sum(dim=1)  # (Batch, Hidden_Size)
        
        return context_vector

class ShoulderRCNN(nn.Module):
    def __init__(self, num_channels=Config.NUM_CHANNELS, num_outputs=Config.NUM_OUTPUTS):
        super(ShoulderRCNN, self).__init__()

        # --- 1. Multiscale Spatial Feature Extraction (Inception) ---
        self.inception1 = MultiscaleInception1D(in_channels=num_channels, out_channels_per_branch=16)
        self.pool1 = nn.MaxPool1d(kernel_size=5) 
        self.eca1 = ECABlock(kernel_size=3)
        self.drop1 = nn.Dropout(p=0.2)

        # Layer 2
        self.inception2 = MultiscaleInception1D(in_channels=48, out_channels_per_branch=32)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        self.eca2 = ECABlock(kernel_size=3)
        self.drop2 = nn.Dropout(p=0.2)

        # --- 3. Temporal Sequence Learning (RNN/LSTM) ---
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, num_layers=1, batch_first=True)

        # --- 3.5. Temporal Attention Mechanism ---
        # self.temporal_attention = TemporalAttention(hidden_size=64)

        # --- 4. Kinematic Regression Head (DECOUPLED HEADS) ---
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        # Four completely independent linear layers for each degree of freedom
        self.fc_yaw = nn.Linear(32, 1)
        self.fc_pitch = nn.Linear(32, 1)
        self.fc_roll = nn.Linear(32, 1)
        self.fc_elbow = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)

        # Multiscale + Attention Block 1
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.eca1(x)
        x = self.drop1(x)

        # Multiscale + Attention Block 2
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.eca2(x)
        x = self.drop2(x)

        # Prepare for LSTM
        x = x.permute(0, 2, 1) # Shape: (Batch, Seq_Len, Channels)

        # LSTM Temporal processing
        lstm_out, _ = self.lstm(x)

        # Temporal Attention Mechanism: Dynamically weight the sequence
        # context_vector = self.temporal_attention(lstm_out)
        last_time_step = lstm_out[:, -1, :]

        # Pass through the shared dense layer
        # out = self.fc1(context_vector)
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.drop3(out)

        # Pass through the independent, decoupled heads
        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        elbow = self.fc_elbow(out)

        # Concatenate them back together to match the (Batch, 4) target tensor shape
        return torch.cat([yaw, pitch, roll, elbow], dim=1)