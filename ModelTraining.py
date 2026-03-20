import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

import DataPreparation
import ControllerConfiguration as Config

# GPU Optimization Settings
torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels
if hasattr(torch.backends, 'xpu'):
    torch.backends.xpu.benchmark = True  # Auto-tune Intel GPU kernels

# ====================================================================================
# ============================== CUSTOM LOSS FUNCTION ================================
# ====================================================================================

class AsymmetricKinematicLoss(nn.Module):
    def __init__(self, phantom_pitch_penalty=5.0):
        super(AsymmetricKinematicLoss, self).__init__()
        self.penalty = phantom_pitch_penalty
        # reduction='none' allows us to modify specific elements before averaging
        self.mse = nn.MSELoss(reduction='none') 

    def forward(self, predictions, targets):
        # 1. Calculate standard MSE for all 4 DOFs: [Yaw, Pitch, Roll, Elbow]
        base_loss = self.mse(predictions, targets)

        # 2. Create a mask: It equals 1.0 ONLY when the Target Pitch is exactly 0.0
        # (This isolates pure Flexion, Hyperextension, and Rest)
        zero_pitch_mask = (targets[:, 1] == 0.0).float()

        # 3. Apply the heavy penalty multiplier ONLY to the Pitch error
        pitch_loss = base_loss[:, 1] * (1.0 + (self.penalty - 1.0) * zero_pitch_mask)

        # 4. Replace the original Pitch loss with the newly penalized one
        base_loss[:, 1] = pitch_loss

        # 5. Return the final scalar loss so the network can backpropagate
        return base_loss.mean()

# ====================================================================================
# ============================== RCNN MODEL DEFINITION ===============================
# ====================================================================================

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
        self.drop1 = nn.Dropout1d(p=0.2)
        
        # Layer 2
        self.inception2 = MultiscaleInception1D(in_channels=48, out_channels_per_branch=32)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        self.eca2 = ECABlock(kernel_size=3)
        self.drop2 = nn.Dropout1d(p=0.2)
        
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

# ====================================================================================
# ============================== TRAINING PIPELINE ===================================
# ====================================================================================

def plot_training_history(train_losses, val_losses):
    """Generates and saves a learning curve plot after training."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss (RMSE)', color='tab:blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss (RMSE)', color='tab:orange', linewidth=2)
    
    plt.title('RCNN Regression Learning Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Root Mean Squared Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_loss_curve.png', dpi=150)
    print("\n[Visuals] Learning curve saved as 'training_loss_curve.png'.")
    plt.show()

def train_model(X_train, y_train, X_val, y_val, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, patience=Config.PATIENCE):
    """
    Trains the PyTorch RCNN model using hardware-accelerated optimizations.
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # --- OPTIMIZATION 1: Multi-threaded Data Loading ---
    # Parallel workers keep GPU fed with data while computation happens
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=Config.NUM_DATA_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=Config.PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=Config.NUM_DATA_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=Config.PREFETCH_FACTOR
    )
    
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    device_type = "xpu" if torch.xpu.is_available() else "cpu"
        
    print(f"\n[{'-'*10} SYSTEM CHECK {'-'*10}]")
    print(f"Training on device: {device}")
    print(f"Batch size: {batch_size} | Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Data workers: {Config.NUM_DATA_WORKERS} | Prefetch: {Config.PREFETCH_FACTOR}")
    print(f"Effective batch size: {batch_size * Config.GRADIENT_ACCUMULATION_STEPS}")
    
    model = ShoulderRCNN(num_channels=X_train.shape[1], num_outputs=y_train.shape[1]).to(device)

    # optimizer_criterion = AsymmetricKinematicLoss(phantom_pitch_penalty=5.0) 
    optimizer_criterion = nn.MSELoss()
    tracker_criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # --- OPTIMIZATION 1.5: Learning Rate Scheduler ---
    # Reduces learning rate by factor when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE,
        min_lr=1e-6
    )
    
    # --- OPTIMIZATION 2: Initialize AMP Scaler ---
    # Enable AMP only if hardware is GPU (XPU or CUDA)
    use_amp = device_type in ["xpu", "cuda"]
    
    amp_dtype = torch.bfloat16 if device_type == "xpu" else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history_train_mae, history_val_mae = [], []
    
    # Start training timer
    training_start_time = time.time()

    print(f"\n[{'-'*10} STARTING TRAINING {'-'*10}]")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_mae = 0.0
        accum_step = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            # Zero gradients only at the start of accumulation
            if accum_step == 0:
                optimizer.zero_grad(set_to_none=True) 
            
            # --- OPTIMIZATION 3: Autocast for Mixed Precision ---
            # Pass the smart dtype into autocast
            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                predictions = model(batch_X)
                loss = optimizer_criterion(predictions, batch_y)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            
            # Only use the scaler if we are on float16 (NVIDIA). Otherwise, normal backward pass!
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            accum_step += 1
            
            # Optimizer step only after accumulating enough gradients or at epoch end
            if accum_step == Config.GRADIENT_ACCUMULATION_STEPS or batch_idx == len(train_loader) - 1:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                accum_step = 0
            
            mae = tracker_criterion(predictions, batch_y)
            train_mae += mae.item() * batch_X.size(0)
            
        train_mae /= len(train_loader.dataset)
        history_train_mae.append(train_mae)
        
        # --- Validation Phase ---
        model.eval()
        val_mae, val_mse_loss = 0.0, 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                
                # We also use AMP for validation to speed it up!
                with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                    predictions = model(batch_X)
                    val_mse_loss += optimizer_criterion(predictions, batch_y).item() * batch_X.size(0)
                    
                val_mae += tracker_criterion(predictions, batch_y).item() * batch_X.size(0)
                
        val_mse_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        history_val_mae.append(val_mae)
        
        status = ""
        if val_mse_loss < best_val_loss:
            best_val_loss = val_mse_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            status = f"--> Saved Best Model"
        else:
            epochs_no_improve += 1
            status = f"--> No improvement ({epochs_no_improve}/{patience})"
        
        # Update learning rate based on validation loss plateau
        scheduler.step(val_mse_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Error: {train_mae:6.2f}° | Val Error: {val_mae:6.2f}° | Time: {epoch_time:6.2f}s {status}")
        
        if epochs_no_improve >= patience:
            print(f"\n[STOP] Early stopping triggered.")
            break
    
    # Calculate and display total training time
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    avg_epoch_time = total_training_time / (epoch + 1)
                
    print(f"\n[{'-'*10} TRAINING COMPLETE {'-'*10}]")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    plot_training_history(history_train_mae, history_val_mae)
    
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    return model

# ====================================================================================
# ============================== DEBUG/DUMMY TEST ====================================
# ====================================================================================

if __name__ == "__main__":
    print("Loading actual dataset from secondary files...")
    
    # 1. Load the real data using our new script
    X_full, y_full = DataPreparation.load_and_prepare_dataset(base_path=Config.BASE_DATA_PATH)
    
    if len(X_full) == 0:
        print("ERROR: No data loaded. Check your file paths!")
    else:
        # 2. Shuffle and split into Training (80%) and Validation (20%)
        # random_state ensures reproducibility 
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=Config.TEST_SPLIT, random_state=42, shuffle=True
        )
        
        print(f"\n[{'-'*10} DATASET DISTRIBUTION {'-'*10}]")
        print(f"{'Class Name':<18} | {'Train':<8} | {'Validation':<8}")
        print("-" * 42)
        
        total_train = 0
        total_val = 0
        
        for class_idx, target_angles in Config.TARGET_MAPPING.items():
            target_vec = np.array(target_angles, dtype=np.float32)
            
            # Count how many rows exactly match this target vector
            train_count = np.sum(np.all(y_train == target_vec, axis=1))
            val_count = np.sum(np.all(y_val == target_vec, axis=1))
            
            total_train += train_count
            total_val += val_count
            
            class_name = f"Movement {class_idx}" if class_idx != 9 else "Rest (Class 9)"
            print(f"{class_name:<18} | {train_count:<8} | {val_count:<8}")
            
        print("-" * 42)
        print(f"{'TOTAL':<18} | {total_train:<8} | {total_val:<8}\n")
        
        print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples...")
        
        # 3. Train the model for real
        trained_model = train_model(X_train, y_train, X_val, y_val, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, patience=Config.PATIENCE)