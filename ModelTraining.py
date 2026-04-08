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

# ====================================================================================
# ============================== TRAINING PIPELINE ===================================
# ====================================================================================

def save_dataset_distribution(y_train, y_val, output_file='training_dataset_distribution.txt'):
    """
    Calculates and saves the dataset distribution across movements to a text file.
    
    Args:
        y_train (np.ndarray): Training targets [num_samples, 4]
        y_val (np.ndarray): Validation targets [num_samples, 4]
        output_file (str): Path to save the distribution file
    """
    movement_counts_train = {}
    movement_counts_val = {}
    
    # Count samples per movement in training set
    for class_idx, target_angles in Config.TARGET_MAPPING.items():
        target_arr = np.array(target_angles, dtype=np.float32)
        matches = np.all(np.isclose(y_train, target_arr, atol=0.1), axis=1)
        movement_counts_train[class_idx] = np.sum(matches)
    
    # Count samples per movement in validation set
    for class_idx, target_angles in Config.TARGET_MAPPING.items():
        target_arr = np.array(target_angles, dtype=np.float32)
        matches = np.all(np.isclose(y_val, target_arr, atol=0.1), axis=1)
        movement_counts_val[class_idx] = np.sum(matches)
    
    # Write to file
    total_train = np.sum(list(movement_counts_train.values()))
    total_val = np.sum(list(movement_counts_val.values()))
    
    with open(output_file, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("DATASET DISTRIBUTION ACROSS MOVEMENTS\n")
        f.write("=" * 65 + "\n\n")
        
        f.write(f"{'Movement':<20} {'Index':<8} {'Train':<12} {'Val':<12} {'Total':<8}\n")
        f.write("-" * 65 + "\n")
        
        for class_idx in sorted(Config.TARGET_MAPPING.keys()):
            movement_name = Config.MOVEMENT_NAMES.get(class_idx, f"Movement {class_idx}")
            train_count = movement_counts_train.get(class_idx, 0)
            val_count = movement_counts_val.get(class_idx, 0)
            total_count = train_count + val_count
            
            f.write(f"{movement_name:<20} {class_idx:<8} {train_count:<12} {val_count:<12} {total_count:<8}\n")
        
        f.write("-" * 65 + "\n")
        f.write(f"{'TOTAL':<20} {'':<8} {total_train:<12} {total_val:<12} {total_train + total_val:<8}\n")
        f.write("=" * 65 + "\n\n")
        
        f.write(f"Training samples: {total_train}\n")
        f.write(f"Validation samples: {total_val}\n")
        f.write(f"Total samples: {total_train + total_val}\n")
        f.write(f"Train/Val split: {100*total_train/(total_train+total_val):.1f}% / {100*total_val/(total_train+total_val):.1f}%\n")
    
    print(f"\n[Dataset Distribution] Saved to '{output_file}'")

def plot_training_history(train_losses, val_losses, best_epoch=None):
    """Generates and saves a learning curve plot after training."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss (RMSE)', color='tab:blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss (RMSE)', color='tab:orange', linewidth=2)

    # Add vertical dotted line at best epoch if provided
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='tab:red', linestyle=':', linewidth=2, 
                   label=f'Best Model (Epoch {best_epoch + 1})', alpha=0.8)

    plt.title('RCNN Regression Learning Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Root Mean Squared Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_loss_curve.png', dpi=150)
    print("\n[Visuals] Learning curve saved as 'training_loss_curve.png'.")
    # plt.show()

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
        train_dataset, batch_size=batch_size, shuffle=False, 
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
    best_epoch = None
    
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
            best_epoch = epoch
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
    plot_training_history(history_train_mae, history_val_mae, best_epoch=best_epoch)
    save_dataset_distribution(y_train, y_val, output_file='training_dataset_distribution.txt')
    
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    return model

# ====================================================================================
# ============================== TRANSFER LEARNING ==================================
# ====================================================================================

def freeze_backbone(model, num_unfreeze_layers=2):
    """
    Freezes all layers except the last `num_unfreeze_layers` layers.
    The unfrozen layers are typically the regression heads and final dense layer.
    
    Args:
        model (nn.Module): The ShoulderRCNN model
        num_unfreeze_layers (int): Number of final layers to keep trainable
    """
    # Get all named parameters
    named_params = list(model.named_parameters())
    
    # Freeze all but the last num_unfreeze_layers layers
    num_to_freeze = len(named_params) - num_unfreeze_layers
    
    for i, (name, param) in enumerate(named_params):
        if i < num_to_freeze:
            param.requires_grad = False
            # print(f"FROZEN: {name}")
        else:
            param.requires_grad = True
            # print(f"TRAINABLE: {name}")
    
    print(f"\n[Transfer Learning] Froze {num_to_freeze} layers, unfroze {num_unfreeze_layers} layers")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

def train_transfer_learning(X_train, y_train, X_val, y_val, pretrained_model_path, 
                           batch_size=None, epochs=None, patience=None, freeze_layers=True):
    """
    Fine-tunes a pretrained model on new collected data.
    
    Args:
        X_train, y_train: Training data (usually from collected_data/training)
        X_val, y_val: Validation data (could be combined from multiple sources)
        pretrained_model_path (str): Path to pretrained model weights
        batch_size: Training batch size (uses Config.TRANSFER_LEARNING_BATCH_SIZE if None)
        epochs: Number of epochs (uses Config.TRANSFER_LEARNING_EPOCHS if None)
        patience: Early stopping patience (uses Config.TRANSFER_LEARNING_PATIENCE if None)
        freeze_layers (bool): Whether to freeze backbone layers
    """
    # Use transfer learning config if not specified
    if batch_size is None:
        batch_size = Config.TRANSFER_LEARNING_BATCH_SIZE
    if epochs is None:
        epochs = Config.TRANSFER_LEARNING_EPOCHS
    if patience is None:
        patience = Config.TRANSFER_LEARNING_PATIENCE
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
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
    
    print(f"\n[{'-'*10} TRANSFER LEARNING SETUP {'-'*10}]")
    print(f"Loading pretrained model from: {pretrained_model_path}")
    print(f"Training on device: {device}")
    print(f"Batch size: {batch_size} | Epochs: {epochs} | Patience: {patience}")
    print(f"Learning rate: {Config.TRANSFER_LEARNING_LEARNING_RATE}")
    
    # Load pretrained model
    model = ShoulderRCNN(num_channels=X_train.shape[1], num_outputs=y_train.shape[1]).to(device)
    try:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"✓ Successfully loaded pretrained weights from {pretrained_model_path}")
    except Exception as e:
        print(f"✗ ERROR loading pretrained model: {e}")
        return None
    
    # Freeze backbone if requested
    if freeze_layers and Config.FREEZE_BACKBONE_LAYERS:
        freeze_backbone(model, num_unfreeze_layers=Config.NUM_LAYERS_TO_UNFREEZE)
    else:
        print("[Transfer Learning] Training all layers (no freezing)")
    
    optimizer_criterion = nn.MSELoss()
    tracker_criterion = nn.L1Loss()
    
    # Only optimize the parameters that require gradients
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=Config.TRANSFER_LEARNING_LEARNING_RATE
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE,
        min_lr=1e-7
    )
    
    # AMP Setup
    use_amp = device_type in ["xpu", "cuda"]
    amp_dtype = torch.bfloat16 if device_type == "xpu" else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history_train_mae, history_val_mae = [], []
    best_epoch = None
    
    training_start_time = time.time()
    
    print(f"\n[{'-'*10} STARTING TRANSFER LEARNING {'-'*10}]")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_mae = 0.0
        accum_step = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            if accum_step == 0:
                optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                predictions = model(batch_X)
                loss = optimizer_criterion(predictions, batch_y)
            
            scaled_loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            accum_step += 1
            
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
        
        # Validation Phase
        model.eval()
        val_mae, val_mse_loss = 0.0, 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                
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
            best_epoch = epoch
            torch.save(model.state_dict(), Config.TRANSFER_LEARNING_MODEL_SAVE_PATH)
            status = f"--> Saved Best Model"
        else:
            epochs_no_improve += 1
            status = f"--> No improvement ({epochs_no_improve}/{patience})"
        
        scheduler.step(val_mse_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Error: {train_mae:6.2f}° | Val Error: {val_mae:6.2f}° | Time: {epoch_time:6.2f}s {status}")
        
        if epochs_no_improve >= patience:
            print(f"\n[STOP] Early stopping triggered.")
            break
    
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    avg_epoch_time = total_training_time / (epoch + 1)
    
    print(f"\n[{'-'*10} TRANSFER LEARNING COMPLETE {'-'*10}]")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print(f"Best model saved to: {Config.TRANSFER_LEARNING_MODEL_SAVE_PATH}")
    plot_training_history(history_train_mae, history_val_mae, best_epoch=best_epoch)
    save_dataset_distribution(y_train, y_val, output_file='transfer_learning_dataset_distribution.txt')
    
    model.load_state_dict(torch.load(Config.TRANSFER_LEARNING_MODEL_SAVE_PATH))
    return model

# ====================================================================================
# ============================== DEBUG/DUMMY TEST ====================================
# ====================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ShoulderRCNN model")
    parser.add_argument('--mode', type=str, choices=['loso', 'transfer', 'standard'], default='loso',
                       help='Training mode: loso (Leave-One-Subject-Out), transfer (Transfer Learning), or standard (80-20 split)')
    parser.add_argument('--pretrained', type=str, default=Config.MODEL_SAVE_PATH,
                       help='Path to pretrained model (for transfer learning)')
    parser.add_argument('--freeze', action='store_true', default=False,
                       help='Freeze backbone layers during transfer learning')
    args = parser.parse_args()
    
    # ====================================================================================
    # LEAVE-ONE-SUBJECT-OUT (LOSO) TRAINING
    # ====================================================================================
    if args.mode == 'loso':
        print("Loading dataset with Leave-One-Subject-Out Validation...")
        
        # 1. Load training data (subjects 1-7)
        print("\n[LOSO Fold] Loading TRAINING data (Subjects 1-7)...")
        X_train, y_train = DataPreparation.load_and_prepare_dataset(
            base_path=Config.SECONDARY_DATA_PATH, 
            include_subjects=list(range(1, 8)),
            include_noise_aug=True,
        )
        
        # 2. Load validation data (subject 8)
        print("\n[LOSO Fold] Loading VALIDATION data (Subject 8)...")
        X_val, y_val = DataPreparation.load_and_prepare_dataset(
            base_path=Config.SECONDARY_DATA_PATH, 
            include_subjects=[8],
            include_noise_aug=False,
        )
        
        if len(X_train) == 0 or len(X_val) == 0:
            print("ERROR: No data loaded. Check your file paths!")
        else:
            
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
    
    # ====================================================================================
    # TRANSFER LEARNING TRAINING
    # ====================================================================================
    elif args.mode == 'transfer':
        print("Starting Transfer Learning Training...")
        print(f"Pretrained model: {args.pretrained}")
        print(f"Freeze backbone: {args.freeze}")
        
        # 1. Load collected training data
        print("\n[Transfer Learning] Loading TRAINING data from collected_data/training...")
        X_train, y_train = DataPreparation.load_collected_data(
            folder_path='./collected_data/training',
            augment=True,
            include_noise_aug=True,
        )
        
        if X_train is None or len(X_train) == 0:
            print("\n✗ ERROR: No training data found in ./collected_data/training/")
            print("STOPPING: Transfer learning requires training data. Please add .mat files to ./collected_data/training/")
            exit(1)
        
        # 2. Load collected validation data (if available)
        print("\n[Transfer Learning] Checking for validation data in collected_data/validation...")
        X_val_collected, y_val_collected = DataPreparation.load_collected_data(
            folder_path='./collected_data/validation',
            augment=True,
            include_noise_aug=False,
        )
        
        # 3. Always load secondary data validation (from subject 8 LOSO fold)
        print("\n[Transfer Learning] Loading VALIDATION data from secondary_data (Subject 8)...")
        X_val_secondary, y_val_secondary = DataPreparation.load_and_prepare_dataset(
            base_path=Config.BASE_DATA_PATH,
            include_subjects=[8],
            include_noise_aug=False,
        )
        
        if X_val_secondary is None or len(X_val_secondary) == 0:
            print("✗ ERROR: Could not load secondary validation data!")
            exit(1)
        
        # 4. Combine validation datasets
        if X_val_collected is not None and len(X_val_collected) > 0:
            print(f"\n[Transfer Learning] Combining validation sets:")
            print(f"  - Collected: {len(X_val_collected)} samples")
            print(f"  - Secondary: {len(X_val_secondary)} samples")
            X_val = np.concatenate([X_val_secondary, X_val_collected], axis=0)
            y_val = np.concatenate([y_val_secondary, y_val_collected], axis=0)
            print(f"  - Total: {len(X_val)} samples")
        else:
            print(f"\n[Transfer Learning] No collected validation data found. Using only secondary data validation.")
            print(f"  - Secondary: {len(X_val_secondary)} samples")
            X_val = X_val_secondary
            y_val = y_val_secondary
        
        # 5. Print dataset distribution
        print(f"\n[{'-'*10} TRANSFER LEARNING DATASET DISTRIBUTION {'-'*10}]")
        print(f"{'Class Name':<18} | {'Train':<8} | {'Validation':<8}")
        print("-" * 42)
        
        total_train = 0
        total_val = 0
        
        for class_idx, target_angles in Config.TARGET_MAPPING.items():
            target_vec = np.array(target_angles, dtype=np.float32)
            
            train_count = np.sum(np.all(y_train == target_vec, axis=1))
            val_count = np.sum(np.all(y_val == target_vec, axis=1))
            
            total_train += train_count
            total_val += val_count
            
            class_name = f"Movement {class_idx}" if class_idx != 9 else "Rest (Class 9)"
            print(f"{class_name:<18} | {train_count:<8} | {val_count:<8}")
        
        print("-" * 42)
        print(f"{'TOTAL':<18} | {total_train:<8} | {total_val:<8}\n")
        
        print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples...")
        
        # 6. Train with transfer learning
        trained_model = train_transfer_learning(
            X_train, y_train, X_val, y_val,
            pretrained_model_path=args.pretrained,
            freeze_layers=args.freeze
        )
    
    # ====================================================================================
    # STANDARD 80-20 TRAINING
    # ====================================================================================
    elif args.mode == 'standard':
        print("Loading dataset with Standard 80-20 Training-Validation Split...")
        
        # 1. Load all data from all subjects
        print("\n[Standard Mode] Loading ALL data (all subjects)...")
        X_all, y_all = DataPreparation.load_and_prepare_dataset(
            base_path=Config.BASE_DATA_PATH, 
            include_subjects=list(range(1, 9)),
            include_noise_aug=True,
        )
        
        if len(X_all) == 0:
            print("ERROR: No data loaded. Check your file paths!")
        else:
            # 2. Split 80-20
            print("\n[Standard Mode] Performing 80-20 train-validation split...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, shuffle=False
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
            
            # 3. Train the model
            trained_model = train_model(X_train, y_train, X_val, y_val, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, patience=Config.PATIENCE)