import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import DataPreparation
import ControllerConfiguration as Config

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
        last_time_step = lstm_out[:, -1, :] 
        
        # Pass through the shared dense layer
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
    Trains the PyTorch RCNN model with early stopping.
    """
    # Convert numpy arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    # Initialize Model, Loss (MSE for regression), and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{'-'*10} SYSTEM CHECK {'-'*10}]")
    print(f"Training on device: {device}")
    print(f"Training Samples: {len(X_train)} | Validation Samples: {len(X_val)}")
    
    model = ShoulderRCNN(num_channels=X_train.shape[1], num_outputs=y_train.shape[1]).to(device)

    optimizer_criterion = nn.MSELoss() 
    tracker_criterion = nn.L1Loss() # L1 Loss is exactly Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # History tracking for our visual graph
    history_train_mae = []
    history_val_mae = []

    print(f"\n[{'-'*10} STARTING TRAINING {'-'*10}]")
    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_mae = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = optimizer_criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            # Track the MAE for visuals
            mae = tracker_criterion(predictions, batch_y)
            train_mae += mae.item() * batch_X.size(0)
            
        train_mae /= len(train_loader.dataset)
        history_train_mae.append(train_mae)
        
        # --- Validation Phase ---
        model.eval()
        val_mae = 0.0
        val_mse_loss = 0.0 # We still use MSE to determine the "Best Model"
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                
                val_mse_loss += optimizer_criterion(predictions, batch_y).item() * batch_X.size(0)
                val_mae += tracker_criterion(predictions, batch_y).item() * batch_X.size(0)
                
        val_mse_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        history_val_mae.append(val_mae)
        
        # --- Dynamic Console Print ---
        status = ""
        if val_mse_loss < best_val_loss:
            best_val_loss = val_mse_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            status = f"--> Saved Best Model"
        else:
            epochs_no_improve += 1
            status = f"--> No improvement ({epochs_no_improve}/{Config.PATIENCE})"
            
        # Print the MAE (Degrees) to the terminal so it makes sense to human eyes!
        print(f"Epoch {epoch+1:02d}/{Config.EPOCHS} | Train Error: {train_mae:6.2f}° | Val Error: {val_mae:6.2f}° {status}")
        
        if epochs_no_improve >= Config.PATIENCE:
            print(f"\n[STOP] Early stopping triggered.")
            break
                
    print(f"\n[{'-'*10} TRAINING COMPLETE {'-'*10}]")
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
        
        # 3. Train the model for real (using epochs=50 and patience=10)
        trained_model = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)