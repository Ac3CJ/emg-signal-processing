import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

import DataPreparation
import ControllerConfiguration

# ====================================================================================
# ============================== RCNN MODEL DEFINITION ===============================
# ====================================================================================

class ShoulderRCNN(nn.Module):
    def __init__(self, num_channels=8, num_outputs=4):
        super(ShoulderRCNN, self).__init__()
        
        # --- 1. Spatial Feature Extraction (CNN) ---
        # PyTorch Conv1d expects (Batch, Channels, Sequence_Length)
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=10)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.2)
        
        # --- 2. Temporal Sequence Learning (RNN/LSTM) ---
        # batch_first=True makes input/output tensors structured as (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.drop_lstm = nn.Dropout(p=0.3)
        
        # --- 3. Continuous Regression Output (Dense) ---
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        # No activation function on the final layer for continuous regression
        self.fc2 = nn.Linear(32, num_outputs) 

    def forward(self, x):
        # x is originally (Batch, TimeSteps=500, Channels=8)
        
        # CNN Block 1
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # CNN Block 2
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Prepare for LSTM
        # LSTM wants (Batch, TimeSteps, Features), so we permute back
        x = x.permute(0, 2, 1)
        
        # LSTM Block
        # lstm_out contains all hidden states, (hn, cn) contains the final state
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We only care about the very last time step of the LSTM output for our prediction
        last_time_step = lstm_out[:, -1, :]
        x = self.drop_lstm(last_time_step)
        
        # Dense Output Block
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # Linear continuous output
        
        return x

# ====================================================================================
# ============================== TRAINING PIPELINE ===================================
# ====================================================================================

def train_model(X_train, y_train, X_val, y_val, batch_size=64, epochs=50, patience=10):
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss (MSE for regression), and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = ShoulderRCNN(num_channels=X_train.shape[2], num_outputs=y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("Starting training phase...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_shoulder_rcnn.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
                
    print("Training complete! Best model saved as 'best_shoulder_rcnn.pth'")
    
    # Load the best weights before returning
    model.load_state_dict(torch.load('best_shoulder_rcnn.pth'))
    return model

# ====================================================================================
# ============================== DEBUG/DUMMY TEST ====================================
# ====================================================================================

if __name__ == "__main__":
    print("Loading actual dataset from secondary files...")
    
    # 1. Load the real data using our new script
    X_full, y_full = DataPreparation.load_and_prepare_dataset(base_path='./secondary_data')
    
    if len(X_full) == 0:
        print("ERROR: No data loaded. Check your file paths!")
    else:
        # 2. Shuffle and split into Training (80%) and Validation (20%)
        # random_state ensures reproducibility 
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples...")
        
        # 3. Train the model for real (using epochs=50 and patience=10)
        trained_model = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)