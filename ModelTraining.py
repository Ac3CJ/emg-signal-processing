import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os
import re

import DataPreparation
import ControllerConfiguration as Config
import NeuralNetworkModels as NNModels
from FileRepository import DataRepository

REPOSITORY = DataRepository()

# GPU Optimization Settings
torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels
if hasattr(torch.backends, 'xpu'):
    torch.backends.xpu.benchmark = True  # Auto-tune Intel GPU kernels

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

def _is_continuous_emg_layout(X_data, y_data):
    """Checks if data is in continuous layout: X=(channels, samples), y=(samples, outputs)."""
    if not isinstance(X_data, np.ndarray) or not isinstance(y_data, np.ndarray):
        return False
    if X_data.ndim != 2 or y_data.ndim != 2:
        return False
    if X_data.shape[0] == 0 or X_data.shape[1] == 0:
        return False
    if y_data.shape[0] != X_data.shape[1] or y_data.shape[1] == 0:
        return False
    return True


def _build_data_loader(dataset, batch_size, shuffle, num_workers=None):
    """Builds DataLoader with worker settings that work for both 0 and >0 workers."""
    resolved_workers = Config.NUM_DATA_WORKERS if num_workers is None else max(0, int(num_workers))

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": resolved_workers,
        "pin_memory": True,
    }

    if resolved_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR

    return DataLoader(dataset, **loader_kwargs)


def _prepare_training_datasets(
    X_train,
    y_train,
    X_val,
    y_val,
    use_on_the_fly=False,
    window_size=None,
    step_size=None,
    active_channels=None,
):
    """Builds train/val datasets and returns metadata required by the model."""
    can_use_on_the_fly = _is_continuous_emg_layout(X_train, y_train) and _is_continuous_emg_layout(X_val, y_val)

    if use_on_the_fly and can_use_on_the_fly:
        resolved_window_size = int(window_size if window_size is not None else Config.WINDOW_SIZE)
        resolved_step_size = int(step_size if step_size is not None else Config.INCREMENT)

        train_dataset = NNModels.ContinuousEMGDataset(
            continuous_X=np.asarray(X_train, dtype=np.float32),
            continuous_y=np.asarray(y_train, dtype=np.float32),
            window_size=resolved_window_size,
            step_size=resolved_step_size,
            active_channels=active_channels,
        )
        val_dataset = NNModels.ContinuousEMGDataset(
            continuous_X=np.asarray(X_val, dtype=np.float32),
            continuous_y=np.asarray(y_val, dtype=np.float32),
            window_size=resolved_window_size,
            step_size=resolved_step_size,
            active_channels=active_channels,
        )

        num_channels = len(train_dataset.active_channels)
        num_outputs = int(np.asarray(y_train).shape[1])
        return train_dataset, val_dataset, num_channels, num_outputs, True

    if use_on_the_fly and not can_use_on_the_fly:
        print(
            "\n[On-The-Fly] Requested, but inputs are not continuous arrays "
            "(expected X=(channels,samples), y=(samples,outputs))."
        )
        print("[On-The-Fly] Falling back to pre-windowed TensorDataset for compatibility.")

    X_train_tensor = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    y_train_tensor = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    X_val_tensor = torch.as_tensor(np.asarray(X_val, dtype=np.float32))
    y_val_tensor = torch.as_tensor(np.asarray(y_val, dtype=np.float32))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    num_channels = int(X_train_tensor.shape[1])
    num_outputs = int(y_train_tensor.shape[1])
    return train_dataset, val_dataset, num_channels, num_outputs, False


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=Config.BATCH_SIZE,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
    use_on_the_fly=False,
    window_size=None,
    step_size=None,
    active_channels=None,
):
    """
    Trains the PyTorch RCNN model using hardware-accelerated optimizations.
    """
    train_dataset, val_dataset, model_num_channels, model_num_outputs, using_on_the_fly = _prepare_training_datasets(
        X_train,
        y_train,
        X_val,
        y_val,
        use_on_the_fly=use_on_the_fly,
        window_size=window_size,
        step_size=step_size,
        active_channels=active_channels,
    )

    # --- OPTIMIZATION 1: Multi-threaded Data Loading ---
    # Parallel workers keep GPU fed with data while computation happens
    loader_workers = Config.NUM_DATA_WORKERS
    if using_on_the_fly and os.name == "nt" and loader_workers > 0:
        # Windows spawns worker processes; each process loading Torch/XPU DLLs can exhaust resources.
        loader_workers = 0

    train_loader = _build_data_loader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)
    val_loader = _build_data_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)

    device, device_type = _resolve_training_device()

    print(f"\n[{'-'*10} SYSTEM CHECK {'-'*10}]")
    print(f"Training on device: {device}")
    print(f"Window slicing mode: {'on-the-fly' if using_on_the_fly else 'pre-windowed'}")
    if using_on_the_fly:
        resolved_window_size = int(window_size if window_size is not None else Config.WINDOW_SIZE)
        resolved_step_size = int(step_size if step_size is not None else Config.INCREMENT)
        print(f"Window size: {resolved_window_size} | Step size: {resolved_step_size}")
    print(f"Batch size: {batch_size} | Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Data workers: {loader_workers} | Prefetch: {Config.PREFETCH_FACTOR if loader_workers > 0 else 'N/A'}")
    print(f"Effective batch size: {batch_size * Config.GRADIENT_ACCUMULATION_STEPS}")

    model = NNModels.ShoulderRCNN(num_channels=model_num_channels, num_outputs=model_num_outputs).to(device)

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
                           batch_size=None, epochs=None, patience=None, freeze_layers=True,
                           use_on_the_fly=False, window_size=None, step_size=None, active_channels=None):
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
    
    train_dataset, val_dataset, model_num_channels, model_num_outputs, using_on_the_fly = _prepare_training_datasets(
        X_train,
        y_train,
        X_val,
        y_val,
        use_on_the_fly=use_on_the_fly,
        window_size=window_size,
        step_size=step_size,
        active_channels=active_channels,
    )

    loader_workers = Config.NUM_DATA_WORKERS
    if using_on_the_fly and os.name == "nt" and loader_workers > 0:
        # Windows spawns worker processes; each process loading Torch/XPU DLLs can exhaust resources.
        loader_workers = 0

    train_loader = _build_data_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers)
    val_loader = _build_data_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)
    
    device, device_type = _resolve_training_device()
    
    print(f"\n[{'-'*10} TRANSFER LEARNING SETUP {'-'*10}]")
    print(f"Loading pretrained model from: {pretrained_model_path}")
    print(f"Training on device: {device}")
    print(f"Window slicing mode: {'on-the-fly' if using_on_the_fly else 'pre-windowed'}")
    if using_on_the_fly:
        resolved_window_size = int(window_size if window_size is not None else Config.WINDOW_SIZE)
        resolved_step_size = int(step_size if step_size is not None else Config.INCREMENT)
        print(f"Window size: {resolved_window_size} | Step size: {resolved_step_size}")
    print(f"Batch size: {batch_size} | Epochs: {epochs} | Patience: {patience}")
    print(f"Data workers: {loader_workers} | Prefetch: {Config.PREFETCH_FACTOR if loader_workers > 0 else 'N/A'}")
    print(f"Learning rate: {Config.TRANSFER_LEARNING_LEARNING_RATE}")
    
    # Load pretrained model
    model = NNModels.ShoulderRCNN(num_channels=model_num_channels, num_outputs=model_num_outputs).to(device)
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


def _resolve_training_device():
    """Selects the best available training device in priority order."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "xpu"

    return torch.device("cpu"), "cpu"


def _parse_participant_list(participant_list_raw):
    """
    Parses participant IDs from strings like "[1,2,4]" or "1,2,4".
    Returns empty list for empty input (control case: no collected participants).
    """
    if participant_list_raw is None:
        return None

    text = str(participant_list_raw).strip()
    
    # Handle empty input: return empty list for control case (no collected data)
    if not text or text == "[]":
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    tokens = [token for token in re.split(r"[,\s]+", text) if token]
    if len(tokens) == 0:
        # Empty after bracket removal: return empty list
        return []

    participants = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"Invalid participant token '{token}'. Use numeric IDs, e.g. [1,2,4]."
            )

        participant_id = int(token)
        if participant_id <= 0:
            raise ValueError(f"Participant IDs must be positive. Got: {participant_id}")

        if participant_id not in participants:
            participants.append(participant_id)

    return participants


def _parse_channel_list(channel_list_raw):
    """Parses channel indices from strings like "[0,1,4]" or "0,1,4"."""
    if channel_list_raw is None:
        return None

    if isinstance(channel_list_raw, (list, tuple, np.ndarray)):
        tokens = [str(token).strip() for token in channel_list_raw if str(token).strip()]
    else:
        text = str(channel_list_raw).strip()
        if not text:
            raise ValueError("--active_channels was provided but is empty.")

        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()

        tokens = [token for token in re.split(r"[,\s]+", text) if token]

    if len(tokens) == 0:
        raise ValueError("No channels were found in --active_channels.")

    channels = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"Invalid channel token '{token}'. Use channel indices, e.g. [0,1,4]."
            )

        channel_idx = int(token)
        if channel_idx < 0 or channel_idx >= Config.NUM_CHANNELS:
            raise ValueError(
                f"Channel indices must be in range [0, {Config.NUM_CHANNELS - 1}]. Got: {channel_idx}"
            )

        if channel_idx not in channels:
            channels.append(channel_idx)

    return channels


def _count_samples(X_data, y_data=None):
    """Returns logical sample count for both pre-windowed and continuous layouts."""
    if X_data is None:
        return 0

    if y_data is not None and _is_continuous_emg_layout(X_data, y_data):
        window_size = int(getattr(Config, "ON_THE_FLY_WINDOW_SIZE", Config.WINDOW_SIZE))
        step_size = int(getattr(Config, "ON_THE_FLY_STEP_SIZE", Config.INCREMENT))
        total_samples = int(X_data.shape[1])
        if total_samples < window_size:
            return 0
        return ((total_samples - window_size) // step_size) + 1

    return int(len(X_data))


def _extract_window_end_labels(y_data, window_size, step_size):
    """Returns labels sampled at each sliding-window end index for reporting/counting."""
    y_arr = np.asarray(y_data, dtype=np.float32)
    if y_arr.ndim != 2 or y_arr.shape[0] < window_size:
        num_outputs = y_arr.shape[1] if y_arr.ndim == 2 else Config.NUM_OUTPUTS
        return np.zeros((0, num_outputs), dtype=np.float32)

    end_indices = np.arange(window_size - 1, y_arr.shape[0], step_size, dtype=np.int64)
    return y_arr[end_indices]


def _concat_dataset_parts(dataset_parts):
    """Concatenates non-empty dataset tuples [(X, y), ...]."""
    valid_parts = []
    for X_part, y_part in dataset_parts:
        if X_part is None or y_part is None:
            continue
        if len(X_part) == 0 or len(y_part) == 0:
            continue
        valid_parts.append((X_part, y_part))

    if len(valid_parts) == 0:
        return None, None

    if len(valid_parts) == 1:
        return valid_parts[0]

    is_continuous = _is_continuous_emg_layout(valid_parts[0][0], valid_parts[0][1])

    if is_continuous:
        if not all(_is_continuous_emg_layout(part[0], part[1]) for part in valid_parts):
            raise ValueError("Cannot concatenate mixed continuous and pre-windowed dataset parts.")

        X_concat = np.concatenate([part[0] for part in valid_parts], axis=1)
        y_concat = np.concatenate([part[1] for part in valid_parts], axis=0)
    else:
        X_concat = np.concatenate([part[0] for part in valid_parts], axis=0)
        y_concat = np.concatenate([part[1] for part in valid_parts], axis=0)

    return X_concat, y_concat


def _print_dataset_distribution(y_train, y_val, title="DATASET DISTRIBUTION", continuous=False, window_size=None, step_size=None):
    """Prints class-wise train/validation counts for quick sanity checking."""
    if continuous:
        resolved_window_size = int(window_size if window_size is not None else Config.WINDOW_SIZE)
        resolved_step_size = int(step_size if step_size is not None else Config.INCREMENT)
        y_train_eval = _extract_window_end_labels(y_train, resolved_window_size, resolved_step_size)
        y_val_eval = _extract_window_end_labels(y_val, resolved_window_size, resolved_step_size)
    else:
        y_train_eval = np.asarray(y_train, dtype=np.float32)
        y_val_eval = np.asarray(y_val, dtype=np.float32)

    print(f"\n[{'-'*10} {title} {'-'*10}]")
    print(f"{'Class Name':<18} | {'Train':<8} | {'Validation':<8}")
    print("-" * 42)

    total_train = 0
    total_val = 0

    for class_idx, target_angles in Config.TARGET_MAPPING.items():
        target_vec = np.array(target_angles, dtype=np.float32)

        train_count = np.sum(np.all(np.isclose(y_train_eval, target_vec, atol=0.1), axis=1))
        val_count = np.sum(np.all(np.isclose(y_val_eval, target_vec, atol=0.1), axis=1))

        total_train += train_count
        total_val += val_count

        class_name = f"Movement {class_idx}" if class_idx != 9 else "Rest (Class 9)"
        print(f"{class_name:<18} | {train_count:<8} | {val_count:<8}")

    print("-" * 42)
    print(f"{'TOTAL':<18} | {total_train:<8} | {total_val:<8}\n")

# ====================================================================================
# ============================== DEBUG/DUMMY TEST ====================================
# ====================================================================================

if __name__ == "__main__":
    import argparse

    on_the_fly_enabled = bool(getattr(Config, 'ON_THE_FLY_WINDOW_SLICING', False))
    on_the_fly_window_size = int(getattr(Config, 'ON_THE_FLY_WINDOW_SIZE', Config.WINDOW_SIZE))
    on_the_fly_step_size = int(getattr(Config, 'ON_THE_FLY_STEP_SIZE', Config.INCREMENT))
    on_the_fly_active_channels_raw = getattr(Config, 'ON_THE_FLY_ACTIVE_CHANNELS', None)
    
    parser = argparse.ArgumentParser(description="Train ShoulderRCNN model")
    parser.add_argument('--mode', type=str, choices=['loso', 'transfer', 'standard'], default='loso',
                       help='Training mode: loso (Leave-One-Subject-Out), transfer (Transfer Learning), or standard (80-20 split)')
    parser.add_argument('--pretrained', type=str, default=Config.MODEL_SAVE_PATH,
                       help='Path to pretrained model (for transfer learning)')
    parser.add_argument('--freeze', action='store_true', default=False,
                       help='Freeze backbone layers during transfer learning')
    parser.add_argument(
        '--include_collected',
        action='store_true',
        default=False,
        help='Include collected participants in loso/standard training datasets.',
    )
    parser.add_argument(
        '--collected_participants',
        type=str,
        default=None,
        help='Optional collected participant list for loso/standard, e.g. "[1,2,4]". '
             'If omitted with --include_collected, all discovered collected participants are used.',
    )
    parser.add_argument(
        '--collected_train_participants',
        type=str,
        default=None,
        help='Optional participant list for collected transfer split, e.g. "[1,2,4]". '
             'These participants are used for training; unlisted collected participants are excluded from training validation.',
    )
    args = parser.parse_args()

    try:
        selected_active_channels = _parse_channel_list(on_the_fly_active_channels_raw)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        exit(1)

    if on_the_fly_enabled:
        print("\n[On-The-Fly Windowing] Enabled.")
        print(f"  - window_size: {on_the_fly_window_size}")
        print(f"  - step_size: {on_the_fly_step_size}")
        print(f"  - active_channels: {selected_active_channels if selected_active_channels is not None else 'ALL'}")

    try:
        selected_collected_for_main_modes = _parse_participant_list(args.collected_participants)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        exit(1)

    include_collected_main_modes = args.include_collected or (selected_collected_for_main_modes is not None and len(selected_collected_for_main_modes) > 0)
    collected_raw_path = REPOSITORY.raw_root('collected')
    collected_edited_path = REPOSITORY.edited_root('collected')

    if include_collected_main_modes and args.mode in ('loso', 'standard'):
        available_collected = REPOSITORY.discover_participants('collected')
        if len(available_collected) == 0:
            print(f"\nERROR: --include_collected requested but no collected files were found at {collected_raw_path}")
            exit(1)

        if selected_collected_for_main_modes is None:
            selected_collected_for_main_modes = available_collected
        elif len(selected_collected_for_main_modes) > 0:
            # Only validate if non-empty list was explicitly provided
            missing_participants = sorted(set(selected_collected_for_main_modes) - set(available_collected))
            if missing_participants:
                print(f"\nERROR: Requested collected participants not found: {missing_participants}")
                print(f"Available collected participants: {available_collected}")
                exit(1)

        if len(selected_collected_for_main_modes) > 0:
            print("\n[Collected Integration] Enabled for main training mode.")
            print(f"  - Included collected participants: {selected_collected_for_main_modes}")
        else:
            print("\n[Control Case] No collected participants selected (empty list).")
    elif selected_collected_for_main_modes is not None and args.mode == 'transfer':
        print("\n[Info] --collected_participants is ignored in transfer mode.")
        print("[Info] Use --collected_train_participants for transfer participant splits.")
        selected_collected_for_main_modes = []
    else:
        selected_collected_for_main_modes = []
    
    # ====================================================================================
    # LEAVE-ONE-SUBJECT-OUT (LOSO) TRAINING
    # ====================================================================================
    if args.mode == 'loso':
        print("Loading dataset with Leave-One-Subject-Out Validation...")

        secondary_subjects = REPOSITORY.discover_participants('secondary')
        if len(secondary_subjects) == 0:
            secondary_subjects = list(range(1, 9))
            print("[WARNING] Could not auto-discover secondary subjects. Falling back to [1..8].")

        if len(secondary_subjects) < 2:
            print("ERROR: LOSO requires at least 2 secondary participants.")
            exit(1)

        # Keep original LOSO behavior: train on all-but-last subject, validate on last subject.
        train_secondary_subjects = secondary_subjects[:-1]
        val_secondary_subject = secondary_subjects[-1]

        print(f"\n[LOSO] Training secondary subjects: {train_secondary_subjects}")
        print(f"[LOSO] Validation secondary subject: {val_secondary_subject}")
        if len(selected_collected_for_main_modes) > 0:
            print(f"[LOSO] Appending collected participants to TRAINING only: {selected_collected_for_main_modes}")

        X_train_secondary, y_train_secondary = DataPreparation.load_and_prepare_dataset(
            base_path=Config.SECONDARY_DATA_PATH,
            include_subjects=train_secondary_subjects,
            include_noise_aug=True,
            return_continuous=on_the_fly_enabled,
        )

        train_parts = [(X_train_secondary, y_train_secondary)]
        if len(selected_collected_for_main_modes) > 0:
            X_train_collected, y_train_collected = DataPreparation.load_collected_data(
                folder_path=collected_raw_path,
                labelled_folder_path=collected_edited_path,
                augment=True,
                include_noise_aug=True,
                include_participants=selected_collected_for_main_modes,
                return_continuous=on_the_fly_enabled,
            )
            train_parts.append((X_train_collected, y_train_collected))

        X_train, y_train = _concat_dataset_parts(train_parts)

        print(f"\n[LOSO] Loading secondary VALIDATION data from Subject {val_secondary_subject}...")
        X_val, y_val = DataPreparation.load_and_prepare_dataset(
            base_path=Config.SECONDARY_DATA_PATH,
            include_subjects=[val_secondary_subject],
            include_noise_aug=False,
            return_continuous=on_the_fly_enabled,
        )

        if X_train is None or X_val is None or _count_samples(X_train, y_train) == 0 or _count_samples(X_val, y_val) == 0:
            print("ERROR: No data loaded. Check your file paths and labels.")
            exit(1)

        _print_dataset_distribution(
            y_train,
            y_val,
            title="LOSO DATASET DISTRIBUTION",
            continuous=on_the_fly_enabled,
            window_size=on_the_fly_window_size,
            step_size=on_the_fly_step_size,
        )
        print(f"Training on {_count_samples(X_train, y_train)} samples, Validating on {_count_samples(X_val, y_val)} samples...")

        train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            patience=Config.PATIENCE,
            use_on_the_fly=on_the_fly_enabled,
            window_size=on_the_fly_window_size,
            step_size=on_the_fly_step_size,
            active_channels=selected_active_channels,
        )
    
    # ====================================================================================
    # TRANSFER LEARNING TRAINING
    # ====================================================================================
    elif args.mode == 'transfer':
        print("Starting Transfer Learning Training...")
        print(f"Pretrained model: {args.pretrained}")
        print(f"Freeze backbone: {args.freeze}")

        try:
            selected_train_participants = _parse_participant_list(args.collected_train_participants)
        except ValueError as exc:
            print(f"\n✗ ERROR: {exc}")
            exit(1)

        # Optional collected participant selection for TRAINING only.
        if selected_train_participants is not None and len(selected_train_participants) > 0:
            collected_raw_path = REPOSITORY.raw_root('collected')
            collected_edited_path = REPOSITORY.edited_root('collected')

            available_participants = REPOSITORY.discover_participants('collected')
            if len(available_participants) == 0:
                print(f"\n✗ ERROR: No collected files were found at {collected_raw_path}")
                exit(1)

            missing_participants = sorted(set(selected_train_participants) - set(available_participants))
            if missing_participants:
                print(
                    f"\n✗ ERROR: Requested training participants not found in collected data: {missing_participants}"
                )
                print(f"Available collected participants: {available_participants}")
                exit(1)

            print("\n[Transfer Learning] Using selected collected participants for TRAINING:")
            print(f"  - Train participants: {selected_train_participants}")
            print("  - Unlisted collected participants are excluded from training validation")
            print(f"  - Raw source: {collected_raw_path}")

            print("\n[Transfer Learning] Loading collected TRAINING data from selected participants...")
            X_train, y_train = DataPreparation.load_collected_data(
                folder_path=collected_raw_path,
                labelled_folder_path=collected_edited_path,
                augment=True,
                include_noise_aug=True,
                include_participants=selected_train_participants,
                return_continuous=on_the_fly_enabled,
            )
            X_val_collected, y_val_collected = None, None
        elif selected_train_participants is None:
            # Legacy folder-based split (no collected participants specified, try legacy folders)
            print("\n[Transfer Learning] Loading TRAINING data from collected_data/training...")
            X_train, y_train = DataPreparation.load_collected_data(
                folder_path='./collected_data/training',
                augment=True,
                include_noise_aug=True,
                return_continuous=on_the_fly_enabled,
            )

            print("\n[Transfer Learning] Checking for validation data in collected_data/validation...")
            X_val_collected, y_val_collected = DataPreparation.load_collected_data(
                folder_path='./collected_data/validation',
                augment=True,
                include_noise_aug=False,
                return_continuous=on_the_fly_enabled,
            )
        else:
            # Control case: empty list provided, use SECONDARY DATA ONLY (no collected data)
            print("\n[Control Case] Empty collected participant list provided.")
            print("[Transfer Learning] Using SECONDARY DATA for both TRAINING and VALIDATION (control case)...")
            X_train, y_train = DataPreparation.load_and_prepare_dataset(
                base_path=Config.SECONDARY_DATA_PATH,
                include_subjects=[1, 2, 3, 4, 5, 6, 7],  # All but subject 8
                include_noise_aug=True,
                return_continuous=on_the_fly_enabled,
            )
            X_val_collected, y_val_collected = None, None
        
        if X_train is None or _count_samples(X_train, y_train) == 0:
            print("\n✗ ERROR: No training data available.")
            print(f"Selected train participants: {selected_train_participants}")
            exit(1)
        
        # 3. Always load secondary data validation (from subject 8 LOSO fold)
        print("\n[Transfer Learning] Loading VALIDATION data from secondary_data (Subject 8)...")
        X_val_secondary, y_val_secondary = DataPreparation.load_and_prepare_dataset(
            base_path=Config.SECONDARY_DATA_PATH,
            include_subjects=[8],
            include_noise_aug=False,
            return_continuous=on_the_fly_enabled,
        )
        
        if X_val_secondary is None or _count_samples(X_val_secondary, y_val_secondary) == 0:
            print("✗ ERROR: Could not load secondary validation data!")
            exit(1)
        
        # 4. Combine validation datasets
        if X_val_collected is not None and _count_samples(X_val_collected, y_val_collected) > 0:
            print(f"\n[Transfer Learning] Combining validation sets:")
            print(f"  - Collected: {_count_samples(X_val_collected, y_val_collected)} samples")
            print(f"  - Secondary: {_count_samples(X_val_secondary, y_val_secondary)} samples")
            X_val, y_val = _concat_dataset_parts([
                (X_val_secondary, y_val_secondary),
                (X_val_collected, y_val_collected),
            ])
            print(f"  - Total: {_count_samples(X_val, y_val)} samples")
        else:
            print(f"\n[Transfer Learning] No collected validation data found. Using only secondary data validation.")
            print(f"  - Secondary: {_count_samples(X_val_secondary, y_val_secondary)} samples")
            X_val = X_val_secondary
            y_val = y_val_secondary
        
        # 5. Print dataset distribution
        _print_dataset_distribution(
            y_train,
            y_val,
            title="TRANSFER LEARNING DATASET DISTRIBUTION",
            continuous=on_the_fly_enabled,
            window_size=on_the_fly_window_size,
            step_size=on_the_fly_step_size,
        )
        
        print(f"Training on {_count_samples(X_train, y_train)} samples, Validating on {_count_samples(X_val, y_val)} samples...")
        
        # 6. Train with transfer learning
        trained_model = train_transfer_learning(
            X_train, y_train, X_val, y_val,
            pretrained_model_path=args.pretrained,
            freeze_layers=args.freeze,
            use_on_the_fly=on_the_fly_enabled,
            window_size=on_the_fly_window_size,
            step_size=on_the_fly_step_size,
            active_channels=selected_active_channels,
        )
    
    # ====================================================================================
    # STANDARD 80-20 TRAINING
    # ====================================================================================
    elif args.mode == 'standard':
        print("Loading dataset with Standard 80-20 Training-Validation Split...")

        secondary_subjects = REPOSITORY.discover_participants('secondary')
        if len(secondary_subjects) == 0:
            secondary_subjects = list(range(1, 9))
            print("[WARNING] Could not auto-discover secondary subjects. Falling back to [1..8].")

        print("\n[Standard Mode] Loading secondary data...")
        X_secondary, y_secondary = DataPreparation.load_and_prepare_dataset(
            base_path=Config.SECONDARY_DATA_PATH,
            include_subjects=secondary_subjects,
            include_noise_aug=True,
            return_continuous=on_the_fly_enabled,
        )

        dataset_parts = [(X_secondary, y_secondary)]

        if len(selected_collected_for_main_modes) > 0:
            print("\n[Standard Mode] Loading selected collected participants...")
            X_collected, y_collected = DataPreparation.load_collected_data(
                folder_path=collected_raw_path,
                labelled_folder_path=collected_edited_path,
                augment=True,
                include_noise_aug=True,
                include_participants=selected_collected_for_main_modes,
                return_continuous=on_the_fly_enabled,
            )
            dataset_parts.append((X_collected, y_collected))

        X_all, y_all = _concat_dataset_parts(dataset_parts)

        if X_all is None or _count_samples(X_all, y_all) == 0:
            print("ERROR: No data loaded. Check your file paths and labels.")
            exit(1)

        print("\n[Standard Mode] Performing 80-20 train-validation split...")
        if _is_continuous_emg_layout(X_all, y_all):
            split_idx = int(X_all.shape[1] * (1.0 - Config.TEST_SPLIT))
            split_idx = max(1, min(split_idx, X_all.shape[1] - 1))

            X_train = X_all[:, :split_idx]
            X_val = X_all[:, split_idx:]
            y_train = y_all[:split_idx]
            y_val = y_all[split_idx:]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=Config.TEST_SPLIT, random_state=42, shuffle=False
            )

        _print_dataset_distribution(
            y_train,
            y_val,
            title="DATASET DISTRIBUTION",
            continuous=on_the_fly_enabled,
            window_size=on_the_fly_window_size,
            step_size=on_the_fly_step_size,
        )

        print(f"Training on {_count_samples(X_train, y_train)} samples, Validating on {_count_samples(X_val, y_val)} samples...")

        train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            patience=Config.PATIENCE,
            use_on_the_fly=on_the_fly_enabled,
            window_size=on_the_fly_window_size,
            step_size=on_the_fly_step_size,
            active_channels=selected_active_channels,
        )