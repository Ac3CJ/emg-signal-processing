import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import re
import csv

import DataPreparation
import ControllerConfiguration as Config
import NeuralNetworkModels as NNModels
from FileRepository import DataRepository

REPOSITORY = DataRepository()

# GPU Optimization Settings
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends, 'xpu'):
    torch.backends.xpu.benchmark = True

# ====================================================================================
# ============================== TRAINING PIPELINE ===================================
# ====================================================================================

def _resolve_run_paths(run_name):
    """Builds the canonical output paths for a named training run.

    Layout:
        ./neural-network-models/<name>/
            training/   <-- model weights, loss curve, distribution dump
            validation/ <-- (populated by ModelValidator.py)
    """
    sanitized = str(run_name).strip()
    if not sanitized:
        raise ValueError("Run name (--name) must be a non-empty string.")
    run_dir = os.path.join('./neural-network-models', sanitized)
    training_dir = os.path.join(run_dir, 'training')
    validation_dir = os.path.join(run_dir, 'validation')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    return {
        'name': sanitized,
        'run_dir': run_dir,
        'training_dir': training_dir,
        'validation_dir': validation_dir,
        'model_path': os.path.join(training_dir, 'best_shoulder_rcnn.pth'),
        'transfer_model_path': os.path.join(training_dir, 'best_shoulder_rcnn_transfer.pth'),
        'loss_curve_path': os.path.join(training_dir, 'training_loss_curve.png'),
        'distribution_path': os.path.join(training_dir, 'training_dataset_distribution.txt'),
        'training_log_path': os.path.join(training_dir, 'training_log.csv'),
    }


def save_dataset_distribution(train_classes, val_classes, output_file='training_dataset_distribution.txt'):
    """Saves contraction-segment counts per movement class to a text file."""
    train_counts = {idx: 0 for idx in Config.TARGET_MAPPING}
    val_counts = {idx: 0 for idx in Config.TARGET_MAPPING}
    for cls in train_classes:
        if cls in train_counts:
            train_counts[cls] += 1
    for cls in val_classes:
        if cls in val_counts:
            val_counts[cls] += 1

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_combined = total_train + total_val if (total_train + total_val) > 0 else 1

    with open(output_file, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("DATASET DISTRIBUTION ACROSS MOVEMENTS (contraction segments)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"{'Movement':<20} {'Index':<8} {'Train':<12} {'Val':<12} {'Total':<8}\n")
        f.write("-" * 65 + "\n")
        for class_idx in sorted(Config.TARGET_MAPPING.keys()):
            movement_name = Config.MOVEMENT_NAMES.get(class_idx, f"Movement {class_idx}")
            f.write(
                f"{movement_name:<20} {class_idx:<8} "
                f"{train_counts[class_idx]:<12} {val_counts[class_idx]:<12} "
                f"{train_counts[class_idx] + val_counts[class_idx]:<8}\n"
            )
        f.write("-" * 65 + "\n")
        f.write(f"{'TOTAL':<20} {'':<8} {total_train:<12} {total_val:<12} {total_train + total_val:<8}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Training segments: {total_train}\n")
        f.write(f"Validation segments: {total_val}\n")
        f.write(f"Total segments: {total_train + total_val}\n")
        f.write(f"Train/Val split: {100*total_train/total_combined:.1f}% / {100*total_val/total_combined:.1f}%\n")

    print(f"\n[Dataset Distribution] Saved to '{output_file}'")


def plot_training_history(train_losses, val_losses, best_epoch=None, output_path='training_loss_curve.png',
                          total_time_seconds=None):
    """Generates and saves a learning curve plot after training."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss (RMSE)', color='tab:blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss (RMSE)', color='tab:orange', linewidth=2)
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='tab:red', linestyle=':', linewidth=2,
                   label=f'Best Model (Epoch {best_epoch + 1})', alpha=0.8)
    title = 'RCNN Regression Learning Curve'
    if total_time_seconds is not None:
        hours = int(total_time_seconds // 3600)
        minutes = int((total_time_seconds % 3600) // 60)
        seconds = int(total_time_seconds % 60)
        title += f'  |  Total Training Time: {hours}h {minutes}m {seconds}s'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Root Mean Squared Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\n[Visuals] Learning curve saved as '{output_path}'.")


def save_training_log_csv(rows, output_path):
    """Writes per-epoch training history to a CSV.

    Columns: epoch, train_mae_deg, val_mae_deg, val_mse, learning_rate, epoch_time_s, is_best.
    """
    if not rows:
        return
    fieldnames = ['epoch', 'train_mae_deg', 'val_mae_deg', 'val_mse',
                  'learning_rate', 'epoch_time_s', 'is_best']
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Training Log] Saved to '{output_path}'.")


# ====================================================================================
# =========================== CONTRACTION-AWARE LOADER ===============================
# ====================================================================================

def _resolve_contraction_block_shuffle_seed():
    raw_seed = getattr(Config, "CONTRACTION_BLOCK_SHUFFLE_SEED", None)
    if raw_seed is None:
        return None
    try:
        return int(raw_seed)
    except (TypeError, ValueError):
        return None


def _is_contraction_block_shuffle_enabled():
    return bool(getattr(Config, "CONTRACTION_BLOCK_SHUFFLE", True))


class ContractionBlockSampler(Sampler):
    """Shuffles segment order each epoch while preserving intra-segment window order.

    This is the LSTM-friendly shuffling regime requested for the new pipeline:
    contractions reorder between epochs (preventing local minima from fixed
    presentation order), but within each contraction the windows stream in
    rising → falling temporal order so the LSTM sees coherent kinematic edges.
    """

    def __init__(self, segment_window_spans, seed=None):
        self.segment_window_spans = [
            (int(start_idx), int(end_idx))
            for start_idx, end_idx in segment_window_spans
            if int(end_idx) > int(start_idx)
        ]
        self.seed = None if seed is None else int(seed)
        self._epoch = 0
        self._length = int(sum(end_idx - start_idx for start_idx, end_idx in self.segment_window_spans))

    def __iter__(self):
        if self._length == 0:
            return iter(())
        rng_seed = None if self.seed is None else self.seed + self._epoch
        rng = np.random.default_rng(rng_seed)
        order = rng.permutation(len(self.segment_window_spans))
        self._epoch += 1
        ordered_spans = [self.segment_window_spans[int(idx)] for idx in order]
        return (
            window_idx
            for start_idx, end_idx in ordered_spans
            for window_idx in range(start_idx, end_idx)
        )

    def __len__(self):
        return self._length


def _build_data_loader(dataset, batch_size, shuffle, num_workers=None, sampler=None):
    """Builds DataLoader with worker settings that work for both 0 and >0 workers."""
    resolved_workers = Config.NUM_DATA_WORKERS if num_workers is None else max(0, int(num_workers))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": resolved_workers,
        "pin_memory": True,
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle
    if resolved_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
    return DataLoader(dataset, **loader_kwargs)


def _build_continuous_dataset(continuous_X, continuous_y, segment_bounds, window_size, step_size, active_channels):
    """Wraps inputs into a ContinuousEMGDataset using explicit segment bounds."""
    return NNModels.ContinuousEMGDataset(
        continuous_X=np.asarray(continuous_X, dtype=np.float32),
        continuous_y=np.asarray(continuous_y, dtype=np.float32),
        window_size=int(window_size),
        step_size=int(step_size),
        active_channels=active_channels,
        segment_bounds=list(segment_bounds) if segment_bounds else None,
    )


def _resolve_training_device():
    """Selects the best available training device in priority order."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "xpu"
    return torch.device("cpu"), "cpu"


# ====================================================================================
# ============================== TRAINING ENTRY POINTS ===============================
# ====================================================================================

def train_model(
    X_train, y_train, train_bounds,
    X_val, y_val, val_bounds,
    batch_size=Config.BATCH_SIZE,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
    window_size=None,
    step_size=None,
    active_channels=None,
    model_save_path=None,
    loss_curve_path=None,
    training_log_path=None,
):
    """Trains the PyTorch RCNN model with on-the-fly windowing and contraction-block shuffling."""
    resolved_model_path = model_save_path or Config.MODEL_SAVE_PATH
    resolved_loss_curve_path = loss_curve_path or 'training_loss_curve.png'
    resolved_training_log_path = training_log_path or 'training_log.csv'
    resolved_window_size = int(window_size if window_size is not None else Config.WINDOW_SIZE)
    resolved_step_size = int(step_size if step_size is not None else Config.INCREMENT)

    train_dataset = _build_continuous_dataset(
        X_train, y_train, train_bounds, resolved_window_size, resolved_step_size, active_channels
    )
    val_dataset = _build_continuous_dataset(
        X_val, y_val, val_bounds, resolved_window_size, resolved_step_size, active_channels
    )

    model_num_channels = len(train_dataset.active_channels)
    model_num_outputs = int(np.asarray(y_train).shape[1])

    loader_workers = Config.NUM_DATA_WORKERS
    if os.name == "nt" and loader_workers > 0:
        loader_workers = 0

    train_sampler = None
    if _is_contraction_block_shuffle_enabled():
        train_sampler = ContractionBlockSampler(
            segment_window_spans=train_dataset.get_segment_window_spans(),
            seed=_resolve_contraction_block_shuffle_seed(),
        )

    train_loader = _build_data_loader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=loader_workers, sampler=train_sampler,
    )
    val_loader = _build_data_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)

    device, device_type = _resolve_training_device()

    print(f"\n[{'-'*10} SYSTEM CHECK {'-'*10}]")
    print(f"Training on device: {device}")
    print(f"Window size: {resolved_window_size} | Step size: {resolved_step_size}")
    print(f"Train segments: {len(train_dataset.get_segment_window_spans())} | "
          f"Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")
    print(f"Batch size: {batch_size} | Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Data workers: {loader_workers} | Prefetch: {Config.PREFETCH_FACTOR if loader_workers > 0 else 'N/A'}")
    print(f"Contraction-block shuffling: {'enabled' if train_sampler is not None else 'disabled'}")
    print(f"Effective batch size: {batch_size * Config.GRADIENT_ACCUMULATION_STEPS}")

    model = NNModels.ShoulderRCNN(num_channels=model_num_channels, num_outputs=model_num_outputs).to(device)

    optimizer_criterion = nn.MSELoss()
    tracker_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE,
        min_lr=1e-6,
    )

    use_amp = device_type in ["xpu", "cuda"]
    amp_dtype = torch.bfloat16 if device_type == "xpu" else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history_train_mae, history_val_mae = [], []
    training_log_rows = []
    best_epoch = None
    training_start_time = time.time()

    print(f"\n[{'-'*10} STARTING TRAINING {'-'*10}]")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_lr = optimizer.param_groups[0]['lr']
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

        is_best = val_mse_loss < best_val_loss
        status = ""
        if is_best:
            best_val_loss = val_mse_loss
            epochs_no_improve = 0
            best_epoch = epoch
            os.makedirs(os.path.dirname(resolved_model_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), resolved_model_path)
            status = "--> Saved Best Model"
        else:
            epochs_no_improve += 1
            status = f"--> No improvement ({epochs_no_improve}/{patience})"

        scheduler.step(val_mse_loss)
        epoch_time = time.time() - epoch_start_time
        training_log_rows.append({
            'epoch': epoch + 1,
            'train_mae_deg': round(train_mae, 6),
            'val_mae_deg': round(val_mae, 6),
            'val_mse': round(val_mse_loss, 6),
            'learning_rate': epoch_lr,
            'epoch_time_s': round(epoch_time, 4),
            'is_best': is_best,
        })
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Error: {train_mae:6.2f}° | Val Error: {val_mae:6.2f}° | Time: {epoch_time:6.2f}s {status}")

        if epochs_no_improve >= patience:
            print(f"\n[STOP] Early stopping triggered.")
            break

    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    avg_epoch_time = total_training_time / (epoch + 1)

    print(f"\n[{'-'*10} TRAINING COMPLETE {'-'*10}]")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print(f"Best model saved to: {resolved_model_path}")
    save_training_log_csv(training_log_rows, resolved_training_log_path)
    plot_training_history(history_train_mae, history_val_mae, best_epoch=best_epoch,
                          output_path=resolved_loss_curve_path,
                          total_time_seconds=total_training_time)

    model.load_state_dict(torch.load(resolved_model_path))
    return model


# ====================================================================================
# ============================== TRANSFER LEARNING ===================================
# ====================================================================================

def freeze_backbone(model, num_unfreeze_layers=2):
    """Freezes all but the last `num_unfreeze_layers` layers."""
    named_params = list(model.named_parameters())
    num_to_freeze = len(named_params) - num_unfreeze_layers
    for i, (name, param) in enumerate(named_params):
        param.requires_grad = i >= num_to_freeze
    print(f"\n[Transfer Learning] Froze {num_to_freeze} layers, unfroze {num_unfreeze_layers} layers")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")


def train_transfer_learning(
    X_train, y_train, train_bounds,
    X_val, y_val, val_bounds,
    pretrained_model_path,
    batch_size=None, epochs=None, patience=None, freeze_layers=True,
    window_size=None, step_size=None, active_channels=None,
    model_save_path=None,
    loss_curve_path=None,
    training_log_path=None,
):
    """Fine-tunes a pretrained model on new collected data."""
    resolved_model_path = model_save_path or Config.TRANSFER_LEARNING_MODEL_SAVE_PATH
    resolved_loss_curve_path = loss_curve_path or 'training_loss_curve.png'
    resolved_training_log_path = training_log_path or 'training_log.csv'
    if batch_size is None:
        batch_size = Config.TRANSFER_LEARNING_BATCH_SIZE
    if epochs is None:
        epochs = Config.TRANSFER_LEARNING_EPOCHS
    if patience is None:
        patience = Config.TRANSFER_LEARNING_PATIENCE

    resolved_window_size = int(window_size if window_size is not None else Config.WINDOW_SIZE)
    resolved_step_size = int(step_size if step_size is not None else Config.INCREMENT)

    train_dataset = _build_continuous_dataset(
        X_train, y_train, train_bounds, resolved_window_size, resolved_step_size, active_channels
    )
    val_dataset = _build_continuous_dataset(
        X_val, y_val, val_bounds, resolved_window_size, resolved_step_size, active_channels
    )

    model_num_channels = len(train_dataset.active_channels)
    model_num_outputs = int(np.asarray(y_train).shape[1])

    loader_workers = Config.NUM_DATA_WORKERS
    if os.name == "nt" and loader_workers > 0:
        loader_workers = 0

    train_sampler = None
    if _is_contraction_block_shuffle_enabled():
        train_sampler = ContractionBlockSampler(
            segment_window_spans=train_dataset.get_segment_window_spans(),
            seed=_resolve_contraction_block_shuffle_seed(),
        )

    train_loader = _build_data_loader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=loader_workers, sampler=train_sampler,
    )
    val_loader = _build_data_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)

    device, device_type = _resolve_training_device()

    print(f"\n[{'-'*10} TRANSFER LEARNING SETUP {'-'*10}]")
    print(f"Loading pretrained model from: {pretrained_model_path}")
    print(f"Training on device: {device}")
    print(f"Window size: {resolved_window_size} | Step size: {resolved_step_size}")
    print(f"Batch size: {batch_size} | Epochs: {epochs} | Patience: {patience}")
    print(f"Data workers: {loader_workers} | Prefetch: {Config.PREFETCH_FACTOR if loader_workers > 0 else 'N/A'}")
    print(f"Contraction-block shuffling: {'enabled' if train_sampler is not None else 'disabled'}")
    print(f"Learning rate: {Config.TRANSFER_LEARNING_LEARNING_RATE}")

    model = NNModels.ShoulderRCNN(num_channels=model_num_channels, num_outputs=model_num_outputs).to(device)
    try:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"✓ Successfully loaded pretrained weights from {pretrained_model_path}")
    except Exception as e:
        print(f"✗ ERROR loading pretrained model: {e}")
        return None

    if freeze_layers and Config.FREEZE_BACKBONE_LAYERS:
        freeze_backbone(model, num_unfreeze_layers=Config.NUM_LAYERS_TO_UNFREEZE)
    else:
        print("[Transfer Learning] Training all layers (no freezing)")

    optimizer_criterion = nn.MSELoss()
    tracker_criterion = nn.L1Loss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.TRANSFER_LEARNING_LEARNING_RATE,
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE,
        min_lr=1e-7,
    )

    use_amp = device_type in ["xpu", "cuda"]
    amp_dtype = torch.bfloat16 if device_type == "xpu" else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history_train_mae, history_val_mae = [], []
    training_log_rows = []
    best_epoch = None
    training_start_time = time.time()

    print(f"\n[{'-'*10} STARTING TRANSFER LEARNING {'-'*10}]")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_lr = optimizer.param_groups[0]['lr']
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

        is_best = val_mse_loss < best_val_loss
        status = ""
        if is_best:
            best_val_loss = val_mse_loss
            epochs_no_improve = 0
            best_epoch = epoch
            os.makedirs(os.path.dirname(resolved_model_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), resolved_model_path)
            status = "--> Saved Best Model"
        else:
            epochs_no_improve += 1
            status = f"--> No improvement ({epochs_no_improve}/{patience})"

        scheduler.step(val_mse_loss)
        epoch_time = time.time() - epoch_start_time
        training_log_rows.append({
            'epoch': epoch + 1,
            'train_mae_deg': round(train_mae, 6),
            'val_mae_deg': round(val_mae, 6),
            'val_mse': round(val_mse_loss, 6),
            'learning_rate': epoch_lr,
            'epoch_time_s': round(epoch_time, 4),
            'is_best': is_best,
        })
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
    print(f"Best model saved to: {resolved_model_path}")
    save_training_log_csv(training_log_rows, resolved_training_log_path)
    plot_training_history(history_train_mae, history_val_mae, best_epoch=best_epoch,
                          output_path=resolved_loss_curve_path,
                          total_time_seconds=total_training_time)

    model.load_state_dict(torch.load(resolved_model_path))
    return model


# ====================================================================================
# ============================== CLI / PIPELINE PLUMBING =============================
# ====================================================================================

def _parse_participant_list(participant_list_raw):
    """Parses participant IDs from strings like '[1,2,4]' or '1,2,4'."""
    if participant_list_raw is None:
        return None
    text = str(participant_list_raw).strip()
    if not text or text == "[]":
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    tokens = [token for token in re.split(r"[,\s]+", text) if token]
    if len(tokens) == 0:
        return []
    participants = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(f"Invalid participant token '{token}'. Use numeric IDs, e.g. [1,2,4].")
        participant_id = int(token)
        if participant_id <= 0:
            raise ValueError(f"Participant IDs must be positive. Got: {participant_id}")
        if participant_id not in participants:
            participants.append(participant_id)
    return participants


def _parse_channel_list(channel_list_raw):
    """Parses channel indices from strings like '[0,1,4]' or '0,1,4'."""
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
            raise ValueError(f"Invalid channel token '{token}'. Use channel indices, e.g. [0,1,4].")
        channel_idx = int(token)
        if channel_idx < 0 or channel_idx >= Config.NUM_CHANNELS:
            raise ValueError(f"Channel indices must be in range [0, {Config.NUM_CHANNELS - 1}]. Got: {channel_idx}")
        if channel_idx not in channels:
            channels.append(channel_idx)
    return channels


def _concat_dataset_bundles(bundles):
    """Concatenates [(X, y, bounds, classes), ...] into a single bundle.

    bounds are offset by the running total_samples of preceding bundles so the
    contraction segments stay contiguous in the merged continuous array.
    """
    valid = []
    for bundle in bundles:
        if bundle is None:
            continue
        X, y, bounds, classes = bundle
        if X is None or y is None or X.shape[1] == 0:
            continue
        valid.append(bundle)

    if len(valid) == 0:
        return None, None, [], []
    if len(valid) == 1:
        X, y, bounds, classes = valid[0]
        return X, y, list(bounds), list(classes)

    merged_X = []
    merged_y = []
    merged_bounds = []
    merged_classes = []
    cursor = 0
    for X, y, bounds, classes in valid:
        merged_X.append(X)
        merged_y.append(y)
        for start, end in bounds:
            merged_bounds.append((start + cursor, end + cursor))
        merged_classes.extend(classes)
        cursor += X.shape[1]

    return (
        np.concatenate(merged_X, axis=1),
        np.concatenate(merged_y, axis=0),
        merged_bounds,
        merged_classes,
    )


def _print_dataset_distribution(train_classes, val_classes, title="DATASET DISTRIBUTION"):
    """Prints contraction-segment counts per movement class."""
    train_counts = {idx: 0 for idx in Config.TARGET_MAPPING}
    val_counts = {idx: 0 for idx in Config.TARGET_MAPPING}
    for cls in train_classes:
        if cls in train_counts:
            train_counts[cls] += 1
    for cls in val_classes:
        if cls in val_counts:
            val_counts[cls] += 1

    print(f"\n[{'-'*10} {title} {'-'*10}]")
    print(f"{'Class Name':<18} | {'Train':<8} | {'Validation':<8}")
    print("-" * 42)
    total_train = 0
    total_val = 0
    for class_idx in sorted(Config.TARGET_MAPPING.keys()):
        class_name = f"Movement {class_idx}" if class_idx != 9 else "Rest (Class 9)"
        print(f"{class_name:<18} | {train_counts[class_idx]:<8} | {val_counts[class_idx]:<8}")
        total_train += train_counts[class_idx]
        total_val += val_counts[class_idx]
    print("-" * 42)
    print(f"{'TOTAL':<18} | {total_train:<8} | {total_val:<8}\n")


# ====================================================================================
# ============================== TRAINING PIPELINES ==================================
# ====================================================================================

def _resolve_collected_participants_for_main_modes(args, selected_collected_for_main_modes, collected_raw_path):
    """Validates and resolves collected participant selection for loso/standard modes."""
    include_collected_main_modes = args.include_collected or (
        selected_collected_for_main_modes is not None and len(selected_collected_for_main_modes) > 0
    )

    if include_collected_main_modes and args.mode in ('loso', 'standard'):
        available_collected = REPOSITORY.discover_participants('collected')
        if len(available_collected) == 0:
            print(f"\nERROR: --include_collected requested but no collected files were found at {collected_raw_path}")
            exit(1)

        if selected_collected_for_main_modes is None:
            selected_collected_for_main_modes = available_collected
        elif len(selected_collected_for_main_modes) > 0:
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
        return selected_collected_for_main_modes

    if selected_collected_for_main_modes is not None and args.mode == 'transfer':
        print("\n[Info] --collected_participants is ignored in transfer mode.")
        print("[Info] Use --collected_train_participants for transfer participant splits.")
        return []

    if selected_collected_for_main_modes is None:
        return []
    return selected_collected_for_main_modes


def loso_pipeline(
    selected_collected_for_main_modes,
    collected_raw_path,
    collected_edited_path,
    on_the_fly_window_size,
    on_the_fly_step_size,
    selected_active_channels,
    run_paths,
):
    """Implements the Leave-One-Subject-Out training pipeline."""
    print("Loading dataset with Leave-One-Subject-Out Validation...")

    secondary_subjects = REPOSITORY.discover_participants('secondary')
    if len(secondary_subjects) == 0:
        secondary_subjects = list(range(1, 9))
        print("[WARNING] Could not auto-discover secondary subjects. Falling back to [1..8].")
    if len(secondary_subjects) < 2:
        print("ERROR: LOSO requires at least 2 secondary participants.")
        exit(1)

    train_secondary_subjects = secondary_subjects[:-1]
    val_secondary_subject = secondary_subjects[-1]

    print(f"\n[LOSO] Training secondary subjects: {train_secondary_subjects}")
    print(f"[LOSO] Validation secondary subject: {val_secondary_subject}")
    if len(selected_collected_for_main_modes) > 0:
        print(f"[LOSO] Appending collected participants to TRAINING only: {selected_collected_for_main_modes}")

    secondary_bundle = DataPreparation.load_and_prepare_dataset(
        base_path=Config.SECONDARY_DATA_PATH,
        include_subjects=train_secondary_subjects,
        include_noise_aug=True,
    )

    bundles = [secondary_bundle]
    if len(selected_collected_for_main_modes) > 0:
        collected_bundle = DataPreparation.load_collected_data(
            folder_path=collected_raw_path,
            labelled_folder_path=collected_edited_path,
            augment=True,
            include_noise_aug=True,
            include_participants=selected_collected_for_main_modes,
        )
        bundles.append(collected_bundle)

    X_train, y_train, train_bounds, train_classes = _concat_dataset_bundles(bundles)

    print(f"\n[LOSO] Loading secondary VALIDATION data from Subject {val_secondary_subject}...")
    X_val, y_val, val_bounds, val_classes = DataPreparation.load_and_prepare_dataset(
        base_path=Config.SECONDARY_DATA_PATH,
        include_subjects=[val_secondary_subject],
        include_noise_aug=False,
        augment=False,
    )

    if X_train is None or X_val is None or len(train_bounds) == 0 or len(val_bounds) == 0:
        print("ERROR: No data loaded. Check your file paths and labels.")
        exit(1)

    _print_dataset_distribution(train_classes, val_classes, title="LOSO DATASET DISTRIBUTION")
    print(f"Training on {len(train_bounds)} segments, Validating on {len(val_bounds)} segments...")

    train_model(
        X_train, y_train, train_bounds,
        X_val, y_val, val_bounds,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        patience=Config.PATIENCE,
        window_size=on_the_fly_window_size,
        step_size=on_the_fly_step_size,
        active_channels=selected_active_channels,
        model_save_path=run_paths['model_path'],
        loss_curve_path=run_paths['loss_curve_path'],
        training_log_path=run_paths['training_log_path'],
    )
    save_dataset_distribution(train_classes, val_classes, output_file=run_paths['distribution_path'])


def transfer_pipeline(
    args,
    on_the_fly_window_size,
    on_the_fly_step_size,
    selected_active_channels,
    run_paths,
):
    """Implements the transfer learning training pipeline."""
    print("Starting Transfer Learning Training...")
    print(f"Pretrained model: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze}")

    try:
        selected_train_participants = _parse_participant_list(args.collected_train_participants)
    except ValueError as exc:
        print(f"\n✗ ERROR: {exc}")
        exit(1)

    if selected_train_participants is not None and len(selected_train_participants) > 0:
        collected_raw_path = REPOSITORY.raw_root('collected')
        collected_edited_path = REPOSITORY.edited_root('collected')

        available_participants = REPOSITORY.discover_participants('collected')
        if len(available_participants) == 0:
            print(f"\n✗ ERROR: No collected files were found at {collected_raw_path}")
            exit(1)

        missing_participants = sorted(set(selected_train_participants) - set(available_participants))
        if missing_participants:
            print(f"\n✗ ERROR: Requested training participants not found in collected data: {missing_participants}")
            print(f"Available collected participants: {available_participants}")
            exit(1)

        print("\n[Transfer Learning] Using selected collected participants for TRAINING:")
        print(f"  - Train participants: {selected_train_participants}")

        X_train, y_train, train_bounds, train_classes = DataPreparation.load_collected_data(
            folder_path=collected_raw_path,
            labelled_folder_path=collected_edited_path,
            augment=True,
            include_noise_aug=True,
            include_participants=selected_train_participants,
        )
    else:
        print("\n[Control Case] Empty collected participant list provided.")
        print("[Transfer Learning] Using SECONDARY DATA for TRAINING (control case)...")
        X_train, y_train, train_bounds, train_classes = DataPreparation.load_and_prepare_dataset(
            base_path=Config.SECONDARY_DATA_PATH,
            include_subjects=[1, 2, 3, 4, 5, 6, 7],
            include_noise_aug=True,
        )

    if X_train is None or len(train_bounds) == 0:
        print("\n✗ ERROR: No training data available.")
        exit(1)

    print("\n[Transfer Learning] Loading VALIDATION data from secondary_data (Subject 8)...")
    X_val, y_val, val_bounds, val_classes = DataPreparation.load_and_prepare_dataset(
        base_path=Config.SECONDARY_DATA_PATH,
        include_subjects=[8],
        include_noise_aug=False,
        augment=False,
    )

    if X_val is None or len(val_bounds) == 0:
        print("✗ ERROR: Could not load secondary validation data!")
        exit(1)

    _print_dataset_distribution(train_classes, val_classes, title="TRANSFER LEARNING DATASET DISTRIBUTION")
    print(f"Training on {len(train_bounds)} segments, Validating on {len(val_bounds)} segments...")

    trained_model = train_transfer_learning(
        X_train, y_train, train_bounds,
        X_val, y_val, val_bounds,
        pretrained_model_path=args.pretrained,
        freeze_layers=args.freeze,
        window_size=on_the_fly_window_size,
        step_size=on_the_fly_step_size,
        active_channels=selected_active_channels,
        model_save_path=run_paths['transfer_model_path'],
        loss_curve_path=run_paths['loss_curve_path'],
        training_log_path=run_paths['training_log_path'],
    )
    save_dataset_distribution(train_classes, val_classes, output_file=run_paths['distribution_path'])
    return trained_model


def standard_pipeline(
    selected_collected_for_main_modes,
    collected_raw_path,
    collected_edited_path,
    on_the_fly_window_size,
    on_the_fly_step_size,
    selected_active_channels,
    run_paths,
):
    """Implements the standard 80-20 train/validation pipeline."""
    print("Loading dataset with Standard 80-20 Training-Validation Split...")

    secondary_subjects = REPOSITORY.discover_participants('secondary')
    if len(secondary_subjects) == 0:
        secondary_subjects = list(range(1, 9))
        print("[WARNING] Could not auto-discover secondary subjects. Falling back to [1..8].")

    print("\n[Standard Mode] Loading secondary data...")
    secondary_bundle = DataPreparation.load_and_prepare_dataset(
        base_path=Config.SECONDARY_DATA_PATH,
        include_subjects=secondary_subjects,
        include_noise_aug=True,
    )

    bundles = [secondary_bundle]
    if len(selected_collected_for_main_modes) > 0:
        print("\n[Standard Mode] Loading selected collected participants...")
        collected_bundle = DataPreparation.load_collected_data(
            folder_path=collected_raw_path,
            labelled_folder_path=collected_edited_path,
            augment=True,
            include_noise_aug=True,
            include_participants=selected_collected_for_main_modes,
        )
        bundles.append(collected_bundle)

    X_all, y_all, all_bounds, all_classes = _concat_dataset_bundles(bundles)

    if X_all is None or len(all_bounds) == 0:
        print("ERROR: No data loaded. Check your file paths and labels.")
        exit(1)

    print("\n[Standard Mode] Performing 80-20 train-validation split (segment-aware)...")
    split_segment_idx = max(1, int(len(all_bounds) * (1.0 - Config.TEST_SPLIT)))
    split_segment_idx = min(split_segment_idx, len(all_bounds) - 1)
    split_sample_idx = all_bounds[split_segment_idx][0]

    X_train = X_all[:, :split_sample_idx]
    X_val = X_all[:, split_sample_idx:]
    y_train = y_all[:split_sample_idx]
    y_val = y_all[split_sample_idx:]

    train_bounds = all_bounds[:split_segment_idx]
    val_bounds = [(start - split_sample_idx, end - split_sample_idx) for start, end in all_bounds[split_segment_idx:]]
    train_classes = all_classes[:split_segment_idx]
    val_classes = all_classes[split_segment_idx:]

    _print_dataset_distribution(train_classes, val_classes, title="DATASET DISTRIBUTION")
    print(f"Training on {len(train_bounds)} segments, Validating on {len(val_bounds)} segments...")

    train_model(
        X_train, y_train, train_bounds,
        X_val, y_val, val_bounds,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        patience=Config.PATIENCE,
        window_size=on_the_fly_window_size,
        step_size=on_the_fly_step_size,
        active_channels=selected_active_channels,
        model_save_path=run_paths['model_path'],
        loss_curve_path=run_paths['loss_curve_path'],
        training_log_path=run_paths['training_log_path'],
    )
    save_dataset_distribution(train_classes, val_classes, output_file=run_paths['distribution_path'])


# ====================================================================================
# ============================== BENCHMARK PIPELINE ==================================
# ====================================================================================

BENCHMARK_TRAIN_COLLECTED = [1, 2, 4, 8]
BENCHMARK_TEST_COLLECTED = [3, 6, 7, 9]


def benchmark_pipeline(
    on_the_fly_window_size,
    on_the_fly_step_size,
    selected_active_channels,
    run_paths,
):
    """Train on all valid secondary + handpicked collected; validate on held-out collected.

    The held-out collected participants (BENCHMARK_TEST_COLLECTED) are intentionally
    excluded from training so they can act as unseen benchmark trials in the report.
    Run ModelValidator.py with the same --name to populate the validation/ folder.
    """
    print("Loading dataset for Benchmark Mode (all secondary + selected collected)...")

    secondary_subjects = REPOSITORY.discover_participants('secondary')
    if len(secondary_subjects) == 0:
        secondary_subjects = list(range(1, 9))
        print("[WARNING] Could not auto-discover secondary subjects. Falling back to [1..8].")

    collected_raw_path = REPOSITORY.raw_root('collected')
    collected_edited_path = REPOSITORY.edited_root('collected')
    available_collected = REPOSITORY.discover_participants('collected')

    train_collected = [p for p in BENCHMARK_TRAIN_COLLECTED if p in available_collected]
    test_collected = [p for p in BENCHMARK_TEST_COLLECTED if p in available_collected]

    missing_train = sorted(set(BENCHMARK_TRAIN_COLLECTED) - set(available_collected))
    missing_test = sorted(set(BENCHMARK_TEST_COLLECTED) - set(available_collected))
    if missing_train:
        print(f"[WARNING] Benchmark train participants missing on disk: {missing_train}")
    if missing_test:
        print(f"[WARNING] Benchmark test participants missing on disk: {missing_test}")

    print(f"\n[Benchmark] Secondary training subjects (all valid): {secondary_subjects}")
    print(f"[Benchmark] Collected training participants: {train_collected}")
    print(f"[Benchmark] Collected validation (held out) participants: {test_collected}")
    print(f"[Benchmark] COLLECTED_BLACKLIST honored: {sorted(getattr(Config, 'COLLECTED_BLACKLIST', []) or [])}")

    secondary_bundle = DataPreparation.load_and_prepare_dataset(
        base_path=Config.SECONDARY_DATA_PATH,
        include_subjects=secondary_subjects,
        include_noise_aug=True,
    )

    bundles = [secondary_bundle]
    if len(train_collected) > 0:
        collected_bundle = DataPreparation.load_collected_data(
            folder_path=collected_raw_path,
            labelled_folder_path=collected_edited_path,
            augment=True,
            include_noise_aug=True,
            include_participants=train_collected,
        )
        bundles.append(collected_bundle)

    X_train, y_train, train_bounds, train_classes = _concat_dataset_bundles(bundles)

    print(f"\n[Benchmark] Loading collected VALIDATION participants: {test_collected}")
    if len(test_collected) > 0:
        X_val, y_val, val_bounds, val_classes = DataPreparation.load_collected_data(
            folder_path=collected_raw_path,
            labelled_folder_path=collected_edited_path,
            augment=False,
            include_noise_aug=False,
            include_participants=test_collected,
        )
    else:
        X_val, y_val, val_bounds, val_classes = None, None, [], []

    if X_train is None or len(train_bounds) == 0:
        print("ERROR: No training data loaded. Check your file paths and labels.")
        exit(1)
    if X_val is None or len(val_bounds) == 0:
        print("ERROR: No validation data loaded. Check that test participants exist on disk.")
        exit(1)

    _print_dataset_distribution(train_classes, val_classes, title="BENCHMARK DATASET DISTRIBUTION")
    print(f"Training on {len(train_bounds)} segments, Validating on {len(val_bounds)} segments...")
    print(f"Run output dir: {run_paths['run_dir']}")

    train_model(
        X_train, y_train, train_bounds,
        X_val, y_val, val_bounds,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        patience=Config.PATIENCE,
        window_size=on_the_fly_window_size,
        step_size=on_the_fly_step_size,
        active_channels=selected_active_channels,
        model_save_path=run_paths['model_path'],
        loss_curve_path=run_paths['loss_curve_path'],
        training_log_path=run_paths['training_log_path'],
    )
    save_dataset_distribution(train_classes, val_classes, output_file=run_paths['distribution_path'])
    print(f"\n[Benchmark] Done. Model + training artifacts saved under: {run_paths['training_dir']}")
    print(f"[Benchmark] Run ModelValidator.py with --name {run_paths['name']} to populate validation/ CSVs.")


def _build_arg_parser():
    """Builds the CLI parser for training mode selection."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ShoulderRCNN model")
    parser.add_argument('--name', type=str, required=True,
                       help='Run name. Outputs go to ./neural-network-models/<name>/training/.')
    parser.add_argument('--mode', type=str, choices=['loso', 'transfer', 'standard', 'benchmark'], default='loso',
                       help='Training mode: loso, transfer, standard, or benchmark.')
    parser.add_argument('--pretrained', type=str, default=Config.MODEL_SAVE_PATH,
                       help='Path to pretrained model (for transfer learning)')
    parser.add_argument('--freeze', action='store_true', default=False,
                       help='Freeze backbone layers during transfer learning')
    parser.add_argument('--include_collected', action='store_true', default=False,
                       help='Include collected participants in loso/standard training datasets.')
    parser.add_argument('--collected_participants', type=str, default=None,
                       help='Optional collected participant list, e.g. "[1,2,4]".')
    parser.add_argument('--collected_train_participants', type=str, default=None,
                       help='Optional participant list for collected transfer split, e.g. "[1,2,4]".')
    return parser


def main():
    """Entry point for training mode dispatch."""
    on_the_fly_window_size = int(getattr(Config, 'ON_THE_FLY_WINDOW_SIZE', Config.WINDOW_SIZE))
    on_the_fly_step_size = int(getattr(Config, 'ON_THE_FLY_STEP_SIZE', Config.INCREMENT))
    on_the_fly_active_channels_raw = getattr(Config, 'ON_THE_FLY_ACTIVE_CHANNELS', None)

    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        run_paths = _resolve_run_paths(args.name)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        exit(1)

    try:
        selected_active_channels = _parse_channel_list(on_the_fly_active_channels_raw)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        exit(1)

    print(f"\n[Run] name='{run_paths['name']}'  → {run_paths['run_dir']}")
    print("\n[Pipeline] Order: load → segment → augment → filter → train (on-the-fly windowing)")
    print(f"  - window_size: {on_the_fly_window_size}")
    print(f"  - step_size: {on_the_fly_step_size}")
    print(f"  - active_channels: {selected_active_channels if selected_active_channels is not None else 'ALL'}")

    try:
        selected_collected_for_main_modes = _parse_participant_list(args.collected_participants)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        exit(1)

    collected_raw_path = REPOSITORY.raw_root('collected')
    collected_edited_path = REPOSITORY.edited_root('collected')
    selected_collected_for_main_modes = _resolve_collected_participants_for_main_modes(
        args, selected_collected_for_main_modes, collected_raw_path,
    )

    if args.mode == 'loso':
        loso_pipeline(
            selected_collected_for_main_modes,
            collected_raw_path,
            collected_edited_path,
            on_the_fly_window_size,
            on_the_fly_step_size,
            selected_active_channels,
            run_paths,
        )
    elif args.mode == 'transfer':
        transfer_pipeline(
            args,
            on_the_fly_window_size,
            on_the_fly_step_size,
            selected_active_channels,
            run_paths,
        )
    elif args.mode == 'standard':
        standard_pipeline(
            selected_collected_for_main_modes,
            collected_raw_path,
            collected_edited_path,
            on_the_fly_window_size,
            on_the_fly_step_size,
            selected_active_channels,
            run_paths,
        )
    elif args.mode == 'benchmark':
        benchmark_pipeline(
            on_the_fly_window_size,
            on_the_fly_step_size,
            selected_active_channels,
            run_paths,
        )


if __name__ == "__main__":
    main()
