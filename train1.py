import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    GRAD_CLIP,
    NUMERIC_COLS,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
)
from dataset import YamaPADataset, yama_collate
from model import YamamotoPitchRNN

# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -----------------------------
# Hyperparameters for regularization
# -----------------------------
WEIGHT_DECAY = 1e-4       # L2 regularization
LABEL_SMOOTH = 0.1        # label smoothing for noisy pitch labels
PATIENCE     = 8          # epochs with no val-loss improvement before early stop

BEST_MODEL_PATH = "yamamoto_rnn_v3_best.pt"
LOSS_CURVE_PATH = "loss_curve_v3.png"
ACC_CURVE_PATH  = "accuracy_curve_v3.png"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("yamamoto_v3_pitches_2025.csv")
df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)

# -----------------------------
# Pitch Type Distribution (Ground Truth)
# -----------------------------
pitch_counts = df["pitch_type"].value_counts(normalize=True).sort_index()
pitch_counts_raw = df["pitch_type"].value_counts().sort_index()

print("\n--- TRUE PITCH DISTRIBUTION (V3) ---")
for pitch in pitch_counts.index:
    pct = 100 * pitch_counts[pitch]
    raw = pitch_counts_raw[pitch]
    print(f"{pitch:>3s} : {pct:6.2f}%  ({raw} pitches)")

# -----------------------------
# Train / Val split by game
# -----------------------------
game_ids = df["game_pk"].unique()
np.random.shuffle(game_ids)

split_idx = int(0.8 * len(game_ids))
train_games = set(game_ids[:split_idx])
val_games   = set(game_ids[split_idx:])

train_df = df[df["game_pk"].isin(train_games)].copy()
val_df   = df[df["game_pk"].isin(val_games)].copy()

train_pas = train_df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()
val_pas   = val_df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()

train_dataset = YamaPADataset(train_df, train_pas, NUMERIC_COLS)
val_dataset   = YamaPADataset(val_df, val_pas, NUMERIC_COLS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=yama_collate,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=yama_collate,
)

# -----------------------------
# Model setup
# -----------------------------
num_pitch_types       = int(df["pitch_type_idx"].max() + 1)
num_prev_pitch_tokens = int(df["prev_pitch_idx"].max() + 1)
num_batter_hands      = int(df["batter_hand_idx"].max() + 1)
num_prev_result_tokens = int(df["prev_pitch_result_idx"].max() + 1)

model = YamamotoPitchRNN(
    num_pitch_types,
    num_prev_pitch_tokens,
    num_batter_hands,
    num_prev_result_tokens,
    input_numeric_dim=len(NUMERIC_COLS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

# -----------------------------
# Class weights for imbalance
# -----------------------------
pitch_freq = df["pitch_type_idx"].value_counts().sort_index()
inv_freq = 1.0 / pitch_freq.values
alpha = 0.25  # slightly softer reweighting than 0.5 to reduce overfitting on rare pitches
weights = inv_freq ** alpha
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# -----------------------------
# Loss, optimizer, scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(
    weight=weights,
    ignore_index=-100,
    label_smoothing=LABEL_SMOOTH,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Cosine annealing over full training; you can also use StepLR or ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS
)

# -----------------------------
# Tracking
# -----------------------------
train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []

best_val_loss = float("inf")
epochs_no_improve = 0

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(1, NUM_EPOCHS + 1):
    # -----------------------------
    # Train phase
    # -----------------------------
    model.train()
    total_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch in train_loader:
        prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch
        prev_b, hand_b, prev_res_b, num_b, labels_b = (
            prev_b.to(DEVICE),
            hand_b.to(DEVICE),
            prev_res_b.to(DEVICE),
            num_b.to(DEVICE),
            labels_b.to(DEVICE),
        )

        logits = model(prev_b, hand_b, prev_res_b, num_b)
        B, T, C = logits.shape

        loss = criterion(
            logits.view(B * T, C),
            labels_b.view(B * T),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

        # Training accuracy for this batch
        with torch.no_grad():
            preds = logits.argmax(dim=-1)          # (B, T)
            valid_mask = labels_b != -100          # ignore padded labels
            correct = (preds[valid_mask] == labels_b[valid_mask]).sum().item()
            total = valid_mask.sum().item()

            running_correct += correct
            running_total += total

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc  = running_correct / running_total if running_total > 0 else 0.0

    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    # -----------------------------
    # Validation phase
    # -----------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch
            prev_b, hand_b, prev_res_b, num_b, labels_b = (
                prev_b.to(DEVICE),
                hand_b.to(DEVICE),
                prev_res_b.to(DEVICE),
                num_b.to(DEVICE),
                labels_b.to(DEVICE),
            )

            logits = model(prev_b, hand_b, prev_res_b, num_b)
            B, T, C = logits.shape

            loss = criterion(
                logits.view(B * T, C),
                labels_b.view(B * T),
            )
            val_loss += loss.item()

            preds = logits.argmax(dim=-1)
            valid_mask = labels_b != -100
            correct += (preds[valid_mask] == labels_b[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc = correct / total if total > 0 else 0.0
    val_accs.append(val_acc)

    # -----------------------------
    # Scheduler step
    # -----------------------------
    scheduler.step()

    # -----------------------------
    # Early stopping & best model saving
    # -----------------------------
    improved = avg_val_loss < best_val_loss - 1e-4  # small tolerance
    if improved:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_flag = "*"
    else:
        epochs_no_improve += 1
        best_flag = " "

    # Logging
    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch:02d} | "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Train Acc: {avg_train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} {best_flag}"
    )

    if epochs_no_improve >= PATIENCE:
        print(
            f"\nEarly stopping triggered after {epoch} epochs "
            f"(no val-loss improvement for {PATIENCE} epochs)."
        )
        break

print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Best model saved to: {BEST_MODEL_PATH}")

# -----------------------------
# Plots
# -----------------------------
# Loss curve
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.tight_layout()
plt.savefig(LOSS_CURVE_PATH, dpi=200)
plt.close()
print(f"Saved loss curve to {LOSS_CURVE_PATH}")

# Accuracy curve
plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs,   label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.tight_layout()
plt.savefig(ACC_CURVE_PATH, dpi=200)
plt.close()
print(f"Saved accuracy curve to {ACC_CURVE_PATH}")
