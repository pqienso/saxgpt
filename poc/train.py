import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast # Correct import for AMP
import typing as tp
import argparse
import random
from time import time
from tqdm import tqdm

from model import MHAModel

# --- Global Constants ---
RANDOM_SEED = 364298472
TRAIN = 0.8
VALIDATION = 0.1
EFFECTIVE_BSZ = 128
MODEL_CONFIG = dict(
    d_model=128,
    nhead=8,
    num_decoder_layers=8,
    num_encoder_layers=8,
    dim_feedforward=512,
    vocab_size=2049, # Added vocab_size to config for consistency
)
# Note: The vocab_size should be 2048 (for tokens) + 1 (for padding_idx) = 2049

# --- Dataset Class with fixes ---
class SequenceDataset(TensorDataset):
    def __init__(
        self,
        data: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        seq_len: int = 1500,
        stride: int = 750,
        padding_idx: int = 2048, # Added padding_idx as an argument
    ):
        src = []
        tgt = []
        for index, (backing, lead) in enumerate(data):
            if backing.shape[-1] < seq_len:
                continue
            src.append(backing.unfold(-1, seq_len, stride).transpose(0, 1))
            tgt.append(lead.unfold(-1, seq_len, stride).transpose(0, 1))
        src = torch.concat(src)
        tgt = torch.concat(tgt)
        # Use a lambda to pass the padding_idx to the static method via vmap
        src = torch.vmap(lambda s: self.add_delay_interleaving(s, padding_idx))(src).to(device)
        tgt = torch.vmap(lambda t: self.add_delay_interleaving(t, padding_idx))(tgt).to(device)
        return super().__init__(src, tgt)

    @staticmethod
    def add_delay_interleaving(
        streams: torch.Tensor, padding_idx: int = 2048
    ) -> torch.Tensor:
        num_streams = len(streams)
        new_streams = []
        for index, stream in enumerate(streams):
            new_streams.append(
                F.pad(stream, (index + 1, num_streams - index), value=padding_idx)
            )
        return torch.stack(new_streams)

    @staticmethod
    def remove_delay_interleaving(streams: torch.Tensor) -> torch.Tensor:
        num_streams = len(streams)
        stream_length = streams.shape[-1]
        new_streams = []
        for index, stream in enumerate(streams):
            new_streams.append(
                torch.narrow(
                    stream, -1, 1 + index, stream_length - (num_streams - 1) - 2
                )
            )
        return torch.stack(new_streams)


# --- Checkpoint Functions ---
def save_checkpoint(model, optimizer, epoch, loss, config, filepath):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(lr: float, filepath, device: torch.device, optimizer=None):
    checkpoint = torch.load(filepath, map_location=device) # Map to the correct device
    config = checkpoint["config"]
    # Pass the device to the model constructor
    model = MHAModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Move optimizer state to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss, config


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset containing audio codes")
    parser.add_argument("checkpoint", type=str, help="checkpoint file location")
    parser.add_argument("hours", type=float, help="hours to run")
    parser.add_argument(
        "--lr", type=float, help="learning rate of optimizer", default=1e-3
    )
    parser.add_argument("--bsz", type=int, help="batch size", default=32)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=100)
    parser.add_argument("--warmup-steps", type=int, default=500, help="Number of warmup steps for the learning rate")
    args = parser.parse_args()

    # --- Device Setup ---
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Exiting.")
        exit()
    device = torch.device("cuda")
    print(f"Using device {device}")

    # --- Dataset Loading and Splitting ---
    dataset = torch.load(args.dataset)
    train_len = int(TRAIN * len(dataset))
    val_len = int(VALIDATION * len(dataset))
    random.seed(RANDOM_SEED)
    random.shuffle(dataset)
    train_codes, val_codes, test_codes = (
        dataset[:train_len],
        dataset[train_len : train_len + val_len],
        dataset[train_len + val_len :],
    )

    # --- Model, Optimizer, and Checkpoint Loading ---
    # Use the model config from the global variable if no checkpoint exists
    try:
        model, optimizer, previous_epochs, loss, model_config = load_checkpoint(
            args.lr, args.checkpoint, device
        )
        print(f"Loaded checkpoint from epoch {previous_epochs} with loss {loss:.4f}")
    except FileNotFoundError:
        print("Checkpoint not found. Starting training from scratch.")
        # Ensure vocab_size is in MODEL_CONFIG for PositionalEncoding
        model = MHAModel(**MODEL_CONFIG).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        previous_epochs = 0
        loss = float('inf') # Initialize loss
        model_config = MODEL_CONFIG
        
    # Get padding index from model config for consistency
    padding_idx = model_config.get('vocab_size', 2048)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx).to(device)
    
    # Pass padding_idx to the dataset class
    train_ds = SequenceDataset(train_codes, device=device, padding_idx=padding_idx)
    val_ds = SequenceDataset(val_codes, device=device, padding_idx=padding_idx)

    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.bsz)
    scaler = GradScaler(device="cuda")
    start_time = time()

    # --- Training Loop ---
    # Initialize these outside the loop for the progress bar to access
    batch_acc = 0.0
    total_norm = 0.0
    current_batch_count = 0
    
    # For cumulative warmup steps
    total_batches_processed = previous_epochs * len(train_loader)

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_non_pad = 0
        
        # Move zero_grad() to after the accumulation step

        progress_bar = tqdm(train_loader, leave=False)

        for batch_idx, (src, tgt) in enumerate(progress_bar):
            
            # --- Learning Rate Warmup ---
            total_batches_processed += 1
            lr_scale = min(total_batches_processed / args.warmup_steps, 1.0)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * lr_scale
            
            # Update progress bar description with running averages
            progress_bar.set_description(
                f"Epoch {epoch + 1 + previous_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Batch Loss: {loss:.4f} | Batch Acc: {batch_acc:.2f}% | Grad Norm: {total_norm:.2f}"
            )

            # --- Forward Pass with AMP ---
            with autocast(device_type="cuda"):
                output = model(src, tgt[:, :, :-1])
                output = output.view(-1, model.vocab_size)
                targets = tgt[:, :, 1:].contiguous().view(-1)
                loss = criterion(output, targets)
            
            # --- Backward Pass and Gradient Accumulation ---
            # Scale the loss for AMP
            scaler.scale(loss).backward()
            
            current_batch_count += args.bsz
            
            if current_batch_count >= EFFECTIVE_BSZ or batch_idx == len(train_loader) - 1:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                # print(f"Clipped Grad Norm: {total_norm:.4f}") # Uncomment for debugging
                
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()
                current_batch_count = 0

            # --- Metrics Calculation (on the current batch) ---
            epoch_loss += loss.item() # This accumulates the loss for the epoch average
            
            with torch.no_grad():
                predicted = output.argmax(-1)
                non_pad_mask = targets != padding_idx
                total_correct += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
                total_non_pad += non_pad_mask.sum().item()
                
                # Calculate current batch accuracy for the progress bar
                batch_acc = 100 * (predicted[non_pad_mask] == targets[non_pad_mask]).float().mean().item()


        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                output = model(src, tgt[:, :, :-1])
                output = output.view(-1, model.vocab_size)
                targets = tgt[:, :, 1:].contiguous().view(-1)

                val_loss += criterion(output, targets).item()

                predicted = output.argmax(-1)
                non_pad_mask = targets != padding_idx
                val_correct += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
                val_total += non_pad_mask.sum().item()

        # --- Epoch Summary ---
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * total_correct / (total_non_pad + 1e-8)
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / (val_total + 1e-8)

        print(f"\nEpoch {epoch + 1 + previous_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("______________________________________________\n")

        # --- Save Checkpoint ---
        save_checkpoint(
            model,
            optimizer,
            epoch + 1 + previous_epochs,
            val_loss,
            model_config,
            filepath=args.checkpoint,
        )
        
        # --- Time-based Early Stopping ---
        if time() - start_time > args.hours * 3600:
            print(f"Stopping training after {args.hours} hours.")
            break