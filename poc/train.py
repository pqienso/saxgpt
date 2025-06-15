import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import typing as tp
import argparse
import random
from time import time
from tqdm import tqdm

from model import MHAModel

RANDOM_SEED = 364298472
TRAIN = 0.8
VALIDATION = 0.1
MODEL_CONFIG = dict(
    d_model=128,
    nhead=8,
    num_decoder_layers=8,
    num_encoder_layers=8,
    dim_feedforward=512,
)


class SequenceDataset(TensorDataset):
    def __init__(
        self,
        data: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        seq_len: int = 1500,
        stride: int = 750,
    ):
        src = []
        tgt = []
        for index, (backing, lead) in enumerate(data):
            if backing.shape[-1] < seq_len:
                # print(f"Index {index} has seq_len {backing.shape[-1]}, skipping")
                continue
            src.append(backing.unfold(-1, seq_len, stride).transpose(0, 1))
            tgt.append(lead.unfold(-1, seq_len, stride).transpose(0, 1))
        src = torch.concat(src)
        tgt = torch.concat(tgt)
        src = torch.vmap(self.add_delay_interleaving)(src).to(device)
        tgt = torch.vmap(self.add_delay_interleaving)(tgt).to(device)
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


def save_checkpoint(model, optimizer, epoch, loss, config, filepath):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(lr: float, filepath, optimizer=None):
    checkpoint = torch.load(filepath)
    config = checkpoint["config"]
    model = MHAModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])  # Load model state
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )  # Load optimizer state
    epoch = checkpoint["epoch"]  # Get saved epoch
    loss = checkpoint["loss"]  # Get saved loss (optional)
    return model, optimizer, epoch, loss, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset containing audio codes")
    parser.add_argument("checkpoint", type=str, help="checkpoint file location")
    parser.add_argument("hours", type=float, help="hours to run")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--lr", type=float, help="learning rate of optimizer", default=1e-3
    )
    parser.add_argument("--bsz", type=int, help="batch size", default=32)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=100)
    args = parser.parse_args()

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

    if args.cuda and not torch.cuda.is_available():
        print("Warning: CUDA not available.")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    train_ds = SequenceDataset(train_codes, device=device)
    val_ds = SequenceDataset(val_codes, device=device)
    # test_ds = SequenceDataset(test_codes, device=device)

    model, optimizer, previous_epochs, loss, model_config = load_checkpoint(
        args.lr, args.checkpoint
    )
    criterion = nn.CrossEntropyLoss(ignore_index=2048).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.bsz)
    start_time = time()
    batch_acc = None

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_non_pad = 0

        progress_bar = tqdm(train_loader, leave=False)

        for batch_idx, (src, tgt) in enumerate(progress_bar):
            if batch_idx != 0:
                progress_bar.set_description(
                    # f"Epoch {epoch + 1 + previous_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    # f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Train loss: {loss.item():.4f} | Train acc: {batch_acc:.2f}%"
                )

            optimizer.zero_grad()

            output = model(src, tgt[:, :, :-1])
            output = output.view(-1, model.vocab_size)
            targets = tgt[:, :, 1:].contiguous().view(-1)

            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                predicted = output.argmax(-1)
                non_pad_mask = targets != 2048
                total_correct += (
                    (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
                )
                total_non_pad += non_pad_mask.sum().item()

            batch_acc = (
                100
                * (predicted[non_pad_mask] == targets[non_pad_mask])
                .float()
                .mean()
                .item()
            )

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
                non_pad_mask = targets != 2048
                val_correct += (
                    (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
                )
                val_total += non_pad_mask.sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * total_correct / (total_non_pad + 1e-8)
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / (val_total + 1e-8)

        print(f"\nEpoch {epoch + 1 + previous_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        # print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("______________________________________________\n")

        save_checkpoint(
            model,
            optimizer,
            epoch + 1 + previous_epochs,
            val_loss,
            model_config,
            filepath=args.checkpoint,
        )
        if time() - start_time > args.hours * 3600:
            break
