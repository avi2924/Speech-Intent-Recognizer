import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from scripts.dataset import FSCIntentDataset
from models.models import CNNAudioGRU
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json
import numpy as np


# Set up logging
log_file = "training.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # Append to the log file
)
logger = logging.getLogger()


def collate_fn(batch):
    """
    Custom collate function to pad or truncate Mel spectrograms to a fixed size.
    """
    max_length = 200  # Set a fixed length for the time dimension
    mel_specs, labels = [], []

    for mel, label in batch:
        # Pad or truncate the Mel spectrogram
        if mel.size(1) > max_length:
            mel = mel[:, :max_length]  # Truncate
        else:
            padding = max_length - mel.size(1)
            mel = torch.nn.functional.pad(mel, (0, padding))  # Pad

        mel_specs.append(mel)
        labels.append(label)

    # Stack the tensors into a batch
    mel_specs = torch.stack(mel_specs)
    labels = torch.tensor(labels)

    return mel_specs, labels


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Load full dataset
    full_dataset = FSCIntentDataset(
        csv_path=args.csv,
        label_map_path=args.label_map
    )

    # Train/Validation split using indices
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=args.val_split, random_state=42, shuffle=True)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load model
    model = CNNAudioGRU(num_classes=len(full_dataset.label_map)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        all_preds, all_labels = [], []

        for mel, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        val_acc = evaluate(model, val_loader, device)

        # Log training and validation metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {sum(train_losses)/len(train_losses):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Epoch {epoch+1} | Train Loss: {sum(train_losses)/len(train_losses):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            logger.info(f"✅ Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
            print("✅ Best model saved!")


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, label in loader:
            mel, label = mel.to(device), label.to(device)
            output = model(mel)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/processed/processed.csv')
    parser.add_argument('--label_map', type=str, default='data/processed/label_map.json')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.2, help="Proportion of validation data")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    args = parser.parse_args()

    # Log the start of training
    logger.info(f"Starting training with args: {args}")
    train(args)
    logger.info("Training completed.")