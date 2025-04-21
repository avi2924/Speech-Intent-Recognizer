import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
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

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Load train dataset
    train_dataset = FSCIntentDataset(
        csv_path=args.train_csv,
        label_map_path=args.label_map
    )

    # Load validation dataset
    val_dataset = FSCIntentDataset(
        csv_path=args.val_csv,
        label_map_path=args.label_map
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    # Load model
    model = CNNAudioGRU(num_classes=config['num_labels']).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_val_acc = 0
    early_stop_counter = 0
    os.makedirs(config['save_path'], exist_ok=True)

    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        all_preds, all_labels = [], []

        for mel, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training"):
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
        logger.info(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {sum(train_losses)/len(train_losses):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Epoch {epoch+1} | Train Loss: {sum(train_losses)/len(train_losses):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_path'], "best_model.pt"))
            logger.info(f"✅ Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
            print("✅ Best model saved!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= config['early_stop_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            print(f"Early stopping at epoch {epoch+1}")
            break


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
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--train_csv', type=str, default='data/processed/train_data.csv',
                       help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='data/processed/valid_data.csv',
                       help='Path to validation CSV file')
    parser.add_argument('--label_map', type=str, default='data/processed/label_map.json',
                       help='Path to label map JSON file')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Log the start of training
    logger.info(f"Starting training with config: {config}")
    train(args, config)
    logger.info("Training completed.")