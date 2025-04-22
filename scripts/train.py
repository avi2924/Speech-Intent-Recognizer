import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from scripts.dataset import FSCIntentDataset
import sys
sys.path.append('E:/Speech-Intent-Recognition')
from models.models import CNNAudioGRU
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import json
import numpy as np

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

def log_gpu_memory(message=""):
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        logger.info(f"{message} - GPU Memory: {used_mem/1024/1024:.1f}MB / {total_mem/1024/1024:.1f}MB")

def collate_fn(batch):
    max_length = 200
    mel_specs, labels = [], []

    for mel, label in batch:
        if mel is None or mel.shape[0] == 0 or mel.shape[1] == 0:
            mel = torch.zeros((64, max_length))
        
        if mel.size(1) > max_length:
            mel = mel[:, :max_length]
        else:
            padding = max_length - mel.size(1)
            mel = torch.nn.functional.pad(mel, (0, padding))

        mel_specs.append(mel)
        labels.append(label)

    try:
        mel_specs = torch.stack(mel_specs)
        labels = torch.tensor(labels)
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        mel_specs = torch.zeros((len(mel_specs), 64, max_length))
        labels = torch.tensor(labels)

    return mel_specs, labels

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(args, config):
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_dataset = FSCIntentDataset(
        csv_path=args.train_csv,
        label_map_path=args.label_map
    )
    
    val_dataset = FSCIntentDataset(
        csv_path=args.val_csv,
        label_map_path=args.label_map
    )
    
    logger.info(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    num_workers = config.get('num_workers', 4)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Pin memory for faster transfers
        prefetch_factor=2  # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    model = CNNAudioGRU(num_classes=config['num_labels']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    learning_rate = float(config['lr'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=False
    )
    
    os.makedirs(config['save_path'], exist_ok=True)
    checkpoint_path = os.path.join(config['save_path'], "latest_checkpoint.pt")
    start_epoch = 0
    best_val_acc = 0
    
    if os.path.exists(checkpoint_path) and config.get('resume', False):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
    
    logger.info("Starting training...")
    early_stop_counter = 0
    
    for epoch in range(start_epoch, config['epochs']):
        # Training phase
        model.train()
        train_losses = []
        all_preds, all_labels = [], []
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} (Train)")
        for batch_idx, (mel, label) in enumerate(train_loader_tqdm):
            # Move data to device immediately
            mel = mel.to(device, non_blocking=True)  # non_blocking speeds up transfers
            label = label.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            avg_loss = np.mean(train_losses[-100:]) if train_losses else 0
            train_loader_tqdm.set_postfix(loss=f"{avg_loss:.4f}")
        
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation phase
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        result_msg = (
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        logger.info(result_msg)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'config': config
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_path'], "best_model.pt"))
            logger.info(f"[BEST] Model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= config.get('early_stop_patience', 3):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_losses = []
    all_preds, all_labels = [], []
    
    val_loader_tqdm = tqdm(data_loader, desc="Validation")
    
    with torch.no_grad():
        for mel, label in val_loader_tqdm:
            mel, label = mel.to(device), label.to(device)
            outputs = model(mel)
            loss = criterion(outputs, label)
            
            val_losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    val_loss = sum(val_losses) / len(val_losses)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_acc, val_loss

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
    
    config = load_config(args.config)
    train(args, config)