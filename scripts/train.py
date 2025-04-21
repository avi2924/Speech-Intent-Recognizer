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
import sys
import time

# Set up logging
log_file = "training.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # Append to the log file
)
logger = logging.getLogger()

# Update your collate_fn function
def collate_fn(batch):
    """
    Custom collate function to pad or truncate Mel spectrograms to a fixed size.
    """
    max_length = 200  # Set a fixed length for the time dimension
    mel_specs, labels = [], []

    for mel, label in batch:
        # Check if we actually got a valid tensor
        if mel is None or mel.shape[0] == 0 or mel.shape[1] == 0:
            print(f"Warning: Invalid mel shape for label {label}, using zeros")
            mel = torch.zeros((64, max_length))
        
        # Pad or truncate the Mel spectrogram
        if mel.size(1) > max_length:
            mel = mel[:, :max_length]  # Truncate
        else:
            padding = max_length - mel.size(1)
            mel = torch.nn.functional.pad(mel, (0, padding))  # Pad

        mel_specs.append(mel)
        labels.append(label)

    # Stack the tensors into a batch
    try:
        mel_specs = torch.stack(mel_specs)
        labels = torch.tensor(labels)
    except Exception as e:
        print(f"Error stacking tensors: {e}")
        # Print the shapes for debugging
        print(f"Mel spec shapes: {[m.shape for m in mel_specs]}")
        # Fall back to a dummy batch
        mel_specs = torch.zeros((len(mel_specs), 64, max_length))
        labels = torch.tensor(labels)

    return mel_specs, labels

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Modify the train function to implement progress bars correctly
def train(args, config):
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    
    # Enable CuDNN benchmarking for performance
    torch.backends.cudnn.benchmark = True
    
    # Load datasets
    print("Loading train dataset...")
    train_dataset = FSCIntentDataset(
        csv_path=args.train_csv,
        label_map_path=args.label_map
    )
    print(f"Train dataset loaded with {len(train_dataset)} samples")
    logger.info(f"Train dataset loaded with {len(train_dataset)} samples")

    print("Loading validation dataset...")
    val_dataset = FSCIntentDataset(
        csv_path=args.val_csv,
        label_map_path=args.label_map
    )
    print(f"Validation dataset loaded with {len(val_dataset)} samples")
    logger.info(f"Validation dataset loaded with {len(val_dataset)} samples")

    # Create dataloaders
    num_workers = config.get('num_workers', 0)
    print(f"Creating data loaders with num_workers={num_workers}...")
    logger.info(f"Creating data loaders with num_workers={num_workers}...")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # Initialize model
    print("Initializing model...")
    logger.info("Initializing model...")
    model = CNNAudioGRU(num_classes=config['num_labels']).to(device)
    print("Model initialized and moved to device")
    logger.info("Model initialized and moved to device")
    
    # Loss and optimizer
    print("Initializing loss function and optimizer...")
    logger.info("Initializing loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    learning_rate = float(config['lr'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Using Adam optimizer with learning rate: {learning_rate}")
    
    # Learning rate scheduler - set verbose=False to avoid warnings
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=False
    )
    logger.info("Using ReduceLROnPlateau scheduler")
    
    # Check for checkpoints
    os.makedirs(config['save_path'], exist_ok=True)
    checkpoint_path = os.path.join(config['save_path'], "latest_checkpoint.pt")
    start_epoch = 0
    best_val_acc = 0
    
    if os.path.exists(checkpoint_path) and config.get('resume', False):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
        print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
        logger.info(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
    
    # Training loop
    print("Starting full training...")
    logger.info("Starting full training...")
    early_stop_counter = 0
    
    for epoch in range(start_epoch, config['epochs']):
        # Training phase
        model.train()
        train_losses = []
        all_preds, all_labels = [], []
        
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{config['epochs']} - Training...")
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - Training...")
        
        # Training loop with custom progress display
        batch_start_time = time.time()
        for batch_idx, (mel, label) in enumerate(train_loader):
            # Move data to device
            mel, label = mel.to(device), label.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, label)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            train_losses.append(loss.item())
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            # Log progress at intervals
            if batch_idx % 100 == 0:
                avg_loss = np.mean(train_losses[-100:]) if len(train_losses) > 0 else 0
                progress = (batch_idx / len(train_loader)) * 100
                elapsed = time.time() - batch_start_time
                batch_time = elapsed / (batch_idx + 1)
                eta = batch_time * (len(train_loader) - batch_idx)
                
                progress_msg = f"Batch {batch_idx}/{len(train_loader)} [{progress:.1f}%] - Loss: {avg_loss:.4f}, {batch_time*1000:.1f}ms/batch, ETA: {eta:.0f}s"
                print(f"\r{progress_msg}", end="", flush=True)
                
                # Log to file at wider intervals
                if batch_idx % 500 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {avg_loss:.4f}")
            
            # Clean up memory
            del mel, label, output, loss, preds
            
            # Periodic memory cleanup
            if batch_idx % 200 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
        
        # Print newline after progress display
        print()
        
        # Calculate training metrics
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation phase
        print(f"Epoch {epoch+1}/{config['epochs']} - Validating...")
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - Validating...")
        
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Training time
        epoch_time = time.time() - epoch_start_time
        
        # Log results
        result_msg = (
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        logger.info(result_msg)
        print(result_msg)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'config': config
        }
        torch.save(checkpoint, os.path.join(config['save_path'], "latest_checkpoint.pt"))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_path'], "best_model.pt"))
            # No Unicode character, just plain ASCII
            best_msg = f"[BEST] Model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}"
            logger.info(best_msg)
            print(best_msg)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= config.get('early_stop_patience', 3):
            early_stop_msg = f"Early stopping at epoch {epoch+1}"
            logger.info(early_stop_msg)
            print(early_stop_msg)
            break
            
        # Clean up between epochs
        torch.cuda.empty_cache()
    
    completion_msg = f"Training completed. Best validation accuracy: {best_val_acc:.4f}"
    print(completion_msg)
    logger.info(completion_msg)
    return best_val_acc

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on the provided data loader
    """
    model.eval()
    val_losses = []
    all_preds, all_labels = [], []
    
    # Start evaluation time
    eval_start_time = time.time()
    print("Starting validation...")
    
    # Track batch progress
    total_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_idx, (mel, label) in enumerate(data_loader):
            # Log progress
            if batch_idx % 20 == 0:
                progress = (batch_idx / total_batches) * 100
                print(f"\rValidation: {batch_idx}/{total_batches} [{progress:.1f}%]", end="", flush=True)
                
            # Process batch
            mel, label = mel.to(device), label.to(device)
            outputs = model(mel)
            loss = criterion(outputs, label)
            
            val_losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            # Clean up memory
            del mel, label, outputs, preds
    
    # Print newline after progress display
    print()
    
    # Calculate metrics
    val_loss = sum(val_losses) / len(val_losses)
    val_acc = accuracy_score(all_labels, all_preds)
    
    # Report validation time
    eval_time = time.time() - eval_start_time
    print(f"Validation completed in {eval_time:.2f}s")
    
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
    
    # Load configuration from YAML file
    config = load_config(args.config)
    config['lr'] = float(config['lr'])  # Ensure learning rate is a float
    
    # Log the start of training
    logger.info(f"Starting training with config: {config}")
    train(args, config)
    logger.info("Training completed.")