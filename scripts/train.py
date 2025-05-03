import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchaudio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import CNNAudioGRU
from scripts.dataset import FSCIntentDataset

# Force GPU selection and setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
import torch

def setup_gpu():
    """Setup GPU and print diagnostics"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Optimize performance
        torch.backends.cudnn.enabled = True
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Test GPU with a small tensor operation
        test_tensor = torch.rand(1000, 1000).to(device)
        test_result = test_tensor @ test_tensor.t()
        del test_tensor, test_result  # Clean up
        torch.cuda.empty_cache()
        
        print(f"GPU test successful. CUDA version: {torch.version.cuda}")
        return device
    else:
        print("No GPU available, using CPU.")
        return torch.device("cpu")

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    """Basic collate function for DataLoader"""
    max_length = 200
    mel_specs, labels = [], []
    
    for mel, label in batch:
        if mel is None or mel.shape[0] == 0 or mel.shape[1] == 0:
            continue
        
        if mel.size(1) > max_length:
            mel = mel[:, :max_length]
        elif mel.size(1) < max_length:
            padding = max_length - mel.size(1)
            mel = torch.nn.functional.pad(mel, (0, padding))
        
        mel_specs.append(mel)
        labels.append(label)
    
    if not mel_specs:  # Handle empty batch
        return None, None
    
    return torch.stack(mel_specs), torch.tensor(labels, dtype=torch.long)

def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """Training loop with fixed GPU usage"""
    model.train()
    train_losses = []
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (mel, label) in enumerate(pbar):
        # Skip bad batches
        if mel is None or label is None or mel.size(0) == 0:
            continue
        
        # Explicitly move data to GPU
        mel = mel.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient than False
        
        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(mel)
                loss = criterion(output, label)
                
            # Scale loss and backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            output = model(mel)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        
        # Monitor GPU usage every 10 batches
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / 1024**2
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "GPU": f"{gpu_usage:.1f}MB"})
        else:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_losses.append(loss.item())
            
    return sum(train_losses) / max(len(train_losses), 1)

def validate(model, val_loader, criterion, device, scaler=None):
    """Validation with fixed GPU usage"""
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc="Validating"):
            if mel is None or label is None or mel.size(0) == 0:
                continue
                
            # Move to device
            mel = mel.to(device, non_blocking=True)  
            label = label.to(device, non_blocking=True)
            
            # Forward pass (no need for AMP in validation, but included for completeness)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(mel)
                    loss = criterion(output, label)
            else:
                output = model(mel)
                loss = criterion(output, label)
            
            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            val_losses.append(loss.item())
    
    # Final metrics
    accuracy = correct / max(total, 1)
    avg_loss = sum(val_losses) / max(len(val_losses), 1)
    
    return avg_loss, accuracy

def clear_gpu_memory():
    """Clear GPU cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def train(args, config):
    """Main training function with fixed GPU usage"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # GPU setup
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Force a small tensor operation to initialize CUDA
        torch.cuda.FloatTensor(1).fill_(1)
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Feature caching setup
    cache_dir = config.get('cache_dir', 'data/cached_features')
    use_cache = config.get('use_feature_cache', True)
    
    # Load datasets with caching
    train_dataset = FSCIntentDataset(
        csv_path=args.train_csv,
        label_map_path=args.label_map,
        is_training=True,
        augment_prob=config.get('augment_prob', 0.5),
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    val_dataset = FSCIntentDataset(
        csv_path=args.val_csv,
        label_map_path=args.label_map,
        is_training=False,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    print(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'] * 2,  # Double batch size for validation
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model properly
    model = CNNAudioGRU(num_classes=config.get('num_labels', 31))
    model = model.to(device)  # Explicitly move model to GPU
    
    # Verify model is on GPU
    if next(model.parameters()).device.type != 'cuda' and torch.cuda.is_available():
        print("WARNING: Model is not on GPU! Moving model to GPU now.")
        model = model.to(device)
    
    # Print model memory usage
    if torch.cuda.is_available():
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"Model size: {size_all_mb:.2f} MB")
    
    # Setup optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = float(config.get('lr', 0.0003))
    weight_decay = float(config.get('weight_decay', 0.0001))
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup AMP correctly
    use_amp = config.get('use_amp', True) and torch.cuda.is_available()

    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")
        # Use the new PyTorch 2.0+ GradScaler API
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        print("Mixed precision disabled")
    
    # Training loop
    epochs = config.get('epochs', 20)
    patience = config.get('early_stop_patience', 5)
    best_val_acc = 0
    no_improve_count = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, scaler)
        
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        
        # Save if model improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            
            # Save the model
            save_path = config.get('save_path', 'checkpoints/')
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            print(f"New best model saved with accuracy: {val_acc:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
            
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train intent recognition model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--train_csv', type=str, default=None,
                       help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, default=None,
                       help='Path to validation CSV')
    parser.add_argument('--label_map', type=str, default='data/processed/label_map.json',
                       help='Path to label map JSON file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set args from config if not provided in command line
    if not hasattr(args, 'train_csv') or args.train_csv is None:
        args.train_csv = config.get('train_csv')
    if not hasattr(args, 'val_csv') or args.val_csv is None:
        args.val_csv = config.get('valid_csv')
    
    # Quick GPU test
    if torch.cuda.is_available():
        print("CUDA is available. Testing GPU...")
        with torch.cuda.amp.autocast():
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
        print(f"GPU test result: tensor on {z.device}, shape {z.shape}")
        print("GPU test passed!")
    else:
        print("CUDA is not available. Training will use CPU.")
    
    train(args, config)