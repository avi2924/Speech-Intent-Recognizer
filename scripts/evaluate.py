import os
import json
import torch
import argparse
import logging
import numpy as np
import yaml
import time
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


import sys
import os
# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataset import FSCIntentDataset
from models.models import CNNAudioGRU
from scripts.train import collate_fn  # Reuse the collate_fn from training

# Set up logging
log_file = "evaluation.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # Append to the log file
)
logger = logging.getLogger()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate(args, config):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = FSCIntentDataset(
        csv_path=args.test_csv,
        label_map_path=args.label_map
    )
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn
    )
    logger.info("Test DataLoader created")
    
    # Load model
    model = CNNAudioGRU(num_classes=config['num_labels']).to(device)
    
    try:
        # Load model weights
        model_path = args.model_path
        if model_path.endswith('.pt'):
            # Direct model state dict
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # Full checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    logger.info("Model set to evaluation mode")
    
    # Perform evaluation
    all_preds = []
    all_targets = []
    
    # Load label map to get class names
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    
    # Invert label map to get class names
    id_to_class = {idx: cls for cls, idx in label_map.items()}
    
    logger.info("Running evaluation...")
    print("Starting evaluation...")
    
    # Evaluate without gradients
    total_batches = len(test_loader)
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (mel, label) in enumerate(test_loader):
            # Display progress
            if batch_idx % 10 == 0:
                progress = (batch_idx / total_batches) * 100
                elapsed = time.time() - eval_start_time
                batch_time = elapsed / (batch_idx + 1) if batch_idx > 0 else 0
                eta = batch_time * (total_batches - batch_idx) if batch_idx > 0 else 0
                
                print(f"\rEvaluating: {batch_idx}/{total_batches} [{progress:.1f}%] - ETA: {eta:.0f}s", 
                      end="", flush=True)
                
                # Log occasionally to file
                if batch_idx % 50 == 0:
                    logger.info(f"Processing batch {batch_idx}/{total_batches}")
            
            # Move data to device
            mel, label = mel.to(device), label.to(device)
            
            # Forward pass
            outputs = model(mel)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
            
            # Clean up memory
            del mel, label, outputs, preds
            
            # Periodic memory cleanup
            if batch_idx % 100 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
    
    # Print newline after progress display
    print()
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Generate classification report
    target_names = list(label_map.keys())
    class_report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Save results
    results_dir = os.path.join(config['save_path'], "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save classification report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(class_report)
    
    # Save confusion matrix
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)
    
    # Print summary
    total_time = time.time() - eval_start_time
    eval_summary = f"Evaluation completed in {total_time:.2f}s. Test accuracy: {accuracy:.4f}"
    logger.info(eval_summary)
    print(eval_summary)
    logger.info(f"Detailed results saved to {results_dir}")
    print(f"Detailed results saved to {results_dir}")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained intent recognition model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration YAML file")
    parser.add_argument("--test_csv", type=str, default="data/processed/test_data.csv",
                       help="Path to the test CSV file")
    parser.add_argument("--label_map", type=str, default="data/processed/label_map.json",
                       help="Path to the label map JSON file")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to the saved model (defaults to save_path/best_model.pt from config)")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Set default model path if not provided
    if args.model_path is None:
        args.model_path = os.path.join(config['save_path'], "best_model.pt")
    
    # Log the start of evaluation
    logger.info(f"Starting evaluation with config: {config}")
    evaluate(args, config)
    logger.info("Evaluation completed.")