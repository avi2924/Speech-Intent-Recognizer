import os
import sys
import torch
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import FSCIntentDataset
from models.models import CNNAudioGRU
from scripts.train import collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate(args, config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load label map
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    
    # Create inverse label map for readable results
    inv_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    # Load test dataset
    test_dataset = FSCIntentDataset(
        csv_path=args.test_csv,
        label_map_path=args.label_map,
        is_training=False
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = CNNAudioGRU(num_classes=num_classes).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.info(f"Loaded model from {args.model_path}")
    
    # Evaluate model
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for mel, label in tqdm(test_loader, desc="Evaluating"):
            mel, label = mel.to(device), label.to(device)
            outputs = model(mel)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Create classification report with class names
    target_names = [inv_label_map[i] for i in range(num_classes)]
    cls_report = classification_report(all_labels, all_preds, target_names=target_names)
    logger.info(f"Classification Report:\n{cls_report}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create results directory
    results_dir = os.path.join(config['save_path'], 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save classification report
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(cls_report)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    
    logger.info(f"Evaluation results saved to {results_dir}")
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate speech intent recognition model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--label_map', type=str, required=True, help='Path to label map JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    evaluate(args, config)