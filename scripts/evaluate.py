import os
import json
import torch
import argparse
import logging
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

def evaluate_model(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Load the test dataset
    test_dataset = FSCIntentDataset(
        csv_path=args.test_csv,
        label_map_path=args.label_map
    )
    
    # Load label map for reporting
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    
    # Invert the label map for interpretation
    inv_label_map = {v: k for k, v in label_map.items()}
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,  # Reduced for stability
        collate_fn=collate_fn
    )

    # Load the model
    model = CNNAudioGRU(num_classes=config['num_labels']).to(device)
    
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        return

    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []
    
    # Add error handling to the evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                mel, label = batch
                mel, label = mel.to(device), label.to(device)
                outputs = model(mel)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue  # Skip this batch and continue
    
    # Check if we successfully processed any samples
    if len(all_preds) == 0:
        print("No samples were successfully processed. Cannot calculate metrics.")
        logger.error("No samples were successfully processed. Cannot calculate metrics.")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=list(range(len(label_map))), 
        target_names=[inv_label_map[i] for i in range(len(label_map))],
        digits=4
    )
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Log and print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + report)
    
    # Save results to file
    save_dir = os.path.dirname(args.model_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save confusion matrix
    np.savetxt(
        os.path.join(save_dir, "confusion_matrix.csv"), 
        conf_matrix, 
        delimiter=",", 
        fmt="%d"
    )
    
    # Generate per-class accuracy
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    class_results = {inv_label_map[i]: float(acc) for i, acc in enumerate(class_acc)}
    
    with open(os.path.join(save_dir, "per_class_accuracy.json"), "w") as f:
        json.dump(class_results, f, indent=2)
    
    # Save misclassified examples for analysis
    misclassified = []
    for i, (true, pred) in enumerate(zip(all_labels, all_preds)):
        if true != pred:
            try:
                audio_path = test_dataset.data.iloc[i]['audio_path']
                misclassified.append({
                    "audio_path": audio_path,
                    "true_label": int(true),
                    "true_intent": inv_label_map[int(true)],
                    "pred_label": int(pred),
                    "pred_intent": inv_label_map[int(pred)]
                })
            except Exception as e:
                print(f"Error processing misclassified example {i}: {str(e)}")
                continue
    
    with open(os.path.join(save_dir, "misclassified.json"), "w") as f:
        json.dump(misclassified, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")
    logger.info(f"Results saved to {save_dir}")
    
    # Optional: Generate a more visually appealing confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                   xticklabels=[inv_label_map[i] for i in range(len(label_map))],
                   yticklabels=[inv_label_map[i] for i in range(len(label_map))])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        print("Confusion matrix visualization saved.")
    except Exception as e:
        print(f"Could not generate confusion matrix visualization: {str(e)}")


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
    evaluate_model(args, config)
    logger.info("Evaluation completed.")