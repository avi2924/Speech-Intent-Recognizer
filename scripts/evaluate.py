import os
import json
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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


def evaluate_model(args):
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Load the model
    model = CNNAudioGRU(num_classes=len(label_map)).to(device)
    
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
    
    with torch.no_grad():
        for mel, label in tqdm(test_loader, desc="Evaluating"):
            mel, label = mel.to(device), label.to(device)
            outputs = model(mel)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
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
            audio_path = test_dataset.data.iloc[i]['audio_path']
            misclassified.append({
                "audio_path": audio_path,
                "true_label": int(true),
                "true_intent": inv_label_map[int(true)],
                "pred_label": int(pred),
                "pred_intent": inv_label_map[int(pred)]
            })
    
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
    except:
        print("Could not generate confusion matrix visualization. Make sure matplotlib and seaborn are installed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained intent recognition model")
    parser.add_argument("--test_csv", type=str, default="data/processed/processed.csv",
                       help="Path to the test CSV file")
    parser.add_argument("--label_map", type=str, default="data/processed/label_map.json",
                       help="Path to the label map JSON file")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt",
                       help="Path to the saved model")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Log the start of evaluation
    logger.info(f"Starting evaluation with args: {args}")
    evaluate_model(args)
    logger.info("Evaluation completed.")