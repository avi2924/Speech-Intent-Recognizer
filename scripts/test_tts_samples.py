import os
import sys
import torch
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test_model module instead of specific functions
import scripts.test_model as test_model

def setup_report_folder(folder_name="model_analysis"):
    """Create folder for storing report visuals"""
    report_dir = os.path.join("checkpoints", folder_name)
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

def load_model(model_path, label_map_path, device):
    """Load model and label map"""
    # Load the label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    # Determine the number of classes from the checkpoint, not the label map
    # First load a temporary model to check the weights
    from models.models import CNNAudioGRU
    
    # Load the model weights temporarily to check the number of classes
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is raw state dict or has a state_dict key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Extract number of classes from the final layer weights
    if 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
        print(f"Detected {num_classes} classes in the model checkpoint")
    else:
        # Fallback to label map count
        num_classes = len(label_map)
        print(f"Using {num_classes} classes from label map")
    
    # Create model with the correct number of classes
    model = CNNAudioGRU(num_classes=num_classes)
    
    # Now load the weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Check if the label map has enough classes
    if len(label_map) < num_classes:
        print(f"Warning: Label map has {len(label_map)} classes, but model has {num_classes} classes.")
        print("This may cause prediction errors if the model predicts a class not in your label map.")
    
    return model, label_map

def process_single_audio(model, audio_path, inv_label_map, device):
    """Process a single audio file and return predictions"""
    # Use test_model module's functions directly
    features = test_model.extract_features(audio_path)
    
    if features is None:
        return None
    
    # Convert to tensor and add batch dimension
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predictions
        confidence, prediction = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        prediction = prediction.item()
        
        # Get top 3 predictions
        values, indices = torch.topk(probabilities, min(3, len(inv_label_map)), dim=1)
        top_predictions = []
        
        for i, (value, index) in enumerate(zip(values[0], indices[0])):
            intent = inv_label_map[index.item()]
            top_predictions.append({
                'rank': i+1,
                'label': intent,
                'probability': value.item()
            })
    
    # Create result dictionary
    result = {
        'intent': inv_label_map[prediction],
        'confidence': confidence,
        'top_predictions': top_predictions
    }
    
    return result

def test_audio_files(model_path, audio_dir, label_map_path, details_csv=None, report_dir="tts_test_results"):
    """Test model on all audio files in a directory"""
    # Create report directory
    report_dir = setup_report_folder(report_dir)
    print(f"Results will be saved to: {report_dir}")
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map = load_model(model_path, label_map_path, device)
    
    # Create inverse label map
    inv_label_map = {v: k for k, v in label_map.items()}
    
    # Find all audio files
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav"))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    # Load details if available
    if details_csv:
        details_path = Path(details_csv)
        if not details_path.exists() and (audio_dir / "details.csv").exists():
            details_path = audio_dir / "details.csv"
        
        if details_path.exists():
            details_df = pd.read_csv(details_path)
            # Create a mapping from filename to details
            filename_to_details = {row['filename']: row for _, row in details_df.iterrows()}
        else:
            filename_to_details = None
    else:
        # Check if details.csv exists in audio_dir
        if (audio_dir / "details.csv").exists():
            details_df = pd.read_csv(audio_dir / "details.csv")
            filename_to_details = {row['filename']: row for _, row in details_df.iterrows()}
        else:
            filename_to_details = None
    
    # Process each audio file
    results = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        if audio_file.name == "details.csv":
            continue
            
        # Get expected label if available
        expected_label = None
        if filename_to_details and audio_file.name in filename_to_details:
            expected_label = filename_to_details[audio_file.name]['class']
            expected_text = filename_to_details[audio_file.name]['text']
        else:
            expected_text = None
        
        # Process audio
        try:
            # Make prediction using our custom function
            result = process_single_audio(model, str(audio_file), inv_label_map, device)
            
            if result:
                # Add to results
                results.append({
                    'filename': audio_file.name,
                    'text': expected_text,
                    'expected_label': expected_label,
                    'predicted_label': result['intent'],
                    'confidence': result['confidence'],
                    'correct': expected_label == result['intent'] if expected_label else None,
                    'top_predictions': result.get('top_predictions', [])
                })
            else:
                print(f"Failed to process audio file: {audio_file}")
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    results_df.to_csv(os.path.join(report_dir, "detailed_results.csv"), index=False)
    
    # Only continue with evaluation if we have expected labels
    if 'expected_label' in results_df.columns and results_df['expected_label'].notna().any():
        # Filter out rows without expected labels
        eval_df = results_df.dropna(subset=['expected_label'])
        
        # Calculate accuracy
        accuracy = eval_df['correct'].mean() * 100
        print(f"\nOverall accuracy: {accuracy:.2f}%")
        
        # Generate classification report
        report = classification_report(
            eval_df['expected_label'], 
            eval_df['predicted_label'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(report_dir, "classification_report.csv"))
        
        # Create confusion matrix
        labels = sorted(eval_df['expected_label'].unique())
        cm = confusion_matrix(
            eval_df['expected_label'],
            eval_df['predicted_label'],
            labels=labels
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "confusion_matrix.png"), dpi=300)
        plt.close()
        
        # Plot class accuracy
        plt.figure(figsize=(14, 8))
        class_accuracy = eval_df.groupby('expected_label')['correct'].mean().sort_values(ascending=False)
        sns.barplot(x=class_accuracy.index, y=class_accuracy.values)
        plt.xlabel('Intent Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "class_accuracy.png"), dpi=300)
        plt.close()
        
        # Plot confidence distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=eval_df, x='confidence', hue='correct', bins=20, 
                     multiple='stack', palette=['red', 'green'])
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution (Green: Correct, Red: Incorrect)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "confidence_distribution.png"), dpi=300)
        plt.close()
        
    print(f"Results and visualizations saved to: {report_dir}")
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Test model on TTS generated audio files")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--label_map", type=str, required=True, help="Path to the label map JSON file")
    parser.add_argument("--details_csv", type=str, help="Path to the details CSV file (optional)")
    parser.add_argument("--report_dir", type=str, default="tts_test_results", help="Directory to save test results")
    
    args = parser.parse_args()
    test_audio_files(args.model, args.audio_dir, args.label_map, args.details_csv, args.report_dir)

if __name__ == "__main__":
    main()