import os
import sys
import json
import torch
import librosa
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.models import CNNAudioGRU

def setup_logging():
    """Set up basic logging to console"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

# Set up logger
logger = setup_logging()

def load_model(model_path, num_classes, device):
    """Load the trained model"""
    try:
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {device}")
        
        model = CNNAudioGRU(num_classes=num_classes).to(device)
        
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

def extract_features(audio_path):
    """Extract mel spectrogram features from audio file"""
    try:
        logger.info(f"Processing audio file: {audio_path}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
            
        # Load audio file
        logger.info("Loading audio file...")
        y, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"Audio loaded: duration={len(y)/sr:.2f}s, sample rate={sr}Hz")
        
        # Extract mel spectrogram
        logger.info("Extracting mel spectrogram...")
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=1024, 
            hop_length=512, 
            n_mels=64
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Convert to tensor
        mel_spec = torch.FloatTensor(mel_spec)
        logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
        
        # Add batch dimension
        mel_spec = mel_spec.unsqueeze(0)
        
        return mel_spec
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

def predict(model, audio_path, label_map, device):
    """Make a prediction on a single audio file"""
    try:
        # Extract features
        mel_spec = extract_features(audio_path)
        if mel_spec is None:
            return None
        
        # Pad or truncate to expected length
        max_length = 200
        if mel_spec.size(2) > max_length:
            mel_spec = mel_spec[:, :, :max_length]
        else:
            padding = max_length - mel_spec.size(2)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        
        # Move to device
        mel_spec = mel_spec.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(mel_spec)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # Get label from class index
        inv_label_map = {v: k for k, v in label_map.items()}
        predicted_label = inv_label_map.get(pred_class, "Unknown")
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_predictions": get_top_predictions(probs, inv_label_map, k=3)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

def get_top_predictions(probs, inv_label_map, k=3):
    """Get top k predictions with probabilities"""
    probs = probs.cpu().numpy()[0]
    top_indices = np.argsort(probs)[::-1][:k]
    
    return [
        {
            "label": inv_label_map.get(idx, "Unknown"),
            "probability": float(probs[idx])
        }
        for idx in top_indices
    ]

def interactive_test(model, label_map, device):
    """Test the model interactively"""
    print("\n===== INTERACTIVE TESTING =====")
    print("Enter the path to an audio file (or 'q' to quit):")
    
    while True:
        user_input = input("\nAudio file path (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            break
        
        # Check if file exists
        if not os.path.exists(user_input):
            print(f"File not found: {user_input}")
            continue
        
        # Make prediction
        result = predict(model, user_input, label_map, device)
        
        if result is None:
            print("Failed to make prediction. Check logs for details.")
            continue
        
        # Display results
        print("\n----- PREDICTION RESULTS -----")
        print(f"Predicted intent: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        
        print("\nTop predictions:")
        for i, pred in enumerate(result['top_predictions']):
            print(f"  {i+1}. {pred['label']} ({pred['probability']*100:.2f}%)")

def batch_test(model, audio_dir, label_map, device):
    """Test the model on a directory of audio files"""
    print(f"\n===== BATCH TESTING on {audio_dir} =====")
    
    audio_files = [
        os.path.join(audio_dir, f) 
        for f in os.listdir(audio_dir) 
        if f.endswith(('.wav', '.mp3', '.flac'))
    ]
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    results = []
    for audio_path in audio_files:
        print(f"Processing: {os.path.basename(audio_path)}")
        result = predict(model, audio_path, label_map, device)
        
        if result is None:
            print(f"Failed to process {audio_path}")
            continue
            
        result['file'] = os.path.basename(audio_path)
        results.append(result)
    
    # Display summary
    print("\n----- BATCH RESULTS SUMMARY -----")
    for result in results:
        print(f"{result['file']}: {result['predicted_label']} ({result['confidence']*100:.2f}%)")
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Test speech intent recognition model')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                       help='Path to the trained model')
    parser.add_argument('--label_map', type=str, default='data/processed/label_map.json',
                       help='Path to the label map')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to an audio file or directory for testing')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if model and label map exist
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.label_map):
        logger.error(f"Label map not found: {args.label_map}")
        return
    
    # Load label map
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.model, num_classes=31, device=device)  # Use 31 to match the trained model
    if model is None:
        return
    
    # Test mode
    if args.interactive:
        interactive_test(model, label_map, device)
    elif args.audio:
        if os.path.isdir(args.audio):
            batch_test(model, args.audio, label_map, device)
        elif os.path.isfile(args.audio):
            result = predict(model, args.audio, label_map, device)
            if result:
                print("\n----- PREDICTION RESULTS -----")
                print(f"Predicted intent: {result['predicted_label']}")
                print(f"Confidence: {result['confidence']*100:.2f}%")
                
                print("\nTop predictions:")
                for i, pred in enumerate(result['top_predictions']):
                    print(f"  {i+1}. {pred['label']} ({pred['probability']*100:.2f}%)")
        else:
            logger.error(f"Audio path not found: {args.audio}")
    else:
        interactive_test(model, label_map, device)

if __name__ == "__main__":
    main()