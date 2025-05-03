import os
import json
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Extract audio features using torchaudio (faster than librosa)"""
    
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize torchaudio transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Initialize amplitude to dB converter
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
    def extract_features(self, audio_path, max_duration=5.0):
        """Extract mel spectrogram features from audio file"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"File not found: {audio_path}")
                return None
                
            # Load audio with torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Limit duration if needed
            max_samples = int(max_duration * self.sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            # Extract mel spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # Convert to dB scale (like librosa's power_to_db)
            mel_spec = self.amplitude_to_db(mel_spec)
            
            # Remove batch dimension
            mel_spec = mel_spec.squeeze(0)
            
            # Normalize (similar to how librosa does it)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
            
            return mel_spec
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None

def precompute_dataset_features(csv_path, output_dir, label_map_path=None, max_duration=5.0):
    """Precompute and cache all features from a dataset"""
    # Load dataset
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    
    # Print column names for debugging
    logger.info(f"CSV columns: {df.columns.tolist()}")
    logger.info(f"First row sample: {df.iloc[0].to_dict()}")
    
    # Create feature extractor
    extractor = AudioFeatureExtractor()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dataset name from CSV path
    dataset_name = os.path.basename(csv_path).replace('.csv', '')
    
    # Create cache file path
    cache_file = os.path.join(output_dir, f"{dataset_name}_features.pt")
    
    # Process each audio file
    features_dict = {}
    error_count = 0
    
    # Determine which column has the labels
    if 'label' in df.columns:
        label_column = 'label'
    elif 'intent' in df.columns:
        label_column = 'intent'
    elif 'action' in df.columns and 'object' in df.columns:
        # Create label from action and object
        df['label'] = df['action'] + '_' + df['object']
        label_column = 'label'
    else:
        # Use a dummy label
        df['label'] = 'unknown'
        label_column = 'label'
        logger.warning("Could not find label column, using 'unknown' as label")
    
    logger.info(f"Using '{label_column}' column for labels")
    
    for idx in tqdm(range(len(df)), desc=f"Processing {dataset_name}"):
        row = df.iloc[idx]
        audio_path = row['path']
        label = row[label_column]
        
        # Extract features
        mel_spec = extractor.extract_features(audio_path, max_duration)
        
        if mel_spec is not None:
            # Store features
            features_dict[audio_path] = {
                'features': mel_spec,
                'label': label
            }
        else:
            error_count += 1
    
    # Save features
    torch.save(features_dict, cache_file)
    logger.info(f"Saved {len(features_dict)} features to {cache_file}")
    logger.info(f"Failed to process {error_count} files")
    
    # Return the path to the cached features
    return cache_file

def main():
    parser = argparse.ArgumentParser(description="Precompute audio features using torchaudio")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--valid_csv", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output_dir", type=str, default="data/cached_features", help="Output directory for cached features")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label map JSON file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    logger.info("Starting feature precomputation...")
    
    train_cache = precompute_dataset_features(args.train_csv, args.output_dir, args.label_map)
    valid_cache = precompute_dataset_features(args.valid_csv, args.output_dir, args.label_map)
    test_cache = precompute_dataset_features(args.test_csv, args.output_dir, args.label_map)
    
    # Create a cache info file
    cache_info = {
        'train_features': train_cache,
        'valid_features': valid_cache,
        'test_features': test_cache
    }
    
    with open(os.path.join(args.output_dir, "cache_info.json"), 'w') as f:
        json.dump(cache_info, f, indent=2)
    
    logger.info("Feature precomputation complete!")

if __name__ == "__main__":
    main()