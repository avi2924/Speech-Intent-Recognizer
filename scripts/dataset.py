import os
import torch
import json
import pandas as pd
import numpy as np
import librosa
import logging
from torch.utils.data import Dataset

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FSCIntentDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Fluent Speech Commands (FSC) data.
    """
    def __init__(self, csv_path, label_map_path, is_training=True, cache_features=True):
        """
        Args:
            csv_path (str): Path to the preprocessed CSV file
            label_map_path (str): Path to the label map JSON file
            is_training (bool): Flag indicating if the dataset is for training
            cache_features (bool): Flag indicating if features should be cached
        """
        self.data = pd.read_csv(csv_path)
        self.is_training = is_training
        self.cache_features = cache_features
        self.feature_cache = {}
        
        # Handle different column names
        # Map common column names to expected ones
        column_mappings = {
            'file_path': 'path',
            'audio_path': 'path',
            'filepath': 'path',
            'audio_file': 'path',
            'wav_path': 'path',
            'wav_file': 'path',
            'intent': 'label',
            'class': 'label',
            'intent_label': 'label',
            'intent_class': 'label',
        }
        
        # Rename columns if needed
        for old_col, new_col in column_mappings.items():
            if old_col in self.data.columns and new_col not in self.data.columns:
                self.data = self.data.rename(columns={old_col: new_col})
        
        # Create label column if it doesn't exist but we have action and object
        if 'label' not in self.data.columns and 'action' in self.data.columns and 'object' in self.data.columns:
            self.data['label'] = self.data['action'] + '_' + self.data['object']
        
        # Ensure we have the necessary columns
        if 'path' not in self.data.columns:
            raise ValueError(f"CSV file {csv_path} missing 'path' column or equivalent")
        if 'label' not in self.data.columns:
            raise ValueError(f"CSV file {csv_path} missing 'label' column or equivalent")
        
        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
            
        logger.info(f"Loaded dataset with {len(self.data)} samples")
        logger.info(f"Number of classes: {len(self.label_map)}")
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    
    def extract_features(self, audio_path):
        """
        Extracts Mel Spectrogram features from the given audio file.
        """
        # Add caching
        if self.cache_features and audio_path in self.feature_cache:
            return self.feature_cache[audio_path]
        
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None
                
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract mel spectrogram
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
            
            # Cache result before returning
            if self.cache_features:
                self.feature_cache[audio_path] = mel_spec
            
            return torch.FloatTensor(mel_spec)
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        """
        row = self.data.iloc[idx]
        audio_path = row['path']
        
        # Handle label mapping
        label_str = row['label']
        if label_str in self.label_map:
            label = self.label_map[label_str]
        else:
            # If label not in map, use a default or error
            logger.warning(f"Label '{label_str}' not found in label map, using 0")
            label = 0
        
        # Extract features
        mel_spec = self.extract_features(audio_path)
        
        # Handle extraction failures
        if mel_spec is None:
            logger.warning(f"Failed to extract features from {audio_path}, using zeros")
            mel_spec = torch.zeros((64, 100))
            
        return mel_spec, label


if __name__ == '__main__':
    # Example usage of the FSCIntentDataset
    dataset = FSCIntentDataset(
        csv_path='data/processed/train_data.csv',
        label_map_path='data/processed/label_map.json'
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Fetch and display the first sample
    first_sample = dataset[0]
    print(f"First sample Mel Spectrogram shape: {first_sample[0].shape}")
    print(f"First sample Label: {first_sample[1]}")
