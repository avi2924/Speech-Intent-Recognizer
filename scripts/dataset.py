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
    def __init__(self, csv_path, label_map_path, is_training=True):
        """
        Args:
            csv_path (str): Path to the preprocessed CSV file
            label_map_path (str): Path to the label map JSON file
            is_training (bool): Flag indicating if the dataset is for training
        """
        self.data = pd.read_csv(csv_path)
        self.is_training = is_training
        
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
        try:
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
        label = self.label_map[row['label']]
        
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
