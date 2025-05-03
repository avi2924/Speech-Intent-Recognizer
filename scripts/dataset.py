import os
import json
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class FSCIntentDataset(Dataset):
    """Fluent Speech Commands dataset with feature caching and torchaudio support"""
    
    def __init__(self, csv_path, label_map_path, is_training=True, augment_prob=0.5, 
                 use_cache=True, cache_dir="data/cached_features", mel_spec_length=200):
        """
        Initialize dataset with caching support
        Args:
            csv_path: Path to dataset CSV
            label_map_path: Path to label map JSON
            is_training: Whether this is training set (for augmentation)
            augment_prob: Probability of applying augmentation
            use_cache: Whether to use cached features
            cache_dir: Directory for cached features
            mel_spec_length: Target length for mel spectrograms
        """
        self.data = pd.read_csv(csv_path)
        self.sample_rate = 16000
        self.is_training = is_training
        self.augment_prob = augment_prob if is_training else 0.0
        self.n_mels = 64
        self.mel_spec_length = mel_spec_length
        self.use_cache = use_cache
        
        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Setup feature caching
        self.in_memory_cache = {}
        
        # Check for precomputed features
        if use_cache:
            dataset_name = os.path.basename(csv_path).replace('.csv', '')
            self.cache_file = os.path.join(cache_dir, f"{dataset_name}_features.pt")
            
            if os.path.exists(self.cache_file):
                logger.info(f"Loading cached features from {self.cache_file}")
                self.features_dict = torch.load(self.cache_file)
                logger.info(f"Loaded {len(self.features_dict)} cached features")
            else:
                logger.info(f"No cached features found at {self.cache_file}")
                self.features_dict = {}
        else:
            self.features_dict = {}
            
        # Initialize torchaudio transforms for feature extraction
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Initialize augmentations
        if is_training:
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        
        logger.info(f"Initialized dataset with {len(self.data)} samples, {len(self.label_map)} classes")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item with caching support"""
        audio_path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['label']
        
        # Map label text to integer
        label_id = self.label_map.get(label, 0)
        
        # Check in-memory cache first
        if audio_path in self.in_memory_cache:
            mel_spec = self.in_memory_cache[audio_path]
        
        # Check disk cache next
        elif audio_path in self.features_dict:
            mel_spec = self.features_dict[audio_path]['features']
            # Store in memory for faster access next time
            self.in_memory_cache[audio_path] = mel_spec
            
        # Extract features if not cached
        else:
            mel_spec = self.extract_features(audio_path)
            
            # Add to in-memory cache
            if mel_spec is not None:
                self.in_memory_cache[audio_path] = mel_spec
        
        # Apply augmentation during training
        if self.is_training and np.random.random() < self.augment_prob:
            mel_spec = self.augment_features(mel_spec)
        
        # Handle pad/trim to target length
        if mel_spec.size(1) > self.mel_spec_length:
            mel_spec = mel_spec[:, :self.mel_spec_length]
        elif mel_spec.size(1) < self.mel_spec_length:
            padding = self.mel_spec_length - mel_spec.size(1)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        
        return mel_spec, label_id
    
    def extract_features(self, audio_path):
        """Extract features using torchaudio (fallback if not cached)"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"File not found: {audio_path}")
                return torch.zeros((self.n_mels, self.mel_spec_length))
            
            # Load audio with torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Limit duration if needed (5 seconds max)
            max_samples = int(5.0 * self.sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            # Extract mel spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # Convert to dB scale
            mel_spec = self.amplitude_to_db(mel_spec)
            
            # Remove batch dimension
            mel_spec = mel_spec.squeeze(0)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
            
            return mel_spec
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return torch.zeros((self.n_mels, self.mel_spec_length))
    
    def augment_features(self, mel_spec):
        """Apply augmentations to mel spectrogram using torchaudio"""
        # Add batch dimension for torchaudio transforms
        mel_spec = mel_spec.unsqueeze(0)
        
        # Random time masking
        if np.random.random() < 0.5:
            mel_spec = self.time_masking(mel_spec)
        
        # Random frequency masking
        if np.random.random() < 0.5:
            mel_spec = self.freq_masking(mel_spec)
        
        # Remove batch dimension
        mel_spec = mel_spec.squeeze(0)
        
        return mel_spec


if __name__ == '__main__':
    pass
