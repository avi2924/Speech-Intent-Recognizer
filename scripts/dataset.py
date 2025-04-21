import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import json

class FSCIntentDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Fluent Speech Commands (FSC) data.
    """
    def __init__(self, csv_path, label_map_path, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file containing audio paths and labels.
            label_map_path (str): Path to the JSON file containing the label map.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.data = pd.read_csv(csv_path)
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.transform = transform
        # Set the root directory for audio files
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        """
        try:
            # Get relative audio path and label
            rel_audio_path = self.data.iloc[idx]['audio_path']
            label = self.data.iloc[idx]['label']
            
            # Convert to absolute path
            audio_path = os.path.normpath(os.path.join(self.root_dir, rel_audio_path))
            
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return torch.zeros((64, 100)), label

            # Try to load the audio file
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as audio_error:
                print(f"Error loading audio {audio_path}: {audio_error}")
                return torch.zeros((64, 100)), label

            # Handle mono/stereo conversion if needed
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                
            # Handle empty or corrupted audio
            if waveform.size(1) == 0:
                print(f"Empty audio file: {audio_path}")
                return torch.zeros((64, 100)), label

            # Convert waveform to Mel Spectrogram
            mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64
            )
            mel_spec = mel_spec_transform(waveform).squeeze(0)  # [n_mels, time]

            # Apply any additional transformations
            if self.transform:
                mel_spec = self.transform(mel_spec)

            return mel_spec, label
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a placeholder tensor in case of error
            return torch.zeros((64, 100)), label


if __name__ == '__main__':
    # Example usage of the FSCIntentDataset
    dataset = FSCIntentDataset(
        csv_path='data/processed/train_data.csv',  # Updated to use train_data.csv
        label_map_path='data/processed/label_map.json'
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Fetch and display the first sample
    first_sample = dataset[0]
    print(f"First sample Mel Spectrogram shape: {first_sample[0].shape}")
    print(f"First sample Label: {first_sample[1]}")
