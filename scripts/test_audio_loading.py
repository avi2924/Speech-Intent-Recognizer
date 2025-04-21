import torch
import torchaudio
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import time

def collate_fn_audio(batch):
    """Custom collate function to handle variable-length audio files"""
    waveforms = []
    indices = []
    max_length = 0
    
    # Find the maximum length in this batch
    for waveform, idx in batch:
        max_length = max(max_length, waveform.shape[1])
    
    # Pad all waveforms to the maximum length
    for waveform, idx in batch:
        if waveform.shape[1] < max_length:
            padding = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        waveforms.append(waveform)
        indices.append(idx)
    
    # Stack the tensors
    waveforms = torch.stack(waveforms)
    indices = torch.tensor(indices)
    
    return waveforms, indices

class SimpleAudioDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        try:
            start_time = time.time()
            rel_audio_path = self.data.iloc[idx]['audio_path']
            audio_path = os.path.normpath(os.path.join(self.root_dir, rel_audio_path))
            
            print(f"Loading audio file {idx}: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
            load_time = time.time() - start_time
            print(f"Loaded audio file {idx} in {load_time:.2f}s: {audio_path}")
            
            return waveform, idx
        except Exception as e:
            print(f"Error loading audio {idx}: {e}")
            return torch.zeros(1, 1000), idx

# Main test
if __name__ == "__main__":
    start_total = time.time()
    print("Creating dataset...")
    dataset = SimpleAudioDataset("data/processed/train_data.csv")
    print(f"Dataset created with {len(dataset)} samples")
    
    print("Creating DataLoader with custom collate function...")
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn_audio
    )
    print("DataLoader created")
    
    print("Loading first 5 batches...")
    try:
        for i, (audio, idx) in enumerate(loader):
            print(f"Batch {i+1}: Audio shape {audio.shape}")
            if i >= 4:  # Just test 5 batches
                break
    except Exception as e:
        print(f"Error during batch loading: {e}")
    
    total_time = time.time() - start_total
    print(f"Test completed in {total_time:.2f} seconds")