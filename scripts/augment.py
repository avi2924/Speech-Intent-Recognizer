import numpy as np
import torch
import torchaudio
import random

def time_shift(waveform, shift_limit=0.1):
    """
    Shifts the waveform in time by a random amount
    
    Args:
        waveform: Tensor of shape [1, length]
        shift_limit: Maximum shift as a fraction of length
    
    Returns:
        Shifted waveform of same shape
    """
    length = waveform.shape[1]
    shift = int(random.uniform(-shift_limit, shift_limit) * length)
    
    if shift > 0:
        # Shift right (padding left)
        padded = torch.nn.functional.pad(waveform, (shift, 0))
        return padded[:, :length]
    else:
        # Shift left (padding right)
        shift = abs(shift)
        padded = torch.nn.functional.pad(waveform, (0, shift))
        return padded[:, shift:]

def pitch_shift(waveform, sample_rate, pitch_factor_range=(-2.0, 2.0)):
    """
    Shifts the pitch of waveform by a random amount
    
    Args:
        waveform: Tensor of shape [1, length]
        sample_rate: Audio sample rate
        pitch_factor_range: Range of pitch shift in semitones
    
    Returns:
        Pitch-shifted waveform of same shape
    """
    pitch_factor = float(random.uniform(*pitch_factor_range))
    
    # Convert to effects format (n_steps is in semitones)
    effects = [
        ["pitch", str(pitch_factor * 100)],  # Scale by 100 for semitones
        ["rate", str(sample_rate)]
    ]
    
    augmented, new_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    
    return augmented

def speed_change(waveform, sample_rate, speed_factor_range=(0.85, 1.15)):
    """
    Changes the speed of waveform by a random amount
    
    Args:
        waveform: Tensor of shape [1, length]
        sample_rate: Audio sample rate
        speed_factor_range: Range of speed factors
    
    Returns:
        Speed-changed waveform
    """
    speed_factor = float(random.uniform(*speed_factor_range))
    
    # For time stretching without pitch change, use "tempo"
    effects = [
        ["tempo", str(speed_factor)],
        ["rate", str(sample_rate)]
    ]
    
    augmented, new_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    
    return augmented

def add_noise(waveform, noise_level_range=(0.001, 0.01)):
    """
    Adds random noise to the waveform
    
    Args:
        waveform: Tensor of shape [1, length]
        noise_level_range: Range of noise levels
    
    Returns:
        Noisy waveform of same shape
    """
    noise_level = float(random.uniform(*noise_level_range))
    noise = torch.randn_like(waveform) * noise_level
    
    return waveform + noise

def apply_augmentation(waveform, sample_rate, augment_prob=0.7):
    """
    Apply random augmentations with probability
    
    Args:
        waveform: Tensor of shape [1, length]
        sample_rate: Audio sample rate
        augment_prob: Probability of applying augmentations
    
    Returns:
        Augmented waveform
    """
    if not isinstance(waveform, torch.Tensor):
        # Convert numpy array to tensor if needed
        waveform = torch.tensor(waveform).float()
        
        # Add channel dimension if missing
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
    
    if random.random() < augment_prob:
        # Apply time shift
        if random.random() < 0.5:
            waveform = time_shift(waveform)
            
        # Apply pitch shift
        if random.random() < 0.5:
            waveform = pitch_shift(waveform, sample_rate)
            
        # Apply speed change
        if random.random() < 0.5:
            waveform = speed_change(waveform, sample_rate)
            
        # Apply noise
        if random.random() < 0.5:
            waveform = add_noise(waveform)
    
    return waveform

def apply_spec_augmentation(mel_spec, time_mask_param=20, freq_mask_param=10):
    """
    Apply spectrogram augmentation
    
    Args:
        mel_spec: Mel spectrogram tensor [freq, time]
        time_mask_param: Maximum time steps to mask
        freq_mask_param: Maximum frequency steps to mask
    
    Returns:
        Augmented spectrogram
    """
    # Add batch dimension for torchaudio transforms
    mel_spec = mel_spec.unsqueeze(0)
    
    # Time masking
    if random.random() < 0.5:
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        mel_spec = time_masking(mel_spec)
    
    # Frequency masking
    if random.random() < 0.5:
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        mel_spec = freq_masking(mel_spec)
    
    # Remove batch dimension
    mel_spec = mel_spec.squeeze(0)
    
    return mel_spec