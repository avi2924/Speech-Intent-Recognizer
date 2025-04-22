"""
Utility functions for handling file paths in the Speech Intent Recognition project.
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def normalize_audio_path(path, base_path):
    """
    Normalize audio file paths in the dataset
    """
    if os.path.isabs(path):
        # If it's already an absolute path, just use it
        return path
        
    # Try to find the audio file in several potential locations
    potential_locations = [
        path,  # As-is
        os.path.join(base_path, path),  # Relative to project root
        os.path.join(base_path, 'data', 'FSC', 'fluent_speech_commands_dataset', path),  # Relative to FSC dataset
        os.path.join(base_path, 'data', 'FSC', 'fluent_speech_commands_dataset', 'wavs', path),  # Common subfolder for wavs
    ]
    
    for loc in potential_locations:
        if os.path.exists(loc):
            return loc
            
    # If we couldn't find it, log a warning and return the original path
    print(f"Warning: Could not find audio file at {path}")
    return path