"""
Utility functions for handling file paths in the Speech Intent Recognition project.
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def normalize_audio_path(path):
    """
    Normalize audio file paths by:
    1. Converting to Windows path format
    2. Removing duplicate 'wavs' directory
    3. Ensuring consistent path separators
    4. Handling relative paths
    
    Args:
        path (str or Path): Input path to normalize
        
    Returns:
        str: Normalized absolute path
    """
    try:
        # Convert to Path object and resolve any relative parts
        path = Path(path).resolve()
        
        # Fix double wavs issue
        parts = list(path.parts)
        if 'wavs' in parts and parts.count('wavs') > 1:
            # Keep only the first occurrence of 'wavs'
            first_wavs = parts.index('wavs')
            filtered_parts = []
            seen_wavs = False
            
            for part in parts:
                if part == 'wavs' and seen_wavs:
                    continue
                if part == 'wavs':
                    seen_wavs = True
                filtered_parts.append(part)
                
            path = Path(*filtered_parts)
        
        # Convert to Windows-style absolute path
        return str(path)
        
    except Exception as e:
        logger.warning(f"Failed to normalize path {path}: {str(e)}")
        return str(path)