import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from scripts.utils.path_utils import normalize_audio_path
    print("Successfully imported path_utils")
    
    # Test the function
    test_path = r"E:\Speech-Intent-Recognition\data\FSC\fluent_speech_commands_dataset\wavs\wavs\test.wav"
    normalized = normalize_audio_path(test_path)
    print(f"Original path: {test_path}")
    print(f"Normalized path: {normalized}")
    
except ImportError as e:
    print(f"Import failed: {e}")