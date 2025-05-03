import os
import sys
from pathlib import Path
import pandas as pd
import json
import logging
import argparse
from tqdm import tqdm
import soundfile as sf
import torchaudio  # Add this import

# Get the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(ROOT_DIR))

from scripts.utils.path_utils import normalize_audio_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_audio(audio_path, use_torchaudio=False):
    """Validate audio file exists and can be read"""
    try:
        if not os.path.exists(audio_path):
            logger.warning(f"File not found: {audio_path}")
            return False
        
        if use_torchaudio:
            # Use torchaudio for validation (faster)
            try:
                waveform, sr = torchaudio.load(audio_path)
                # Check if audio has content
                if waveform.shape[1] < 100:  # Very short audio
                    logger.warning(f"Audio too short: {audio_path}")
                    return False
                return True
            except Exception as e:
                logger.warning(f"Error with torchaudio: {e}, trying soundfile")
                # Fall back to soundfile if torchaudio fails
                pass
        
        # Use soundfile as default or fallback
        data, sr = sf.read(audio_path)
        if len(data) < 100:  # Very short audio
            logger.warning(f"Audio too short: {audio_path}")
            return False
                
        return True
    except Exception as e:
        logger.warning(f"Invalid audio file: {audio_path} - {e}")
        return False

def process_dataset(csv_path, base_path, use_torchaudio=False):
    """
    Process dataset CSV and validate audio files
    
    Args:
        csv_path: Path to dataset CSV
        base_path: Base path for audio files
        use_torchaudio: Whether to use torchaudio for validation
        
    Returns:
        Processed dataframe with valid audio files
    """
    logger.info(f"Processing {csv_path}")
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return None
    
    logger.info(f"Loaded {len(df)} examples from {csv_path}")
    
    # Check if CSV has the required columns
    required_columns = ['path'] if 'path' in df.columns else []
    if not required_columns:
        # Try to find path column with alternative names
        path_columns = ['file_path', 'audio_path', 'filepath', 'audio_file', 'wav_path', 'wav_file']
        for col in path_columns:
            if col in df.columns:
                df = df.rename(columns={col: 'path'})
                required_columns = ['path']
                break
    
    # If action and object columns exist, we'll create a label
    if 'action' in df.columns and 'object' in df.columns:
        required_columns.extend(['action', 'object'])
    # If we have a label column already, add it to required columns
    elif 'label' in df.columns or 'intent' in df.columns or 'class' in df.columns:
        if 'intent' in df.columns:
            df = df.rename(columns={'intent': 'label'})
        if 'class' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'class': 'label'})
        required_columns.append('label')
    
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logger.error(f"CSV file {csv_path} missing required columns: {missing}")
        return None
    
    # Create label if needed
    if 'label' not in df.columns and 'action' in df.columns and 'object' in df.columns:
        df['label'] = df['action'] + '_' + df['object']
    
    # Normalize paths
    df['path'] = df['path'].apply(lambda p: normalize_audio_path(p, base_path))
    
    # Validate audio files
    valid_files = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating audio"):
        if validate_audio(row['path'], use_torchaudio):
            valid_files.append(idx)
        else:
            logger.warning(f"Invalid audio file: {row['path']}")
            
    if len(valid_files) == 0:
        logger.error(f"No valid audio files found in {csv_path}")
        return None
            
    df = df.iloc[valid_files].reset_index(drop=True)
    logger.info(f"Kept {len(df)} valid audio files out of {len(valid_files)}")
    
    return df

def create_label_map(df):
    """Create label map from unique labels"""
    label_column = 'label' if 'label' in df.columns else 'intent'
    
    if label_column not in df.columns:
        # Try to create label from action and object if available
        if 'action' in df.columns and 'object' in df.columns:
            df['label'] = df['action'] + '_' + df['object']
            label_column = 'label'
        else:
            logger.error("Could not find label column in dataframe")
            return {}
    
    unique_labels = sorted(df[label_column].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return label_map

def preprocess_dataset(train_csv, valid_csv, test_csv, output_dir, label_map_path=None, use_torchaudio=False):
    """Wrapper function to preprocess the FSC dataset for pipeline use"""
    # Create an Args object to pass to main
    class Args:
        def __init__(self):
            self.train_csv = train_csv
            self.valid_csv = valid_csv
            self.test_csv = test_csv
            self.output_dir = output_dir
            self.label_map_path = label_map_path
            self.use_torchaudio = use_torchaudio  # New parameter
    
    args = Args()
    return main(args)  # Return the result

def main(args):
    """Main preprocessing function"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    train_df = process_dataset(args.train_csv, ROOT_DIR, args.use_torchaudio)
    valid_df = process_dataset(args.valid_csv, ROOT_DIR, args.use_torchaudio)
    test_df = process_dataset(args.test_csv, ROOT_DIR, args.use_torchaudio)
    
    if train_df is None or valid_df is None or test_df is None:
        logger.error("Failed to process one or more datasets")
        return
    
    # Create label map from training data
    label_map = create_label_map(train_df)
    logger.info(f"Created label map with {len(label_map)} classes")
    
    # Save processed data
    train_output = os.path.join(args.output_dir, 'train_data.csv')
    valid_output = os.path.join(args.output_dir, 'valid_data.csv')
    test_output = os.path.join(args.output_dir, 'test_data.csv')
    
    train_df.to_csv(train_output, index=False)
    valid_df.to_csv(valid_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    logger.info(f"Saved processed CSV files to {args.output_dir}")
    
    # Save label map
    label_map_path = args.label_map_path or os.path.join(args.output_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    logger.info(f"Saved label map to {label_map_path}")
    
    # Return paths for pipeline use
    logger.info(f"Total samples: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    return {
        'train_csv': train_output,
        'valid_csv': valid_output,
        'test_csv': test_output,
        'label_map': label_map_path
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess FSC dataset')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train CSV')
    parser.add_argument('--valid_csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--label_map_path', type=str, help='Path to save label map')
    parser.add_argument('--use_torchaudio', action='store_true', help='Use torchaudio for validation')
    
    args = parser.parse_args()
    main(args)