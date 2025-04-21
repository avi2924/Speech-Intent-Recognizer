import os
import sys
from pathlib import Path
import pandas as pd
import json
import logging
import argparse
from tqdm import tqdm
import soundfile as sf

# Get the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(ROOT_DIR))

from scripts.utils.path_utils import normalize_audio_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_audio(audio_path):
    """Validate audio file"""
    try:
        data, sr = sf.read(audio_path)
        return True
    except Exception as e:
        logger.warning(f"Invalid audio file: {audio_path} - {e}")
        return False

def process_dataset(csv_path, base_path):
    """Process dataset from CSV file"""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    
    # Ensure all required columns exist
    required_columns = ['path', 'action', 'object', 'location']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"CSV file {csv_path} missing required columns")
        return None
    
    # Normalize paths and create label
    df['path'] = df['path'].apply(lambda p: normalize_audio_path(p, base_path))
    df['label'] = df['action'] + '_' + df['object']
    
    # Validate audio files
    valid_files = []
    logger.info(f"Validating {len(df)} audio files...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating audio"):
        if validate_audio(row['path']):
            valid_files.append(idx)
            
    df = df.iloc[valid_files].reset_index(drop=True)
    logger.info(f"Kept {len(df)} valid audio files out of {len(valid_files)}")
    
    return df

def create_label_map(df):
    """Create label map from unique labels"""
    unique_labels = sorted(df['label'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return label_map

def preprocess_dataset(train_csv, valid_csv, test_csv, output_dir, label_map_path=None):
    """Wrapper function to preprocess the FSC dataset for pipeline use"""
    class Args:
        def __init__(self):
            self.train_csv = train_csv
            self.valid_csv = valid_csv
            self.test_csv = test_csv
            self.output_dir = output_dir
            self.label_map_path = label_map_path
    
    args = Args()
    main(args)
    
    return {
        'train_csv': os.path.join(output_dir, 'train_data.csv'),
        'valid_csv': os.path.join(output_dir, 'valid_data.csv'),
        'test_csv': os.path.join(output_dir, 'test_data.csv'),
        'label_map': os.path.join(output_dir, 'label_map.json')
    }

def main(args):
    """Main preprocessing function"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    train_df = process_dataset(args.train_csv, ROOT_DIR)
    valid_df = process_dataset(args.valid_csv, ROOT_DIR)
    test_df = process_dataset(args.test_csv, ROOT_DIR)
    
    if train_df is None or valid_df is None or test_df is None:
        logger.error("Failed to process one or more datasets")
        return
    
    # Create label map from training data
    label_map = create_label_map(train_df)
    
    # Save label map
    label_map_path = args.label_map_path or os.path.join(args.output_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=4)
    
    logger.info(f"Label map saved to {label_map_path}")
    
    # Save processed data
    train_df.to_csv(os.path.join(args.output_dir, 'train_data.csv'), index=False)
    valid_df.to_csv(os.path.join(args.output_dir, 'valid_data.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_data.csv'), index=False)
    
    logger.info(f"Preprocessing complete. Files saved to {args.output_dir}")
    logger.info(f"Total samples: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess FSC dataset')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train CSV')
    parser.add_argument('--valid_csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--label_map_path', type=str, help='Path to save label map')
    
    args = parser.parse_args()
    main(args)