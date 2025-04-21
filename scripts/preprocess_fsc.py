import os
import csv
import json
import argparse
import pandas as pd

def build_label_map(intents):
    unique_intents = sorted(set(intents))
    return {intent: idx for idx, intent in enumerate(unique_intents)}

def preprocess_fsc_dataset(train_csv, valid_csv, test_csv, output_dir, label_map_path):
    all_intents = []
    base_dir = 'data/FSC/fluent_speech_commands_dataset'
    
    # Process each split and collect all intents
    train_data = process_split(train_csv, base_dir, all_intents)
    valid_data = process_split(valid_csv, base_dir, all_intents)
    test_data = process_split(test_csv, base_dir, all_intents)
    
    # Create label map from all intents across all splits
    label_map = build_label_map(all_intents)
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    # Save each split to separate CSV files
    save_csv(os.path.join(output_dir, 'train_data.csv'), train_data, label_map)
    save_csv(os.path.join(output_dir, 'valid_data.csv'), valid_data, label_map)
    save_csv(os.path.join(output_dir, 'test_data.csv'), test_data, label_map)
    
    print(f"Preprocessing complete.")
    print(f"Train CSV saved to {os.path.join(output_dir, 'train_data.csv')}")
    print(f"Validation CSV saved to {os.path.join(output_dir, 'valid_data.csv')}")
    print(f"Test CSV saved to {os.path.join(output_dir, 'test_data.csv')}")
    print(f"Label map saved to {label_map_path}")

def process_split(csv_path, base_dir, all_intents):
    """Process a single data split and return processed data"""
    data = []
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        wav_path = row['path']
        # Add the base directory to the path
        full_path = os.path.join(base_dir, wav_path).replace('\\', '/')
        # Create intent string from slots
        intent = f"{row['action']} {row['object']} {row['location']}"
        data.append((full_path, intent))
        all_intents.append(intent)
    
    return data

def save_csv(output_path, data, label_map):
    """Save processed data to CSV file"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_path', 'label'])
        for path, intent in data:
            writer.writerow([path, label_map[intent]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='data/FSC/fluent_speech_commands_dataset/data/train_data.csv')
    parser.add_argument('--valid_csv', type=str, default='data/FSC/fluent_speech_commands_dataset/data/valid_data.csv')
    parser.add_argument('--test_csv', type=str, default='data/FSC/fluent_speech_commands_dataset/data/test_data.csv')
    parser.add_argument('--output_dir', type=str, default='data/processed/')
    parser.add_argument('--label_map_path', type=str, default='data/processed/label_map.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    preprocess_fsc_dataset(
        args.train_csv,
        args.valid_csv,
        args.test_csv,
        args.output_dir,
        args.label_map_path
    )
