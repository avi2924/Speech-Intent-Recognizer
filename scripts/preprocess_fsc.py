import os
import csv
import json
import argparse
import pandas as pd

def build_label_map(intents):
    unique_intents = sorted(set(intents))
    return {intent: idx for idx, intent in enumerate(unique_intents)}

def preprocess_fsc_dataset(meta_csv_paths, output_csv, label_map_path):
    data = []
    all_intents = []
    base_dir = 'data/FSC/fluent_speech_commands_dataset'  # Add this line

    for csv_path in meta_csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            wav_path = row['path']
            # Add the base directory to the path
            full_path = os.path.join(base_dir, wav_path).replace('\\', '/')
            # Create intent string from slots
            intent = f"{row['action']} {row['object']} {row['location']}"
            data.append((full_path, intent))  # Store the full path
            all_intents.append(intent)

    # Create label map
    label_map = build_label_map(all_intents)
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)

    # Save CSV with audio path and encoded label
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_path', 'label'])
        for path, intent in data:
            writer.writerow([path, label_map[intent]])

    print(f"Preprocessing complete. CSV saved to {output_csv}")
    print(f"Label map saved to {label_map_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='data/FSC/fluent_speech_commands_dataset/data/train_data.csv')
    parser.add_argument('--valid_csv', type=str, default='data/FSC/fluent_speech_commands_dataset/data/valid_data.csv')
    parser.add_argument('--test_csv', type=str, default='data/FSC/fluent_speech_commands_dataset/data/test_data.csv')
    parser.add_argument('--output_csv', type=str, default='data/processed/processed.csv')
    parser.add_argument('--label_map_path', type=str, default='data/processed/label_map.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    preprocess_fsc_dataset(
        [args.train_csv, args.valid_csv, args.test_csv],
        args.output_csv,
        args.label_map_path
    )
