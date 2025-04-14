import os
import pandas as pd
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define paths
data_dir = 'data/FSC/'
output_dir = 'data/processed/'

def preprocess_data():
    audio_files = []
    labels = []

    for label in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, label)):
            for file in os.listdir(os.path.join(data_dir, label)):
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(data_dir, label, file))
                    labels.append(label)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split data into training and validation
    train_files, val_files, train_labels, val_labels = train_test_split(audio_files, encoded_labels, test_size=0.1)

    # Save processed data
    processed_data = {
        'file': train_files + val_files,
        'label': list(train_labels) + list(val_labels)
    }

    df = pd.DataFrame(processed_data)
    df.to_csv(os.path.join(output_dir, 'processed.csv'), index=False)

    print("Preprocessing completed. Data saved to:", os.path.join(output_dir, 'processed.csv'))

if __name__ == "__main__":
    preprocess_data()
