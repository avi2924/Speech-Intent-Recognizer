import os
import pandas as pd
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from gtts import gTTS
import time

def sanitize_filename(filename):
    """Convert text to a valid filename"""
    # Remove invalid filename characters and replace with underscore
    valid_filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # Limit filename length
    if len(valid_filename) > 100:
        valid_filename = valid_filename[:97] + "..."
    return valid_filename

def generate_audio_files(csv_file, output_dir, accent="en", speed=False):
    """Generate audio files from transcriptions in CSV using gTTS"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a details.csv file for visualization later
    details_file = os.path.join(output_dir, "details.csv")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    if 'transcription' not in df.columns or 'label' not in df.columns:
        print(f"Error: Required columns not found in {csv_file}")
        return
    
    print(f"Generating {len(df)} audio files...")
    
    # Prepare details dataframe
    details = []
    
    # Generate audio for each transcription
    for i, row in tqdm(df.iterrows(), total=len(df)):
        transcription = row['transcription']
        label = row['label']
        
        # Create a valid filename
        base_filename = sanitize_filename(transcription)
        filename = f"{i+1:03d}_{base_filename}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Generate audio file
        try:
            # Create a gTTS object
            tts = gTTS(text=transcription, lang=accent, slow=speed)
            tts.save(output_path)
            
            # Add to details
            details.append({
                'filename': filename,
                'text': transcription,
                'class': label
            })
            
            # Small delay to avoid API rate limits
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error generating audio for '{transcription}': {str(e)}")
    
    # Save details CSV
    pd.DataFrame(details).to_csv(details_file, index=False)
    
    print(f"Audio files generated in: {output_dir}")
    print(f"Details saved to: {details_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate audio files from transcriptions using Google TTS")
    parser.add_argument('--csv_file', type=str, default="fsc_custom_intents_100_sentences.csv",
                        help='Path to CSV file containing transcriptions')
    parser.add_argument('--output_dir', type=str, default="mic_recordings",
                        help='Directory to save audio files')
    parser.add_argument('--accent', type=str, default="en", 
                        choices=["en", "en-us", "en-uk", "en-au"],
                        help='Accent to use for TTS')
    parser.add_argument('--slow', action='store_true',
                        help='Use slower speech rate')
    
    args = parser.parse_args()
    generate_audio_files(args.csv_file, args.output_dir, args.accent, args.slow)

if __name__ == "__main__":
    main()