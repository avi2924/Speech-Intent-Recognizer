import pyttsx3
import os
from datetime import datetime

def text_to_wav(text, output_file=None, output_dir="mic_recordings"):
    """
    Convert text to speech and save as a WAV file in the specified directory.
    
    Args:
        text: The text to convert to speech
        output_file: Optional filename for the output WAV file. If None, a timestamp-based name will be used.
        output_dir: Directory to save the WAV file (default: "mic_recordings")
    
    Returns:
        The path to the created WAV file
    """
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"speech_{timestamp}.wav"
    
    # Make sure it has .wav extension
    if not output_file.lower().endswith('.wav'):
        output_file += '.wav'
    
    # Combine directory and filename
    output_path = os.path.join(output_dir, output_file)
    
    # Convert text to speech and save to file
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    
    full_path = os.path.abspath(output_path)
    print(f"Text converted to speech and saved as: {full_path}")
    return full_path

def main():
    print("Text-to-Speech Converter")
    print("Type your text and press Enter. Type 'quit' to exit.")
    print("Files will be saved to the 'mic_recordings' directory.")
    print("----------------------------------------------")
    
    # Create the mic_recordings directory if it doesn't exist
    os.makedirs("mic_recordings", exist_ok=True)
    
    while True:
        text = input("\nEnter text to convert (or 'quit' to exit): ")
        
        if text.lower() == 'quit':
            print("Exiting program.")
            break
        
        if text:
            # Optional: Ask for custom filename
            custom_filename = input("Enter filename (optional, press Enter for auto-generated): ")
            
            if custom_filename:
                text_to_wav(text, custom_filename)
            else:
                text_to_wav(text)
        else:
            print("No text entered. Please try again.")

if __name__ == "__main__":
    main()