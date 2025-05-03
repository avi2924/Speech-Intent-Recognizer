import os
import sys
import json
import torch
import librosa
import numpy as np
import argparse
import pyaudio
import wave
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model
from models.models import CNNAudioGRU

class MicrophoneListener:
    def __init__(self, sample_rate=16000, chunk_size=1024, threshold=0.01, 
                 silence_limit=1, prior_recording=0.5, record_dir='mic_recordings'):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.silence_limit = silence_limit
        self.prior_recording = prior_recording
        self.record_dir = record_dir
        
        # Create recording directory if it doesn't exist
        os.makedirs(self.record_dir, exist_ok=True)
        
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        print(f"Microphone Listener initialized")
    
    def _calculate_energy(self, data):
        """Calculate energy of audio chunk"""
        data_int16 = np.frombuffer(data, dtype=np.int16)
        data_float = data_int16.astype(np.float32) / 32768.0
        return np.mean(np.abs(data_float))
    
    def _is_speech(self, data, threshold):
        """Detect if audio chunk contains speech"""
        energy = self._calculate_energy(data)
        return energy > threshold
    
    def listen(self, callback=None, save_audio=True):
        """Listen for speech and process when detected"""
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("Listening... (Press Ctrl+C to stop)")
            
            # Calculate number of chunks for prior recording buffer
            prior_chunks = int(self.prior_recording * self.sample_rate / self.chunk_size)
            
            # Buffer for storing prior audio chunks
            prior_buffer = []
            
            # Audio recording variables
            recording = False
            recorded_chunks = []
            silence_chunks = 0
            start_time = None
            
            # Main listening loop
            while True:
                # Read chunk from stream
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Update prior buffer
                prior_buffer.append(data)
                if len(prior_buffer) > prior_chunks:
                    prior_buffer.pop(0)
                
                # If we're not already recording, check if we should start
                if not recording:
                    if self._is_speech(data, self.threshold):
                        print("\n[Speech detected]")
                        recording = True
                        # Add prior buffer to recorded chunks
                        recorded_chunks.extend(prior_buffer)
                        silence_chunks = 0
                        start_time = datetime.now()
                
                # If we're recording, add chunk to recorded audio
                if recording:
                    recorded_chunks.append(data)
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    
                    # Display elapsed time
                    sys.stdout.write(f"\rRecording: {elapsed_time:.1f}s")
                    sys.stdout.flush()
                    
                    # Check if current chunk is silence
                    if not self._is_speech(data, self.threshold):
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    
                    # Stop recording if silence limit reached
                    silence_duration = silence_chunks * (self.chunk_size / self.sample_rate)
                    if silence_duration >= self.silence_limit:
                        print(f"\n[End of speech detected - {elapsed_time:.1f}s]")
                        
                        # Stop recording and process audio
                        recording = False
                        
                        # Convert recorded audio to numpy array
                        audio_data = np.frombuffer(b''.join(recorded_chunks), dtype=np.int16)
                        audio_float = audio_data.astype(np.float32) / 32768.0
                        
                        # Save audio if requested
                        if save_audio:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(self.record_dir, f"recording_{timestamp}.wav")
                            self._save_audio(recorded_chunks, filename)
                            print(f"Saved to: {filename}")
                        
                        # Call callback with processed audio data
                        if callback:
                            callback(audio_float, self.sample_rate)
                        
                        # Reset for next recording
                        recorded_chunks = []
        
        except KeyboardInterrupt:
            print("\nStopping listener...")
        
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            print("Listener stopped")
    
    def _save_audio(self, chunks, filename):
        """Save audio chunks to a WAV file"""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(chunks))
        wf.close()
    
    def close(self):
        """Close PyAudio"""
        self.p.terminate()

class IntentRecognizer:
    def __init__(self, model_path, label_map_path, device=None):
        """Initialize the intent recognizer"""
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load label map
        try:
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            print(f"Loaded label map with {len(self.label_map)} intents")
        except Exception as e:
            print(f"Error loading label map: {e}")
            sys.exit(1)
        
        # Load model
        try:
            self.model = CNNAudioGRU(num_classes=31).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
            
        # Global normalization values - same as in dataset.py
        self.global_mean = -30.1
        self.global_std = 12.7
    
    def extract_features(self, audio_data, sample_rate):
        """Extract mel spectrogram features from audio data"""
        try:
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=sample_rate, 
                n_fft=1024, 
                hop_length=512, 
                n_mels=64
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Apply global normalization
            mel_spec = (mel_spec - self.global_mean) / self.global_std
            
            # Convert to tensor
            mel_spec = torch.FloatTensor(mel_spec)
            
            # Add batch dimension
            mel_spec = mel_spec.unsqueeze(0)
            
            return mel_spec
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_data, sample_rate):
        """Predict intent from audio data"""
        # Extract features
        mel_spec = self.extract_features(audio_data, sample_rate)
        if mel_spec is None:
            return None
        
        # Pad or truncate to expected length
        max_length = 200
        if mel_spec.size(2) > max_length:
            mel_spec = mel_spec[:, :, :max_length]
        else:
            padding = max_length - mel_spec.size(2)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        
        # Move to device
        mel_spec = mel_spec.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                output = self.model(mel_spec)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred_class = torch.argmax(output, dim=1).item()
                confidence = probs[0][pred_class].item()
        
        # Get label from class index
        predicted_label = self.inv_label_map.get(pred_class, "Unknown")
        
        # Get top 3 predictions
        probs = probs.cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = [
            {
                "label": self.inv_label_map.get(idx, "Unknown"),
                "probability": float(probs[idx])
            }
            for idx in top_indices
        ]
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_predictions": top_predictions
        }
    
    def process_audio(self, audio_data, sample_rate):
        """Process audio data and display results"""
        result = self.predict(audio_data, sample_rate)
        
        if result:
            print("\n=== INTENT RECOGNITION RESULTS ===")
            print(f"Predicted Intent: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            
            print("\nTop Predictions:")
            for i, pred in enumerate(result['top_predictions']):
                print(f"  {i+1}. {pred['label']} ({pred['probability']*100:.2f}%)")
            
            print("=" * 35)

class GPUPrefetcher:
    """Memory-efficient GPU prefetcher"""
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self._preload()
    
    def _preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        except Exception as e:
            print(f"Error in prefetcher: {e}")
            self.next_data = None
            return
        
        with torch.cuda.stream(self.stream):
            # Handle tuple/list data
            if isinstance(self.next_data, (list, tuple)):
                self.next_data = [
                    t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                    for t in self.next_data
                ]
            # Handle tensor data
            elif isinstance(self.next_data, torch.Tensor):
                self.next_data = self.next_data.to(self.device, non_blocking=True)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self._preload()
        return data
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = self.next()
        if data is None:
            raise StopIteration
        return data

def collate_fn(batch):
    """Handle possible corrupted/missing audio files in batches"""
    # Filter out None values (from errors)
    valid_batch = [(mel, label) for mel, label in batch if mel is not None and isinstance(mel, torch.Tensor)]
    
    # Return empty batch if all items were filtered out
    if len(valid_batch) == 0:
        return torch.zeros((0, 64, 200)), torch.zeros(0, dtype=torch.long)
    
    # Unpack the valid batch
    mel_specs, labels = zip(*valid_batch)
    
    # Stack mel spectrograms
    mel_specs = torch.stack(mel_specs)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return mel_specs, labels

def main():
    parser = argparse.ArgumentParser(description='Speech Intent Recognition from Microphone')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                       help='Path to the trained model')
    parser.add_argument('--label_map', type=str, default='data/processed/label_map.json',
                       help='Path to the label map')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Energy threshold for speech detection')
    parser.add_argument('--silence_limit', type=float, default=1.0,
                       help='Seconds of silence before stopping recording')
    
    args = parser.parse_args()
    
    # Initialize intent recognizer
    recognizer = IntentRecognizer(args.model, args.label_map)
    
    # Initialize microphone listener
    listener = MicrophoneListener(threshold=args.threshold, silence_limit=args.silence_limit)
    
    try:
        # Start listening
        listener.listen(callback=recognizer.process_audio)
    finally:
        # Clean up
        listener.close()

if __name__ == "__main__":
    main()