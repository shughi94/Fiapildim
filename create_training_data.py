import pyaudio
import numpy as np
import librosa
import os, time, datetime, json

class ChordTrainer:

    TRAINING_STEPS = 1
    CHORDS = ['A', 'C', 'D', 'E', 'G', 'Am', 'Em', 'Dm', 'F']

    DATA_DIR = "training_data"

    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    TRAIN_RECORD_SECONDS = 2
    SPEC_SHAPE = (128, 86)

    def __init__(self):
        pass
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
    def record_audio(self):
        """Record audio for 2 seconds and return as numpy array"""
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                rate=self.RATE, input=True,
                                frames_per_buffer=self.CHUNK)
        print("Recording...")
        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * self.TRAIN_RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        print("Finished recording")
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data.astype(np.float32) / 32768.0
    
    def update_metadata(self, chord_name, filepath):
        """Update metadata file with information about training examples"""
        metadata_path = os.path.join(self.DATA_DIR, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
            
        if chord_name not in metadata:
            metadata[chord_name] = []
            
        metadata[chord_name].append({
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_training_example(self, features, chord_name):
        """Save training example to file"""
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chord_name}_{timestamp}.npz"
        filepath = os.path.join(self.DATA_DIR, filename)
        
        # Save features and label
        np.savez(filepath, features=features, label=chord_name)
        
        # Update metadata
        self.update_metadata(chord_name, filepath)
        
        print(f"Saved training example for {chord_name}")
        return filepath

    def extract_features(self, audio_data):
        """
        Extract features from audio data using Mel spectrogram
        Taken straight from AI no idea what tf is a melspectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.RATE, 
            n_mels=self.SPEC_SHAPE[0],
            n_fft=2048,
            hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to fixed shape
        if mel_spec_db.shape[1] < self.SPEC_SHAPE[1]:
            pad_width = self.SPEC_SHAPE[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :self.SPEC_SHAPE[1]]
            
        return mel_spec_db
    
    def run_training_session(self):

        for i in range(0, self.TRAINING_STEPS):
            for chord in self.CHORDS:
                print(f"Get ready to play chord: {chord}")
                time.sleep(2)
                print("PLAY NOW!")
                audio_data = self.record_audio()
                features = self.extract_features(audio_data)
                self.save_training_example(features, chord)
                print(f"Added {chord} chord to training data.")

        self.audio.terminate()

class ModelTrainer:

    pass



c = ChordTrainer()
c.run_training_session()
            