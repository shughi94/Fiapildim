import pyaudio
import numpy as np
import librosa
from datetime import datetime
import os, time, json
from tensorflow.keras import models, layers, utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "training_data"
MODEL_PATH = "ohmygodmyfirstmodel.h5"

SPEC_SHAPE = (128, 86)

class ChordTrainer:

    TRAINING_STEPS = 2
    CHORDS = ['A', 'C', 'D', 'E', 'G', 'Am', 'Em', 'Dm', 'F']

    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    TRAIN_RECORD_SECONDS = 2
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        os.makedirs(DATA_DIR, exist_ok=True)
        
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
        metadata_path = os.path.join(DATA_DIR, "metadata.json")
        
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
        filepath = os.path.join(DATA_DIR, filename)
        
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
            n_mels=SPEC_SHAPE[0],
            n_fft=2048,
            hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to fixed shape
        if mel_spec_db.shape[1] < SPEC_SHAPE[1]:
            pad_width = SPEC_SHAPE[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :SPEC_SHAPE[1]]
            
        return mel_spec_db
    
    def run_training_session(self):

        for i in range(0, self.TRAINING_STEPS):
            for chord in self.CHORDS:
                print(f"Get ready to play chord: {chord}")
                time.sleep(2)
                audio_data = self.record_audio()
                features = self.extract_features(audio_data)
                self.save_training_example(features, chord)
                print(f"Added {chord} chord to training data.")

        self.audio.terminate()

class ModelTrainer:

    def build_model(self, input_shape, num_classes):
        """Build the neural network model"""
        model = models.Sequential([
            layers.Reshape((*input_shape, 1), input_shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    
    def load_training_data(self):
        """Load all training data from the data directory"""
        features = []
        labels = []

        metadata_path = os.path.join(DATA_DIR, "metadata.json")
        if not os.path.exists(metadata_path):
            print("No training data found. Please run chord_trainer.py first.")
            return None, None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for chord_name, examples in metadata.items():
            for example in examples:
                filepath = example['filepath']
                if os.path.exists(filepath):
                    data = np.load(filepath)
                    features.append(data['features'])
                    labels.append(chord_name)

        return np.array(features), np.array(labels)
    
    def train_model(self):
        """Train the chord detection model"""
        # Load training data
        X, y = self.load_training_data()
        if X is None:
            return

        print(f"Loaded {len(X)} training samples")

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Build model
        model = self.build_model(SPEC_SHAPE, len(label_encoder.classes_))

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Save model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


print("Starting chord training session...")
c = ChordTrainer()
c.run_training_session()
m = ModelTrainer()
m.train_model()
print("ALL DONE!!")
            