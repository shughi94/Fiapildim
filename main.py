import numpy as np
import pyaudio
from collections import deque
import time, librosa
from tensorflow.keras import models

IDK_NO_TIME_TO_FIGURE_OUT_WHY_I_NEED_THIS_MAPPING = {
    0: 'A',
    2: 'C',
    3: 'D',
    5: 'E',
    8: 'G',
    1: 'Am',
    6: 'Em',
    4: 'Dm',
    7: 'F',
}

class GuitarChordDetector:

    def __init__(self):
        
        # Audio parameters
        self.SPEC_SHAPE = (128, 86)
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 3
        self.model = models.load_model("ohmygodmyfirstmodel.h5")
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Buffer for audio data
        self.buffer = deque(maxlen=int(self.RATE * self.RECORD_SECONDS / self.CHUNK))

    def estimate_chord_fft(self, audio_array):
            """Estimate chord non ML way using FFT (frequencies)"""
            fft = np.fft.rfft(audio_array)
            freqs = np.fft.rfftfreq(len(audio_array), 1/self.RATE)
            magnitude = np.abs(fft)

            # Find the strongest frequencies
            top_indices = np.argsort(magnitude)[-6:]  # Top 6 peaks
            top_freqs = freqs[top_indices]

            chord_freqs = {
                "A": [110, 220, 440],
                "C": [130.81, 261.63, 523.25],
                "D": [146.83, 293.66, 587.33],
                "E": [164.81, 329.63, 659.25],
                "G": [196, 392, 784],
            }

            for chord, freqs_list in chord_freqs.items():
                if any(np.any(np.isclose(top_freqs, f, atol=5)) for f in freqs_list):
                    return chord
            return "Unknown"
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        self.buffer.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_listening(self):
        """Start listening to audio"""
        if self.stream is None:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("Started listening...")
    
    def stop_listening(self):
        """Stop listening to audio"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("Stopped listening.")
    
    def process_audio(self):
        """Process audio data to detect chords"""
        if len(self.buffer) < self.buffer.maxlen:
            return
        
        # Combine audio chunks
        audio_data = b''.join(self.buffer)
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        mel = librosa.feature.melspectrogram(y=audio_array, sr=self.RATE, n_mels=self.SPEC_SHAPE[0], n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Ensure correct shape
        if mel_db.shape[1] < self.SPEC_SHAPE[1]:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.SPEC_SHAPE[1] - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :self.SPEC_SHAPE[1]]
        
        # Predict
        prediction = self.model.predict(np.expand_dims(mel_db, axis=0), verbose=0)
        chord_idx = np.argmax(prediction)
        confidence = prediction[0][chord_idx]
        
        # Display result
        print(f"Detected: {IDK_NO_TIME_TO_FIGURE_OUT_WHY_I_NEED_THIS_MAPPING[chord_idx]} ({confidence*100:.1f}%)")

        #chord = self.estimate_chord_fft(audio_array)
        #print(f"Estimated chord: {chord}")
    

guitarchord = GuitarChordDetector()

while True:
    guitarchord.start_listening()
    time.sleep(1)
    guitarchord.process_audio()
    guitarchord.stop_listening
    time.sleep(0.5)
    
