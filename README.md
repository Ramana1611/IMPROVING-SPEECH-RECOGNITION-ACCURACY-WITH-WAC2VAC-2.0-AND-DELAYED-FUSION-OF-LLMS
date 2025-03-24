# IMPROVING-SPEECH-RECOGNITION-ACCURACY-WITH-WAC2VAC-2.0-AND-DELAYED-FUSION-OF-LLMS

import librosa
import numpy as np
import soundfile as sf
import os

#Load file
def load_audio(file_path, sr=16000):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

#Noise reduction
def reduce_noise(audio, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, audio.shape)
    audio_with_noise = audio + noise
    return audio_with_noise

#Save preprocess audio
def save_audio(file_path, audio, sr=16000):
    sf.write(file_path, audio, sr)

#Set path
directory_path = r"D:\cv-corpus-21.0-delta-2025-03-14-en"
output_dir = r"D:\cv-corpus-21.0-delta-2025-03-14-en\processed"

#Create directories
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Created directory: {directory_path}. Please add .wav files to it.")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Process files
try:
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            audio, sr = load_audio(file_path)
            audio_clean = reduce_noise(audio)
            output_file = os.path.join(output_dir, f"preprocessed_{filename}")
            save_audio(output_file, audio_clean)
    print("Done processing all files.")
except FileNotFoundError:
    print(f"Error: Couldn’t access '{directory_path}'. Check if it exists.")
except Exception as e:
    print(f"An error occurred: {e}")
