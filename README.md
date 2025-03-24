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














IMPROVING SPEECH RECOGNITION ACCURACY WITH WAC2VAC 2.0 AND DELAYED FUSION OF LLMS





 
Project Supervisor:
S.Akilandeswari ME.(Ph.D)
 
TEAM MEMBERS:
RAMANA .R ARAVINDHAN .M SELVEM .A
 
Motivation
1.	Why this project?
Traditional	speech	recognition	systems	struggle	with	noise,	accents,	and complex sentences.

2.	How was the need identified?
Analysis of speech recognition failures in real-world applications. Need for better contextual understanding and error correction.
 
Base Paper Details

Title of the Paper: Delayed Fusion: Integrating Large Language Models into First- Pass Decoding in End-to-End Speech Recognition
Journal Name: IEEE Xplore
Publication Date: Jan 16,2025
Authors: Takaaki Hori (Apple)Martin Kocour (Brno University of Technology)Adnan Haider (Apple)Erik McDermott (Apple)Xiaodan Zhuang (Apple)
 
Expected Outcomes
1.	What problem this project is trying to solve?
Problem	this	project	is	solving	Improve	speech-to-text	accuracy	in	noisy
environments. Reduce errors caused by similar-sounding words (homophones).

2.	Who are the beneficiaries?
Speech assistants, transcription services, accessibility tools, etc.

3.	How this is going to contribute for further development?
Combining	WAC2VAC	2.0	with	Transformers	for	improved	context-aware transcription.

4.	Scope for viability?
High, as AI-driven speech processing is in high demand.
 
Problem Identified
1.	What is the process adopted to identify the problem??
Existing speech recognition struggles with noise, accents, and errors.

2.	What is the need to solve it?
Improves accuracy for voice assistants, transcription, and accessibility.

3.	SCOPE WITH SOLUTION??
Scope:
Useful in AI assistants, transcription services, and multilingual
speech recognition.
Solution:
WAC2VAC 2.0 for better speech feature extraction.Transformers for understanding context.Delayed Fusion with LLMs for final correction.
