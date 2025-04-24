import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertTokenizer, BertModel, pipeline
from gtts import gTTS
import os
import librosa
from concurrent.futures import ThreadPoolExecutor

class RealTimeSpeechProcessor:
    def __init__(self):
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Speech Recognition (Wav2Vec 2.0)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(self.device)
        
        # Text Correction (BERT + LLM)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.llm = pipeline("text-generation", model="gpt-3.5-turbo-instruct", device=0 if self.device == "cuda" else -1)
        
        # Audio Config
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 500  # Adjust based on mic sensitivity

    def capture_audio(self, duration=5):
        """Capture live audio with voice activity detection"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        
        print("\n Listening... (Speak now)")
        frames = []
        silent_chunks = 0
        
        for _ in range(0, int(self.RATE / self.CHUNK * duration)):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            
            # Voice Activity Detection
            if np.abs(audio_chunk).mean() < self.SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > 10:  # Stop after 10 silent chunks
                    break
            else:
                silent_chunks = 0
                frames.append(audio_chunk)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if len(frames) == 0:
            return None, self.RATE
        return np.concatenate(frames), self.RATE

    def speech_to_text(self, audio):
        """Convert speech to text with Wav2Vec2"""
        inputs = self.processor(audio, sampling_rate=self.RATE, return_tensors="pt", padding=True).input_values.to(self.device)
        with torch.no_grad():
            logits = self.asr_model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(predicted_ids)[0]

    def correct_text(self, text):
        """Correct and enhance text using BERT + LLM"""
        # BERT for context understanding
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            bert_output = self.bert_model(**inputs)
        
        # LLM for grammatical correction
        prompt = f"Correct this speech transcription while keeping the original meaning:\n{text}\nCorrected version:"
        corrected = self.llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        return corrected.split("Corrected version:")[-1].strip()

    def text_to_speech(self, text):
        """Convert text to speech"""
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        os.system("start output.mp3" if os.name == 'nt' else "afplay output.mp3")

    def process_realtime(self):
        """Main processing loop"""
        try:
            while True:
                # 1. Capture audio
                audio, sr = self.capture_audio(duration=10)
                if audio is None:
                    print("No speech detected. Trying again...")
                    continue
                
                # 2. Speech-to-Text
                raw_text = self.speech_to_text(audio)
                print(f"\n Raw: {raw_text}")
                
                # 3. Parallel correction
                with ThreadPoolExecutor() as executor:
                    corrected_future = executor.submit(self.correct_text, raw_text)
                    corrected_text = corrected_future.result()
                
                print(f" Corrected: {corrected_text}")
                
                # 4. Text-to-Speech
                self.text_to_speech(corrected_text)
                
        except KeyboardInterrupt:
            print("\nProcessing stopped.")

if __name__ == "__main__":
    processor = RealTimeSpeechProcessor()
    processor.process_realtime()
