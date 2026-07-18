"""
Live Microphone Inference Script.

This script records audio from your computer's microphone for a specified duration,
saves it to a temporary file, and then runs the trained SLID pipeline to predict the language.

Usage:
    python src/live_predict.py [seconds]
    (e.g., python src/live_predict.py 5)
"""

import sys
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path

# Import the prediction logic from our existing script
from predict import predict_language

def record_and_predict(duration=5, sample_rate=16000):
    """
    Records audio from the microphone and passes it to the prediction pipeline.
    """
    temp_wav = Path("live_recording.wav")
    
    print("\n" + "="*50)
    print(f"🎙️ GET READY TO SPEAK (or play audio from your phone)")
    print("="*50)
    
    # Simple countdown
    for i in range(3, 0, -1):
        print(f"Recording starts in {i}...")
        time.sleep(1)
        
    print("\n🔴 RECORDING NOW...")
    print(f"(Speak/Play for {duration} seconds)")
    
    # Record audio (mono channel)
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    
    print("✅ Recording finished!")
    print("="*50 + "\n")
    
    # Save to a temporary WAV file
    sf.write(temp_wav, recording, sample_rate)
    
    # Pass the temporary file to our pipeline
    try:
        predict_language(temp_wav)
    finally:
        pass

if __name__ == "__main__":
    # Allow user to specify duration via command line argument, default to 5
    duration_seconds = 5
    if len(sys.argv) > 1:
        try:
            duration_seconds = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid number of seconds. Using default: 5")
            
    record_and_predict(duration=duration_seconds)
