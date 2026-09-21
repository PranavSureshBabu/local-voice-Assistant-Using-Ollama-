import webrtcvad
import sounddevice as sd
import numpy as np

vad = webrtcvad.Vad(2)

SAMPLE_RATE = 48000  # changed from 16000 to match your mic
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

print("Listening... speak or stay silent. Press Ctrl+C to stop.\n")

def callback(indata, frames, time, status):
    audio_bytes = indata.tobytes()
    is_speech = vad.is_speech(audio_bytes, SAMPLE_RATE)
    print("🗣️ Speech" if is_speech else "... silence")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    blocksize=FRAME_SIZE,
    device=9,
    callback=callback
):
    while True:
        sd.sleep(1000)