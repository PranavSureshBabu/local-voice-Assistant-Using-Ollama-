import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

print(sd.query_devices())

# --- Step 1: Record audio ---
duration = 5  # seconds
sample_rate = 48000  # matches the mic's native rate — avoids WASAPI errors

print("Recording for 5 seconds... speak now!")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16", device=9)
sd.wait()
write("test_audio.wav", sample_rate, audio)
print("Recording finished, saved as test_audio.wav")

print("Loading Whisper model...")
model = WhisperModel("small.en", compute_type="int8")

print("Transcribing...")
segments, info = model.transcribe("test_audio.wav")

print("\n--- Transcription ---")
for segment in segments:
    print(segment.text)