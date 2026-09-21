import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import ollama
#import pyttsx3
import webrtcvad
import numpy as np
import collections
import re
import time
from kokoro import KPipeline
import sounddevice as sd
import threading
from scipy.signal import resample
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["HF_HUB_OFFLINE"] = "1" 

# --- Config ---
SAMPLE_RATE = 48000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
DEVICE_IN = 2 
DEVICE_OUT = 10
AUDIO_FILE = "input.wav"
SILENCE_LIMIT_FRAMES = 50  # ~0.9 sec of silence ends recording (30 frames * 30ms)

vad = webrtcvad.Vad(2)

kokoro_pipeline = KPipeline(lang_code='a')  # load once at startup, not per call
interrupt_flag = {"stop": False}

print("Loading Whisper model...")
whisper_model = WhisperModel("medium.en", compute_type="int8")

def record_until_silence():
    print("\n🎤 Listening... start speaking.")
    frames = []
    state = {"silence_count": 0, "triggered": False, "stop": False}

    def callback(indata, frame_count, time_info, status):
        if state["stop"]:
            return
        audio_bytes = indata.tobytes()
        is_speech = vad.is_speech(audio_bytes, SAMPLE_RATE)

        if is_speech:
            state["triggered"] = True
            state["silence_count"] = 0
            frames.append(indata.copy())
        elif state["triggered"]:
            state["silence_count"] += 1
            frames.append(indata.copy())
            if state["silence_count"] > SILENCE_LIMIT_FRAMES:
                state["stop"] = True

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                         blocksize=FRAME_SIZE, device=DEVICE_IN, callback=callback):
        while not state["stop"]:
            sd.sleep(50)

    print("Recording finished.")
    audio_data = np.concatenate(frames, axis=0)
    write(AUDIO_FILE, SAMPLE_RATE, audio_data)

    print("Recording finished.")
    audio_data = np.concatenate(frames, axis=0)
    write(AUDIO_FILE, SAMPLE_RATE, audio_data)

def transcribe_audio():
    segments, info = whisper_model.transcribe(AUDIO_FILE)
    text = " ".join(segment.text for segment in segments).strip()
    return text

def ask_ollama(conversation_history):
    print("🤖 Thinking...\n")
    t_start = time.time()
    start_mic_monitor()

    response = ollama.chat(
        model="llama3.2",
        messages=conversation_history,
        stream=True
    )

    full_reply = ""
    buffer = ""
    first_token_time = None
    first_speech_time = None

    for chunk in response:
        piece = chunk["message"]["content"]
        if first_token_time is None:
            first_token_time = time.time()
            print(f"[⏱ First token: {first_token_time - t_start:.2f}s]")

        print(piece, end="", flush=True)
        full_reply += piece
        buffer += piece

        if interrupt_flag["stop"]:
            break 

        match = re.search(r'([.!?])\s', buffer)
        if match:
            sentence_end = match.end()
            sentence = buffer[:sentence_end].strip()
            buffer = buffer[sentence_end:]

            if first_speech_time is None:
                first_speech_time = time.time()
                print(f"\n[⏱ First speech starts: {first_speech_time - t_start:.2f}s]")

            speak(sentence)

    if buffer.strip() and not interrupt_flag["stop"]:
        speak(buffer.strip())

    stop_mic_monitor()
    print()
    return full_reply


EXIT_PHRASES = ["bye", "goodbye", "i'm good", "i am good", "that's all", 
                "gotta go", "i have to go", "thank you", "thanks", "no thanks"]

def is_exit(text):
    text_lower = text.lower().strip().rstrip(".!?")
    return any(text_lower.endswith(phrase) for phrase in EXIT_PHRASES)

interrupt_flag = {"stop": False, "active": False}
mic_stream = None

def start_mic_monitor():
    global mic_stream
    interrupt_flag["stop"] = False
    interrupt_flag["active"] = True
    speech_frame_count = {"count": 0}

    def callback(indata, frame_count, time_info, status):
        if not interrupt_flag["active"]:
            return
        if vad.is_speech(indata.tobytes(), SAMPLE_RATE):
            speech_frame_count["count"] += 1
            if speech_frame_count["count"] >= 5:
                interrupt_flag["stop"] = True
                sd.stop()  # cuts off current playback immediately
            else:
                speech_frame_count["count"] = 0

    mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                 blocksize=FRAME_SIZE, device=DEVICE_IN, callback=callback)
    mic_stream.start()

def stop_mic_monitor():
    global mic_stream
    interrupt_flag["active"] = False
    if mic_stream:
        mic_stream.stop()
        mic_stream.close()
        mic_stream = None

def speak(text):
    if interrupt_flag["stop"]:
        return
    generator = kokoro_pipeline(text, voice='af_heart')
    for gs, ps, audio in generator:
        if interrupt_flag["stop"]:
            break
        audio_resampled = resample(audio, int(len(audio) * 48000 / 24000))
        sd.play(audio_resampled, samplerate=48000, device=DEVICE_OUT)
        sd.wait()


if __name__ == "__main__":
    print("Voice assistant ready. Press Ctrl+C to quit.\n")

    conversation_history = [
        {"role": "system", "content": "You are a voice assistant. Keep answers short, natural, and conversational — 1 to 3 sentences unless the user explicitly asks for more detail."}
    ]

    while True:
        record_until_silence()
        user_text = transcribe_audio()

        if not user_text:
            print("Didn't catch anything, try again.")
            continue

        print(f"📝 You said: {user_text}")

        if is_exit(user_text):
            speak("Goodbye! Have a great day.")
            print("Ending conversation.")
            break

        conversation_history.append({"role": "user", "content": user_text})
        reply = ask_ollama(conversation_history)
        conversation_history.append({"role": "assistant", "content": reply})