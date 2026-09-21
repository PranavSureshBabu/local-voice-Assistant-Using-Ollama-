from kokoro import KPipeline
import sounddevice as sd

pipeline = KPipeline(lang_code='a')  # 'a' = American English

text = "Hello! This is a test of the Kokoro text to speech engine."
generator = pipeline(text, voice='af_heart')

for gs, ps, audio in generator:
    sd.play(audio, samplerate=24000)
    sd.wait()