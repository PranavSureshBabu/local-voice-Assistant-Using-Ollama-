# local-voice-Assistant-Using-Ollama-

Architecture
 Mic → VAD (webrtcvad) → Faster-Whisper (STT) → Ollama / Llama 3.2 (LLM) → Kokoro-82M (TTS) → Speakers
                                                        ↑
                                          Conversation history (multi-turn memory)
                                          
Voice Activity Detection decides when you start and stop talking — no fixed recording window.

Speech-to-Text transcribes your speech locally.

The LLM generates a response, streamed token-by-token.

Text-to-Speech speaks each completed sentence as soon as it's ready, while the LLM keeps generating the rest — reducing perceived latency.

Barge-in lets you interrupt the assistant mid-reply by speaking over it.

<img width="680" height="608" alt="image" src="https://github.com/user-attachments/assets/f91e74fa-49ff-44e2-b83a-503ed0e7df04" />


## Step-by-Step Build Process

Day 1 — Environment & baseline
Installed Ollama, pulled llama3.2:3b
Set up Python venv and core packages (faster-whisper, sounddevice, ollama)
Validated Ollama's streaming API and Whisper transcription independently before combining them

Day 2 — End-to-end pipeline
Combined recording → transcription → LLM → TTS (pyttsx3 initially) into one script
Fixed a silent-recording bug caused by the default MME audio driver; switched to the WASAPI device index
Resolved a PortAudioError by recording at the microphone's native sample rate (48000 Hz) instead of forcing 16000 Hz — Faster-Whisper resamples internally

Day 3 — Voice Activity Detection
Replaced fixed-duration recording with webrtcvad, so the assistant starts/stops recording based on actual speech
Rewrote the recording function to use a callback-based audio stream (more reliable than manual polling, which caused distorted/silent captures)
Tuned silence-timeout and VAD aggressiveness to avoid cutting off longer sentences with natural pauses

Day 4 — Latency optimization
Streamed LLM responses sentence-by-sentence into TTS instead of waiting for the full reply — the assistant starts speaking within ~1 second instead of waiting for full generation
Added timing instrumentation (first_token, first_speech) to measure and reason about latency
Added a system prompt instructing the LLM to give short, conversational answers suited for voice output

Day 5 — Robustness
Conversation memory: passed full message history to Ollama so follow-up questions ("what are examples of it?") resolve correctly
Graceful exit: detects goodbye phrases and ends the session without an unnecessary LLM call
Barge-in: switched from pyttsx3 to Kokoro-82M (played via sounddevice) specifically because pyttsx3 cannot be reliably interrupted on Windows; implemented a background mic-monitoring thread that stops playback and halts further generation when the user starts speaking
Fixed a PortAudioError caused by opening a new mic stream per sentence — the mic stream is now opened once per conversational turn

Day 6 — Cloud deployment (partial)
Explored offloading the LLM (Ollama) to a remote AWS EC2 instance, keeping STT/TTS local
Launched a free-tier t2.micro EC2 instance for learning purposes (see Deployment Notes below)


## Future Improvements

True interrupt-to-transcription: continuously buffer audio through the assistant's speaking turn so that words spoken during a barge-in become the next transcribed input, rather than being discarded

Wake word activation (e.g., openWakeWord) for hands-free "always listening" operation instead of continuous VAD

Retrieval-augmented responses for factual accuracy — connecting the LLM to a live search tool for questions outside its training knowledge

Full cloud deployment with a browser-based client (WebRTC mic capture + FastAPI/WebSocket backend) so the assistant is accessible from any device via a shareable link, rather than only via a local Python script

Acoustic echo cancellation to allow reliable barge-in over speakers instead of requiring headphones

Model benchmarking across model sizes (1b vs 3b) and Whisper model sizes (small.en vs medium.en) to quantify the latency/accuracy tradeoff with real numbers
