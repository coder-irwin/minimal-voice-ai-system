<p align="center">
  <h1 align="center">🎙️ Realtime Voice AI System</h1>
  <p align="center">
    <strong>Streaming STT → Local LLM → Live TTS — All in Real Time</strong>
  </p>
  <p align="center">
    A production-grade, fully local, realtime conversational AI backend with streaming speech-to-text, local LLM inference, and live text-to-speech — all orchestrated through a single persistent WebSocket connection.
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start (Local Development)](#-quick-start-local-development)
- [Configuration & Deployment Profiles](#-configuration--deployment-profiles)
- [Hardware-Aware Optimization](#-hardware-aware-optimization)
- [Component Deep Dive](#-component-deep-dive)
- [WebSocket API Reference](#-websocket-api-reference)
- [Frontend Integration](#-frontend-integration)
- [Docker & Production Deployment](#-docker--production-deployment)
- [Performance Tuning](#-performance-tuning)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🌟 Overview

This system creates a **fully local, privacy-first voice AI assistant** that runs entirely on your machine — no cloud APIs required for core functionality. It's designed for ultra-low-latency conversational AI where a user speaks into their microphone and hears an AI-generated voice response in real time.

### Key Features

| Feature | Description |
|---------|-------------|
| **Realtime Streaming** | Audio flows through the entire pipeline in a single persistent WebSocket connection |
| **Local LLM Inference** | Uses Ollama for local model inference — no API keys, no cloud dependency |
| **Dual TTS Engines** | Kokoro (high quality, neural) and Piper (lightweight, fast) — hot-swappable at runtime |
| **Hardware-Aware** | Automatically detects Apple Silicon, NVIDIA CUDA, or CPU-only and optimizes accordingly |
| **Barge-In / Interruption** | User can interrupt the AI mid-response — the system cancels LLM + TTS immediately |
| **Eager First Chunk TTS** | TTS starts speaking after just 2-3 words from the LLM, not after a full sentence |
| **Multi-Layer Memory** | Recent turns, persistent user facts, conversation summaries, active task tracking |
| **Session Persistence** | Full transcript + audio WAV saved per session to disk |
| **WebRTC VAD** | Google's WebRTC Voice Activity Detection with volume gating for accurate speech detection |
| **Deployment Profiles** | Pre-configured profiles for local dev, CPU server, GPU server, and cloud GPU |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        BROWSER (Frontend)                        │
│                                                                  │
│  Microphone ──► PCM 16kHz mono ──► WebSocket ──► Backend         │
│                                                                  │
│  Speaker   ◄── Base64 Audio    ◄── WebSocket ◄── Backend         │
└──────────────────────────────────────────────────────────────────┘
                              │
                    Persistent WebSocket
                      /ws/stream
                              │
┌──────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (Python)                       │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────────┐  │
│  │   VAD   │───►│   STT   │───►│   LLM   │───►│  TTS Engine  │  │
│  │(WebRTC) │    │(Whisper)│    │(Ollama) │    │(Kokoro/Piper)│  │
│  └─────────┘    └─────────┘    └─────────┘    └──────────────┘  │
│       │              │              │                │           │
│       │              │              │                │           │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐   ┌──────────┐     │
│  │ Session │    │ Memory  │    │Persistence│   │ Hardware │     │
│  │ Manager │    │ Manager │    │  Layer    │   │ Manager  │     │
│  └─────────┘    └─────────┘    └──────────┘   └──────────┘     │
└──────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   OLLAMA SERVER    │
                    │  (Local or Remote) │
                    │                   │
                    │  qwen2.5:1.5b     │
                    │  llama3.2:3b      │
                    │  gemma3:4b        │
                    │  etc.             │
                    └───────────────────┘
```

### Data Flow (Single Turn)

```
1. User speaks into microphone
2. Browser captures PCM 16kHz mono audio chunks
3. Chunks stream over WebSocket to backend
4. VAD (WebRTC) detects speech onset/offset
5. During speech: partial STT every 0.5s (fire-and-forget)
6. After 0.5s silence: final STT transcription
7. Transcription + conversation history → LLM (Ollama)
8. LLM streams tokens back
9. SmartSentenceChunker splits tokens into TTS-sized chunks
   - FIRST chunk sent after ~2-3 words (eager mode)
   - Subsequent chunks wait for sentence boundaries
10. TTS synthesizes audio chunks in parallel
11. Audio chunks stream back to browser via WebSocket
12. Browser plays audio through speaker
13. Session, transcript, and audio persisted to disk
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI + Uvicorn | Async HTTP + WebSocket server |
| **Speech-to-Text** | Faster-Whisper (CTranslate2) | Local STT with Silero VAD integration |
| **Voice Activity Detection** | WebRTC VAD (webrtcvad) | Accurate speech/silence detection with volume gating |
| **Language Model** | Ollama (local) | Local LLM inference (Qwen, Llama, Gemma, etc.) |
| **Text-to-Speech** | Kokoro ONNX + Piper TTS | Dual neural TTS engines, hot-swappable |
| **Frontend** | Vanilla HTML/JS (single file) | Browser-based UI with mic capture + audio playback |
| **Containerization** | Docker + Docker Compose | Production deployment packaging |

---

## 📁 Project Structure

```
voice_ai_backend/
├── app/                          # Core application modules
│   ├── main.py                   # FastAPI app, WebSocket handler, pipeline orchestration
│   ├── stt.py                    # Faster-Whisper STT engine (singleton)
│   ├── llm.py                    # Ollama/Gemini LLM agent with streaming
│   ├── vad.py                    # WebRTC-based Voice Activity Detector
│   ├── session.py                # Per-user session state + system prompts
│   ├── memory_manager.py         # Multi-layer conversational memory
│   ├── persistence.py            # Disk-based session/transcript/audio storage
│   ├── tts_manager.py            # TTS engine router (Kokoro ↔ Piper)
│   ├── hardware_manager.py       # Runtime GPU/CPU detection singleton
│   ├── deployment_config.py      # Deployment profiles (local/cpu/gpu/cloud)
│   └── tts/                      # TTS provider implementations
│       ├── base.py               # Abstract base TTS provider
│       ├── kokoro_provider.py    # Kokoro ONNX neural TTS
│       └── piper_provider.py     # Piper lightweight TTS
├── models/                       # Pre-downloaded TTS model files
│   ├── kokoro/                   # Kokoro ONNX model + voice pack
│   │   ├── kokoro-v0_19.onnx     # ~325 MB neural TTS model
│   │   └── voices.bin            # ~5.7 MB multi-voice pack
│   └── piper/                    # Piper ONNX model
│       ├── en_US-libritts_r-medium.onnx      # ~78 MB voice model
│       └── en_US-libritts_r-medium.onnx.json # Voice config
├── sessions/                     # Persisted session data (auto-created)
│   └── session_<id>/
│       ├── metadata.json         # Session config & timestamps
│       ├── transcript.json       # Full conversation transcript
│       └── audio/                # Recorded audio files
│           └── conversation.wav  # Full session audio recording
├── index.html                    # Browser-based frontend UI
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Production container image
├── test_client.py                # CLI WebSocket test client
└── docker-compose.yml            # Multi-service production deployment (root level)
```

---

## 📦 Prerequisites

### Required Software

| Software | Version | Purpose | Install |
|----------|---------|---------|---------|
| **Python** | 3.10+ | Runtime | [python.org](https://www.python.org/downloads/) |
| **Ollama** | Latest | Local LLM server | [ollama.com](https://ollama.com/download) |
| **espeak-ng** | Latest | TTS phonemizer (Kokoro) | `brew install espeak-ng` (macOS) / `apt install espeak-ng` (Linux) |
| **ffmpeg** | Latest | Audio processing | `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux) |
| **portaudio** | Latest | Audio I/O | `brew install portaudio` (macOS) / `apt install portaudio19-dev` (Linux) |

### Hardware Requirements

| Hardware | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 8 GB | 16+ GB |
| **CPU** | 4 cores | 8+ cores (Apple M-series or modern Intel/AMD) |
| **GPU** | Not required | NVIDIA with 8+ GB VRAM for faster inference |
| **Disk** | 2 GB free | 5+ GB (for models + session recordings) |

---

## 🚀 Quick Start (Local Development)

### Step 1: Install Ollama and Pull a Model

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull a lightweight model (in a separate terminal)
ollama pull qwen2.5:1.5b-instruct
```

> **Model Recommendations:**
> - **Apple Silicon M1-M4 / CPU:** `qwen2.5:1.5b-instruct` (fastest, ~1 GB)
> - **NVIDIA GPU 8GB+:** `llama3.2:3b-instruct` (better quality, ~2 GB)
> - **NVIDIA GPU 12GB+:** `gemma3:4b` (highest quality, ~3 GB)

### Step 2: Clone and Set Up the Project

```bash
# Clone the repository
git clone <your-repo-url>
cd voice_ai_backend

# Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download TTS Models

#### Kokoro TTS (High Quality Neural Voice)

```bash
mkdir -p models/kokoro
cd models/kokoro

# Download the ONNX model (~325 MB)
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v0.1/kokoro-v0_19.onnx

# Download the voice pack (~5.7 MB)
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v0.1/voices.bin

cd ../..
```

#### Piper TTS (Lightweight Fast Voice)

```bash
mkdir -p models/piper
cd models/piper

# Download the ONNX model (~78 MB)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx

# Download the config
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json

cd ../..
```

### Step 4: Start the Server

```bash
# Make sure Ollama is running in another terminal: ollama serve
python -m app.main
```

You should see startup telemetry like this:

```
======================================================
⚙️  HARDWARE & DEPLOYMENT TELEMETRY
======================================================
Deployment Profile: local_dev
Active Platform:    apple_silicon
CPU Cores:          10
System RAM:         24 GB
CUDA GPU Support:   False
STT Model Size:     base.en
LLM Dynamic Model:  qwen2.5:1.5b-instruct
TTS Backend Profile: onnx_coreml
======================================================
Backend ready.
```

### Step 5: Open the Frontend

Open your browser and navigate to:

```
http://localhost:8000
```

Click **"Start Listening"** and speak into your microphone. The AI will respond in real time.

---

## ⚙️ Configuration & Deployment Profiles

The system uses **deployment profiles** defined in `app/deployment_config.py`. Set the active profile via environment variable:

```bash
export DEPLOYMENT_PROFILE=local_dev  # default
```

### Available Profiles

| Profile | STT Model | Context Window | Max Tokens | TTS Engine | Max Sessions | Best For |
|---------|-----------|---------------|------------|------------|-------------|----------|
| `local_dev` | `base.en` | 512 | 60 | Piper | 5 | MacBook / local dev |
| `cpu_server` | `small.en` | 2048 | 64 | Kokoro | 20 | Cloud CPU instances |
| `gpu_server` | `medium.en` | 4096 | 128 | Kokoro | 50 | On-premise GPU servers |
| `cloud_gpu` | `large-v3` | 4096 | 128 | Kokoro | 100 | Cloud GPU (A100, T4, etc.) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPLOYMENT_PROFILE` | `local_dev` | Active configuration profile |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | Auto-detected | Override LLM model selection |
| `ONNX_PROVIDER` | Auto-detected | Force ONNX execution provider (e.g., `CUDAExecutionProvider`) |
| `GEMINI_API_KEY` | None | Optional Google Gemini API key for cloud LLM fallback |

---

## 🖥️ Hardware-Aware Optimization

The system automatically detects your hardware at startup and optimizes the entire pipeline:

```python
# Automatic detection happens in hardware_manager.py
# You don't need to configure anything manually
```

### Detection Matrix

| Hardware | STT Device | LLM Backend | TTS Backend | Auto-Selected Model |
|----------|-----------|-------------|-------------|-------------------|
| **Apple Silicon** (M1/M2/M3/M4) | CPU (int8) | Metal (via Ollama) | ONNX CoreML | `qwen2.5:1.5b-instruct` |
| **NVIDIA GPU** (8+ GB VRAM) | CUDA (fp16) | CUDA (via Ollama) | ONNX CUDA | `llama3.2:3b-instruct` |
| **NVIDIA GPU** (12+ GB VRAM) | CUDA (fp16) | CUDA (via Ollama) | ONNX CUDA | `gemma3:4b` |
| **CPU Only** | CPU (int8) | CPU (via Ollama) | ONNX CPU | `qwen2.5:1.5b-instruct` |

---

## 🔍 Component Deep Dive

### 1. Voice Activity Detection (`app/vad.py`)

Uses **Google's WebRTC VAD** (aggressiveness level 3) with a volume gate to filter out background noise and only trigger on the primary speaker.

- **Frame size:** 30ms (960 bytes at 16kHz)
- **Onset:** 2 consecutive speech frames to start
- **Grace period:** 12 frames (360ms) of silence before stopping
- **Volume gate:** RMS > 1200.0 threshold filters background speakers

### 2. Speech-to-Text (`app/stt.py`)

**Faster-Whisper** (CTranslate2) with two transcription modes:

| Mode | When Used | Beam Size | Purpose |
|------|-----------|-----------|---------|
| `transcribe_chunk()` | Every 0.5s during speech | 1 | Fast partial transcripts for live display |
| `transcribe_final()` | After silence detected | 1 | Final high-quality transcription |

Both run in a thread pool (`asyncio.to_thread`) so they never block the event loop.

### 3. Language Model (`app/llm.py`)

**AIAgent** handles streaming LLM inference with automatic backend selection:

- **Primary:** Ollama (local) — streams via NDJSON over HTTP
- **Fallback:** Google Gemini (cloud) — only if `GEMINI_API_KEY` is set and Ollama is unavailable
- **Dynamic model routing:** Automatically selects the best model based on available VRAM/hardware
- **Keep-alive:** Models stay loaded in memory for 30+ minutes to avoid cold-start latency

### 4. Multi-Layer Memory (`app/memory_manager.py`)

The conversation memory system is designed for **zero background LLM cost**:

| Layer | Contents | Token Cost |
|-------|----------|-----------|
| **System Prompt** | Angelina identity + Glocal Assist overview | ~150 tokens (constant) |
| **Dynamic Injection** | Business logic OR scheduling rules (state-dependent) | 0-100 tokens |
| **Recent Turns** | Last 4 messages (2 conversational turns) | ~80 tokens |
| **Active Tasks** | Scheduled meetings/appointments | ~20 tokens |

**Critical design decision:** All background processing (state detection, task extraction) uses deterministic Python heuristics — **zero LLM calls**. This prevents background tasks from blocking the next user turn's TTFT.

### 5. Text-to-Speech (`app/tts_manager.py`)

Dual-engine TTS with runtime hot-swapping:

| Engine | Quality | Speed | Model Size | Best For |
|--------|---------|-------|-----------|----------|
| **Kokoro** | High (neural) | ~2x realtime | 325 MB | Production / final quality |
| **Piper** | Good (neural) | ~5x realtime | 78 MB | Low latency / resource-constrained |

### 6. Eager First Chunk Streaming

The `SmartSentenceChunker` implements a **two-phase strategy**:

```
Phase 1 (EAGER): Send first chunk to TTS after ~8 characters (~2-3 words)
                  → User hears audio almost immediately after TTFT

Phase 2 (NORMAL): Wait for sentence boundaries (. ! ? , ;) for natural prosody
                   → Subsequent chunks sound natural and well-paced
```

This means the user hears audio **200-300ms after the first LLM token**, rather than waiting 1-2 seconds for a complete sentence.

### 7. Barge-In / Interruption

When the user starts speaking while the AI is still talking:

1. VAD detects speech onset → `cancel_active_llm()` called
2. Active LLM `asyncio.Task` is cancelled
3. Active TTS `asyncio.Task` is cancelled
4. Audio streaming stops immediately
5. New STT transcription begins on the user's fresh utterance
6. Pipeline restarts from scratch with the new input

### 8. Session Persistence (`app/persistence.py`)

Every session is automatically saved to disk:

```
sessions/session_<id>/
├── metadata.json      # Session config, timestamps
├── transcript.json    # Full conversation [{role, text, timestamp}, ...]
└── audio/
    └── conversation.wav  # Complete session audio recording
```

Sessions can be **restored** — if a client reconnects with the same session ID, the backend loads prior context.

---

## 📡 WebSocket API Reference

### Endpoint

```
ws://localhost:8000/ws/stream
```

### Client → Server Messages

| Message Type | Format | Description |
|-------------|--------|-------------|
| **Audio chunk** | Binary (bytes) | Raw PCM 16kHz mono int16 audio |
| **Set config** | JSON `{"type": "set_config", "llm_model": "...", "tts_engine": "...", "tts_voice": "...", "session_id": "..."}` | Update session settings |

### Server → Client Events

| Event | Payload | Description |
|-------|---------|-------------|
| `session_started` | `{session_id}` | Session initialized |
| `available_models` | `{llms, tts_engines, tts_voices}` | Available model/voice options |
| `config_updated` | `{}` | Config change acknowledged |
| `user_started_speaking` | `{}` | VAD detected speech onset |
| `partial_transcript` | `{text, latency_ms}` | Live partial STT result |
| `user_stopped_speaking` | `{}` | VAD detected silence |
| `final_transcript` | `{text, latency_ms}` | Final STT result |
| `llm_processing_started` | `{generation_id}` | LLM inference started |
| `llm_partial_response` | `{token, first_token_ms, generation_id}` | Streaming LLM token |
| `llm_final_response` | `{text, first_token_ms, total_ms, generation_id}` | Complete LLM response |
| `assistant_speaking_started` | `{generation_id}` | TTS audio streaming began |
| `assistant_audio_chunk` | `{audio_b64, sample_rate, generation_id}` | Base64 PCM audio chunk |
| `assistant_speaking_finished` | `{generation_id}` | TTS finished for this turn |
| `assistant_finished` | `{generation_id}` | Full turn complete |
| `turn_metrics` | `{metrics}` | Latency breakdown for this turn |
| `system_metrics` | `{cpu_percent, mem_percent, mem_used_gb}` | Live system resource usage |
| `error` | `{message}` | Error occurred |

---

## 🌐 Frontend Integration

The included `index.html` is a complete, self-contained frontend:

### Key Frontend Features

- **Microphone capture:** Uses `getUserMedia` + `AudioWorklet` for low-latency PCM capture
- **Audio playback:** Real-time PCM chunk playback via `AudioContext`
- **Live transcription:** Displays partial + final transcripts in real time
- **Turn metrics:** Shows STT latency, LLM TTFT, TTS first chunk, end-to-end latency
- **System monitoring:** Live CPU and RAM usage display
- **Model selection:** Runtime LLM model and TTS engine/voice switching
- **Session management:** Persistent session IDs with automatic reconnection

### Building Your Own Frontend

Minimal JavaScript example to connect to the backend:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

// Send config on connect
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'set_config',
    llm_model: 'qwen2.5:1.5b-instruct',
    tts_engine: 'kokoro',
    tts_voice: 'af_heart'
  }));
};

// Handle events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'partial_transcript':
      console.log('Partial:', data.text);
      break;
    case 'final_transcript':
      console.log('Final:', data.text);
      break;
    case 'llm_partial_response':
      process.stdout.write(data.token);
      break;
    case 'assistant_audio_chunk':
      // Decode base64 and play through AudioContext
      const audioBytes = atob(data.audio_b64);
      playAudio(audioBytes, data.sample_rate);
      break;
  }
};

// Stream microphone audio
navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } })
  .then(stream => {
    // Process audio and send PCM chunks via ws.send(pcmBytes)
  });
```

---

## 🐳 Docker & Production Deployment

### Option 1: Docker Compose (Recommended)

The `docker-compose.yml` at the project root spins up both the Ollama server and the voice backend:

```bash
# From the project root (one level above voice_ai_backend)
docker compose up -d
```

This starts:
- **ollama** — LLM inference server with GPU passthrough
- **voice-backend** — The complete voice AI backend

### Option 2: Standalone Docker

```bash
# Build the image
cd voice_ai_backend
docker build -t voice-ai-backend .

# Run (assumes Ollama is running separately)
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e DEPLOYMENT_PROFILE=cpu_server \
  -v ./sessions:/data/sessions \
  voice-ai-backend
```

### Option 3: Cloud Deployment (GPU)

For cloud GPU instances (AWS, GCP, Azure):

```bash
# Set environment for cloud GPU
export DEPLOYMENT_PROFILE=cloud_gpu
export OLLAMA_BASE_URL=http://ollama-service:11434

# Run with Docker Compose
docker compose up -d
```

### Health Check

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "components": {
    "llm": {"status": "ok", "backend": "ollama", "default_model": "qwen2.5:1.5b-instruct"},
    "stt": {"status": "ok"},
    "tts": {"status": "ok", "kokoro_loaded": true, "piper_voices_found": 1},
    "persistence": {"status": "ok", "writable": true}
  }
}
```

---

## ⚡ Performance Tuning

### Latency Budget Breakdown

For a typical turn on Apple Silicon M4 (24 GB):

| Stage | Time | Notes |
|-------|------|-------|
| Silence detection | 500ms | Configurable via `SILENCE_TIMEOUT` |
| Final STT | 400-600ms | beam_size=1, base.en model |
| LLM TTFT | 800-1500ms | qwen2.5:1.5b, num_ctx=512 |
| TTS first audio | 200-500ms | Eager first chunk after ~2-3 words |
| **Total end-to-end** | **~1.5-3s** | From speech end to first audio |

### Key Tuning Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `SILENCE_TIMEOUT` | `app/main.py` | 0.5s | Lower = faster response, higher = fewer false triggers |
| `num_ctx` | `app/llm.py` | 512 | Lower = faster TTFT, but less context available |
| `num_predict` | `app/llm.py` | 60 | Max tokens generated per response |
| `num_thread` | `app/llm.py` | 8 | CPU threads for Ollama inference |
| `beam_size` | `app/stt.py` | 1 | Higher = better accuracy, slower |
| `first_chunk_chars` | `app/main.py` | 8 | Characters before first TTS chunk fires |
| `min_volume_threshold` | `app/vad.py` | 1200.0 | RMS threshold for speech detection |
| `grace_frames` | `app/vad.py` | 12 | Silence frames (30ms each) before speech end |

### Tips for Faster TTFT

1. **Use a smaller model:** `qwen2.5:0.5b-instruct` is 2-3x faster than `1.5b`
2. **Reduce `num_ctx`:** Every token in the context adds ~1-2ms of prefill time on CPU
3. **Keep Ollama warm:** The `keep_alive` setting keeps models loaded in memory
4. **Avoid background LLM calls:** The memory manager uses zero-cost heuristics for this reason
5. **Use NVIDIA GPU:** Even a GTX 1060 will be 5-10x faster than M4 CPU for LLM inference

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `Address already in use` | Port 8000 occupied | `lsof -ti:8000 \| xargs kill -9` |
| `Ollama connection refused` | Ollama not running | Start with `ollama serve` in another terminal |
| No audio from TTS | Missing espeak-ng | Install: `brew install espeak-ng` (macOS) |
| STT returns empty | Audio too quiet / short | Lower `min_volume_threshold` in VAD, check mic permissions |
| TTFT > 5 seconds | Context too large / model too big | Reduce `num_ctx`, use smaller model, check for background LLM calls |
| Responses cut off mid-sentence | `num_predict` too low | Increase `num_predict` in `app/llm.py` |
| "AI language model" responses | System prompt too generic | Check `SYSTEM_PROMPT` in `app/session.py` |
| Kokoro model not found | Missing ONNX files | Download to `models/kokoro/` (see Quick Start) |
| WebSocket disconnects | Timeout or memory | Check `session_timeout_seconds` in deployment config |

### Checking Logs

The backend outputs detailed turn timelines:

```
=== 🕒 TURN TIMELINE [abc12345] ===
1. Speech Started:     +0ms
2. Speech Ended:       +2304ms
3. STT Completed:      +2714ms
4. LLM 1st Token:      +3527ms
5. LLM Completed:      +5267ms
6. TTS 1st Audio Sent: +3867ms
=========================================
```

### Verifying Hardware Detection

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

Check the `llm.backend` and `llm.default_model` fields to confirm the system detected your hardware correctly.

---

## 📜 License

This project is proprietary to Glocal Assist. All rights reserved.

---

<p align="center">
  <strong>Built with ❤️ for ultra-low-latency conversational AI</strong>
</p>
