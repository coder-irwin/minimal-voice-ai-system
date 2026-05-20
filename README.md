# Minimal Voice AI Backend

A production-ready, ultra-low latency real-time voice AI backend using FastAPI, WebSockets, Faster-Whisper, and LLM integrations.

## Prerequisites
- Python 3.9+
- A working microphone
- (Optional) `GEMINI_API_KEY` for conversational LLM responses

## Installation

1. Navigate to the directory:
   ```bash
   cd "minimal voice ai system/voice_ai_backend"
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Mac/Linux
   # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have issues with `pyaudio` on Mac, you may need to run `brew install portaudio` before installing.*

4. (Optional) Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   # or create a .env file with GEMINI_API_KEY=...
   ```

## Running the Server
Start the FastAPI server:
```bash
python -m app.main
```
The first time you run this, it will download the Faster-Whisper `base.en` model (very quick).

## Running the Client
Open a new terminal window, activate the virtual environment, and run:
```bash
python test_client.py
```
Start speaking! The terminal will print your transcribed speech and the AI's response in real-time.

## How it Works
- `test_client.py` captures 16kHz PCM audio and streams it to the server via WebSockets.
- `app/main.py` is a FastAPI WebSocket server that buffers audio chunks.
- `app/stt.py` uses `faster-whisper` for fast, local transcription of audio chunks.
- `app/llm.py` takes the transcript and passes it to an AI Agent (Gemini by default, falling back to an Echo bot if no key is provided).
