"""
main.py — End-to-end realtime Voice AI pipeline.

Architecture:
  User Voice → Browser Mic (PCM 16kHz)
    → Persistent WebSocket
      → VAD (energy-based, server-side)
        → Partial STT (every 0.5s, fire-and-forget)
        → Silence > 2s detected
          → Final STT (beam_size=3)
            → Ollama LLM (streaming tokens)
              → llm_partial_response events (each token)
              → llm_final_response event
              → assistant_finished
            → Auto-resume listening

Events emitted to client:
  - session_started
  - user_started_speaking
  - partial_transcript     {text, latency_ms}
  - user_stopped_speaking
  - final_transcript       {text, latency_ms}
  - llm_processing_started
  - llm_partial_response   {token, first_token_ms?}
  - llm_final_response     {text, total_ms}
  - assistant_finished
  - error                  {message}
"""

import os
import time
import uuid
import asyncio
import logging
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()

from app.stt import STTEngine
from app.llm import AIAgent
from app.vad import EnergyVAD
from app.session import Session, session_manager
from app.memory_manager import MemoryManager
from app.piper_manager import PiperManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SEC = SAMPLE_RATE * BYTES_PER_SAMPLE  # 32000

# Partial transcription interval (bytes ≈ 0.5s of audio)
PARTIAL_CHUNK_BYTES = int(BYTES_PER_SEC * 0.5)

# Silence threshold to finalize (seconds)
SILENCE_TIMEOUT = 2.0

# Minimum speech bytes before we bother with transcription
MIN_SPEECH_BYTES = int(BYTES_PER_SEC * 0.3)


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown — load singleton engines."""
    logger.info("Starting Voice AI Backend (End-to-End Pipeline)…")

    # STT engine (singleton, model loaded once)
    app.state.stt = STTEngine(model_size="base.en")

    # LLM agent (singleton, persistent Ollama session)
    app.state.llm = AIAgent()
    await app.state.llm.startup()

    # Memory Manager (orchestrates multi-layer context)
    app.state.memory_manager = MemoryManager(app.state.llm)

    # Piper TTS Manager (singleton)
    app.state.piper = PiperManager()

    logger.info("Backend ready.")
    yield

    # Cleanup
    await app.state.llm.shutdown()
    logger.info("Backend shut down.")


app = FastAPI(title="Voice AI Backend", lifespan=lifespan)


# ── Frontend ─────────────────────────────────────────────────────────
@app.get("/")
async def get_frontend():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())


# ── Helpers ──────────────────────────────────────────────────────────
async def _send(ws: WebSocket, event_type: str, **kwargs):
    """Send a JSON event to the client. Silently ignores closed sockets."""
    try:
        await ws.send_json({"type": event_type, **kwargs})
    except Exception:
        pass


async def _do_partial_transcription(
    stt: STTEngine, session: Session, ws: WebSocket
):
    """Transcribe current audio buffer and emit partial_transcript if new."""
    buf = bytes(session.live_audio_buffer)
    if len(buf) < MIN_SPEECH_BYTES:
        return

    t0 = time.perf_counter()
    text = await stt.transcribe_chunk(buf)
    latency = (time.perf_counter() - t0) * 1000

    if text and text != session.last_partial:
        session.speech_buffer_text = text
        session.last_partial = text
        await _send(ws, "partial_transcript", text=text, latency_ms=round(latency, 1))
        logger.info(f"[partial] {text}  ({latency:.0f}ms)")


async def _do_finalize_and_stream(
    stt: STTEngine, llm: AIAgent, memory_manager: MemoryManager, piper: PiperManager, session: Session, ws: WebSocket
):
    """
    Called when silence > SILENCE_TIMEOUT.
    1. Final STT on full utterance buffer
    2. Stream LLM tokens to client
    3. Triggers memory processing in the background
    4. Reset for next utterance
    """
    buf = bytes(session.live_audio_buffer)
    if len(buf) < MIN_SPEECH_BYTES:
        session.reset_speech_state()
        return

    # ── 1. Final STT ──
    t_stt = time.perf_counter()
    session.turn_metrics['stt_start_time'] = time.time()
    text = await stt.transcribe_final(buf)
    session.turn_metrics['stt_end_time'] = time.time()
    stt_ms = (time.perf_counter() - t_stt) * 1000

    if not text.strip():
        session.reset_speech_state()
        return

    await _send(ws, "final_transcript", text=text, latency_ms=round(stt_ms, 1))
    logger.info(f"[FINAL] {text}  ({stt_ms:.0f}ms)")

    # ── 2. Stream LLM response ──
    generation_id = f"gen_{uuid.uuid4().hex[:8]}"
    session.active_generation_id = generation_id

    await _send(ws, "llm_processing_started", generation_id=generation_id)
    t_llm = time.perf_counter()

    # Synchronously persist the user turn for perfect conversational continuity
    session.recent_turns.append({"role": "user", "text": text})
    if len(session.recent_turns) > 100:
        session.recent_turns = session.recent_turns[-100:]

    # Build optimized messages from multi-layer memory
    messages = memory_manager.build_messages(session)

    full_response = []
    first_token_ms = None

    try:
        async for chunk in llm.stream_response(messages, generation_id=generation_id):
            if session.active_generation_id != generation_id:
                logger.info(f"[{session.session_id}] Stale stream invalidated")
                break

            token = chunk.get("token", "")
            done = chunk.get("done", False)
            ftm = chunk.get("first_token_ms")

            if ftm is not None:
                first_token_ms = round(ftm, 1)

            if token:
                if 'llm_first_token_time' not in session.turn_metrics:
                    session.turn_metrics['llm_first_token_time'] = time.time()
                full_response.append(token)
                await _send(
                    ws,
                    "llm_partial_response",
                    token=token,
                    first_token_ms=first_token_ms if ftm is not None else None,
                    generation_id=generation_id,
                )

            if done:
                break

    except asyncio.CancelledError:
        logger.info(f"[{session.session_id}] LLM streaming cancelled")
        session.reset_speech_state()
        return

    session.turn_metrics['llm_end_time'] = time.time()
    total_llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)
    response_text = "".join(full_response).strip()

    if response_text and session.active_generation_id == generation_id:
        # Clean up repetitive closers for premium UX
        for phrase in ["How can I help you today?", "Feel free to ask.", "Let me know if", "How can I help you?"]:
            if response_text.endswith(phrase):
                response_text = response_text.replace(phrase, "").strip()

        # Synchronously persist the assistant turn
        session.recent_turns.append({"role": "assistant", "text": response_text})
        if len(session.recent_turns) > 100:
            session.recent_turns = session.recent_turns[-100:]

        await _send(
            ws,
            "llm_final_response",
            text=response_text,
            first_token_ms=first_token_ms,
            total_ms=total_llm_ms,
            generation_id=generation_id,
        )
        logger.info(
            f"[LLM] {response_text[:80]}… "
            f"(first_token={first_token_ms}ms, total={total_llm_ms}ms)"
        )
        
        # ── 3. Background Memory Processing ──
        # Offload layer 2 and 3 updates to the background (facts & summary)
        asyncio.create_task(
            memory_manager.process_turn_background(session, text, response_text)
        )
        
        # ── 4. TTS Synthesis and Streaming ──
        await _send(ws, "assistant_speaking_started", generation_id=generation_id)
        try:
            first_tts_chunk = True
            async for chunk_bytes, sample_rate in piper.synthesize_stream(response_text):
                if first_tts_chunk:
                    session.turn_metrics['tts_first_chunk_time'] = time.time()
                    first_tts_chunk = False
                    
                # Check for barge-in every chunk
                if session.active_generation_id != generation_id:
                    logger.info(f"[{session.session_id}] TTS streaming interrupted by barge-in")
                    break
                
                chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")
                await _send(
                    ws, 
                    "assistant_audio_chunk", 
                    audio_b64=chunk_b64, 
                    sample_rate=sample_rate, 
                    generation_id=generation_id
                )
        except Exception as e:
            logger.error(f"[{session.session_id}] TTS streaming error: {e}")
            
        if session.active_generation_id == generation_id:
            await _send(ws, "assistant_speaking_finished", generation_id=generation_id)

    # ── Print Turn Timeline Log ──
    t0 = session.turn_metrics.get('speech_start_time', 0)
    def ms_diff(key, raw=False):
        t = session.turn_metrics.get(key)
        if raw: return int((t - t0) * 1000) if t else None
        return f"+{(t - t0)*1000:.0f}ms" if t else "N/A"
        
    timeline = (
        f"\n\n=== 🕒 TURN TIMELINE [{session.session_id}] ===\n"
        f"1. Speech Started:     +0ms\n"
        f"2. Speech Ended:       {ms_diff('speech_end_time')}\n"
        f"3. STT Completed:      {ms_diff('stt_end_time')}\n"
        f"4. LLM 1st Token:      {ms_diff('llm_first_token_time')}\n"
        f"5. LLM Completed:      {ms_diff('llm_end_time')}\n"
        f"6. TTS 1st Audio Sent: {ms_diff('tts_first_chunk_time')}\n"
        f"=========================================\n"
    )
    logger.info(timeline)

    await _send(ws, "turn_metrics", metrics={
        "speech_start": 0,
        "speech_end": ms_diff('speech_end_time', raw=True),
        "stt_end": ms_diff('stt_end_time', raw=True),
        "llm_first_token": ms_diff('llm_first_token_time', raw=True),
        "llm_end": ms_diff('llm_end_time', raw=True),
        "tts_first_chunk": ms_diff('tts_first_chunk_time', raw=True)
    })

    await _send(ws, "assistant_finished", generation_id=generation_id)

    # ── 5. Reset for next utterance ──
    session.reset_speech_state()


# ── WebSocket endpoint ───────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())[:8]
    session = session_manager.create(session_id)
    vad = EnergyVAD(sample_rate=SAMPLE_RATE)

    stt: STTEngine = app.state.stt
    llm: AIAgent = app.state.llm
    memory_manager: MemoryManager = app.state.memory_manager
    piper: PiperManager = app.state.piper

    # ── Background System Monitoring ──
    async def _system_monitor_loop():
        import psutil
        psutil.cpu_percent(interval=None) # Init
        try:
            while True:
                await asyncio.sleep(1)
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                await _send(
                    websocket, 
                    "system_metrics", 
                    cpu_percent=round(cpu, 1), 
                    mem_percent=round(mem.percent, 1), 
                    mem_used_gb=round(mem.used / (1024**3), 2)
                )
        except asyncio.CancelledError:
            pass

    monitor_task = asyncio.create_task(_system_monitor_loop())

    logger.info(f"Client connected — session {session_id}")
    await _send(websocket, "session_started", session_id=session_id)

    bytes_since_last_partial = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            # ── VAD ──
            is_speech = vad.process(data)
            now = time.monotonic()

            if is_speech:
                # ── Speech detected ──
                if not session.is_speaking:
                    session.is_speaking = True
                    session.speech_start = now
                    session.silence_start = None
                    session._llm_triggered = False
                    session.turn_metrics = {'speech_start_time': time.time()}

                    # Cancel any in-progress LLM generation (barge-in)
                    session.cancel_active_llm()

                    await _send(websocket, "user_started_speaking")
                    logger.info(f"[{session_id}] speech start")

                session.live_audio_buffer.extend(data)
                bytes_since_last_partial += len(data)

                # ── Periodic partial STT ──
                if bytes_since_last_partial >= PARTIAL_CHUNK_BYTES:
                    bytes_since_last_partial = 0
                    asyncio.create_task(
                        _do_partial_transcription(stt, session, websocket)
                    )

            else:
                # ── Silence ──
                if session.is_speaking:
                    session.live_audio_buffer.extend(data)

                    if session.silence_start is None:
                        session.silence_start = now

                    silence_duration = now - session.silence_start

                    if silence_duration >= SILENCE_TIMEOUT and not session._llm_triggered:
                        session._llm_triggered = True
                        session.is_speaking = False
                        session.turn_metrics['speech_end_time'] = time.time()
                        await _send(websocket, "user_stopped_speaking")
                        logger.info(
                            f"[{session_id}] silence {silence_duration:.1f}s → finalizing"
                        )

                        session.cancel_active_llm()

                        finalize_task = asyncio.create_task(
                            _do_finalize_and_stream(stt, llm, memory_manager, piper, session, websocket)
                        )
                        session._active_llm_task = finalize_task

    except WebSocketDisconnect:
        logger.info(f"Client disconnected — session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error [{session_id}]: {e}", exc_info=True)
        await _send(websocket, "error", message=str(e))
    finally:
        monitor_task.cancel()
        session.cancel_active_llm()
        session_manager.remove(session_id)


# ── Entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
