"""
stt.py — Singleton Faster-Whisper STT engine with streaming-optimised transcription.

Key design decisions:
  - Model loaded ONCE at startup (singleton). Never re-instantiated per request.
  - Uses beam_size=1 for lowest latency.
  - vad_filter=True so Whisper's internal Silero VAD trims leading/trailing silence.
  - Runs transcription in a thread pool so the asyncio loop is never blocked.
  - Two public methods:
      transcribe_chunk()  — fast partial transcript for short audio windows
      transcribe_final()  — higher-quality transcription on the full utterance buffer
"""

import numpy as np
from faster_whisper import WhisperModel
import asyncio
import logging

logger = logging.getLogger(__name__)


class STTEngine:
    """Singleton-friendly STT engine backed by Faster-Whisper."""

    _instance = None  # class-level singleton reference

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_size: str = "base.en", compute_type: str = "default"):
        if self._initialized:
            return
        logger.info(f"Loading Faster-Whisper model: {model_size}")
        # 'auto' picks CUDA when available, else CPU
        self.model = WhisperModel(model_size, device="auto", compute_type=compute_type)
        self._initialized = True
        logger.info("STT model loaded successfully (singleton)")

    # ------------------------------------------------------------------
    # Streaming partial transcription — called frequently on small chunks
    # ------------------------------------------------------------------
    async def transcribe_chunk(self, audio_bytes: bytes) -> str:
        """
        Transcribe a short audio chunk (typically 0.5–2s of PCM 16kHz mono).
        Optimised for speed: beam_size=1, no language detection overhead.
        Returns the concatenated text of all detected segments (may be empty).
        """
        if len(audio_bytes) < 1600:  # less than 0.05s at 16kHz — skip
            return ""
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = await asyncio.to_thread(
                self.model.transcribe,
                audio_np,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=100,
                ),
                without_timestamps=True,
                language="en",
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text
        except Exception as e:
            logger.error(f"Error in chunk transcription: {e}")
            return ""

    # ------------------------------------------------------------------
    # Final high-quality transcription on the full utterance buffer
    # ------------------------------------------------------------------
    async def transcribe_final(self, audio_bytes: bytes) -> str:
        """
        Transcribe the complete utterance buffer after silence is detected.
        Slightly higher quality settings than chunk mode.
        """
        if len(audio_bytes) < 3200:  # less than 0.1s — skip
            return ""
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = await asyncio.to_thread(
                self.model.transcribe,
                audio_np,
                beam_size=3,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
                without_timestamps=True,
                language="en",
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text
        except Exception as e:
            logger.error(f"Error in final transcription: {e}")
            return ""
