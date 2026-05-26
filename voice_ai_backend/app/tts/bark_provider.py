"""
bark_provider.py — Suno Bark TTS integration.

Bark is a transformer-based audio generation model that natively supports
expressive emotional cues like [laughter], [sighs], and [whispers].
Includes robust imports and fallback routing.
"""

import os
import asyncio
import logging
from typing import AsyncGenerator, Tuple
from .base import BaseTTSProvider

logger = logging.getLogger(__name__)

class BarkProvider(BaseTTSProvider):
    def __init__(self):
        self.voices = ["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3", "v2/en_speaker_4"]
        self.is_loaded = False

        try:
            from bark import preload_models
            logger.info("Pre-loading Bark TTS model weights (this might take a few moments)...")
            preload_models()
            self.is_loaded = True
            logger.info("Bark TTS model weights preloaded successfully.")
        except ImportError:
            logger.warning("Bark package (suno-bark) not installed. Bark provider will operate in fallback mode.")
        except Exception as e:
            logger.error(f"Failed to load Bark TTS engine: {e}. Operating in fallback mode.")

    def get_available_voices(self) -> list[str]:
        return self.voices

    def default_voice(self) -> str:
        return "v2/en_speaker_6" if "v2/en_speaker_6" in self.voices else "v2/en_speaker_0"

    async def synthesize_stream(self, text: str, voice: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """
        Synthesize audio via Bark.
        Falls back to Kokoro/Piper gracefully if Bark is not installed.
        """
        if self.is_loaded:
            try:
                from bark import generate_audio
                
                # Bark generation is autoregressive, offload to thread
                def run_inference():
                    # generate_audio returns a numpy float array at 24000 Hz
                    return generate_audio(text, history_prompt=voice)

                audio_float = await asyncio.to_thread(run_inference)
                
                import numpy as np
                # Convert to 16-bit PCM bytes
                audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)
                yield (audio_int16.tobytes(), 24000)
            except Exception as e:
                logger.error(f"Bark synthesis failed: {e}")
                async for chunk in self._fallback_synthesis(text):
                    yield chunk
        else:
            logger.info(f"Bark is in fallback mode. Dynamically routing '{text}' to alternative engine.")
            async for chunk in self._fallback_synthesis(text):
                yield chunk

    async def _fallback_synthesis(self, text: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """Dynamic fallback routing to Kokoro/Piper."""
        try:
            from app.tts_manager import TTSManager
            manager = TTSManager()
            active_engine = "kokoro" if "kokoro" in manager.providers and manager.providers["kokoro"].kokoro else "piper"
            provider = manager.providers.get(active_engine)
            if provider:
                fallback_voice = provider.default_voice()
                async for chunk in provider.synthesize_stream(text, fallback_voice):
                    yield chunk
            else:
                yield (b"\x00" * 4800, 24000)
        except Exception as fe:
            logger.error(f"Bark fallback failed: {fe}")
            yield (b"\x00" * 4800, 24000)
