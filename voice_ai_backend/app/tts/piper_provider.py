"""
piper_provider.py — Piper TTS provider.

Piper's synthesize() is a synchronous blocking call that returns an iterable
of AudioChunk dataclasses. We run it in a thread via run_in_executor and
bridge chunks through an asyncio.Queue.

AudioChunk has:
  - .audio_int16_bytes  (property → bytes)
  - .sample_rate        (int, typically 22050)
"""

import os
import asyncio
import logging
from typing import AsyncGenerator, Tuple
from piper.voice import PiperVoice
from .base import BaseTTSProvider

logger = logging.getLogger(__name__)


class PiperProvider(BaseTTSProvider):
    def __init__(self, models_dir: str = "models/piper"):
        self.models_dir = models_dir
        self.voices = {}

        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".onnx"):
                    voice_name = file.replace(".onnx", "")
                    self.voices[voice_name] = os.path.join(models_dir, file)

        self._loaded_models = {}
        logger.info(f"Piper provider found {len(self.voices)} voice(s): {list(self.voices.keys())}")

    def get_available_voices(self) -> list[str]:
        return list(self.voices.keys())

    def default_voice(self) -> str:
        voices = self.get_available_voices()
        if "en_US-libritts_r-medium" in voices:
            return "en_US-libritts_r-medium"
        return voices[0] if voices else "piper-not-found"

    def _get_model(self, voice: str):
        if voice not in self._loaded_models:
            if voice not in self.voices:
                logger.error(f"Piper voice '{voice}' not found. Available: {list(self.voices.keys())}")
                return None
            
            from app.hardware_manager import hardware_manager
            use_cuda = hardware_manager.supports_cuda
            
            try:
                logger.info(f"Loading Piper voice '{voice}' (use_cuda={use_cuda})")
                self._loaded_models[voice] = PiperVoice.load(self.voices[voice], use_cuda=use_cuda)
                logger.info(f"Loaded Piper voice: {voice}")
            except Exception as e:
                logger.warning(f"Failed to load Piper with use_cuda={use_cuda} ({e}). Retrying with use_cuda=False.")
                try:
                    self._loaded_models[voice] = PiperVoice.load(self.voices[voice], use_cuda=False)
                    logger.info(f"Loaded Piper voice (CPU fallback): {voice}")
                except Exception as ex:
                    logger.error(f"Failsafe Piper voice loading failed completely for {voice}: {ex}")
                    return None
        return self._loaded_models[voice]

    async def synthesize_stream(self, text: str, voice: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        model = self._get_model(voice)
        if not model:
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        def _sync_worker():
            try:
                for chunk in model.synthesize(text):
                    # chunk is a piper.voice.AudioChunk dataclass
                    audio_bytes = chunk.audio_int16_bytes
                    sr = chunk.sample_rate
                    loop.call_soon_threadsafe(queue.put_nowait, (audio_bytes, sr))
            except Exception as e:
                logger.error(f"Piper synthesis error: {e}", exc_info=True)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _sync_worker)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
