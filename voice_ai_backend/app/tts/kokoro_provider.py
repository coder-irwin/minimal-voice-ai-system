"""
kokoro_provider.py — Kokoro TTS via kokoro-onnx.

Key design note: Kokoro's create_stream() is an ASYNC generator.
It internally uses run_in_executor for CPU-bound ONNX inference,
then yields (np.float32 array, sample_rate) tuples.

We consume it directly from the async context — no thread wrapper needed.
"""

import os
import asyncio
import logging
import numpy as np
from typing import AsyncGenerator, Tuple
from .base import BaseTTSProvider

logger = logging.getLogger(__name__)


class KokoroProvider(BaseTTSProvider):
    def __init__(self, models_dir: str = "models/kokoro"):
        self.models_dir = models_dir
        self.kokoro = None
        self.available_voices = []

        onnx_path = os.path.join(models_dir, "kokoro-v0_19.onnx")
        voices_path = os.path.join(models_dir, "voices.bin")

        if os.path.exists(onnx_path) and os.path.exists(voices_path):
            try:
                from kokoro_onnx import Kokoro
                from app.hardware_manager import hardware_manager
                
                # Determine provider trial order based on hardware manager
                providers_to_try = []
                if hardware_manager.tts_backend == "onnx_cuda":
                    providers_to_try = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                elif hardware_manager.tts_backend == "onnx_coreml":
                    providers_to_try = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers_to_try = ["CPUExecutionProvider"]

                for provider in providers_to_try:
                    try:
                        logger.info(f"Attempting to load Kokoro ONNX model with provider: {provider}")
                        os.environ["ONNX_PROVIDER"] = provider
                        self.kokoro = Kokoro(onnx_path, voices_path)
                        self.available_voices = self.kokoro.get_voices()
                        logger.info(f"Loaded Kokoro ONNX model successfully with provider {provider}.")
                        break
                    except Exception as pe:
                        logger.warning(f"Failed to load Kokoro with provider {provider}: {pe}")
                        self.kokoro = None
                
                if not self.kokoro:
                    logger.error("All attempted ONNX providers for Kokoro failed.")
            except ImportError:
                logger.error("kokoro-onnx package not installed.")
            except Exception as e:
                logger.error(f"Failed to load Kokoro TTS: {e}")
        else:
            logger.warning(f"Kokoro model files not found at {models_dir}.")

    def get_available_voices(self) -> list[str]:
        return self.available_voices

    def default_voice(self) -> str:
        if "af_heart" in self.available_voices:
            return "af_heart"
        return self.available_voices[0] if self.available_voices else "kokoro-not-found"

    async def synthesize_stream(self, text: str, voice: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        if not self.kokoro:
            logger.error("Kokoro is not loaded.")
            return

        try:
            # create_stream is an async generator that yields (float32_array, sample_rate)
            async for audio_float, sample_rate in self.kokoro.create_stream(text, voice=voice, speed=1.0):
                # Convert float32 [-1.0, 1.0] → int16 PCM bytes
                audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)
                yield (audio_int16.tobytes(), sample_rate)
        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}", exc_info=True)
