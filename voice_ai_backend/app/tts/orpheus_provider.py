"""
orpheus_provider.py — Orpheus-TTS integration.

Orpheus-TTS is a 3B Llama-based Text-to-Speech engine yielding 24kHz mono 16-bit PCM.
Since vllm/orpheus-speech requires an NVIDIA GPU and is not natively supported on macOS,
this provider includes a robust importing system with a graceful fallback to Kokoro/Piper
to keep the server stable on development environments.
"""

import os
import asyncio
import logging
from typing import AsyncGenerator, Tuple
from .base import BaseTTSProvider

logger = logging.getLogger(__name__)

class OrpheusProvider(BaseTTSProvider):
    def __init__(self):
        self.model = None
        self.voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
        self.is_loaded = False

        # Attempt to import and initialize Orpheus-TTS
        try:
            # We try both names in case of PyPI naming variances
            try:
                import sys
                sys.path.insert(0, 'orpheus_tts_pypi')
                from orpheus_tts import OrpheusModel
            except ImportError:
                from orpheus_speech import OrpheusModel

            logger.info("Initializing Orpheus-TTS model (canopylabs/orpheus-tts-0.1-finetune-prod)...")
            # Max model length set to 2048 as recommended in the canopyai docs
            self.model = OrpheusModel(
                model_name="canopylabs/orpheus-tts-0.1-finetune-prod", 
                max_model_len=2048
            )
            self.is_loaded = True
            logger.info("Orpheus-TTS model loaded successfully.")
        except ImportError:
            logger.warning("Orpheus-TTS package (orpheus-speech/vllm) not installed. Orpheus provider will operate in fallback mode.")
        except Exception as e:
            logger.error(f"Failed to initialize Orpheus-TTS engine: {e}. Operating in fallback mode.")

    def get_available_voices(self) -> list[str]:
        return self.voices

    def default_voice(self) -> str:
        return "tara"

    async def synthesize_stream(self, text: str, voice: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """
        Stream speech synthesis from Orpheus-TTS at 24000 Hz.
        If the primary engine is unavailable, it gracefully routes to Kokoro/Piper internally
        to maintain conversation integrity without crashing.
        """
        if self.is_loaded and self.model:
            try:
                # Prompt formatting as specified in the docs: {voice}: {text}
                prompt = f"{voice}: {text}"
                
                # generate_speech is run inside a thread pool as it performs intensive inference
                def run_inference():
                    return self.model.generate_speech(prompt=prompt, voice=voice)

                syn_tokens = await asyncio.to_thread(run_inference)
                
                # Stream the resulting audio chunks (expected to be raw mono 16-bit PCM at 24kHz)
                for chunk in syn_tokens:
                    if chunk:
                        yield (chunk, 24000)
            except Exception as e:
                logger.error(f"Orpheus synthesis failed: {e}. Falling back to default synthesizers.")
                async for chunk in self._fallback_synthesis(text):
                    yield chunk
        else:
            # Fallback mode: use another loaded system engine (like Kokoro) to synthesize the audio
            logger.info(f"Orpheus is in fallback mode. Dynamically routing '{text}' to alternative engine.")
            async for chunk in self._fallback_synthesis(text):
                yield chunk

    async def _fallback_synthesis(self, text: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """Graceful fallback runner using Kokoro or Piper."""
        try:
            from app.tts_manager import TTSManager
            manager = TTSManager()
            # Try Kokoro first, then Piper
            active_engine = "kokoro" if "kokoro" in manager.providers and manager.providers["kokoro"].kokoro else "piper"
            provider = manager.providers.get(active_engine)
            if provider:
                fallback_voice = provider.default_voice()
                async for chunk in provider.synthesize_stream(text, fallback_voice):
                    yield chunk
            else:
                # Final absolute fallback: yield a tiny bit of silence
                yield (b"\x00" * 4800, 24000)
        except Exception as fe:
            logger.error(f"Orpheus fallback synthesis failed: {fe}")
            yield (b"\x00" * 4800, 24000)
