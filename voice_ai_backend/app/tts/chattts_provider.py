"""
chattts_provider.py — ChatTTS integration.

ChatTTS is a generative speech model optimized for conversational assistants.
It natively supports conversational prompts, pauses, and paralinguistic tags
like [laughter] and [sigh]. Includes robust imports and fallback routing.
"""

import os
import asyncio
import logging
from typing import AsyncGenerator, Tuple
from .base import BaseTTSProvider

logger = logging.getLogger(__name__)

class ChatTTSProvider(BaseTTSProvider):
    def __init__(self):
        self.chat = None
        self.voices = ["voice_1", "voice_2", "voice_3", "voice_4"]
        self.is_loaded = False

        try:
            import ChatTTS
            logger.info("Initializing ChatTTS engine...")
            self.chat = ChatTTS.Chat()
            self.chat.load_models()
            self.is_loaded = True
            logger.info("ChatTTS engine loaded successfully.")
        except ImportError:
            logger.warning("ChatTTS package not installed. ChatTTS provider will operate in fallback mode.")
        except Exception as e:
            logger.error(f"Failed to load ChatTTS engine: {e}. Operating in fallback mode.")

    def get_available_voices(self) -> list[str]:
        return self.voices

    def default_voice(self) -> str:
        return "voice_1"

    async def synthesize_stream(self, text: str, voice: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """
        Stream speech synthesis from ChatTTS.
        Falls back to Kokoro/Piper gracefully if ChatTTS is not installed.
        """
        if self.is_loaded and self.chat:
            try:
                # ChatTTS performs intensive calculations, run in executor
                def run_inference():
                    # returns numpy arrays representing audio waveforms
                    return self.chat.infer([text], use_decoder=True)

                wavs = await asyncio.to_thread(run_inference)
                if wavs and len(wavs) > 0:
                    import numpy as np
                    audio_float = wavs[0]
                    # Convert to 16-bit PCM bytes
                    audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)
                    # Yield as a single chunk at 24000 Hz or the default ChatTTS rate (typically 24000)
                    yield (audio_int16.tobytes(), 24000)
            except Exception as e:
                logger.error(f"ChatTTS synthesis error: {e}")
                async for chunk in self._fallback_synthesis(text):
                    yield chunk
        else:
            logger.info(f"ChatTTS is in fallback mode. Dynamically routing '{text}' to alternative engine.")
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
            logger.error(f"ChatTTS fallback failed: {fe}")
            yield (b"\x00" * 4800, 24000)
