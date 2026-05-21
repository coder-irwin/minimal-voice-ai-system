import logging
from typing import AsyncGenerator, Tuple

from app.tts.piper_provider import PiperProvider
from app.tts.kokoro_provider import KokoroProvider

logger = logging.getLogger(__name__)

class TTSManager:
    """Central registry and router for all TTS engines."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize providers
        self.providers = {
            "piper": PiperProvider(),
            "kokoro": KokoroProvider()
        }
        self._initialized = True
        logger.info("TTS Manager initialized.")

    def get_available_engines(self) -> list[str]:
        return list(self.providers.keys())

    def get_available_voices(self) -> dict[str, list[str]]:
        return {
            engine: provider.get_available_voices()
            for engine, provider in self.providers.items()
        }

    async def synthesize_stream(self, text: str, engine: str, voice: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """Route the TTS request to the active engine."""
        if engine not in self.providers:
            logger.error(f"TTS Engine {engine} not found! Falling back to Piper.")
            engine = "piper"
            
        provider = self.providers[engine]
        
        # Verify voice exists for this engine, fallback if not
        if voice not in provider.get_available_voices():
            default = provider.default_voice()
            logger.warning(f"Voice {voice} not found for {engine}. Falling back to {default}.")
            voice = default
            
        async for chunk in provider.synthesize_stream(text, voice):
            yield chunk
