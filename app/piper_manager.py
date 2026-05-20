"""
piper_manager.py — Local high-quality TTS using Piper.
"""
import os
import asyncio
import logging
from typing import AsyncGenerator, Tuple
from piper.voice import PiperVoice

logger = logging.getLogger(__name__)

class PiperManager:
    """Singleton-friendly Piper TTS engine."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, model_path: str = "models/piper/en_US-libritts_r-medium.onnx"):
        if self._initialized:
            return
            
        logger.info(f"Loading Piper TTS model: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Piper model not found at {model_path}. Did you download it?")
            raise FileNotFoundError(f"Piper model not found: {model_path}")
            
        # load() expects the .onnx path, and it will look for .onnx.json automatically
        self.voice = PiperVoice.load(model_path)
        self._initialized = True
        logger.info("Piper TTS model loaded successfully")
        
    async def synthesize_stream(self, text: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """
        Yields tuples of (raw PCM audio bytes, sample_rate).
        Offloads the blocking synthesis to a background thread.
        """
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=100)
        
        def _sync_worker():
            try:
                # synthesize yields AudioChunk objects
                for chunk in self.voice.synthesize(text):
                    # We must safely push to the async queue from this background thread
                    loop.call_soon_threadsafe(queue.put_nowait, (chunk.audio_int16_bytes, chunk.sample_rate))
            except Exception as e:
                logger.error(f"Piper synthesis error: {e}")
            finally:
                # Send EOF marker
                loop.call_soon_threadsafe(queue.put_nowait, None)
                
        # Start the blocking generator in a thread pool
        loop.run_in_executor(None, _sync_worker)
        
        # Async generator loop to yield chunks to the caller
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
