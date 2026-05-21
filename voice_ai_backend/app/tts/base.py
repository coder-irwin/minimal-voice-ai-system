import abc
from typing import AsyncGenerator, Tuple

class BaseTTSProvider(abc.ABC):
    """Abstract interface for all TTS engines."""

    @abc.abstractmethod
    async def synthesize_stream(self, text: str, voice: str) -> AsyncGenerator[Tuple[bytes, int], None]:
        """
        Yields tuples of (raw PCM audio bytes, sample_rate).
        Should yield audio incrementally as it becomes available.
        """
        pass
        
    @abc.abstractmethod
    def get_available_voices(self) -> list[str]:
        """Return a list of available voice identifiers."""
        pass
        
    @abc.abstractmethod
    def default_voice(self) -> str:
        """Return the default voice identifier."""
        pass
