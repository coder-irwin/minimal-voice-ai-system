"""
vad.py — Lightweight energy-based Voice Activity Detector.

No external ML libraries required. Operates purely on raw PCM int16 frames.

Algorithm:
  1. Compute RMS energy of the incoming chunk.
  2. Maintain a short exponential-moving-average of background noise.
  3. Speech is detected when RMS > (noise_floor * speech_ratio).
  4. State transitions:
       SILENCE → SPEAKING  : if energy above threshold for ≥ onset_frames
       SPEAKING → SILENCE  : if energy below threshold for silence_grace_frames
         (grace period avoids fragmenting natural speech pauses)

Usage:
    vad = EnergyVAD()
    is_active = vad.process(pcm_bytes)   # returns True if speech
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnergyVAD:
    """
    Pure-Python energy-based VAD. No external ML deps.

    Parameters
    ----------
    sample_rate      : int   — audio sample rate (must match incoming PCM)
    frame_ms         : int   — ms per chunk fed to process()
    speech_ratio     : float — RMS must exceed noise_floor * speech_ratio to count as speech
    noise_alpha      : float — EMA coefficient for noise floor adaptation (lower = slower)
    onset_frames     : int   — consecutive speech frames needed to flip SPEAKING
    grace_frames     : int   — consecutive silent frames allowed before flipping SILENCE
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        speech_ratio: float = 2.8,
        noise_alpha: float = 0.05,
        onset_frames: int = 2,
        grace_frames: int = 8,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.speech_ratio = speech_ratio
        self.noise_alpha = noise_alpha
        self.onset_frames = onset_frames
        self.grace_frames = grace_frames

        # State
        self._noise_floor: float = 200.0   # initial guess, adapts quickly
        self._speaking: bool = False
        self._onset_count: int = 0          # consecutive speech frames counter
        self._grace_count: int = 0          # consecutive silence frames counter

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def _rms(self, pcm_bytes: bytes) -> float:
        """Compute RMS amplitude of int16 PCM bytes."""
        if len(pcm_bytes) < 2:
            return 0.0
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(audio ** 2)) + 1e-8)

    def process(self, pcm_bytes: bytes) -> bool:
        """
        Feed one chunk of raw PCM int16 bytes.
        Returns True if the VAD currently considers this speech.
        """
        rms = self._rms(pcm_bytes)
        threshold = self._noise_floor * self.speech_ratio
        is_active = rms > threshold

        if is_active:
            self._grace_count = 0
            self._onset_count += 1
            if not self._speaking and self._onset_count >= self.onset_frames:
                self._speaking = True
                logger.debug(f"VAD: speech onset (rms={rms:.1f}, floor={self._noise_floor:.1f})")
        else:
            self._onset_count = 0
            if self._speaking:
                self._grace_count += 1
                if self._grace_count >= self.grace_frames:
                    self._speaking = False
                    self._grace_count = 0
                    logger.debug(f"VAD: silence (rms={rms:.1f}, floor={self._noise_floor:.1f})")
            else:
                # Adapt noise floor only during clear silence
                self._noise_floor = (
                    self.noise_alpha * rms + (1 - self.noise_alpha) * self._noise_floor
                )

        return self._speaking

    def reset(self):
        """Reset VAD state (call on session disconnect)."""
        self._speaking = False
        self._onset_count = 0
        self._grace_count = 0
        self._noise_floor = 200.0
