"""
session.py — Per-user session state with rolling conversation memory.
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a premium realtime conversational assistant.

You are:
- professional
- warm
- conversational
- concise
- context-aware

Maintain natural conversational continuity.
Reference previous discussion context naturally.
Speak in short spoken-style responses.
Avoid robotic phrasing and repetitive resets.
Keep responses concise and natural for voice interaction.
"""


@dataclass
class Session:
    session_id: str

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # --- audio ---
    live_audio_buffer: bytearray = field(default_factory=bytearray)

    # --- VAD / speech state ---
    is_speaking: bool = False
    silence_start: Optional[float] = None
    speech_start: Optional[float] = None

    # --- transcript state ---
    speech_buffer_text: str = ""
    last_partial: str = ""

    # --- Memory Layers ---
    # Layer 1: Active Working Memory
    recent_turns: list = field(default_factory=list)
    
    # Layer 2: Persistent User Facts
    user_facts: dict = field(default_factory=dict)
    
    # Layer 3: Conversation Summary Memory
    summarized_memory: str = ""

    # Lock for concurrent access
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Duplicate LLM trigger guard
    _llm_triggered: bool = False

    # Active LLM generation tracking for interruption/barge-in
    active_generation_id: Optional[str] = None
    _active_llm_task: Optional[asyncio.Task] = None

    # Performance Tracking & Timestamps
    latency_metrics: dict = field(default_factory=dict)
    turn_metrics: dict = field(default_factory=dict)

    def reset_speech_state(self):
        """Called after a finalized utterance is dispatched to LLM or upon barge-in."""
        self.live_audio_buffer.clear()
        self.speech_buffer_text = ""
        self.last_partial = ""
        self.is_speaking = False
        self.silence_start = None
        self.speech_start = None
        self._llm_triggered = False
        self.active_generation_id = None
        self.updated_at = time.time()

    def cancel_active_llm(self):
        """Cancel any in-progress LLM generation (barge-in support)."""
        self.active_generation_id = None
        if self._active_llm_task and not self._active_llm_task.done():
            self._active_llm_task.cancel()
            logger.info(f"[{self.session_id}] LLM generation cancelled (barge-in)")

class SessionManager:
    """Singleton registry mapping session_id → Session."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str) -> Session:
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def remove(self, session_id: str):
        s = self._sessions.pop(session_id, None)
        if s:
            s.cancel_active_llm()
            logger.info(f"Session removed: {session_id}")

    def __len__(self):
        return len(self._sessions)

# Module-level singleton
session_manager = SessionManager()
