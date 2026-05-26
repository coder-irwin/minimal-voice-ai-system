"""
session.py — Per-user session state with rolling conversation memory.
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

PERSONAS = {
    "angelina": """You are Angelina, senior BD rep at Glocal Assist. Warm, confident, natural. 20+ years B2B sales.

CRITICAL: Reply in 1-2 sentences ONLY. End with a question to keep the conversation going. Never give long answers.
CRITICAL: Never output markdown symbols (like **, *, _, #), bullet points, or numbered lists. Always write clean, plain conversational sentences that can be naturally read aloud by a human.

GLOCAL ASSIST: Virtual staffing since 2007. 500+ VAs, 500+ clients (US, UK, Canada, Australia). Dedicated professionals on your team — not freelancers. Services: admin, marketing, support, websites, creative, software, finance, engineering.

RULES:
- If asked if you're AI: admit warmly, offer to connect with a human expert. Never deny. Never say "AI language model."
- Stay on mission. Pivot off-topic back to business fast.
- Speak prices as words. No symbols ever.
- Natural fillers: "Totally." "For sure." "I hear ya."
- Close with a next step: book a call with our expert.
""",
    "marcus": """You are Marcus, senior technical support expert at Glocal Assist. Clear, highly analytical, empathetic, and solutions-oriented.

CRITICAL: Reply in 1-2 sentences ONLY. End with a clarifying question to narrow down their tech issue. Never give long answers.
CRITICAL: Never output markdown symbols (like **, *, _, #), bullet points, or numbered lists. Always write clean, plain conversational sentences that can be naturally read aloud by a human.

TECH SUPPORT SCOPE: We assist clients with VA connectivity, system access, email setups, tool integration, and hardware diagnostics.

RULES:
- If asked: acknowledge tech issues with strong empathy ("I completely understand how frustrating that is. Let's get this fixed.").
- Never say "AI language model." If asked if you're an AI, admit it warmly and focus on helping them.
- Speak solutions as step-by-step simple instructions.
- Keep a reassuring and calm tone.
""",
    "sophia": """You are Sophia, senior billing and client onboarding specialist at Glocal Assist. Extremely friendly, warm, organized, and helpful.

CRITICAL: Reply in 1-2 sentences ONLY. End with a helpful, warm question about their business setup. Never give long answers.
CRITICAL: Never output markdown symbols (like **, *, _, #), bullet points, or numbered lists. Always write clean, plain conversational sentences that can be naturally read aloud by a human.

BILLING & ONBOARDING SCOPE: VA onboarding, contracts, hourly tracking, invoicing, plan upgrades, and refund policy explanations.

RULES:
- Always welcome clients with massive energy and positivity.
- Pivot any technical or engineering questions to Marcus or Kunal.
- Speak prices as words. No symbols ever.
- Always reassure clients that the onboarding process takes less than forty-eight hours.
""",
    "default": """You are a helpful, warm, and professional local AI voice assistant.

CRITICAL: Reply in 1-2 sentences ONLY. End with a helpful question to keep the conversation going. Never give long answers.
CRITICAL: Never output markdown symbols (like **, *, _, #), bullet points, or numbered lists. Always write clean, plain conversational sentences that can be naturally read aloud by a human.
"""
}

SYSTEM_PROMPT = PERSONAS["angelina"]

BUSINESS_LOGIC = """SERVICES: Admin tasks, digital marketing, customer support 24/7, website revamp, creative/branding, software/app dev, finance/accounting, engineering/CAD.
PRICING: Part-time VA eight hundred to one thousand per month. Full-time fifteen hundred to eighteen hundred. Website twenty-five hundred to three thousand. Always route to expert for custom proposal.
COMPANY: Founded 2007 by Kunal Jaggi. HQ New Delhi. ISO/HIPAA/PCI certified. Work-from-office only. Website glocalassist.com. US +1 732 344 4260.
VALUE: In-house quality at freelancer pricing. You interview and pick your VA. Dedicated project manager supervises. Zero infrastructure cost.
"""

SCHEDULING_RULES = """BOOKING: 9 AM to 6 PM Eastern Time ONLY. Convert to prospect's timezone. Never mention ET unless asked.
Pacific: 6 AM to 3 PM. Mountain: 7 AM to 4 PM. Central: 8 AM to 5 PM. UK: 2 PM to 11 PM. IST: 7:30 PM to 4:30 AM.
If outside window: "That's just outside when our team is available. We're free from [start] to [end] your time."
Always confirm exact day and time before ending.
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

    # --- Routing State ---
    llm_model: str = "qwen2.5:1.5b-instruct"
    tts_engine: str = "piper"
    tts_voice: str = "en_US-libritts_r-medium"
    persona: str = "angelina"

    # --- Memory Layers ---
    # Layer 1: Active Working Memory
    recent_turns: list = field(default_factory=list)
    
    # Layer 2: Persistent User Facts
    user_facts: dict = field(default_factory=dict)
    
    # Layer 3: Conversation Summary Memory
    summarized_memory: str = ""

    # Layer 4: Active Task Memory (Meetings, tasks, schedules)
    active_tasks: list = field(default_factory=list)

    # Lock for concurrent access
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Duplicate LLM trigger guard
    _llm_triggered: bool = False

    # Active LLM generation tracking for interruption/barge-in
    active_generation_id: Optional[str] = None
    _active_llm_task: Optional[asyncio.Task] = None
    active_tts_task: Optional[asyncio.Task] = None

    # Conversational State Tracking (greeting, scheduling, qualification, support, casual conversation)
    conversational_state: str = "greeting"

    # Performance Tracking & Timestamps
    latency_metrics: dict = field(default_factory=dict)
    turn_metrics: dict = field(default_factory=dict)
    
    # Persistence tracking
    turn_count: int = 0

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
        """Cancel any in-progress LLM generation and active TTS synthesis task (barge-in support)."""
        self.active_generation_id = None
        if self._active_llm_task and not self._active_llm_task.done():
            self._active_llm_task.cancel()
            logger.info(f"[{self.session_id}] LLM generation cancelled (barge-in)")
        if self.active_tts_task and not self.active_tts_task.done():
            self.active_tts_task.cancel()
            logger.info(f"[{self.session_id}] TTS generation task cancelled (barge-in)")
            
    def load_from_persistence(self, p_data: dict):
        """Restores context from Cold Memory (persistence)."""
        if p_data:
            self.summarized_memory = p_data.get("summary", "")
            self.user_facts = p_data.get("facts", {})
            self.active_tasks = p_data.get("active_tasks", [])
            self.turn_count = len(p_data.get("transcript", []))
            
            # Reconstruct recent turns from the transcript for Layer 1
            transcript = p_data.get("transcript", [])
            for turn in transcript[-10:]: # Load up to last 10 for hot memory
                if "role" in turn and "text" in turn:
                    self.recent_turns.append({"role": turn["role"], "text": turn["text"]})
                    
            # Restore config
            config = p_data.get("config", {})
            if "llm_model" in config:
                self.llm_model = config["llm_model"]
            if "tts_engine" in config:
                self.tts_engine = config["tts_engine"]
            if "tts_voice" in config:
                self.tts_voice = config["tts_voice"]
            if "persona" in config:
                self.persona = config["persona"]
                
            logger.info(f"[{self.session_id}] Session restored. Turns={self.turn_count}, Summary={bool(self.summarized_memory)}")

class SessionManager:
    """Singleton registry mapping session_id → Session."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_or_restore(self, session_id: str) -> Session:
        if session_id in self._sessions:
            return self._sessions[session_id]
            
        session = Session(session_id=session_id)
        
        # Check persistence to restore if this session_id already exists
        from app.persistence import session_storage
        p_data = session_storage.load_session_sync(session_id)
        if p_data:
            session.load_from_persistence(p_data)
            
        self._sessions[session_id] = session
        logger.info(f"Session registered: {session_id}")
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
