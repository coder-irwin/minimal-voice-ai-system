"""
memory_manager.py — Orchestrates multi-layer conversational memory.

CRITICAL LATENCY DESIGN DECISION:
  All background processing uses ZERO LLM calls.
  Ollama on CPU is single-threaded — any background LLM call
  blocks the NEXT user turn's TTFT by 3-12 seconds.
  State detection and task extraction use Python heuristics instead.
"""
import logging
import json
import asyncio
import uuid
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.session import Session
    from app.llm import AIAgent

logger = logging.getLogger(__name__)


def normalize_date(date_str: str) -> str:
    """Normalize extracted relative date string to YYYY-MM-DD format."""
    date_str_clean = date_str.lower().strip()
    base_date = datetime.now()
    
    if "today" in date_str_clean:
        return base_date.strftime("%Y-%m-%d")
    elif "tomorrow" in date_str_clean:
        return (base_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    for day, day_idx in weekdays.items():
        if day in date_str_clean:
            days_ahead = day_idx - base_date.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (base_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str_clean)
    if match:
        return match.group(0)
        
    return date_str


def validate_and_convert_booking(date_str: str, time_str: str, tz_str: str) -> dict:
    """
    Performs strict backend timezone conversion and validation.
    Converts local timezone time to Eastern Time (ET) and checks if it falls within the 9 AM - 6 PM ET boundary.
    """
    tz = tz_str.upper().strip()
    if tz in ("PST", "PDT"):
        tz = "PT"
    elif tz in ("MST", "MDT"):
        tz = "MT"
    elif tz in ("CST", "CDT"):
        tz = "CT"
    elif tz in ("EST", "EDT"):
        tz = "ET"
        
    if not tz:
        tz = "ET"
        
    time_clean = time_str.lower().strip()
    
    match = re.search(r"(\d+)(?::(\d+))?\s*(am|pm)?", time_clean)
    if not match:
        return {"time_et": "N/A", "is_valid_booking_hours": False, "date_iso": date_str, "timezone_normalized": tz}
        
    hours = int(match.group(1))
    minutes = int(match.group(2)) if match.group(2) else 0
    ampm = match.group(3)
    
    if ampm:
        if ampm == "pm" and hours < 12:
            hours += 12
        elif ampm == "am" and hours == 12:
            hours = 0
    else:
        if hours > 0 and hours < 8:
            hours += 12
            
    offsets = {
        "PT": 180, "MT": 120, "CT": 60, "ET": 0,
        "GMT": -300, "UTC": -300, "CET": -360, "IST": -570, "AEST": -840,
    }
    
    offset_min = offsets.get(tz, 0)
    local_minutes = hours * 60 + minutes
    et_minutes = local_minutes + offset_min
    et_minutes_normalized = et_minutes % 1440
    
    et_hours = et_minutes_normalized // 60
    et_mins = et_minutes_normalized % 60
    
    et_ampm = "PM" if et_hours >= 12 else "AM"
    display_hours = et_hours % 12
    if display_hours == 0:
        display_hours = 12
    time_et = f"{display_hours}:{et_mins:02d} {et_ampm} ET"
    
    is_valid = (540 <= et_minutes_normalized <= 1080)
    date_iso = normalize_date(date_str)
    
    return {
        "time_et": time_et,
        "is_valid_booking_hours": is_valid,
        "date_iso": date_iso,
        "timezone_normalized": tz
    }


class PromptSwitcher:
    """Dynamic speech style configuration registry and prompt switcher."""
    
    SPEECH_STYLES = {
        "orpheus": (
            "\nSPEECH OUTPUT RULE (ORPHEUS MODE):\n"
            "- Write for SPEECH, not writing. Pause naturally, restart thoughts, varying emotional pacing.\n"
            "- Use natural hesitations and conversational rhythm (e.g. 'So yeah... honestly...', 'Um...', 'I mean...').\n"
            "- Allowed tags: <laugh>, <chuckle>, <sigh>, <gasp>. Use them sparingly when emotionally useful.\n"
            "- Focus on absolute emotional realism and natural conversational cadence.\n"
        ),
        "chattts": (
            "\nSPEECH OUTPUT RULE (CHATTTS MODE):\n"
            "- Write for highly conversational, relaxed dialogue. Pause naturally.\n"
            "- Supported paralinguistic tags: [laughter], [sigh], [gasp], [uv_break]. Use them sparingly mid-sentence.\n"
            "- *Example*: 'Honestly, I didn't think this would work [laughter] but it is amazing!'\n"
            "- Keep sentences short and easy to speak aloud.\n"
        ),
        "bark": (
            "\nSPEECH OUTPUT RULE (BARK MODE):\n"
            "- Support rich, expressive paralinguistic cues and creative pacing.\n"
            "- Supported paralinguistic tags: [laughter], [chuckles], [sighs], [gasp], [whispers], [crying].\n"
            "- Use them only when emotionally useful. Keep responses very concise.\n"
        ),
        "fish_speech": (
            "\nSPEECH OUTPUT RULE (FISH SPEECH MODE):\n"
            "- Write for highly expressive, zero-shot voice cloned speech with extreme realism.\n"
            "- Supports natural language paralinguistic tone guidelines and emotive descriptions.\n"
            "- Supported paralinguistic tags: [laugh], [sigh], [whisper in a small voice], [gasp]. Use sparingly mid-sentence.\n"
        ),
        "f5tts": (
            "\nSPEECH OUTPUT RULE (F5-TTS MODE):\n"
            "- Optimised for Flow Matching zero-shot voice cloning.\n"
            "- Mimics the precise emotional state, natural breathing pauses, and whisper of the cloned speaker.\n"
            "- Keep sentences natural, concise, and highly conversational.\n"
        ),
        "kokoro": (
            "\nSPEECH OUTPUT RULE (KOKORO MODE):\n"
            "- Write for stable, clear speech. Use cleaner sentence structure.\n"
            "- Reduce punctuation complexity. Do NOT use stacked punctuation, long ellipses, or chaotic phrasing.\n"
            "- Optimize for low-latency, high-stability synthesis and fast vocoder speed.\n"
        )
    }

    @classmethod
    def get_speech_prompt_rules(cls, engine: str) -> str:
        """Returns the specific paralinguistic and speech rules for the requested TTS engine."""
        return cls.SPEECH_STYLES.get(
            engine.lower().strip(),
            (
                "\nSPEECH OUTPUT RULE:\n"
                "- Keep sentences short, clean, and highly conversational. Easy and natural to say aloud.\n"
            )
        )


class MemoryManager:
    """Manages conversational memory with zero-LLM background processing."""

    def __init__(self, llm: 'AIAgent'):
        self.llm = llm

    def build_messages(self, session: 'Session') -> list[dict]:
        """Construct the optimized prompt using all memory layers."""
        from app.session import PERSONAS, BUSINESS_LOGIC, SCHEDULING_RULES

        messages = []
        
        # 1. Base System Prompt (always ultra-compact)
        persona_name = getattr(session, "persona", "angelina").lower()
        base_prompt = PERSONAS.get(persona_name, PERSONAS.get("angelina"))
        system_content = base_prompt + "\n\n"
        
        # Inject dynamic prompt blocks based on conversational state
        state = getattr(session, "conversational_state", "greeting")
        
        if state == "greeting":
            # MINIMAL — no extra injection. Fastest possible TTFT.
            pass
        elif state == "qualification":
            system_content += BUSINESS_LOGIC + "\n"
        elif state == "scheduling":
            system_content += SCHEDULING_RULES + "\n"
        elif state == "support":
            system_content += "Be empathetic and solution-oriented. Offer to connect with a support expert.\n"
        # casual conversation: no injection
        
        # 2. Active Task Memory (very compact)
        if getattr(session, "active_tasks", None):
            system_content += "\nActive Meetings:\n"
            for task in session.active_tasks:
                system_content += f"- {task.get('topic','')}: {task.get('time_local','')} {task.get('timezone','')} on {task.get('date_iso','')}\n"

        # 2.5 SPEECH OPTIMIZATION ENGINE (Dynamic Prompt Switcher based on active TTS engine)
        active_tts = getattr(session, "tts_engine", "piper").lower()
        system_content += PromptSwitcher.get_speech_prompt_rules(active_tts)

        # Dynamic LLM model size adaptation rules
        llm_model = getattr(session, "llm_model", "qwen2.5:1.5b").lower()
        if "0.5b" in llm_model or "tiny" in llm_model:
            system_content += "- Keep instructions lightweight, minimize emotional complexity and tags, keep phrasing direct and simple.\n"
        elif "8b" in llm_model or "7b" in llm_model or "9b" in llm_model:
            system_content += "- Allow subtle emotional pacing, rich nuance, and conversational variation.\n"

        messages.append({"role": "system", "content": system_content.strip()})

        # 3. Recent Active Turns — ONLY last 4 messages (2 turns) for minimal context
        for turn in session.recent_turns[-4:]:
            role = "assistant" if turn["role"] in ("ai", "assistant") else "user"
            messages.append({"role": role, "content": turn["text"]})

        return messages

    async def process_turn_background(self, session: 'Session', user_text: str, assistant_text: str):
        """
        Background task to update memory layers after a conversational turn.
        
        CRITICAL: Uses ZERO LLM calls. All processing is deterministic Python.
        This ensures Ollama's inference slot is always free for the next user turn.
        """
        # 1. Detect conversational state with keyword heuristics
        self._detect_state_heuristic(session, user_text, assistant_text)
        
        # 2. Extract scheduling tasks with regex
        self._extract_tasks_heuristic(session, user_text, assistant_text)
        
        # 3. Persist updates
        from app.persistence import session_storage
        asyncio.create_task(
            session_storage.update_memory(session.session_id, session.summarized_memory, session.user_facts, getattr(session, 'active_tasks', None))
        )

    def _detect_state_heuristic(self, session: 'Session', user_text: str, assistant_text: str):
        """Deterministic conversational state detection. Zero LLM cost."""
        text = (user_text + " " + assistant_text).lower()
        
        if any(w in text for w in ["schedule", "meeting", "book", "calendar", "time slot", "appointment", "call at"]):
            session.conversational_state = "scheduling"
        elif any(w in text for w in ["pricing", "cost", "service", "virtual assistant", "glocal", "offer", "plan", "package", "rate"]):
            session.conversational_state = "qualification"
        elif any(w in text for w in ["support", "help me", "issue", "problem", "broken", "not working", "complaint"]):
            session.conversational_state = "support"
        elif any(w in text for w in ["hello", "hi ", "hey", "good morning", "good afternoon", "how are you"]):
            session.conversational_state = "greeting"
        else:
            session.conversational_state = "casual conversation"
        
        logger.info(f"[{session.session_id}] State: {session.conversational_state}")

    def _extract_tasks_heuristic(self, session: 'Session', user_text: str, assistant_text: str):
        """Extract scheduling tasks using regex patterns. Zero LLM cost."""
        text = (user_text + " " + assistant_text).lower()
        
        if not any(w in text for w in ["schedule", "book", "meeting", "call at", "appointment", "lock in"]):
            return
            
        time_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))', text)
        if not time_match:
            return
            
        tz_match = re.search(r'(pacific|eastern|central|mountain|pt|et|ct|mt|pst|est|cst|mst|pdt|edt|cdt|mdt|gmt|utc|ist|aest|cet)\s*(?:time)?', text)
        tz = "ET"
        if tz_match:
            tz_map = {
                "pacific": "PT", "pt": "PT", "pst": "PT", "pdt": "PT",
                "mountain": "MT", "mt": "MT", "mst": "MT", "mdt": "MT",
                "central": "CT", "ct": "CT", "cst": "CT", "cdt": "CT",
                "eastern": "ET", "et": "ET", "est": "ET", "edt": "ET",
                "gmt": "GMT", "utc": "UTC", "ist": "IST", "aest": "AEST", "cet": "CET"
            }
            tz = tz_map.get(tz_match.group(1).lower(), "ET")
        
        date_match = re.search(r'(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})', text)
        date_str = date_match.group(1) if date_match else "tomorrow"
        
        time_str = time_match.group(1).strip()
        
        validation = validate_and_convert_booking(date_str, time_str, tz)
        if validation["is_valid_booking_hours"]:
            for existing in session.active_tasks:
                if (existing.get("date_iso") == validation["date_iso"] and 
                    existing.get("time_et") == validation["time_et"]):
                    return
            
            canonical_task = {
                "id": str(uuid.uuid4()),
                "type": "meeting",
                "topic": "Scheduled Call",
                "date_iso": validation["date_iso"],
                "time_local": time_str,
                "timezone": validation["timezone_normalized"],
                "time_et": validation["time_et"],
                "is_valid_booking_hours": True,
                "status": "scheduled",
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
            session.active_tasks.append(canonical_task)
            logger.info(f"[{session.session_id}] Task extracted (heuristic): {canonical_task}")
        else:
            logger.info(f"[{session.session_id}] Booking rejected: {time_str} {tz} -> {validation['time_et']} (outside 9 AM - 6 PM ET)")
