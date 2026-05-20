"""
memory_manager.py — Orchestrates multi-layer conversational memory.
"""
import logging
import json
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.session import Session
    from app.llm import AIAgent

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversational memory, background fact extraction, and summarization."""

    def __init__(self, llm: 'AIAgent'):
        self.llm = llm
        # Number of roundtrips (user + assistant turns) to keep in raw active memory before summarizing
        self.max_recent_roundtrips = 10 

    def build_messages(self, session: 'Session') -> list[dict]:
        """Construct the optimized prompt using all memory layers."""
        from app.session import SYSTEM_PROMPT

        messages = []
        
        # 1. Base System Prompt
        system_content = SYSTEM_PROMPT + "\n\n"
        
        # 2. Layer 2: Persistent User Facts
        if session.user_facts:
            system_content += "Known facts about the user:\n"
            for k, v in session.user_facts.items():
                system_content += f"- {k}: {v}\n"
            system_content += "\n"

        # 3. Layer 3: Summarized Memory (Optional, keep it lightweight)
        if session.summarized_memory:
            system_content += "Prior Conversation Context:\n"
            system_content += session.summarized_memory + "\n"

        messages.append({"role": "system", "content": system_content.strip()})

        # 4. Layer 1: Recent Active Turns (up to 100 messages)
        for turn in session.recent_turns[-100:]:
            # Map 'ai' or 'assistant' to 'assistant'
            role = "assistant" if turn["role"] in ("ai", "assistant") else "user"
            messages.append({"role": role, "content": turn["text"]})

        return messages

    async def process_turn_background(self, session: 'Session', user_text: str, assistant_text: str):
        """
        Background task to update memory layers after a conversational turn.
        Zero latency cost to the user's active interaction.
        """
        # Triggers
        summarize_task = None
        extract_task = None

        # Check if we need to summarize older turns (Layer 1 -> Layer 3)
        # We track how many we've summarized without deleting them from recent_turns
        cursor = getattr(session, '_summarization_cursor', 0)
        if len(session.recent_turns) - cursor > self.max_recent_roundtrips * 2:
            # Take 4 messages (2 roundtrips) to summarize
            turns_to_summarize = session.recent_turns[cursor:cursor+4]
            session._summarization_cursor = cursor + 4
            
            summarize_task = asyncio.create_task(
                self._summarize_older_turns(session, turns_to_summarize)
            )

        # Always try to extract new facts from the latest turn (Layer 2)
        extract_task = asyncio.create_task(
            self._extract_user_facts(session, user_text, assistant_text)
        )

        if summarize_task:
            await summarize_task
        if extract_task:
            await extract_task

    async def _summarize_older_turns(self, session: 'Session', turns: list[dict]):
        """Compress older turns into the summary memory using the LLM."""
        dialogue = ""
        for t in turns:
            role = "User" if t["role"] == "user" else "Assistant"
            dialogue += f"{role}: {t['text']}\n"

        prompt = (
            "Summarize the following conversation dialogue concisely. "
            "Focus on ongoing topics, user goals, active projects, important corrections, "
            "unresolved discussion points, and conversational continuity.\n"
            "Do not output conversational fillers, just a dense summary paragraph.\n\n"
        )
        if session.summarized_memory:
            prompt += f"Existing prior summary:\n{session.summarized_memory}\n\n"
        
        prompt += f"New dialogue to append/integrate:\n{dialogue}"

        try:
            response = await self.llm.generate_background(
                [{"role": "user", "content": prompt}], 
                max_tokens=150
            )
            if response:
                session.summarized_memory = response.strip()
                logger.info(f"[{session.session_id}] Updated Layer 3 Summary: {session.summarized_memory}")
        except Exception as e:
            logger.error(f"Background summarization failed: {e}")

    async def _extract_user_facts(self, session: 'Session', user_text: str, assistant_text: str):
        """Extract lightweight user facts from the latest turn."""
        prompt = (
            "Extract any new, important facts about the user from the dialogue below. "
            "Examples of facts: user's name, preferences, goals, favorite technologies, current project. "
            "If no clear facts are found, return an empty JSON object: {}\n"
            "Otherwise, return a JSON object with key-value pairs representing the facts.\n"
            "Output ONLY valid JSON.\n\n"
            f"User: {user_text}\nAssistant: {assistant_text}"
        )

        try:
            # Tell LLM to force JSON output format
            response = await self.llm.generate_background(
                [{"role": "user", "content": prompt}], 
                max_tokens=150,
                json_format=True
            )
            if response:
                try:
                    # Clean up markdown code blocks if any
                    clean_resp = response.strip()
                    if clean_resp.startswith("```json"):
                        clean_resp = clean_resp[7:-3].strip()
                    elif clean_resp.startswith("```"):
                        clean_resp = clean_resp[3:-3].strip()

                    new_facts = json.loads(clean_resp)
                    if new_facts and isinstance(new_facts, dict):
                        session.user_facts.update(new_facts)
                        if new_facts:
                            logger.info(f"[{session.session_id}] Extracted facts (Layer 2): {new_facts}")
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.error(f"Background fact extraction failed: {e}")
