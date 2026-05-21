import os
import json
import wave
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SessionStorage:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            try:
                from app.deployment_config import deployment_config
                base_dir = deployment_config.get("persistence")["base_dir"]
            except Exception:
                base_dir = "sessions"
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> str:
        return os.path.join(self.base_dir, f"session_{session_id}")

    def _get_audio_dir(self, session_id: str) -> str:
        return os.path.join(self._get_session_dir(session_id), "audio")

    async def initialize_session(self, session_id: str, config: Dict[str, Any]):
        """Creates directory structure and initial metadata."""
        def _sync_init():
            session_dir = self._get_session_dir(session_id)
            audio_dir = self._get_audio_dir(session_id)
            os.makedirs(session_dir, exist_ok=True)
            os.makedirs(audio_dir, exist_ok=True)

            metadata = {
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "config": config,
                "total_turns": 0
            }

            with open(os.path.join(session_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
                
            # Initialize empty transcript
            with open(os.path.join(session_dir, "transcript.json"), "w", encoding="utf-8") as f:
                json.dump([], f)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_init)

    async def append_turn(self, session_id: str, role: str, text: str, audio_filename: str) -> int:
        """Appends a turn to the transcript.json and updates total_turns. Returns the turn index."""
        def _sync_append():
            session_dir = self._get_session_dir(session_id)
            transcript_path = os.path.join(session_dir, "transcript.json")
            metadata_path = os.path.join(session_dir, "metadata.json")

            # Load existing
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = json.load(f)
            except FileNotFoundError:
                transcript = []

            turn_idx = len(transcript) + 1
            turn_data = {
                "turn": turn_idx,
                "role": role,
                "text": text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "audio_file": f"audio/{audio_filename}" if audio_filename else None
            }
            transcript.append(turn_data)

            # Save transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=2)
                
            # Update metadata
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = {}
                
            metadata["total_turns"] = turn_idx
            metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
                
            return turn_idx

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_append)

    async def append_to_conversation(self, session_id: str, audio_bytes: bytes, original_sample_rate: int):
        """Resamples audio to 24000Hz, appends to a continuous PCM file, and writes conversation.wav."""
        def _sync_append():
            if not audio_bytes:
                return
                
            import numpy as np
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            target_sr = 24000
            if original_sample_rate != target_sr:
                duration = len(audio_array) / original_sample_rate
                target_len = int(duration * target_sr)
                x_old = np.linspace(0, duration, len(audio_array))
                x_new = np.linspace(0, duration, target_len)
                audio_array = np.interp(x_new, x_old, audio_array).astype(np.int16)
                
            resampled_bytes = audio_array.tobytes()
            
            audio_dir = self._get_audio_dir(session_id)
            os.makedirs(audio_dir, exist_ok=True)
            
            pcm_path = os.path.join(audio_dir, "conversation.pcm")
            wav_path = os.path.join(audio_dir, "conversation.wav")
            
            # Append to raw PCM
            with open(pcm_path, "ab") as f:
                f.write(resampled_bytes)
                
            # Regenerate WAV file from full PCM
            with open(pcm_path, "rb") as f:
                full_pcm = f.read()
                
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit PCM
                wf.setframerate(target_sr)
                wf.writeframes(full_pcm)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_append)

    async def update_memory(self, session_id: str, summary: str, facts: Dict[str, Any], active_tasks: List[Dict[str, Any]] = None):
        """Persists the summarized memory, user facts, and active tasks."""
        def _sync_update():
            session_dir = self._get_session_dir(session_id)
            if summary:
                with open(os.path.join(session_dir, "summary.txt"), "w", encoding="utf-8") as f:
                    f.write(summary)
            
            if facts:
                with open(os.path.join(session_dir, "user_facts.json"), "w", encoding="utf-8") as f:
                    json.dump(facts, f, indent=2)

            if active_tasks is not None:
                with open(os.path.join(session_dir, "active_tasks.json"), "w", encoding="utf-8") as f:
                    json.dump(active_tasks, f, indent=2)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_update)

    def load_session_sync(self, session_id: str) -> Dict[str, Any]:
        """Synchronously loads session context to reconstruct 'Hot' memory upon connection."""
        session_dir = self._get_session_dir(session_id)
        if not os.path.exists(session_dir):
            return None
            
        result = {
            "transcript": [],
            "summary": "",
            "facts": {},
            "config": {}
        }
        
        try:
            with open(os.path.join(session_dir, "transcript.json"), "r", encoding="utf-8") as f:
                result["transcript"] = json.load(f)
        except Exception:
            pass
            
        try:
            with open(os.path.join(session_dir, "summary.txt"), "r", encoding="utf-8") as f:
                result["summary"] = f.read().strip()
        except Exception:
            pass
            
        try:
            with open(os.path.join(session_dir, "user_facts.json"), "r", encoding="utf-8") as f:
                result["facts"] = json.load(f)
        except Exception:
            pass
            
        try:
            with open(os.path.join(session_dir, "active_tasks.json"), "r", encoding="utf-8") as f:
                result["active_tasks"] = json.load(f)
        except Exception:
            result["active_tasks"] = []
            
        try:
            with open(os.path.join(session_dir, "metadata.json"), "r", encoding="utf-8") as f:
                metadata = json.load(f)
                result["config"] = metadata.get("config", {})
        except Exception:
            pass
            
        return result

session_storage = SessionStorage()
