"""
llm.py — Ultra-low-latency LLM layer with Ollama streaming + Gemini fallback.

Architecture:
  - Primary: Ollama local inference via streaming HTTP API
  - Fallback: Google Gemini (cloud)
  - Singleton aiohttp session (persistent connection pooling)
  - Async streaming generator yields tokens as they arrive
  - Cancellable generation for future barge-in support

Ollama API used: POST /api/chat  (stream: true, NDJSON response)
"""

import os
import json
import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
OLLAMA_KEEP_ALIVE = "30m"

class AIAgent:
    """
    Singleton LLM agent with streaming token generation.

    Primary:  Ollama local (qwen2.5:1.5b-instruct / llama3.1:8b)
    Fallback: Gemini API (if GEMINI_API_KEY set and Ollama unavailable)
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._ollama_available = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._model = OLLAMA_DEFAULT_MODEL

        # Gemini fallback
        self._gemini_client = None
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=api_key)
                logger.info("Gemini fallback initialized.")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        self._initialized = True
        logger.info(f"AIAgent initialized (model={self._model})")

    # ── Lifecycle ────────────────────────────────────────────────────

    async def startup(self):
        """Call once at app startup. Creates persistent aiohttp session and pings Ollama."""
        timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_read=None)
        self._session = aiohttp.ClientSession(timeout=timeout)

        # Ping Ollama
        try:
            async with self._session.get(f"{OLLAMA_BASE_URL}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    self._ollama_available = True
                    logger.info(f"Ollama connected. Available models: {models}")

                    # Warm up the model (load into memory)
                    await self._warmup_model()
                else:
                    logger.warning(f"Ollama ping returned {resp.status}")
        except Exception as e:
            logger.warning(f"Ollama not reachable: {e}. Will use Gemini fallback.")

    async def _warmup_model(self):
        """Send a tiny request to load model into GPU memory."""
        try:
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
            async with self._session.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Model '{self._model}' warmed up and loaded in memory.")
                else:
                    body = await resp.text()
                    logger.warning(f"Warmup failed ({resp.status}): {body}")
        except Exception as e:
            logger.warning(f"Model warmup error: {e}")

    async def shutdown(self):
        """Call at app shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Model switching ──────────────────────────────────────────────

    def set_model(self, model_name: str):
        """Switch Ollama model at runtime."""
        self._model = model_name
        logger.info(f"Model switched to: {model_name}")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_ollama_available(self) -> bool:
        return self._ollama_available

    # ── Streaming generation (primary API) ───────────────────────────

    async def stream_response(
        self,
        messages: list[dict],
        generation_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Stream tokens from the LLM. Yields dicts:
          {"token": str, "done": bool, "first_token_ms": float | None, "generation_id": str | None}

        The first yielded dict includes first_token_ms (time from request to first token).
        Subsequent dicts have first_token_ms=None.

        Falls back to Gemini (non-streaming) if Ollama is unavailable.
        """
        if self._ollama_available and self._session and not self._session.closed:
            async for chunk in self._stream_ollama(messages, generation_id):
                yield chunk
        elif self._gemini_client:
            async for chunk in self._stream_gemini_fallback(messages, generation_id):
                yield chunk
        else:
            yield {"token": "[No LLM backend available]", "done": True, "first_token_ms": 0, "generation_id": generation_id}

    async def _stream_ollama(
        self, messages: list[dict], generation_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Stream from Ollama /api/chat with NDJSON response."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 64,
                "repeat_penalty": 1.05,
                "num_ctx": 2048,
            },
        }

        t0 = time.perf_counter()
        first_token_emitted = False

        try:
            async with self._session.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Ollama error ({resp.status}): {body}")
                    yield {"token": f"[Ollama error: {resp.status}]", "done": True, "first_token_ms": 0}
                    return

                async for line in resp.content:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Ollama chat API: {"message": {"content": "token"}, "done": false}
                    token = chunk.get("message", {}).get("content", "")
                    done = chunk.get("done", False)

                    if token or done:
                        first_token_ms = None
                        if not first_token_emitted and token:
                            first_token_ms = (time.perf_counter() - t0) * 1000
                            first_token_emitted = True
                            logger.info(
                                f"[LLM] first token in {first_token_ms:.0f}ms"
                            )

                        yield {
                            "token": token,
                            "done": done,
                            "first_token_ms": first_token_ms,
                            "generation_id": generation_id,
                        }

                    if done:
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(f"[LLM] generation complete in {total_ms:.0f}ms")
                        return

        except asyncio.CancelledError:
            logger.info("[LLM] generation cancelled (barge-in)")
            raise
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield {"token": f"[Error: {e}]", "done": True, "first_token_ms": 0}

    async def _stream_gemini_fallback(
        self, messages: list[dict], generation_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Non-streaming Gemini fallback. Yields the complete response as one chunk."""
        t0 = time.perf_counter()
        try:
            # Build prompt from messages
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"AI: {content}\n"
            prompt += "AI:"

            response = await asyncio.to_thread(
                self._gemini_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text.strip()
            latency = (time.perf_counter() - t0) * 1000
            yield {"token": text, "done": True, "first_token_ms": latency, "generation_id": generation_id}
        except Exception as e:
            logger.error(f"Gemini fallback error: {e}")
            yield {"token": f"[Gemini error: {e}]", "done": True, "first_token_ms": 0, "generation_id": generation_id}

    # ── Legacy non-streaming API (backward compat) ───────────────────

    async def process_transcript(self, text: str, history: list | None = None) -> str:
        """Non-streaming convenience method. Collects all tokens and returns full response."""
        from app.session import SYSTEM_PROMPT
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for turn in history[-10:]:
                role = "user" if turn["role"] == "user" else "assistant"
                messages.append({"role": role, "content": turn["text"]})
        messages.append({"role": "user", "content": text})

        full_response = []
        async for chunk in self.stream_response(messages):
            if chunk["token"]:
                full_response.append(chunk["token"])
        return "".join(full_response)

    async def generate_background(self, messages: list[dict], max_tokens: int = 150, json_format: bool = False) -> str:
        """Non-streaming generation optimized for background tasks (summarization, extraction)."""
        if not self._ollama_available or not self._session or self._session.closed:
            return ""

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0.4 if json_format else 0.5,
                "top_p": 0.9,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }
        
        if json_format:
            payload["format"] = "json"

        try:
            async with self._session.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("content", "")
                else:
                    body = await resp.text()
                    logger.error(f"Ollama background generation error: {body}")
        except Exception as e:
            logger.error(f"Ollama background task failed: {e}")
        
        return ""
