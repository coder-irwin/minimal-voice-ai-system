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
        self.default_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")

        # Gemini fallback (supports API Key and Google Application Default Credentials in GCP)
        self._gemini_client = None
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key or os.environ.get("K_SERVICE"):
            try:
                from google import genai
                if api_key:
                    self._gemini_client = genai.Client(api_key=api_key)
                else:
                    self._gemini_client = genai.Client()  # Automatically resolves ADC / Vertex credentials
                logger.info("Gemini fallback initialized successfully.")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        self._initialized = True
        logger.info(f"AIAgent initialized")

    # ── Lifecycle ────────────────────────────────────────────────────

    async def startup(self):
        """Call once at app startup. Creates persistent aiohttp session and pings Ollama."""
        timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_read=None)
        self._session = aiohttp.ClientSession(timeout=timeout)

        # Ping Ollama
        try:
            async with self._session.get(f"{OLLAMA_BASE_URL}/api/tags") as resp:
                if resp.status == 200:
                    self._ollama_available = True
                    
                    # 1. Determine dynamic model routing
                    selected_model = os.environ.get("OLLAMA_MODEL")
                    if not selected_model:
                        from app.hardware_manager import hardware_manager
                        if hardware_manager.supports_cuda and hardware_manager.available_vram_gb and hardware_manager.available_vram_gb >= 10:
                            selected_model = "llama3.1:8b"
                            logger.info(f"Dynamic Model Routing: High-end GPU (VRAM={hardware_manager.available_vram_gb} GB) detected -> routed to '{selected_model}'")
                        else:
                            selected_model = "qwen2.5:1.5b-instruct"
                            logger.info(f"Dynamic Model Routing: CPU/MPS or low-end GPU detected -> routed to '{selected_model}'")
                    self.default_model = selected_model
                    
                    # 2. Bootstrap model (automatic pull)
                    await self._bootstrap_model(self.default_model)
                    
                    # 3. Warm up model
                    await self._warmup_model(self.default_model)
                else:
                    logger.warning(f"Ollama ping returned {resp.status}")
        except Exception as e:
            logger.warning(f"Ollama not reachable: {e}. Will use Gemini fallback.")
            self.default_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")

    async def _bootstrap_model(self, model_name: str):
        """Verify model is pulled, if not, pull it."""
        if not self._session:
            return
        
        logger.info(f"Checking if model '{model_name}' is already pulled in Ollama...")
        try:
            async with self._session.get(f"{OLLAMA_BASE_URL}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    
                    model_found = False
                    for m in models:
                        if m == model_name or m.split(":")[0] == model_name.split(":")[0]:
                            model_found = True
                            break
                    
                    if model_found:
                        logger.info(f"Model '{model_name}' already exists in Ollama.")
                        return
                    
                    logger.info(f"Model '{model_name}' not found. Initiating automatic pulling (this may take a few minutes)...")
                    payload = {"name": model_name, "stream": False}
                    async with self._session.post(
                        f"{OLLAMA_BASE_URL}/api/pull", json=payload, timeout=None
                    ) as pull_resp:
                        if pull_resp.status == 200:
                            logger.info(f"Model '{model_name}' pulled successfully.")
                        else:
                            body = await pull_resp.text()
                            logger.error(f"Failed to pull model '{model_name}' ({pull_resp.status}): {body}")
                else:
                    logger.warning(f"Ollama tag check returned status {resp.status}")
        except Exception as e:
            logger.error(f"Error during model bootstrapping for '{model_name}': {e}")

    async def get_available_models(self) -> list[str]:
        """Fetch available models from Ollama."""
        if not self._ollama_available or not self._session:
            return []
        try:
            async with self._session.get(f"{OLLAMA_BASE_URL}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
        return []

    async def _warmup_model(self, model_name: str):
        """Send a tiny request to load model into GPU memory."""
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
            async with self._session.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Model '{model_name}' warmed up and loaded in memory.")
                else:
                    body = await resp.text()
                    logger.warning(f"Warmup failed ({resp.status}): {body}")
        except Exception as e:
            logger.warning(f"Model warmup error: {e}")

    async def shutdown(self):
        """Call at app shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Removed legacy stateful model switching ──

    @property
    def is_ollama_available(self) -> bool:
        return self._ollama_available

    # ── Streaming generation (primary API) ───────────────────────────

    async def stream_response(
        self,
        messages: list[dict],
        model_name: str,
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
            async for chunk in self._stream_ollama(messages, model_name, generation_id):
                yield chunk
        elif self._gemini_client:
            async for chunk in self._stream_gemini_fallback(messages, generation_id):
                yield chunk
        else:
            yield {"token": "[No LLM backend available]", "done": True, "first_token_ms": 0, "generation_id": generation_id}

    async def _stream_ollama(
        self, messages: list[dict], model_name: str, generation_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Stream from Ollama /api/chat with NDJSON response."""
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0.4,
                "top_p": 0.85,
                "num_predict": 150,
                "repeat_penalty": 1.1,
                "num_ctx": 512,
                "num_thread": 8,
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

    async def process_transcript(self, text: str, model_name: str, history: list | None = None) -> str:
        """Non-streaming convenience method. Collects all tokens and returns full response."""
        from app.session import SYSTEM_PROMPT
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for turn in history[-10:]:
                role = "user" if turn["role"] == "user" else "assistant"
                messages.append({"role": role, "content": turn["text"]})
        messages.append({"role": "user", "content": text})

        full_response = []
        async for chunk in self.stream_response(messages, model_name):
            if chunk["token"]:
                full_response.append(chunk["token"])
        return "".join(full_response)

    async def generate_background(self, messages: list[dict], model_name: str, max_tokens: int = 150, json_format: bool = False) -> str:
        """Non-streaming generation optimized for background tasks (summarization, extraction)."""
        if not self._ollama_available or not self._session or self._session.closed:
            return ""

        payload = {
            "model": model_name,
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
