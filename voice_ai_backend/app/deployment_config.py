"""
deployment_config.py — Configuration profiles for local development and cloud servers.
"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Detect environment profile, defaulting to "local_dev" (optimized for Apple Silicon / local CPU)
DEPLOYMENT_PROFILE = os.environ.get("DEPLOYMENT_PROFILE", "local_dev")

PROFILES: Dict[str, Dict[str, Any]] = {
    "local_dev": {
        "stt": {
            "model_size": "base.en",
            "compute_type": "default",  # Will use hardware-aware defaults (e.g. int8 on macOS ARM)
        },
        "tts": {
            "default_engine": "piper",
            "default_voice": "en_US-libritts_r-medium"
        },
        "ollama": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "default_model": os.environ.get("OLLAMA_MODEL", None),  # None means dynamically routed based on VRAM/hardware
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 64,
                "repeat_penalty": 1.05,
                "num_ctx": 1024,
            }
        },
        "persistence": {
            "base_dir": "sessions",
        },
        "limits": {
            "max_websocket_sessions": 5,
            "max_audio_queue_size": 100,
            "session_timeout_seconds": 1800,  # 30 mins
        }
    },
    "cpu_server": {
        "stt": {
            "model_size": "small.en",
            "compute_type": "int8",
        },
        "tts": {
            "default_engine": "kokoro",
            "default_voice": "af_heart"
        },
        "ollama": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "default_model": os.environ.get("OLLAMA_MODEL", None),
            "keep_alive": "60m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 64,
                "repeat_penalty": 1.05,
                "num_ctx": 2048,
            }
        },
        "persistence": {
            "base_dir": "/data/sessions",
        },
        "limits": {
            "max_websocket_sessions": 20,
            "max_audio_queue_size": 200,
            "session_timeout_seconds": 3600,  # 1 hour
        }
    },
    "gpu_server": {
        "stt": {
            "model_size": "medium.en",
            "compute_type": "float16",
        },
        "tts": {
            "default_engine": "kokoro",
            "default_voice": "af_heart"
        },
        "ollama": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "default_model": os.environ.get("OLLAMA_MODEL", None),
            "keep_alive": "60m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 128,
                "repeat_penalty": 1.05,
                "num_ctx": 4096,
            }
        },
        "persistence": {
            "base_dir": "/data/sessions",
        },
        "limits": {
            "max_websocket_sessions": 50,
            "max_audio_queue_size": 500,
            "session_timeout_seconds": 3600,
        }
    },
    "cloud_gpu": {
        "stt": {
            "model_size": "large-v3",
            "compute_type": "float16",
        },
        "tts": {
            "default_engine": "kokoro",
            "default_voice": "af_heart"
        },
        "ollama": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "default_model": os.environ.get("OLLAMA_MODEL", None),
            "keep_alive": "120m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 128,
                "repeat_penalty": 1.05,
                "num_ctx": 4096,
            }
        },
        "persistence": {
            "base_dir": "/data/sessions",
        },
        "limits": {
            "max_websocket_sessions": 100,
            "max_audio_queue_size": 1000,
            "session_timeout_seconds": 7200,  # 2 hours
        }
    }
}

class DeploymentConfig:
    def __init__(self):
        self.profile_name = DEPLOYMENT_PROFILE if DEPLOYMENT_PROFILE in PROFILES else "local_dev"
        self.config = PROFILES[self.profile_name]
        logger.info(f"Initialized dynamic deployment config with profile: '{self.profile_name}'")

    def get(self, key: str) -> Any:
        return self.config.get(key)

# Global active config singleton
deployment_config = DeploymentConfig()
