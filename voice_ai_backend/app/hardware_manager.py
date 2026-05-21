"""
hardware_manager.py — Runtime system/GPU profile detection singleton.
"""
import platform
import logging
import os

logger = logging.getLogger(__name__)

class HardwareManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.platform = "cpu_only"
        self.llm_backend = "cpu"
        self.stt_backend = "cpu_int8"
        self.tts_backend = "onnx_cpu"
        
        self.supports_cuda = False
        self.supports_mps = False
        self.available_ram_gb = 0
        self.available_vram_gb = 0
        self.cpu_cores = 0
        
        self._detect_hardware()
        self._initialized = True

    def _detect_hardware(self):
        # 1. Detect CPU cores and RAM
        try:
            import psutil
            self.available_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
            self.cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
        except Exception:
            self.available_ram_gb = 8
            self.cpu_cores = 4

        # 2. Detect platform and MPS
        sys_platform = platform.system()
        machine = platform.machine()
        
        is_apple_silicon = (sys_platform == "Darwin" and machine in ("arm64", "aarch64"))
        
        # 3. Detect CUDA
        onnx_providers = []
        try:
            import onnxruntime as rt
            onnx_providers = rt.get_available_providers()
        except Exception:
            pass

        self.supports_cuda = "CUDAExecutionProvider" in onnx_providers
        
        # Double check CUDA with ctranslate2 or torch if possible
        if not self.supports_cuda:
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    self.supports_cuda = True
            except Exception:
                pass
                
        if not self.supports_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    self.supports_cuda = True
            except Exception:
                pass

        # 4. Assign Backend Profiles based on detection
        if self.supports_cuda:
            self.platform = "nvidia_cuda"
            self.llm_backend = "cuda"
            self.stt_backend = "cuda_fp16"
            self.tts_backend = "onnx_cuda"
            
            # Detect VRAM
            try:
                import torch
                if torch.cuda.is_available():
                    self.available_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
            except Exception:
                pass
        elif is_apple_silicon:
            self.platform = "apple_silicon"
            self.supports_mps = True
            self.llm_backend = "metal"
            self.stt_backend = "cpu_int8"
            
            # Check for CoreML execution provider in ONNX
            if "CoreMLExecutionProvider" in onnx_providers:
                self.tts_backend = "onnx_coreml"
            else:
                self.tts_backend = "onnx_cpu"
        else:
            self.platform = "cpu_only"
            self.llm_backend = "cpu"
            self.stt_backend = "cpu_int8"
            self.tts_backend = "onnx_cpu"

    def get_summary_dict(self) -> dict:
        return {
            "platform": self.platform,
            "llm_backend": self.llm_backend,
            "stt_backend": self.stt_backend,
            "tts_backend": self.tts_backend,
            "supports_cuda": self.supports_cuda,
            "supports_mps": self.supports_mps,
            "available_ram_gb": self.available_ram_gb,
            "available_vram_gb": self.available_vram_gb if self.supports_cuda else None,
            "cpu_cores": self.cpu_cores
        }

hardware_manager = HardwareManager()
