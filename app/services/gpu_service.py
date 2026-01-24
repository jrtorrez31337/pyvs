import threading
from typing import Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._nvml_initialized = False
        self._init_nvml()
        self._initialized = True

    def _init_nvml(self):
        if not PYNVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except pynvml.NVMLError:
            self._nvml_initialized = False

    def get_gpu_status(self) -> list:
        """Get status of all GPUs."""
        if not self._nvml_initialized:
            return []

        gpus = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                gpus.append({
                    'index': i,
                    'name': name,
                    'memory_used': memory.used // (1024 ** 2),  # MB
                    'memory_total': memory.total // (1024 ** 2),  # MB
                    'memory_percent': round(memory.used / memory.total * 100, 1),
                    'utilization': utilization.gpu,
                    'temperature': temperature,
                })
        except pynvml.NVMLError:
            pass

        return gpus


gpu_service = GPUService()
