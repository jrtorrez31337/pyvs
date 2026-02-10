import threading
import torch
from clearvoice import ClearVoice
from app.config import CLEARVOICE_MODEL


class ClearVoiceService:
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
        self._model_lock = threading.Lock()
        self._model_loaded = False
        self.model = None
        self._initialized = True

    def load_model(self):
        if self._model_loaded:
            return
        with self._model_lock:
            if self._model_loaded:
                return
            print(f"Loading ClearVoice {CLEARVOICE_MODEL}...")
            self.model = ClearVoice(
                task='speech_enhancement',
                model_names=[CLEARVOICE_MODEL],
            )
            # ClearVoice picks the GPU with most free VRAM via get_free_gpu(),
            # which may land on cuda:1. Force all models to cuda:0.
            for net in self.model.models:
                net.model = net.model.to('cuda:0')
                net.device = torch.device('cuda:0')
            self._model_loaded = True
            print("ClearVoice model loaded successfully!")

    def enhance_file(self, audio_path):
        """Enhance audio file, return enhanced numpy array."""
        with self._model_lock:
            return self.model(input_path=audio_path, online_write=False)


clearvoice_service = ClearVoiceService()
