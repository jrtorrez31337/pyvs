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
            torch.cuda.set_device(0)
            self.model = ClearVoice(
                task='speech_enhancement',
                model_names=[CLEARVOICE_MODEL],
            )
            self._model_loaded = True
            print("ClearVoice model loaded successfully!")

    def enhance_file(self, audio_path):
        """Enhance audio file, return enhanced numpy array."""
        with self._model_lock:
            torch.cuda.set_device(0)
            return self.model(input_path=audio_path, online_write=False)


clearvoice_service = ClearVoiceService()
