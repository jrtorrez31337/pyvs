import threading
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from app.config import CHATTERBOX_DEVICE


class ChatterboxService:
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
            print("Loading Chatterbox Multilingual TTS...")
            self.model = ChatterboxMultilingualTTS.from_pretrained(
                device=CHATTERBOX_DEVICE
            )
            self._model_loaded = True
            print("Chatterbox model loaded successfully!")

    @property
    def sample_rate(self):
        return self.model.sr if self.model else 24000

    def generate(self, text, language_id="en", audio_prompt_path=None,
                 exaggeration=0.5, cfg_weight=0.5, temperature=0.8):
        """Generate speech with optional voice cloning and emotion control.

        Returns: (numpy_array, sample_rate)
        """
        with self._model_lock:
            wav = self.model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            # Convert torch tensor to numpy
            audio_np = wav.squeeze().cpu().numpy()
            return audio_np, self.sample_rate


chatterbox_service = ChatterboxService()
