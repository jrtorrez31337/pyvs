import threading
import numpy as np
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from app.config import CHATTERBOX_DEVICE
from app.services.gpu_lock import gpu0_lock
from app.services.audio_utils import apply_post_processing


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
        self._model_loaded = False
        self.model = None
        self._initialized = True

    def load_model(self):
        if self._model_loaded:
            return
        with gpu0_lock:
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

    def _chunk_audio(self, wav, sr, chunk_ms=100):
        """Yield fixed-duration chunks from a waveform."""
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        if wav.ndim > 1:
            wav = np.squeeze(wav)
            if wav.ndim > 1:
                wav = wav[0]

        chunk_samples = max(int(sr * chunk_ms / 1000), 1)
        for i in range(0, len(wav), chunk_samples):
            yield wav[i:i + chunk_samples], sr

    def generate(self, text, language_id="en", audio_prompt_path=None,
                 exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
                 repetition_penalty=None, min_p=None, top_p=None,
                 post_processing=None):
        """Generate speech with optional voice cloning and emotion control.

        Returns: (numpy_array, sample_rate)
        """
        self.load_model()
        with gpu0_lock:
            kwargs = {}
            if repetition_penalty is not None:
                kwargs['repetition_penalty'] = repetition_penalty
            if min_p is not None:
                kwargs['min_p'] = min_p
            if top_p is not None:
                kwargs['top_p'] = top_p

            wav = self.model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                **kwargs,
            )
            # Convert torch tensor to numpy
            audio_np = wav.squeeze().cpu().numpy()

        sr = self.sample_rate
        if post_processing:
            audio_np, sr = apply_post_processing(audio_np, sr, post_processing)

        return audio_np, sr

    def generate_streaming(self, text, language_id="en", audio_prompt_path=None,
                           exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
                           repetition_penalty=None, min_p=None, top_p=None,
                           post_processing=None):
        """Generate speech and yield chunks for streaming.

        Chatterbox doesn't support native streaming, so we generate full
        audio then chunk it for the streaming response.
        """
        wav, sr = self.generate(
            text, language_id=language_id, audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature,
            repetition_penalty=repetition_penalty, min_p=min_p, top_p=top_p,
            post_processing=post_processing,
        )
        yield from self._chunk_audio(wav, sr)


chatterbox_service = ChatterboxService()
