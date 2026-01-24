import threading
from faster_whisper import WhisperModel

MODEL_CACHE_PATH = "/data/models/whisper"

class STTService:
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
        """Load Whisper model onto GPU 1"""
        if self._model_loaded:
            return

        with self._model_lock:
            if self._model_loaded:
                return

            print("Loading Whisper medium model...")
            self.model = WhisperModel(
                "medium",
                device="cuda",
                device_index=1,
                compute_type="float16",
                download_root=MODEL_CACHE_PATH,
            )
            self._model_loaded = True
            print("Whisper model loaded successfully!")

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file to text"""
        with self._model_lock:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="en",  # Primary use is English
                vad_filter=True,
            )

            text = " ".join([segment.text.strip() for segment in segments])

            return {
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
            }

    def transcribe_with_language_detect(self, audio_path: str) -> dict:
        """Transcribe audio with automatic language detection"""
        with self._model_lock:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join([segment.text.strip() for segment in segments])

            return {
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
            }


# Singleton instance
stt_service = STTService()
