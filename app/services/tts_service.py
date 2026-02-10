import torch
import threading
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from app.config import TTS_MODEL_BASE_PATH as MODEL_BASE_PATH

class TTSService:
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
        self._models_loaded = False
        self.tokenizer = None
        self.clone_model = None
        self.custom_model = None
        self.design_model = None

        self._initialized = True

    def load_models(self):
        """Load all TTS models onto GPU 0"""
        if self._models_loaded:
            return

        with self._model_lock:
            if self._models_loaded:
                return

            print("Loading Qwen3-TTS Tokenizer...")
            self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
                f"{MODEL_BASE_PATH}/Qwen3-TTS-Tokenizer-12Hz",
                device_map="cuda:0",
            )

            print("Loading Qwen3-TTS-12Hz-1.7B-Base (voice clone)...")
            self.clone_model = Qwen3TTSModel.from_pretrained(
                f"{MODEL_BASE_PATH}/Qwen3-TTS-12Hz-1.7B-Base",
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )

            print("Loading Qwen3-TTS-12Hz-1.7B-CustomVoice...")
            self.custom_model = Qwen3TTSModel.from_pretrained(
                f"{MODEL_BASE_PATH}/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )

            print("Loading Qwen3-TTS-12Hz-1.7B-VoiceDesign...")
            self.design_model = Qwen3TTSModel.from_pretrained(
                f"{MODEL_BASE_PATH}/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )

            self._models_loaded = True
            print("All TTS models loaded successfully!")

    def generate_clone(self, text: str, language: str, ref_audio_paths, ref_texts=None):
        """Generate speech using voice cloning with one or more reference samples.

        Args:
            text: Text to synthesize
            language: Language for synthesis
            ref_audio_paths: Single path or list of paths to reference audio files
            ref_texts: Single text or list of transcripts (optional, improves quality)
        """
        with self._model_lock:
            # Normalize to lists if single values passed
            if isinstance(ref_audio_paths, str):
                ref_audio_paths = [ref_audio_paths]
            if ref_texts is None:
                ref_texts = [None] * len(ref_audio_paths)
            elif isinstance(ref_texts, str):
                ref_texts = [ref_texts]

            wavs, sr = self.clone_model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_paths,
                ref_text=ref_texts,
            )
            return wavs[0], sr

    def generate_custom(self, text: str, language: str, speaker: str, instruct: str = None):
        """Generate speech using custom voice preset"""
        with self._model_lock:
            kwargs = {
                "text": text,
                "language": language,
                "speaker": speaker,
            }
            if instruct:
                kwargs["instruct"] = instruct

            wavs, sr = self.custom_model.generate_custom_voice(**kwargs)
            return wavs[0], sr

    def generate_design(self, text: str, language: str, instruct: str):
        """Generate speech using voice design"""
        with self._model_lock:
            wavs, sr = self.design_model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
            return wavs[0], sr

    def generate_clone_streaming(self, text: str, language: str, ref_audio_paths, ref_texts=None):
        """Generate speech using voice cloning with streaming output.

        Yields audio chunks as they are generated.
        """
        with self._model_lock:
            if isinstance(ref_audio_paths, str):
                ref_audio_paths = [ref_audio_paths]
            if ref_texts is None:
                ref_texts = [None] * len(ref_audio_paths)
            elif isinstance(ref_texts, str):
                ref_texts = [ref_texts]

            # Use streaming mode
            for chunk, sr in self.clone_model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_paths,
                ref_text=ref_texts,
                non_streaming_mode=False,  # Enable streaming
            ):
                yield chunk, sr

    def generate_custom_streaming(self, text: str, language: str, speaker: str, instruct: str = None):
        """Generate speech using custom voice with streaming output."""
        with self._model_lock:
            kwargs = {
                "text": text,
                "language": language,
                "speaker": speaker,
                "non_streaming_mode": False,
            }
            if instruct:
                kwargs["instruct"] = instruct

            for chunk, sr in self.custom_model.generate_custom_voice(**kwargs):
                yield chunk, sr

    def generate_design_streaming(self, text: str, language: str, instruct: str):
        """Generate speech using voice design with streaming output."""
        with self._model_lock:
            for chunk, sr in self.design_model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                non_streaming_mode=False,
            ):
                yield chunk, sr

    def get_supported_speakers(self):
        """Get list of supported speakers for CustomVoice model"""
        if self.custom_model:
            return self.custom_model.get_supported_speakers()
        return []

    def get_supported_languages(self):
        """Get list of supported languages"""
        return ["Chinese", "English", "Japanese", "Korean", "German",
                "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"]


# Singleton instance
tts_service = TTSService()
