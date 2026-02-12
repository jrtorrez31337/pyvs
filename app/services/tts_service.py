import torch
import threading
import numpy as np
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from app.config import TTS_MODEL_BASE_PATH as MODEL_BASE_PATH
from app.services.gpu_lock import gpu0_lock
from app.services.audio_utils import apply_post_processing

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

        self._models_loaded = False
        self.tokenizer = None
        self.clone_model = None
        self.custom_model = None
        self.design_model = None
        self.clone_model_fast = None
        self.custom_model_fast = None

        self._initialized = True

    def load_models(self):
        """Load all TTS models onto GPU 0"""
        if self._models_loaded:
            return

        with gpu0_lock:
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

            from app.config import TTS_FAST_MODEL_ENABLED
            if TTS_FAST_MODEL_ENABLED:
                print("Loading Qwen3-TTS-12Hz-0.6B-Base (fast clone)...")
                self.clone_model_fast = Qwen3TTSModel.from_pretrained(
                    f"{MODEL_BASE_PATH}/Qwen3-TTS-12Hz-0.6B-Base",
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                )

                print("Loading Qwen3-TTS-12Hz-0.6B-CustomVoice (fast custom)...")
                self.custom_model_fast = Qwen3TTSModel.from_pretrained(
                    f"{MODEL_BASE_PATH}/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                )

            self._models_loaded = True
            print("All TTS models loaded successfully!")

    def _chunk_audio(self, wav: np.ndarray, sr: int, chunk_ms: int = 100):
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

    def _normalize_clone_inputs(self, ref_audio_paths, ref_texts=None):
        """Normalize clone references into aligned lists."""
        if not ref_audio_paths:
            return [], []
        if isinstance(ref_audio_paths, str):
            ref_audio_paths = [ref_audio_paths]
        if ref_texts is None:
            ref_texts = []
        elif isinstance(ref_texts, str):
            ref_texts = [ref_texts]
        else:
            ref_texts = list(ref_texts)
        while len(ref_texts) < len(ref_audio_paths):
            ref_texts.append(None)
        return list(ref_audio_paths), ref_texts

    def _build_voice_clone_conditioning(self, ref_audio_paths, ref_texts=None,
                                         batch_size: int = 1, weights=None):
        """Build `voice_clone_prompt` (+ optional `ref_ids`) for non-base models.

        Args:
            ref_audio_paths: list of reference audio paths
            ref_texts: list of reference transcripts
            batch_size: batch size for duplication
            weights: optional list of floats for weighted speaker embedding blending

        Note: this must run while holding `gpu0_lock` because it invokes the base model.
        """
        ref_audio_paths, ref_texts = self._normalize_clone_inputs(ref_audio_paths, ref_texts)
        if not ref_audio_paths:
            return None, None

        # ICL (ref text + ref code) is only practical with a single reference sample.
        use_icl = (
            len(ref_audio_paths) == 1
            and ref_texts
            and isinstance(ref_texts[0], str)
            and bool(ref_texts[0].strip())
        )
        if use_icl:
            prompt_items = self.clone_model.create_voice_clone_prompt(
                ref_audio=ref_audio_paths[0],
                ref_text=ref_texts[0],
                x_vector_only_mode=False,
            )
        else:
            prompt_items = self.clone_model.create_voice_clone_prompt(
                ref_audio=ref_audio_paths,
                ref_text=None,
                x_vector_only_mode=True,
            )

        if not prompt_items:
            return None, None

        if len(prompt_items) > 1:
            # Multiple refs: weighted average of speaker embeddings.
            embeddings = [torch.as_tensor(item.ref_spk_embedding) for item in prompt_items]
            if weights and len(weights) == len(embeddings):
                # Normalize weights
                w = torch.tensor(weights, dtype=torch.float32)
                w = w / w.sum()
                avg_embed = sum(w_i * emb for w_i, emb in zip(w, embeddings))
            else:
                avg_embed = torch.stack(embeddings, dim=0).mean(dim=0)
            voice_clone_prompt = {
                "ref_code": [None],
                "ref_spk_embedding": [avg_embed],
                "x_vector_only_mode": [True],
                "icl_mode": [False],
            }
            ref_ids = None
        else:
            item = prompt_items[0]
            voice_clone_prompt = {
                "ref_code": [torch.as_tensor(item.ref_code) if item.ref_code is not None else None],
                "ref_spk_embedding": [torch.as_tensor(item.ref_spk_embedding)],
                "x_vector_only_mode": [item.x_vector_only_mode],
                "icl_mode": [item.icl_mode],
            }
            ref_ids = None
            if item.icl_mode and item.ref_text:
                ref_ids = [
                    self.clone_model._tokenize_texts(
                        [self.clone_model._build_ref_text(item.ref_text)]
                    )[0]
                ]

        if batch_size > 1:
            voice_clone_prompt = {
                k: v * batch_size for k, v in voice_clone_prompt.items()
            }
            if ref_ids is not None:
                ref_ids = ref_ids * batch_size

        return voice_clone_prompt, ref_ids

    def _extract_model_kwargs(self, inference_params):
        """Extract model-level kwargs from inference_params dict."""
        kwargs = {}
        if not inference_params:
            return kwargs
        for key in ('temperature', 'top_k', 'top_p', 'repetition_penalty'):
            if key in inference_params and inference_params[key] is not None:
                kwargs[key] = inference_params[key]
        return kwargs

    def _apply_post(self, wav, sr, post_processing):
        """Apply post-processing pipeline if options provided."""
        if post_processing:
            wav, sr = apply_post_processing(wav, sr, post_processing)
        return wav, sr

    def generate_clone(self, text: str, language: str, ref_audio_paths, ref_texts=None,
                       fast=False, inference_params=None, post_processing=None):
        """Generate speech using voice cloning with one or more reference samples.

        Args:
            text: Text to synthesize
            language: Language for synthesis
            ref_audio_paths: Single path or list of paths to reference audio files
            ref_texts: Single text or list of transcripts (optional, improves quality)
            fast: Use 0.6B model for faster generation
            inference_params: dict with temperature, top_k, top_p, repetition_penalty
            post_processing: dict with pitch_shift, speed, volume_normalize, sample_rate
        """
        self.load_models()
        with gpu0_lock:
            model = self.clone_model_fast if (fast and self.clone_model_fast) else self.clone_model
            # Normalize to lists if single values passed
            if isinstance(ref_audio_paths, str):
                ref_audio_paths = [ref_audio_paths]
            if ref_texts is None:
                ref_texts = [None] * len(ref_audio_paths)
            elif isinstance(ref_texts, str):
                ref_texts = [ref_texts]

            model_kwargs = self._extract_model_kwargs(inference_params)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_paths,
                ref_text=ref_texts,
                **model_kwargs,
            )
            return self._apply_post(wavs[0], sr, post_processing)

    def generate_custom(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: str = None,
        fast=False,
        ref_audio_paths=None,
        ref_texts=None,
        ref_weights=None,
        inference_params=None,
        post_processing=None,
    ):
        """Generate speech using custom voice preset"""
        self.load_models()
        with gpu0_lock:
            model = self.custom_model_fast if (fast and self.custom_model_fast) else self.custom_model
            kwargs = {
                "text": text,
                "language": language,
                "speaker": speaker,
            }
            if instruct:
                kwargs["instruct"] = instruct
            if ref_audio_paths:
                voice_clone_prompt, ref_ids = self._build_voice_clone_conditioning(
                    ref_audio_paths, ref_texts, batch_size=1, weights=ref_weights
                )
                if voice_clone_prompt is not None:
                    kwargs["voice_clone_prompt"] = voice_clone_prompt
                if ref_ids is not None:
                    kwargs["ref_ids"] = ref_ids

            kwargs.update(self._extract_model_kwargs(inference_params))
            wavs, sr = model.generate_custom_voice(**kwargs)
            return self._apply_post(wavs[0], sr, post_processing)

    def generate_design(self, text: str, language: str, instruct: str,
                        ref_audio_paths=None, ref_texts=None, ref_weights=None,
                        inference_params=None, post_processing=None):
        """Generate speech using voice design"""
        self.load_models()
        with gpu0_lock:
            kwargs = {
                "text": text,
                "language": language,
                "instruct": instruct,
            }
            if ref_audio_paths:
                voice_clone_prompt, ref_ids = self._build_voice_clone_conditioning(
                    ref_audio_paths, ref_texts, batch_size=1, weights=ref_weights
                )
                if voice_clone_prompt is not None:
                    kwargs["voice_clone_prompt"] = voice_clone_prompt
                if ref_ids is not None:
                    kwargs["ref_ids"] = ref_ids

            kwargs.update(self._extract_model_kwargs(inference_params))
            wavs, sr = self.design_model.generate_voice_design(**kwargs)
            return self._apply_post(wavs[0], sr, post_processing)

    def generate_clone_streaming(self, text: str, language: str, ref_audio_paths, ref_texts=None,
                                  fast=False, inference_params=None, post_processing=None):
        """Generate speech using voice cloning with streaming output.

        Yields audio chunks as they are generated.
        """
        self.load_models()
        model = self.clone_model_fast if (fast and self.clone_model_fast) else self.clone_model
        if isinstance(ref_audio_paths, str):
            ref_audio_paths = [ref_audio_paths]
        if ref_texts is None:
            ref_texts = [None] * len(ref_audio_paths)
        elif isinstance(ref_texts, str):
            ref_texts = [ref_texts]

        model_kwargs = self._extract_model_kwargs(inference_params)
        with gpu0_lock:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_paths,
                ref_text=ref_texts,
                non_streaming_mode=False,
                **model_kwargs,
            )

        wav = wavs[0] if isinstance(wavs, list) else wavs
        wav, sr = self._apply_post(wav, sr, post_processing)
        yield from self._chunk_audio(wav, sr)

    def generate_custom_streaming(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: str = None,
        fast=False,
        ref_audio_paths=None,
        ref_texts=None,
        ref_weights=None,
        inference_params=None,
        post_processing=None,
    ):
        """Generate speech using custom voice with streaming output."""
        self.load_models()
        model = self.custom_model_fast if (fast and self.custom_model_fast) else self.custom_model
        kwargs = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "non_streaming_mode": False,
        }
        if instruct:
            kwargs["instruct"] = instruct

        kwargs.update(self._extract_model_kwargs(inference_params))

        with gpu0_lock:
            if ref_audio_paths:
                voice_clone_prompt, ref_ids = self._build_voice_clone_conditioning(
                    ref_audio_paths, ref_texts, batch_size=1, weights=ref_weights
                )
                if voice_clone_prompt is not None:
                    kwargs["voice_clone_prompt"] = voice_clone_prompt
                if ref_ids is not None:
                    kwargs["ref_ids"] = ref_ids
            wavs, sr = model.generate_custom_voice(**kwargs)

        wav = wavs[0] if isinstance(wavs, list) else wavs
        wav, sr = self._apply_post(wav, sr, post_processing)
        yield from self._chunk_audio(wav, sr)

    def generate_design_streaming(self, text: str, language: str, instruct: str,
                                   ref_audio_paths=None, ref_texts=None,
                                   ref_weights=None,
                                   inference_params=None, post_processing=None):
        """Generate speech using voice design with streaming output."""
        self.load_models()
        kwargs = {
            "text": text,
            "language": language,
            "instruct": instruct,
            "non_streaming_mode": False,
        }
        kwargs.update(self._extract_model_kwargs(inference_params))

        with gpu0_lock:
            if ref_audio_paths:
                voice_clone_prompt, ref_ids = self._build_voice_clone_conditioning(
                    ref_audio_paths, ref_texts, batch_size=1, weights=ref_weights
                )
                if voice_clone_prompt is not None:
                    kwargs["voice_clone_prompt"] = voice_clone_prompt
                if ref_ids is not None:
                    kwargs["ref_ids"] = ref_ids
            wavs, sr = self.design_model.generate_voice_design(**kwargs)

        wav = wavs[0] if isinstance(wavs, list) else wavs
        wav, sr = self._apply_post(wav, sr, post_processing)
        yield from self._chunk_audio(wav, sr)

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
