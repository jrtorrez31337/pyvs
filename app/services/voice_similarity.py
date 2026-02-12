"""Voice similarity scoring service.

Compares speaker embeddings from reference audio and generated audio
to produce a similarity score.
"""
import threading
import numpy as np


class VoiceSimilarityService:
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
        self._model_lock = threading.Lock()
        self.encoder = None
        self._initialized = True

    def _load_encoder(self):
        """Lazy-load a speaker encoder for embedding extraction."""
        if self._model_loaded:
            return
        with self._model_lock:
            if self._model_loaded:
                return
            try:
                from resemblyzer import VoiceEncoder
                print("Loading voice encoder for similarity scoring...")
                self.encoder = VoiceEncoder()
                self._model_loaded = True
                print("Voice encoder loaded.")
            except ImportError:
                # Fall back to basic spectral comparison
                self._model_loaded = True
                print("resemblyzer not available, using spectral similarity fallback")

    def extract_embedding(self, audio_path):
        """Extract a speaker embedding from an audio file.

        Returns numpy array embedding, or None if extraction fails.
        """
        self._load_encoder()

        if self.encoder is not None:
            from resemblyzer import preprocess_wav
            import soundfile as sf
            wav, sr = sf.read(audio_path)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            processed = preprocess_wav(wav, source_sr=sr)
            return self.encoder.embed_utterance(processed)
        else:
            # Spectral fallback: use MFCCs
            return self._spectral_embedding(audio_path)

    def _spectral_embedding(self, audio_path):
        """Fallback embedding using spectral features."""
        import soundfile as sf
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        try:
            import librosa
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)
            return np.mean(mfcc, axis=1)
        except ImportError:
            # Very basic: just use statistical features
            return np.array([
                np.mean(wav), np.std(wav), np.max(np.abs(wav)),
                float(len(wav)) / sr,
            ])

    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings.

        Returns float in range [0, 100] as a percentage.
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        cosine_sim = dot / (norm1 * norm2)
        # Map from [-1, 1] to [0, 100]
        return round(max(0.0, min(100.0, (cosine_sim + 1) * 50)), 1)

    def compare_files(self, reference_path, generated_path):
        """Compare two audio files and return similarity score.

        Returns dict with score (0-100%) and details.
        """
        emb1 = self.extract_embedding(reference_path)
        emb2 = self.extract_embedding(generated_path)
        score = self.compute_similarity(emb1, emb2)

        return {
            'score': score,
            'reference_path': reference_path,
            'generated_path': generated_path,
        }


voice_similarity_service = VoiceSimilarityService()
