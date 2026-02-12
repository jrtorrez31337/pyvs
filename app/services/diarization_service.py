"""Speaker diarization service using pyannote.audio."""
import os
import threading


class DiarizationService:
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
        self._available = None
        self.pipeline = None
        self._initialized = True

    @property
    def available(self):
        """Check if pyannote and HF_TOKEN are available."""
        if self._available is None:
            try:
                import pyannote.audio  # noqa: F401
                self._available = bool(os.environ.get('HF_TOKEN'))
            except ImportError:
                self._available = False
        return self._available

    def load_model(self):
        """Load the pyannote speaker diarization pipeline."""
        if self._model_loaded:
            return
        if not self.available:
            raise RuntimeError(
                "Diarization requires pyannote-audio and HF_TOKEN environment variable"
            )

        with self._model_lock:
            if self._model_loaded:
                return

            from pyannote.audio import Pipeline

            hf_token = os.environ.get('HF_TOKEN')
            print("Loading pyannote speaker-diarization-3.1...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            self._model_loaded = True
            print("Diarization pipeline loaded successfully!")

    def diarize(self, audio_path):
        """Run speaker diarization on an audio file.

        Returns list of segments: [{speaker, start, end}]
        """
        self.load_model()
        with self._model_lock:
            diarization = self.pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'speaker': speaker,
                'start': round(turn.start, 3),
                'end': round(turn.end, 3),
            })

        return segments

    def merge_with_words(self, diarization_segments, word_timestamps):
        """Merge diarization segments with word timestamps to produce
        speaker-attributed transcript.

        Returns list of: [{speaker, start, end, text}]
        """
        if not diarization_segments or not word_timestamps:
            return []

        # Assign each word to the speaker whose segment overlaps most
        speaker_texts = []
        current_speaker = None
        current_words = []
        current_start = None
        current_end = None

        for word_info in word_timestamps:
            word_mid = (word_info['start'] + word_info['end']) / 2.0
            # Find best matching speaker segment
            best_speaker = None
            best_overlap = 0
            for seg in diarization_segments:
                if seg['start'] <= word_mid <= seg['end']:
                    overlap = min(word_info['end'], seg['end']) - max(word_info['start'], seg['start'])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = seg['speaker']

            if best_speaker is None:
                best_speaker = current_speaker or 'UNKNOWN'

            if best_speaker != current_speaker:
                if current_words:
                    speaker_texts.append({
                        'speaker': current_speaker,
                        'start': current_start,
                        'end': current_end,
                        'text': ' '.join(current_words),
                    })
                current_speaker = best_speaker
                current_words = []
                current_start = word_info['start']

            current_words.append(word_info['word'])
            current_end = word_info['end']

        if current_words:
            speaker_texts.append({
                'speaker': current_speaker,
                'start': current_start,
                'end': current_end,
                'text': ' '.join(current_words),
            })

        return speaker_texts


# Singleton instance
diarization_service = DiarizationService()
