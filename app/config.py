"""Application constants and configuration."""
import re

# Audio
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1
STREAMING_CHUNK_BYTES = 4800  # ~100ms at 24kHz int16 mono

# Limits
MAX_TEXT_LENGTH = 5000
MAX_INSTRUCT_LENGTH = 500
MAX_UPLOAD_SIZE_MB = 50
MAX_HISTORY_ITEMS = 50
MAX_CACHED_AUDIO = 100
AUDIO_CACHE_TTL_SECONDS = 3600  # 1 hour
UPLOAD_MAX_AGE_SECONDS = 86400  # 24 hours

# Speech enhancement (ClearerVoice)
CLEARVOICE_MODEL = "MossFormer2_SE_48K"

# Model paths
TTS_MODEL_BASE_PATH = "/data/models/Qwen"
TTS_FAST_MODEL_ENABLED = True  # Load 0.6B models for fast mode
STT_MODEL_CACHE_PATH = "/data/models/whisper"
STT_MODEL_NAME = "large-v3-turbo"
STT_DEVICE_INDEX = 1  # GPU index for Whisper STT

# Chatterbox TTS
CHATTERBOX_DEVICE = "cuda:0"
CHATTERBOX_LANGUAGES = [
    ("ar", "Arabic"), ("da", "Danish"), ("de", "German"), ("el", "Greek"),
    ("en", "English"), ("es", "Spanish"), ("fi", "Finnish"), ("fr", "French"),
    ("he", "Hebrew"), ("hi", "Hindi"), ("it", "Italian"), ("ja", "Japanese"),
    ("ko", "Korean"), ("ms", "Malay"), ("nl", "Dutch"), ("no", "Norwegian"),
    ("pl", "Polish"), ("pt", "Portuguese"), ("ru", "Russian"), ("sv", "Swedish"),
    ("sw", "Swahili"), ("tr", "Turkish"), ("zh", "Chinese"),
]
MAX_EXAGGERATION = 2.0

# Audio ID validation (must be UUID format)
AUDIO_ID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def is_valid_audio_id(audio_id: str) -> bool:
    """Validate that an audio ID is a safe UUID string."""
    return bool(AUDIO_ID_PATTERN.match(audio_id))
