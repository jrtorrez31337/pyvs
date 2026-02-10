"""Shared audio processing utilities."""
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
from app.config import NOISE_REDUCE_STATIONARY, NOISE_REDUCE_PROP_DECREASE


def reduce_noise(audio_data, sample_rate):
    """Apply noise reduction to audio data array.

    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate in Hz

    Returns:
        Noise-reduced audio as float32 numpy array
    """
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    reduced = nr.reduce_noise(
        y=audio_data,
        sr=sample_rate,
        stationary=NOISE_REDUCE_STATIONARY,
        prop_decrease=NOISE_REDUCE_PROP_DECREASE,
    )
    return reduced.astype(np.float32)


def reduce_noise_file(audio_path):
    """Apply noise reduction to an audio file, return path to cleaned file.

    Args:
        audio_path: path to input audio file

    Returns:
        Path to temporary file with noise-reduced audio
    """
    data, sr = sf.read(audio_path)
    reduced = reduce_noise(data, sr)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        sf.write(tmp.name, reduced, sr)
        return tmp.name
