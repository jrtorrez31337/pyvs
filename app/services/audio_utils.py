"""Shared audio processing utilities."""
import os
import tempfile
import numpy as np
import soundfile as sf
from app.services.clearvoice_service import clearvoice_service


def reduce_noise(audio_data, sample_rate):
    """Apply speech enhancement to audio data array.

    Uses ClearerVoice MossFormer2 for superior noise removal.
    Writes to temp file, processes, reads back to maintain sample rate.
    """
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Write to temp file for ClearerVoice processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_in:
        tmp_path = tmp_in.name
        sf.write(tmp_path, audio_data, sample_rate)

    try:
        enhanced = clearvoice_service.enhance_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    # ClearerVoice may output at different sample rate (48kHz for MossFormer2_SE_48K)
    # Resample back to original if needed
    if isinstance(enhanced, np.ndarray):
        if len(enhanced.shape) > 1:
            enhanced = enhanced[0]  # Take first channel/batch
        # If length differs significantly, resample
        expected_len = len(audio_data)
        if abs(len(enhanced) - expected_len) > expected_len * 0.1:
            enhanced = np.interp(
                np.linspace(0, 1, expected_len),
                np.linspace(0, 1, len(enhanced)),
                enhanced,
            )

    return enhanced.astype(np.float32)


def reduce_noise_file(audio_path):
    """Apply speech enhancement to an audio file.

    Returns path to temporary file with enhanced audio.
    """
    data, sr = sf.read(audio_path)
    reduced = reduce_noise(data, sr)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        sf.write(tmp.name, reduced, sr)
        return tmp.name
