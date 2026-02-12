"""Shared audio processing utilities."""
import os
import subprocess
import tempfile
import numpy as np
import scipy.signal
import soundfile as sf
from app.services.clearvoice_service import clearvoice_service


def convert_to_wav(input_path):
    """Convert any audio format to WAV using ffmpeg.

    Returns path to a temporary WAV file. Caller must delete it.
    Returns the original path if already WAV/readable by soundfile.
    """
    # Try reading directly first
    try:
        sf.read(input_path, frames=1)
        return input_path
    except Exception:
        pass

    # Convert via ffmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        wav_path = tmp.name

    try:
        subprocess.run(
            ['ffmpeg', '-i', input_path, '-ar', '24000', '-ac', '1', '-y', wav_path],
            capture_output=True, check=True,
        )
        return wav_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        raise RuntimeError(f"Failed to convert audio to WAV: {e}") from e


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

    # MossFormer2_SE_48K outputs at 48kHz regardless of input rate.
    # Resample back to original sample rate if they differ.
    CLEARVOICE_OUTPUT_SR = 48000
    if isinstance(enhanced, np.ndarray):
        if len(enhanced.shape) > 1:
            enhanced = enhanced[0]  # Take first channel/batch
        if sample_rate != CLEARVOICE_OUTPUT_SR:
            num_samples = round(len(enhanced) * sample_rate / CLEARVOICE_OUTPUT_SR)
            enhanced = scipy.signal.resample(enhanced, num_samples)

    return enhanced.astype(np.float32)


def reduce_noise_file(audio_path):
    """Apply speech enhancement to an audio file.

    Returns path to temporary file with enhanced audio.
    Handles non-WAV formats (e.g. WebM) by converting first.
    """
    wav_path = convert_to_wav(audio_path)
    try:
        data, sr = sf.read(wav_path)
        reduced = reduce_noise(data, sr)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            sf.write(tmp.name, reduced, sr)
            return tmp.name
    finally:
        if wav_path != audio_path and os.path.exists(wav_path):
            os.unlink(wav_path)
