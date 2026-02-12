"""Shared audio processing utilities."""
import os
import subprocess
import tempfile
import numpy as np
import scipy.signal
import soundfile as sf
from app.services.clearvoice_service import clearvoice_service

# Valid output sample rates
VALID_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}


def resample_audio(wav, sr, target_sr):
    """Resample audio to a target sample rate using scipy.signal.resample.

    Args:
        wav: numpy array of audio samples
        sr: current sample rate
        target_sr: desired sample rate

    Returns:
        (resampled_wav, target_sr)
    """
    if sr == target_sr:
        return wav, target_sr
    num_samples = round(len(wav) * target_sr / sr)
    resampled = scipy.signal.resample(wav, num_samples)
    return resampled.astype(np.float32), target_sr


def normalize_volume(wav, target_lufs=-16):
    """Peak-normalize audio with an approximate LUFS target.

    Uses peak normalization scaled to approximate the desired loudness.
    target_lufs should be negative (e.g., -16).

    Args:
        wav: numpy array of audio samples (float32)
        target_lufs: target loudness in LUFS (default -16)

    Returns:
        normalized wav array
    """
    peak = np.max(np.abs(wav))
    if peak < 1e-8:
        return wav

    # Map LUFS to a peak target: -16 LUFS ≈ 0.25 peak, -6 LUFS ≈ 0.80 peak
    # Linear scale: 10^(lufs/20) gives approximate peak target
    target_peak = min(10 ** (target_lufs / 20), 0.99)
    gain = target_peak / peak
    return (wav * gain).astype(np.float32)


def time_stretch(wav, sr, rate):
    """Time-stretch audio by a given rate factor.

    rate > 1.0 = faster (shorter), rate < 1.0 = slower (longer).
    Uses phase vocoder via scipy for basic stretching.

    Args:
        wav: numpy array of audio samples
        sr: sample rate
        rate: stretch factor (0.5 to 2.0)

    Returns:
        (stretched_wav, sr)
    """
    if abs(rate - 1.0) < 0.01:
        return wav, sr

    try:
        import librosa
        stretched = librosa.effects.time_stretch(wav, rate=rate)
        return stretched.astype(np.float32), sr
    except ImportError:
        # Fallback: resample trick (changes pitch too, but works without librosa)
        num_samples = round(len(wav) / rate)
        stretched = scipy.signal.resample(wav, num_samples)
        return stretched.astype(np.float32), sr


def pitch_shift(wav, sr, n_steps):
    """Shift pitch by n_steps semitones.

    Args:
        wav: numpy array of audio samples
        sr: sample rate
        n_steps: number of semitones to shift (-12 to +12)

    Returns:
        (shifted_wav, sr)
    """
    if abs(n_steps) < 0.01:
        return wav, sr

    try:
        import librosa
        shifted = librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps)
        return shifted.astype(np.float32), sr
    except ImportError:
        # Fallback: resample-based pitch shift (changes speed too)
        factor = 2 ** (n_steps / 12.0)
        intermediate = scipy.signal.resample(wav, round(len(wav) / factor))
        result = scipy.signal.resample(intermediate, len(wav))
        return result.astype(np.float32), sr


def apply_post_processing(wav, sr, options):
    """Apply a post-processing pipeline to audio.

    Applies in order: pitch → speed → normalize → resample.

    Args:
        wav: numpy array of audio samples (float32)
        sr: current sample rate
        options: dict with optional keys:
            - pitch_shift: semitones (-12 to +12)
            - speed: rate factor (0.5 to 2.0)
            - volume_normalize: bool or target LUFS (e.g., -16)
            - sample_rate: target sample rate

    Returns:
        (processed_wav, output_sr)
    """
    if not options:
        return wav, sr

    # 1. Pitch shift
    n_steps = options.get('pitch_shift', 0)
    if n_steps and abs(float(n_steps)) >= 0.01:
        wav, sr = pitch_shift(wav, sr, float(n_steps))

    # 2. Speed / time stretch
    speed = options.get('speed', 1.0)
    if speed and abs(float(speed) - 1.0) >= 0.01:
        wav, sr = time_stretch(wav, sr, float(speed))

    # 3. Volume normalization
    vol = options.get('volume_normalize')
    if vol is not None and vol is not False:
        target_lufs = float(vol) if isinstance(vol, (int, float)) and vol < 0 else -16
        wav = normalize_volume(wav, target_lufs)

    # 4. Resample
    target_sr = options.get('sample_rate')
    if target_sr and int(target_sr) != sr and int(target_sr) in VALID_SAMPLE_RATES:
        wav, sr = resample_audio(wav, sr, int(target_sr))

    return wav, sr


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
