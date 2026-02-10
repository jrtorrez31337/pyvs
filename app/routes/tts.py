import io
import os
import time
import uuid
import struct
import threading
from collections import OrderedDict
import numpy as np
import soundfile as sf
from flask import Blueprint, request, jsonify, Response, current_app
from app.services.tts_service import tts_service
from app.config import MAX_CACHED_AUDIO, AUDIO_CACHE_TTL_SECONDS, MAX_TEXT_LENGTH, MAX_INSTRUCT_LENGTH, is_valid_audio_id


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16):
    """Create a WAV header for streaming (size set to max)."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    # Use 0xFFFFFFFF for unknown size (streaming)
    data_size = 0xFFFFFFFF - 36

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        0xFFFFFFFF,  # File size (unknown for streaming)
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1 size
        1,   # Audio format (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )
    return header

bp = Blueprint('tts', __name__, url_prefix='/api/tts')

# Store generated audio for download (TTL-evicting cache)
_generated_audio = OrderedDict()  # job_id -> (wav, sr, timestamp)
_audio_cache_lock = threading.Lock()


def _cache_audio(job_id, wav, sr):
    """Store audio with TTL eviction."""
    now = time.time()
    with _audio_cache_lock:
        _generated_audio[job_id] = (wav, sr, now)
        expired = [
            k for k, (_, _, ts) in _generated_audio.items()
            if now - ts > AUDIO_CACHE_TTL_SECONDS
        ]
        for k in expired:
            del _generated_audio[k]
        while len(_generated_audio) > MAX_CACHED_AUDIO:
            _generated_audio.popitem(last=False)


def get_cached_audio(job_id):
    """Retrieve cached audio, or None if expired/missing."""
    with _audio_cache_lock:
        entry = _generated_audio.get(job_id)
        if entry is None:
            return None
        wav, sr, ts = entry
        if time.time() - ts > AUDIO_CACHE_TTL_SECONDS:
            del _generated_audio[job_id]
            return None
        return wav, sr


def _validate_text(text, field_name="text"):
    if not text:
        return f"{field_name} is required"
    if len(text) > MAX_TEXT_LENGTH:
        return f"{field_name} exceeds maximum length of {MAX_TEXT_LENGTH} characters"
    return None


def _validate_instruct(instruct):
    if instruct and len(instruct) > MAX_INSTRUCT_LENGTH:
        return f"Instruction exceeds maximum length of {MAX_INSTRUCT_LENGTH} characters"
    return None


@bp.route('/clone', methods=['POST'])
def tts_clone():
    """Generate speech using voice cloning with one or more reference samples"""
    data = request.get_json()

    text = data.get('text')
    language = data.get('language', 'English')

    # Support both single and multiple reference samples
    ref_audio_ids = data.get('ref_audio_ids') or []
    ref_texts = data.get('ref_texts') or []

    # Backwards compatibility: single sample
    if not ref_audio_ids and data.get('ref_audio_id'):
        ref_audio_ids = [data.get('ref_audio_id')]
        ref_texts = [data.get('ref_text')]

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    if not ref_audio_ids:
        return jsonify({'error': 'At least one reference audio is required'}), 400

    for audio_id in ref_audio_ids:
        if not is_valid_audio_id(audio_id):
            return jsonify({'error': f'Invalid audio ID: {audio_id}'}), 400

    # Get reference audio paths
    ref_audio_paths = []
    for audio_id in ref_audio_ids:
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")
        if not os.path.exists(path):
            return jsonify({'error': f'Reference audio not found: {audio_id}'}), 404
        ref_audio_paths.append(path)

    # Normalize ref_texts to match length
    while len(ref_texts) < len(ref_audio_paths):
        ref_texts.append(None)

    fast = data.get('fast', False)

    try:
        wav, sr = tts_service.generate_clone(text, language, ref_audio_paths, ref_texts, fast=fast)

        # Store for download
        job_id = str(uuid.uuid4())
        _cache_audio(job_id, wav, sr)

        # Return as streaming audio
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)

        return Response(
            buffer.read(),
            mimetype='audio/wav',
            headers={
                'X-Job-Id': job_id,
                'Content-Disposition': 'inline; filename="output.wav"'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/clone/stream', methods=['POST'])
def tts_clone_stream():
    """Stream speech generation using voice cloning."""
    data = request.get_json()

    text = data.get('text')
    language = data.get('language', 'English')
    ref_audio_ids = data.get('ref_audio_ids') or []
    ref_texts = data.get('ref_texts') or []

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    if not ref_audio_ids:
        return jsonify({'error': 'At least one reference audio is required'}), 400

    for audio_id in ref_audio_ids:
        if not is_valid_audio_id(audio_id):
            return jsonify({'error': f'Invalid audio ID: {audio_id}'}), 400

    ref_audio_paths = []
    for audio_id in ref_audio_ids:
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")
        if not os.path.exists(path):
            return jsonify({'error': f'Reference audio not found: {audio_id}'}), 404
        ref_audio_paths.append(path)

    while len(ref_texts) < len(ref_audio_paths):
        ref_texts.append(None)

    fast = data.get('fast', False)

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in tts_service.generate_clone_streaming(
                text, language, ref_audio_paths, ref_texts, fast=fast
            ):
                if not header_sent:
                    sample_rate = sr
                    yield create_wav_header(sr)
                    header_sent = True

                # Convert float32 to int16
                audio_int16 = (chunk * 32767).astype(np.int16)
                all_chunks.append(chunk)
                yield audio_int16.tobytes()

            # Store complete audio for download
            if all_chunks:
                full_audio = np.concatenate(all_chunks)
                job_id = str(uuid.uuid4())
                _cache_audio(job_id, full_audio, sample_rate)
                # Send job ID as final chunk marker (won't be played as audio)
                yield f"<!--JOB_ID:{job_id}-->".encode()

        except Exception as e:
            print(f"Streaming error: {e}")

    return Response(
        generate(),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
        }
    )


@bp.route('/custom', methods=['POST'])
def tts_custom():
    """Generate speech using custom voice preset"""
    data = request.get_json()

    text = data.get('text')
    language = data.get('language', 'English')
    speaker = data.get('speaker')
    instruct = data.get('instruct')
    fast = data.get('fast', False)

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    err = _validate_instruct(instruct)
    if err:
        return jsonify({'error': err}), 400
    if not speaker:
        return jsonify({'error': 'Speaker is required'}), 400

    try:
        wav, sr = tts_service.generate_custom(text, language, speaker, instruct, fast=fast)

        # Store for download
        job_id = str(uuid.uuid4())
        _cache_audio(job_id, wav, sr)

        # Return as audio
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)

        return Response(
            buffer.read(),
            mimetype='audio/wav',
            headers={
                'X-Job-Id': job_id,
                'Content-Disposition': 'inline; filename="output.wav"'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/custom/stream', methods=['POST'])
def tts_custom_stream():
    """Stream speech generation using custom voice."""
    data = request.get_json()

    text = data.get('text')
    language = data.get('language', 'English')
    speaker = data.get('speaker')
    instruct = data.get('instruct')
    fast = data.get('fast', False)

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    err = _validate_instruct(instruct)
    if err:
        return jsonify({'error': err}), 400
    if not speaker:
        return jsonify({'error': 'Speaker is required'}), 400

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in tts_service.generate_custom_streaming(
                text, language, speaker, instruct, fast=fast
            ):
                if not header_sent:
                    sample_rate = sr
                    yield create_wav_header(sr)
                    header_sent = True

                audio_int16 = (chunk * 32767).astype(np.int16)
                all_chunks.append(chunk)
                yield audio_int16.tobytes()

            if all_chunks:
                full_audio = np.concatenate(all_chunks)
                job_id = str(uuid.uuid4())
                _cache_audio(job_id, full_audio, sample_rate)
                yield f"<!--JOB_ID:{job_id}-->".encode()

        except Exception as e:
            print(f"Streaming error: {e}")

    return Response(
        generate(),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
        }
    )


@bp.route('/design', methods=['POST'])
def tts_design():
    """Generate speech using voice design"""
    data = request.get_json()

    text = data.get('text')
    language = data.get('language', 'English')
    instruct = data.get('instruct')

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    err = _validate_instruct(instruct)
    if err:
        return jsonify({'error': err}), 400
    if not instruct:
        return jsonify({'error': 'Voice design instruction is required'}), 400

    try:
        wav, sr = tts_service.generate_design(text, language, instruct)

        # Store for download
        job_id = str(uuid.uuid4())
        _cache_audio(job_id, wav, sr)

        # Return as audio
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)

        return Response(
            buffer.read(),
            mimetype='audio/wav',
            headers={
                'X-Job-Id': job_id,
                'Content-Disposition': 'inline; filename="output.wav"'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/design/stream', methods=['POST'])
def tts_design_stream():
    """Stream speech generation using voice design."""
    data = request.get_json()

    text = data.get('text')
    language = data.get('language', 'English')
    instruct = data.get('instruct')

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    err = _validate_instruct(instruct)
    if err:
        return jsonify({'error': err}), 400
    if not instruct:
        return jsonify({'error': 'Voice design instruction is required'}), 400

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in tts_service.generate_design_streaming(
                text, language, instruct
            ):
                if not header_sent:
                    sample_rate = sr
                    yield create_wav_header(sr)
                    header_sent = True

                audio_int16 = (chunk * 32767).astype(np.int16)
                all_chunks.append(chunk)
                yield audio_int16.tobytes()

            if all_chunks:
                full_audio = np.concatenate(all_chunks)
                job_id = str(uuid.uuid4())
                _cache_audio(job_id, full_audio, sample_rate)
                yield f"<!--JOB_ID:{job_id}-->".encode()

        except Exception as e:
            print(f"Streaming error: {e}")

    return Response(
        generate(),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
        }
    )


@bp.route('/speakers', methods=['GET'])
def get_speakers():
    """Get list of available speakers for CustomVoice"""
    speakers = [
        {"name": "Vivian", "description": "Bright, slightly edgy young female voice", "language": "Chinese"},
        {"name": "Serena", "description": "Warm, gentle young female voice", "language": "Chinese"},
        {"name": "Uncle_Fu", "description": "Seasoned male voice with a low, mellow timbre", "language": "Chinese"},
        {"name": "Dylan", "description": "Youthful Beijing male voice with clear, natural timbre", "language": "Chinese (Beijing)"},
        {"name": "Eric", "description": "Lively Chengdu male voice with slightly husky brightness", "language": "Chinese (Sichuan)"},
        {"name": "Ryan", "description": "Dynamic male voice with strong rhythmic drive", "language": "English"},
        {"name": "Aiden", "description": "Sunny American male voice with clear midrange", "language": "English"},
        {"name": "Ono_Anna", "description": "Playful Japanese female voice with light, nimble timbre", "language": "Japanese"},
        {"name": "Sohee", "description": "Warm Korean female voice with rich emotion", "language": "Korean"},
    ]
    return jsonify(speakers)


@bp.route('/languages', methods=['GET'])
def get_languages():
    """Get list of supported languages"""
    languages = tts_service.get_supported_languages()
    return jsonify(languages)


@bp.route('/download/<job_id>', methods=['GET'])
def download_audio(job_id):
    """Download generated audio by job ID"""
    if not is_valid_audio_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400

    entry = get_cached_audio(job_id)
    if entry is None:
        return jsonify({'error': 'Audio not found or expired'}), 404

    wav, sr = entry

    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='WAV')
    buffer.seek(0)

    return Response(
        buffer.read(),
        mimetype='audio/wav',
        headers={
            'Content-Disposition': f'attachment; filename="generated_{job_id[:8]}.wav"'
        }
    )
