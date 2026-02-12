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
from app.services.audio_utils import VALID_SAMPLE_RATES


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


def _extract_inference_params(data):
    """Extract and validate inference params from request JSON.

    Returns (params_dict, error_string). error_string is None on success.
    """
    params = {}
    err = None

    if 'temperature' in data and data['temperature'] is not None:
        try:
            v = float(data['temperature'])
            if not (0.1 <= v <= 2.0):
                return None, 'temperature must be between 0.1 and 2.0'
            params['temperature'] = v
        except (TypeError, ValueError):
            return None, 'temperature must be numeric'

    if 'top_k' in data and data['top_k'] is not None:
        try:
            v = int(data['top_k'])
            if not (1 <= v <= 200):
                return None, 'top_k must be between 1 and 200'
            params['top_k'] = v
        except (TypeError, ValueError):
            return None, 'top_k must be an integer'

    if 'top_p' in data and data['top_p'] is not None:
        try:
            v = float(data['top_p'])
            if not (0.0 <= v <= 1.0):
                return None, 'top_p must be between 0.0 and 1.0'
            params['top_p'] = v
        except (TypeError, ValueError):
            return None, 'top_p must be numeric'

    if 'repetition_penalty' in data and data['repetition_penalty'] is not None:
        try:
            v = float(data['repetition_penalty'])
            if not (1.0 <= v <= 2.0):
                return None, 'repetition_penalty must be between 1.0 and 2.0'
            params['repetition_penalty'] = v
        except (TypeError, ValueError):
            return None, 'repetition_penalty must be numeric'

    return params or None, err


def _extract_post_processing(data):
    """Extract and validate post-processing params from request JSON.

    Returns (options_dict, error_string). error_string is None on success.
    """
    options = {}

    if 'pitch_shift' in data and data['pitch_shift'] is not None:
        try:
            v = float(data['pitch_shift'])
            if not (-12 <= v <= 12):
                return None, 'pitch_shift must be between -12 and +12'
            if abs(v) >= 0.01:
                options['pitch_shift'] = v
        except (TypeError, ValueError):
            return None, 'pitch_shift must be numeric'

    if 'speed' in data and data['speed'] is not None:
        try:
            v = float(data['speed'])
            if not (0.5 <= v <= 2.0):
                return None, 'speed must be between 0.5 and 2.0'
            if abs(v - 1.0) >= 0.01:
                options['speed'] = v
        except (TypeError, ValueError):
            return None, 'speed must be numeric'

    if 'volume_normalize' in data and data['volume_normalize'] is not None:
        v = data['volume_normalize']
        if isinstance(v, bool):
            if v:
                options['volume_normalize'] = -16
        elif isinstance(v, (int, float)):
            if not (-24 <= float(v) <= -6):
                return None, 'volume_normalize LUFS must be between -24 and -6'
            options['volume_normalize'] = float(v)

    if 'sample_rate' in data and data['sample_rate'] is not None:
        try:
            v = int(data['sample_rate'])
            if v not in VALID_SAMPLE_RATES:
                return None, f'sample_rate must be one of {sorted(VALID_SAMPLE_RATES)}'
            options['sample_rate'] = v
        except (TypeError, ValueError):
            return None, 'sample_rate must be an integer'

    return options or None, None


def _resolve_reference_audio(data, required=False):
    """Resolve reference audio ids/texts/weights from request JSON.

    Returns (ref_audio_paths, ref_texts, error) tuple.
    Also stores ref_weights in data['_ref_weights'] for downstream use.
    """
    ref_audio_ids = data.get('ref_audio_ids') or []
    ref_texts = data.get('ref_texts') or []
    ref_weights = data.get('ref_weights') or []
    if not isinstance(ref_audio_ids, list):
        ref_audio_ids = []
    if not isinstance(ref_texts, list):
        ref_texts = []

    # Backwards compatibility: single sample fields
    if not ref_audio_ids and data.get('ref_audio_id'):
        ref_audio_ids = [data.get('ref_audio_id')]
        ref_texts = [data.get('ref_text')]

    if required and not ref_audio_ids:
        return None, None, ('At least one reference audio is required', 400)

    for audio_id in ref_audio_ids:
        if not is_valid_audio_id(audio_id):
            return None, None, (f'Invalid audio ID: {audio_id}', 400)

    ref_audio_paths = []
    for audio_id in ref_audio_ids:
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")
        if not os.path.exists(path):
            return None, None, (f'Reference audio not found: {audio_id}', 404)
        ref_audio_paths.append(path)

    while len(ref_texts) < len(ref_audio_paths):
        ref_texts.append(None)

    # Store weights for downstream use
    if isinstance(ref_weights, list) and len(ref_weights) == len(ref_audio_paths):
        try:
            data['_ref_weights'] = [float(w) for w in ref_weights]
        except (TypeError, ValueError):
            data['_ref_weights'] = None
    else:
        data['_ref_weights'] = None

    return ref_audio_paths, ref_texts, None


@bp.route('/clone', methods=['POST'])
def tts_clone():
    """Generate speech using voice cloning with one or more reference samples"""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    text = data.get('text')
    language = data.get('language', 'English')

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    ref_audio_paths, ref_texts, ref_err = _resolve_reference_audio(data, required=True)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    fast = data.get('fast', False)

    inference_params, inf_err = _extract_inference_params(data)
    if inf_err:
        return jsonify({'error': inf_err}), 400
    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    try:
        wav, sr = tts_service.generate_clone(
            text, language, ref_audio_paths, ref_texts, fast=fast,
            inference_params=inference_params, post_processing=post_processing,
        )

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
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    text = data.get('text')
    language = data.get('language', 'English')
    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
    ref_audio_paths, ref_texts, ref_err = _resolve_reference_audio(data, required=True)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    fast = data.get('fast', False)

    inference_params, inf_err = _extract_inference_params(data)
    if inf_err:
        return jsonify({'error': inf_err}), 400
    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in tts_service.generate_clone_streaming(
                text, language, ref_audio_paths, ref_texts, fast=fast,
                inference_params=inference_params, post_processing=post_processing,
            ):
                if not header_sent:
                    sample_rate = sr
                    yield create_wav_header(sr)
                    header_sent = True

                # Convert float32 to int16
                audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
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
            print(f"Streaming clone error: {e}")
            yield f"<!--ERROR:{str(e)}-->".encode()

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
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

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
    ref_audio_paths, ref_texts, ref_err = _resolve_reference_audio(data, required=False)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    inference_params, inf_err = _extract_inference_params(data)
    if inf_err:
        return jsonify({'error': inf_err}), 400
    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    try:
        wav, sr = tts_service.generate_custom(
            text,
            language,
            speaker,
            instruct,
            fast=fast,
            ref_audio_paths=ref_audio_paths,
            ref_texts=ref_texts,
            ref_weights=data.get('_ref_weights'),
            inference_params=inference_params,
            post_processing=post_processing,
        )

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
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

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
    ref_audio_paths, ref_texts, ref_err = _resolve_reference_audio(data, required=False)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    inference_params, inf_err = _extract_inference_params(data)
    if inf_err:
        return jsonify({'error': inf_err}), 400
    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in tts_service.generate_custom_streaming(
                text,
                language,
                speaker,
                instruct,
                fast=fast,
                ref_audio_paths=ref_audio_paths,
                ref_texts=ref_texts,
                ref_weights=data.get('_ref_weights'),
                inference_params=inference_params,
                post_processing=post_processing,
            ):
                if not header_sent:
                    sample_rate = sr
                    yield create_wav_header(sr)
                    header_sent = True

                audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
                all_chunks.append(chunk)
                yield audio_int16.tobytes()

            if all_chunks:
                full_audio = np.concatenate(all_chunks)
                job_id = str(uuid.uuid4())
                _cache_audio(job_id, full_audio, sample_rate)
                yield f"<!--JOB_ID:{job_id}-->".encode()

        except Exception as e:
            print(f"Streaming custom error: {e}")
            yield f"<!--ERROR:{str(e)}-->".encode()

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
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

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
    ref_audio_paths, ref_texts, ref_err = _resolve_reference_audio(data, required=False)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    inference_params, inf_err = _extract_inference_params(data)
    if inf_err:
        return jsonify({'error': inf_err}), 400
    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    try:
        wav, sr = tts_service.generate_design(
            text,
            language,
            instruct,
            ref_audio_paths=ref_audio_paths,
            ref_texts=ref_texts,
            ref_weights=data.get('_ref_weights'),
            inference_params=inference_params,
            post_processing=post_processing,
        )

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
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

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
    ref_audio_paths, ref_texts, ref_err = _resolve_reference_audio(data, required=False)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    inference_params, inf_err = _extract_inference_params(data)
    if inf_err:
        return jsonify({'error': inf_err}), 400
    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in tts_service.generate_design_streaming(
                text,
                language,
                instruct,
                ref_audio_paths=ref_audio_paths,
                ref_texts=ref_texts,
                ref_weights=data.get('_ref_weights'),
                inference_params=inference_params,
                post_processing=post_processing,
            ):
                if not header_sent:
                    sample_rate = sr
                    yield create_wav_header(sr)
                    header_sent = True

                audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
                all_chunks.append(chunk)
                yield audio_int16.tobytes()

            if all_chunks:
                full_audio = np.concatenate(all_chunks)
                job_id = str(uuid.uuid4())
                _cache_audio(job_id, full_audio, sample_rate)
                yield f"<!--JOB_ID:{job_id}-->".encode()

        except Exception as e:
            print(f"Streaming design error: {e}")
            yield f"<!--ERROR:{str(e)}-->".encode()

    return Response(
        generate(),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
        }
    )


@bp.route('/dialogue', methods=['POST'])
def tts_dialogue():
    """Generate multi-speaker dialogue audio.

    Accepts either:
      - {segments: [{text, speaker, language, instruct}], ...} for structured input
      - {ssml: "<speak>...</speak>"} for SSML input

    Returns a single combined audio file with all segments concatenated.
    """
    from app.services.ssml_parser import parse_ssml
    from app.services.audio_utils import apply_post_processing

    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    # Parse input: either SSML or explicit segments
    ssml_text = data.get('ssml')
    segments = data.get('segments')

    if ssml_text:
        try:
            parsed = parse_ssml(ssml_text)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        segments = parsed
    elif segments:
        if not isinstance(segments, list):
            return jsonify({'error': 'segments must be a list'}), 400
    else:
        return jsonify({'error': 'Either ssml or segments is required'}), 400

    # Global post-processing and inference params
    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400
    inference_params, inf_err = _extract_inference_params(data)
    if inf_err:
        return jsonify({'error': inf_err}), 400

    silence_gap_ms = data.get('silence_gap_ms', 300)
    default_language = data.get('language', 'English')
    default_speaker = data.get('speaker', 'Ryan')

    try:
        all_audio = []
        sample_rate = None

        for seg in segments:
            seg_type = seg.get('type', 'speech')

            if seg_type == 'break':
                # Insert silence
                duration_ms = seg.get('duration_ms', 500)
                if sample_rate:
                    silence_samples = int(sample_rate * duration_ms / 1000)
                    all_audio.append(np.zeros(silence_samples, dtype=np.float32))
                continue

            text = seg.get('text', '').strip()
            if not text:
                continue

            err = _validate_text(text)
            if err:
                return jsonify({'error': f'Segment error: {err}'}), 400

            speaker = seg.get('speaker') or default_speaker
            language = seg.get('language') or default_language
            instruct = seg.get('instruct')

            # Per-segment prosody overrides
            seg_prosody = seg.get('prosody', {})

            # Generate with custom voice model (supports speaker + instruct)
            wav, sr = tts_service.generate_custom(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                inference_params=inference_params,
                post_processing=seg_prosody if seg_prosody else None,
            )

            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                from app.services.audio_utils import resample_audio
                wav, _ = resample_audio(wav, sr, sample_rate)

            all_audio.append(wav)

            # Add silence gap between segments
            if silence_gap_ms > 0:
                gap_samples = int(sample_rate * silence_gap_ms / 1000)
                all_audio.append(np.zeros(gap_samples, dtype=np.float32))

        if not all_audio:
            return jsonify({'error': 'No audio generated'}), 400

        # Remove trailing silence gap
        if len(all_audio) > 1 and np.all(all_audio[-1] == 0):
            all_audio.pop()

        combined = np.concatenate(all_audio)

        # Apply global post-processing
        if post_processing:
            combined, sample_rate = apply_post_processing(combined, sample_rate, post_processing)

        job_id = str(uuid.uuid4())
        _cache_audio(job_id, combined, sample_rate)

        buffer = io.BytesIO()
        sf.write(buffer, combined, sample_rate, format='WAV')
        buffer.seek(0)

        return Response(
            buffer.read(),
            mimetype='audio/wav',
            headers={
                'X-Job-Id': job_id,
                'Content-Disposition': 'inline; filename="dialogue.wav"'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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


@bp.route('/similarity', methods=['POST'])
def tts_similarity():
    """Compare voice similarity between reference audio and generated audio.

    Accepts: {ref_audio_id: str, generated_audio_id: str}
    Returns: {score: float (0-100)}
    """
    from app.services.voice_similarity import voice_similarity_service

    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    ref_audio_id = data.get('ref_audio_id')
    generated_job_id = data.get('generated_audio_id')

    if not ref_audio_id or not generated_job_id:
        return jsonify({'error': 'ref_audio_id and generated_audio_id are required'}), 400

    if not is_valid_audio_id(ref_audio_id):
        return jsonify({'error': 'Invalid reference audio ID'}), 400

    ref_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{ref_audio_id}.wav")
    if not os.path.exists(ref_path):
        return jsonify({'error': 'Reference audio not found'}), 404

    # Get generated audio from cache and write to temp file
    entry = get_cached_audio(generated_job_id)
    if entry is None:
        return jsonify({'error': 'Generated audio not found or expired'}), 404

    import tempfile
    wav, sr = entry
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            sf.write(tmp.name, wav, sr)
            gen_path = tmp.name

        result = voice_similarity_service.compare_files(ref_path, gen_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'gen_path' in locals() and os.path.exists(gen_path):
            os.unlink(gen_path)


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
