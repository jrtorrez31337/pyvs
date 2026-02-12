import io
import os
import struct
import uuid
import zipfile
import numpy as np
import soundfile as sf
from flask import Blueprint, request, jsonify, Response, current_app
from app.services.chatterbox_service import chatterbox_service
from app.config import MAX_EXAGGERATION, CHATTERBOX_LANGUAGES, is_valid_audio_id
from app.routes.tts import _cache_audio, _validate_text, _extract_post_processing, create_wav_header

bp = Blueprint('chatterbox', __name__, url_prefix='/api/tts/chatterbox')


def _extract_chatterbox_params(data):
    """Extract and clamp Chatterbox-specific generation params."""
    try:
        exaggeration = max(0.0, min(float(data.get('exaggeration', 0.5)), MAX_EXAGGERATION))
        cfg_weight = max(0.0, min(1.0, float(data.get('cfg_weight', 0.5))))
        temperature = max(0.05, min(2.0, float(data.get('temperature', 0.8))))
    except (TypeError, ValueError):
        return None, 'exaggeration, cfg_weight, and temperature must be numeric'

    extra = {}
    if 'repetition_penalty' in data and data['repetition_penalty'] is not None:
        try:
            v = float(data['repetition_penalty'])
            extra['repetition_penalty'] = max(1.0, min(3.0, v))
        except (TypeError, ValueError):
            return None, 'repetition_penalty must be numeric'

    if 'min_p' in data and data['min_p'] is not None:
        try:
            v = float(data['min_p'])
            extra['min_p'] = max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return None, 'min_p must be numeric'

    if 'top_p' in data and data['top_p'] is not None:
        try:
            v = float(data['top_p'])
            extra['top_p'] = max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return None, 'top_p must be numeric'

    return {
        'exaggeration': exaggeration,
        'cfg_weight': cfg_weight,
        'temperature': temperature,
        **extra,
    }, None


def _resolve_chatterbox_ref(data):
    """Resolve Chatterbox reference audio path from request data."""
    ref_audio_id = data.get('ref_audio_id')
    if not ref_audio_id:
        return None, None

    if not is_valid_audio_id(ref_audio_id):
        return None, ('Invalid audio ID', 400)

    audio_prompt_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{ref_audio_id}.wav")
    if not os.path.exists(audio_prompt_path):
        return None, ('Reference audio not found', 404)

    return audio_prompt_path, None


@bp.route('/generate', methods=['POST'])
def chatterbox_generate():
    """Generate speech using Chatterbox TTS with optional voice cloning."""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    text = data.get('text')
    language_id = data.get('language_id', 'en')

    params, param_err = _extract_chatterbox_params(data)
    if param_err:
        return jsonify({'error': param_err}), 400

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400

    # Validate language
    valid_lang_ids = [code for code, name in CHATTERBOX_LANGUAGES]
    if language_id not in valid_lang_ids:
        return jsonify({'error': f'Invalid language: {language_id}'}), 400

    audio_prompt_path, ref_err = _resolve_chatterbox_ref(data)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    try:
        wav, sr = chatterbox_service.generate(
            text=text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            post_processing=post_processing,
            **params,
        )

        job_id = str(uuid.uuid4())
        _cache_audio(job_id, wav, sr)

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


@bp.route('/stream', methods=['POST'])
def chatterbox_stream():
    """Stream speech generation using Chatterbox TTS."""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    text = data.get('text')
    language_id = data.get('language_id', 'en')

    params, param_err = _extract_chatterbox_params(data)
    if param_err:
        return jsonify({'error': param_err}), 400

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400

    valid_lang_ids = [code for code, name in CHATTERBOX_LANGUAGES]
    if language_id not in valid_lang_ids:
        return jsonify({'error': f'Invalid language: {language_id}'}), 400

    audio_prompt_path, ref_err = _resolve_chatterbox_ref(data)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in chatterbox_service.generate_streaming(
                text=text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                post_processing=post_processing,
                **params,
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
            print(f"Streaming chatterbox error: {e}")
            yield f"<!--ERROR:{str(e)}-->".encode()

    return Response(
        generate(),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
        }
    )


@bp.route('/batch', methods=['POST'])
def chatterbox_batch():
    """Batch generate speech for multiple texts. Returns a zip of WAV files."""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    texts = data.get('texts', [])
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts must be a non-empty array'}), 400
    if len(texts) > 50:
        return jsonify({'error': 'Maximum 50 texts per batch'}), 400

    language_id = data.get('language_id', 'en')

    params, param_err = _extract_chatterbox_params(data)
    if param_err:
        return jsonify({'error': param_err}), 400

    valid_lang_ids = [code for code, name in CHATTERBOX_LANGUAGES]
    if language_id not in valid_lang_ids:
        return jsonify({'error': f'Invalid language: {language_id}'}), 400

    audio_prompt_path, ref_err = _resolve_chatterbox_ref(data)
    if ref_err:
        return jsonify({'error': ref_err[0]}), ref_err[1]

    post_processing, pp_err = _extract_post_processing(data)
    if pp_err:
        return jsonify({'error': pp_err}), 400

    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, text in enumerate(texts):
                text_err = _validate_text(text)
                if text_err:
                    return jsonify({'error': f'Item {i}: {text_err}'}), 400

                wav, sr = chatterbox_service.generate(
                    text=text,
                    language_id=language_id,
                    audio_prompt_path=audio_prompt_path,
                    post_processing=post_processing,
                    **params,
                )

                audio_buf = io.BytesIO()
                sf.write(audio_buf, wav, sr, format='WAV')
                zf.writestr(f'output_{i+1:03d}.wav', audio_buf.getvalue())

        zip_buffer.seek(0)
        return Response(
            zip_buffer.read(),
            mimetype='application/zip',
            headers={
                'Content-Disposition': 'attachment; filename="chatterbox_batch.zip"'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/languages', methods=['GET'])
def chatterbox_languages():
    """Get supported Chatterbox languages."""
    return jsonify([{'id': code, 'name': name} for code, name in CHATTERBOX_LANGUAGES])
