import io
import os
import uuid
import numpy as np
import soundfile as sf
from flask import Blueprint, request, jsonify, Response, current_app
from app.services.chatterbox_service import chatterbox_service
from app.config import MAX_EXAGGERATION, CHATTERBOX_LANGUAGES, is_valid_audio_id
from app.routes.tts import _cache_audio, _validate_text

bp = Blueprint('chatterbox', __name__, url_prefix='/api/tts/chatterbox')


@bp.route('/generate', methods=['POST'])
def chatterbox_generate():
    """Generate speech using Chatterbox TTS with optional voice cloning."""
    data = request.get_json()

    text = data.get('text')
    language_id = data.get('language_id', 'en')
    exaggeration = max(0.0, min(float(data.get('exaggeration', 0.5)), MAX_EXAGGERATION))
    cfg_weight = max(0.0, min(1.0, float(data.get('cfg_weight', 0.5))))
    temperature = max(0.05, min(2.0, float(data.get('temperature', 0.8))))
    ref_audio_id = data.get('ref_audio_id')

    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400

    # Validate language
    valid_lang_ids = [code for code, name in CHATTERBOX_LANGUAGES]
    if language_id not in valid_lang_ids:
        return jsonify({'error': f'Invalid language: {language_id}'}), 400

    # Get reference audio path if provided
    audio_prompt_path = None
    if ref_audio_id:
        if not is_valid_audio_id(ref_audio_id):
            return jsonify({'error': 'Invalid audio ID'}), 400
        audio_prompt_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{ref_audio_id}.wav")
        if not os.path.exists(audio_prompt_path):
            return jsonify({'error': 'Reference audio not found'}), 404

    try:
        wav, sr = chatterbox_service.generate(
            text=text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
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


@bp.route('/languages', methods=['GET'])
def chatterbox_languages():
    """Get supported Chatterbox languages."""
    return jsonify([{'id': code, 'name': name} for code, name in CHATTERBOX_LANGUAGES])
