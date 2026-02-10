import os
import tempfile
from flask import Blueprint, request, jsonify, current_app
from app.services.stt_service import stt_service
from app.services.audio_utils import reduce_noise_file

bp = Blueprint('stt', __name__, url_prefix='/api/stt')


@bp.route('', methods=['POST'])
def transcribe():
    """Transcribe audio to text"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    # Check if noise reduction is requested (default: on)
    denoise = request.form.get('denoise', 'true').lower() == 'true'

    # Save to temporary file
    temp_path = None
    denoised_path = None
    try:
        # Create temp file with appropriate extension
        suffix = os.path.splitext(audio_file.filename)[1] or '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp)
            temp_path = tmp.name

        # Apply noise reduction if requested
        transcribe_path = temp_path
        if denoise:
            try:
                denoised_path = reduce_noise_file(temp_path)
                transcribe_path = denoised_path
            except Exception as e:
                print(f"Noise reduction failed, using original: {e}")

        # Transcribe
        auto_detect = request.form.get('auto_detect', 'false').lower() == 'true'
        language = None if auto_detect else request.form.get('language', 'en')
        result = stt_service.transcribe(transcribe_path, language=language)

        result['denoised'] = denoise and denoised_path is not None
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temp files
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if denoised_path and os.path.exists(denoised_path):
            os.unlink(denoised_path)
