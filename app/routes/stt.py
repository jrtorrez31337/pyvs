import os
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
from flask import Blueprint, request, jsonify, current_app
from app.services.stt_service import stt_service

bp = Blueprint('stt', __name__, url_prefix='/api/stt')


def apply_noise_reduction(audio_path):
    """Apply noise reduction and return path to cleaned audio."""
    # Read audio
    data, sr = sf.read(audio_path)

    # Ensure mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Apply noise reduction
    reduced = nr.reduce_noise(
        y=data,
        sr=sr,
        stationary=True,
        prop_decrease=0.75,
    )

    # Save to new temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        sf.write(tmp.name, reduced.astype(np.float32), sr)
        return tmp.name


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
                denoised_path = apply_noise_reduction(temp_path)
                transcribe_path = denoised_path
            except Exception as e:
                print(f"Noise reduction failed, using original: {e}")

        # Transcribe
        auto_detect = request.form.get('auto_detect', 'false').lower() == 'true'

        if auto_detect:
            result = stt_service.transcribe_with_language_detect(transcribe_path)
        else:
            result = stt_service.transcribe(transcribe_path)

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
