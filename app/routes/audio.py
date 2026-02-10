import os
import uuid
import tempfile
import numpy as np
import soundfile as sf
from flask import Blueprint, request, jsonify, current_app
from app.config import is_valid_audio_id
from app.services.audio_utils import reduce_noise, convert_to_wav

bp = Blueprint('audio', __name__, url_prefix='/api/audio')


@bp.route('/upload', methods=['POST'])
def upload_audio():
    """Upload audio file for voice cloning reference"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    # Check if noise reduction is requested (default: on)
    denoise = request.form.get('denoise', 'true').lower() == 'true'

    try:
        # Generate unique ID
        audio_id = str(uuid.uuid4())

        # Save as WAV for consistency
        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")

        # Read and resave to ensure WAV format
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        wav_path = None
        try:
            # Convert to WAV if needed (e.g. WebM from browser recording)
            wav_path = convert_to_wav(tmp_path)

            # Read audio
            data, sr = sf.read(wav_path)

            # Apply noise reduction if requested
            denoised = False
            if denoise:
                try:
                    data = reduce_noise(data, sr)
                    denoised = True
                except Exception as e:
                    print(f"Noise reduction failed, using original: {e}")

            # Get duration
            duration = len(data) / sr

            # Save as WAV
            sf.write(save_path, data, sr)

        finally:
            os.unlink(tmp_path)
            if wav_path and wav_path != tmp_path and os.path.exists(wav_path):
                os.unlink(wav_path)

        return jsonify({
            'id': audio_id,
            'duration': round(duration, 2),
            'sample_rate': sr,
            'denoised': denoised,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/delete/<audio_id>', methods=['DELETE'])
def delete_audio(audio_id):
    """Delete uploaded audio file"""
    if not is_valid_audio_id(audio_id):
        return jsonify({'error': 'Invalid audio ID'}), 400
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")

    if not os.path.exists(file_path):
        return jsonify({'error': 'Audio not found'}), 404

    try:
        os.unlink(file_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/stream/<audio_id>', methods=['GET'])
def stream_audio(audio_id):
    """Stream an uploaded audio file"""
    if not is_valid_audio_id(audio_id):
        return jsonify({'error': 'Invalid audio ID'}), 400
    from flask import send_file
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")

    if not os.path.exists(file_path):
        return jsonify({'error': 'Audio not found'}), 404

    return send_file(file_path, mimetype='audio/wav')


@bp.route('/trim/<audio_id>', methods=['POST'])
def trim_audio(audio_id):
    """Trim audio file to specified start/end times."""
    if not is_valid_audio_id(audio_id):
        return jsonify({'error': 'Invalid audio ID'}), 400
    data = request.get_json()
    start = data.get('start', 0)  # seconds
    end = data.get('end')  # seconds

    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")
    if not os.path.exists(file_path):
        return jsonify({'error': 'Audio not found'}), 404

    try:
        data, sr = sf.read(file_path)

        start_sample = int(start * sr)
        end_sample = int(end * sr) if end else len(data)

        trimmed = data[start_sample:end_sample]

        # Save trimmed version with new ID
        new_id = str(uuid.uuid4())
        new_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{new_id}.wav")
        sf.write(new_path, trimmed, sr)

        return jsonify({
            'id': new_id,
            'duration': len(trimmed) / sr,
            'sample_rate': sr,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/info/<audio_id>', methods=['GET'])
def get_audio_info(audio_id):
    """Get audio file info (duration, sample rate)."""
    if not is_valid_audio_id(audio_id):
        return jsonify({'error': 'Invalid audio ID'}), 400
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")
    if not os.path.exists(file_path):
        return jsonify({'error': 'Audio not found'}), 404

    try:
        data, sr = sf.read(file_path)
        duration = len(data) / sr
        return jsonify({
            'id': audio_id,
            'duration': duration,
            'sample_rate': sr,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
