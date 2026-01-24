import os
import uuid
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
from flask import Blueprint, request, jsonify, current_app

bp = Blueprint('audio', __name__, url_prefix='/api/audio')


def apply_noise_reduction(audio_data, sample_rate):
    """Apply noise reduction to audio data."""
    # Ensure mono for processing
    if len(audio_data.shape) > 1:
        audio_mono = np.mean(audio_data, axis=1)
    else:
        audio_mono = audio_data

    # Apply noise reduction - stationary noise (good for fans, HVAC, white noise)
    reduced = nr.reduce_noise(
        y=audio_mono,
        sr=sample_rate,
        stationary=True,
        prop_decrease=0.75,  # Reduce noise by 75%
    )

    return reduced.astype(np.float32)


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

        try:
            # Read audio
            data, sr = sf.read(tmp_path)

            # Apply noise reduction if requested
            if denoise:
                data = apply_noise_reduction(data, sr)

            # Get duration
            duration = len(data) / sr

            # Save as WAV
            sf.write(save_path, data, sr)

        finally:
            os.unlink(tmp_path)

        return jsonify({
            'id': audio_id,
            'duration': round(duration, 2),
            'sample_rate': sr,
            'denoised': denoise,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/delete/<audio_id>', methods=['DELETE'])
def delete_audio(audio_id):
    """Delete uploaded audio file"""
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
    from flask import send_file
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")

    if not os.path.exists(file_path):
        return jsonify({'error': 'Audio not found'}), 404

    return send_file(file_path, mimetype='audio/wav')


@bp.route('/trim/<audio_id>', methods=['POST'])
def trim_audio(audio_id):
    """Trim audio file to specified start/end times."""
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
