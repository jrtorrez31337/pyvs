import os
import tempfile
from flask import Blueprint, request, jsonify, current_app
from app.services.stt_service import stt_service
from app.services.audio_utils import reduce_noise_file

bp = Blueprint('stt', __name__, url_prefix='/api/stt')


@bp.route('', methods=['POST'])
def transcribe():
    """Transcribe audio to text with optional word timestamps and diarization."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    # Check if noise reduction is requested (default: on)
    denoise = request.form.get('denoise', 'true').lower() == 'true'
    word_timestamps = request.form.get('word_timestamps', 'false').lower() == 'true'
    diarize = request.form.get('diarize', 'false').lower() == 'true'

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

        # Transcribe with options
        if word_timestamps or diarize:
            auto_detect = request.form.get('auto_detect', 'false').lower() == 'true'
            language = None if auto_detect else request.form.get('language', 'en')
            result = stt_service.transcribe_with_options(
                transcribe_path, language=language,
                word_timestamps=(word_timestamps or diarize),
            )
        else:
            auto_detect = request.form.get('auto_detect', 'false').lower() == 'true'
            language = None if auto_detect else request.form.get('language', 'en')
            result = stt_service.transcribe(transcribe_path, language=language)

        result['denoised'] = denoise and denoised_path is not None

        # Run diarization if requested
        if diarize:
            try:
                from app.services.diarization_service import diarization_service
                if diarization_service.available:
                    diar_segments = diarization_service.diarize(transcribe_path)
                    word_data = result.get('words', [])
                    if word_data and diar_segments:
                        result['speakers'] = diarization_service.merge_with_words(
                            diar_segments, word_data
                        )
                    else:
                        result['speakers'] = diar_segments
                else:
                    result['diarization_error'] = 'pyannote-audio not available or HF_TOKEN not set'
            except Exception as e:
                print(f"Diarization failed: {e}")
                result['diarization_error'] = str(e)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temp files
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if denoised_path and os.path.exists(denoised_path):
            os.unlink(denoised_path)


@bp.route('/capabilities', methods=['GET'])
def stt_capabilities():
    """Return available STT capabilities."""
    try:
        from app.services.diarization_service import diarization_service
        diarization_available = diarization_service.available
    except ImportError:
        diarization_available = False

    return jsonify({
        'word_timestamps': True,
        'diarization': diarization_available,
        'live_transcription': False,  # WebSocket not yet configured
    })
