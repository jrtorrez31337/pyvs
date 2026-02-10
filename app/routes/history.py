import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file
import soundfile as sf
from app.config import MAX_HISTORY_ITEMS as MAX_HISTORY, is_valid_audio_id

bp = Blueprint('history', __name__, url_prefix='/api/history')

# In-memory history (could be persisted to file/db)
_history = []


def get_history_dir():
    history_dir = os.path.join(
        os.path.dirname(current_app.config['UPLOAD_FOLDER']), 'history'
    )
    os.makedirs(history_dir, exist_ok=True)
    return history_dir


@bp.route('', methods=['GET'])
def list_history():
    """List generation history."""
    return jsonify(_history[-MAX_HISTORY:])


@bp.route('', methods=['POST'])
def add_history():
    """Add item to history."""
    data = request.get_json()

    item = {
        'id': str(uuid.uuid4()),
        'mode': data.get('mode'),  # clone, custom, design
        'text': data.get('text'),
        'language': data.get('language'),
        'params': data.get('params', {}),  # mode-specific params
        'audio_id': data.get('audio_id'),
        'created_at': datetime.utcnow().isoformat(),
    }

    _history.append(item)

    # Trim history
    while len(_history) > MAX_HISTORY:
        old_item = _history.pop(0)
        # Optionally delete old audio file
        old_path = os.path.join(get_history_dir(), f"{old_item['audio_id']}.wav")
        if os.path.exists(old_path):
            os.unlink(old_path)

    return jsonify(item)


@bp.route('/<item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    """Delete a history item."""
    global _history
    _history = [h for h in _history if h['id'] != item_id]
    return jsonify({'success': True})


@bp.route('/clear', methods=['POST'])
def clear_history():
    """Clear all history."""
    global _history
    _history = []
    return jsonify({'success': True})


@bp.route('/audio/<audio_id>', methods=['GET'])
def get_history_audio(audio_id):
    """Get audio file from history."""
    if not is_valid_audio_id(audio_id):
        return jsonify({'error': 'Invalid audio ID'}), 400
    from app.routes.tts import get_cached_audio
    entry = get_cached_audio(audio_id)
    if entry:
        import io
        wav, sr = entry
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        return send_file(buffer, mimetype='audio/wav')

    return jsonify({'error': 'Audio not found or expired'}), 404
