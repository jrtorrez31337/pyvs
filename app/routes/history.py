import io
import os
import uuid
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file
import soundfile as sf
from app.config import MAX_HISTORY_ITEMS as MAX_HISTORY, is_valid_audio_id

bp = Blueprint('history', __name__, url_prefix='/api/history')

# In-memory history with thread lock
_history = []
_history_lock = threading.Lock()


def get_history_dir():
    history_dir = os.path.join(
        os.path.dirname(current_app.config['UPLOAD_FOLDER']), 'history'
    )
    os.makedirs(history_dir, exist_ok=True)
    return history_dir


def _persist_audio(audio_id):
    """Copy audio from in-memory cache to disk for persistence."""
    from app.routes.tts import get_cached_audio
    entry = get_cached_audio(audio_id)
    if entry is None:
        return
    wav, sr = entry
    history_dir = get_history_dir()
    path = os.path.join(history_dir, f"{audio_id}.wav")
    if not os.path.exists(path):
        sf.write(path, wav, sr, format='WAV')


@bp.route('', methods=['GET'])
def list_history():
    """List generation history."""
    with _history_lock:
        return jsonify(_history[-MAX_HISTORY:])


@bp.route('', methods=['POST'])
def add_history():
    """Add item to history and persist its audio to disk."""
    data = request.get_json()

    audio_id = data.get('audio_id')

    item = {
        'id': str(uuid.uuid4()),
        'mode': data.get('mode'),  # clone, custom, design
        'text': data.get('text'),
        'language': data.get('language'),
        'params': data.get('params', {}),  # mode-specific params
        'audio_id': audio_id,
        'created_at': datetime.utcnow().isoformat(),
    }

    # Persist audio from cache to disk before it expires
    if audio_id and is_valid_audio_id(audio_id):
        try:
            _persist_audio(audio_id)
        except Exception as e:
            print(f"Warning: failed to persist history audio {audio_id}: {e}")

    with _history_lock:
        _history.append(item)

        # Trim history
        while len(_history) > MAX_HISTORY:
            old_item = _history.pop(0)
            # Delete old audio file from disk
            old_audio_id = old_item.get('audio_id')
            if old_audio_id and is_valid_audio_id(old_audio_id):
                old_path = os.path.join(get_history_dir(), f"{old_audio_id}.wav")
                if os.path.exists(old_path):
                    try:
                        os.unlink(old_path)
                    except OSError:
                        pass

    return jsonify(item)


@bp.route('/<item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    """Delete a history item and its audio file."""
    with _history_lock:
        removed = [h for h in _history if h['id'] == item_id]
        _history[:] = [h for h in _history if h['id'] != item_id]

    # Clean up audio file
    for item in removed:
        audio_id = item.get('audio_id')
        if audio_id and is_valid_audio_id(audio_id):
            path = os.path.join(get_history_dir(), f"{audio_id}.wav")
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return jsonify({'success': True})


@bp.route('/clear', methods=['POST'])
def clear_history():
    """Clear all history and audio files."""
    with _history_lock:
        items = list(_history)
        _history.clear()

    # Clean up all audio files
    for item in items:
        audio_id = item.get('audio_id')
        if audio_id and is_valid_audio_id(audio_id):
            path = os.path.join(get_history_dir(), f"{audio_id}.wav")
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return jsonify({'success': True})


@bp.route('/audio/<audio_id>', methods=['GET'])
def get_history_audio(audio_id):
    """Get audio file from history. Checks in-memory cache first, then disk."""
    if not is_valid_audio_id(audio_id):
        return jsonify({'error': 'Invalid audio ID'}), 400

    # Try in-memory cache first (fastest)
    from app.routes.tts import get_cached_audio
    entry = get_cached_audio(audio_id)
    if entry:
        wav, sr = entry
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        return send_file(buffer, mimetype='audio/wav')

    # Fall back to disk (persisted history audio)
    path = os.path.join(get_history_dir(), f"{audio_id}.wav")
    if os.path.exists(path):
        return send_file(path, mimetype='audio/wav')

    return jsonify({'error': 'Audio not found or expired'}), 404
