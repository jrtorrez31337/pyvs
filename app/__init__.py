import os
import time
from flask import Flask, jsonify
from flask_cors import CORS
from app.config import UPLOAD_MAX_AGE_SECONDS, MAX_UPLOAD_SIZE_MB


def _cleanup_stale_uploads(upload_folder):
    """Remove upload files older than UPLOAD_MAX_AGE_SECONDS."""
    now = time.time()
    try:
        for filename in os.listdir(upload_folder):
            filepath = os.path.join(upload_folder, filename)
            if os.path.isfile(filepath):
                age = now - os.path.getmtime(filepath)
                if age > UPLOAD_MAX_AGE_SECONDS:
                    os.unlink(filepath)
    except OSError:
        pass


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Clean up stale uploads older than 24 hours
    _cleanup_stale_uploads(app.config['UPLOAD_FOLDER'])

    from app.routes import tts, stt, audio, profiles, system, history
    app.register_blueprint(tts.bp)
    app.register_blueprint(stt.bp)
    app.register_blueprint(audio.bp)
    app.register_blueprint(profiles.bp)
    app.register_blueprint(system.bp)
    app.register_blueprint(history.bp)

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500

    return app
