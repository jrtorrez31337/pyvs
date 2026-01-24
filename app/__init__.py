from flask import Flask
from flask_cors import CORS
import os


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

    return app
