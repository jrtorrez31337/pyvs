# pyvs - Python Voice Synthesis

A web application for text-to-speech synthesis and speech-to-text transcription using Qwen3-TTS and Faster-Whisper.

## Features

- **Text-to-Speech (TTS)**: Generate natural-sounding speech using Qwen3-TTS
- **Speech-to-Text (STT)**: Transcribe audio using Faster-Whisper
- **Voice Cloning**: Clone voices from audio samples
- **Voice Profiles**: Save and manage custom voice profiles
- **Web Interface**: Easy-to-use browser-based interface
- **HTTPS Support**: Self-signed certificates for secure microphone access over LAN

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended: 2 GPUs for running TTS and STT simultaneously)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/jrtorrez31337/pyvs.git
cd pyvs

# Install dependencies with uv
uv sync
```

## Usage

```bash
# Run with default settings (HTTPS enabled, models preloaded)
uv run python run.py

# Run without preloading models
uv run python run.py --no-preload

# Run on a specific port
uv run python run.py --port 8080

# Run without SSL (microphone won't work over LAN)
uv run python run.py --no-ssl
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Host to bind to | `0.0.0.0` |
| `--port` | Port to bind to | `5000` |
| `--no-preload` | Don't preload models at startup | False |
| `--no-ssl` | Disable HTTPS | False |
| `--debug` | Enable debug mode | False |

## Project Structure

```
pyvs/
├── app/
│   ├── __init__.py      # Flask app factory
│   ├── routes/          # API endpoints
│   ├── services/        # TTS and STT services
│   ├── static/          # Static assets (CSS, JS)
│   └── templates/       # HTML templates
├── certs/               # Auto-generated SSL certificates
├── profiles/            # Saved voice profiles
├── uploads/             # Temporary audio uploads
├── run.py               # Application entry point
└── pyproject.toml       # Project dependencies
```

## License

MIT License
