# pyvs - Python Voice Synthesis

[![Release](https://img.shields.io/github/v/release/jrtorrez31337/pyvs)](https://github.com/jrtorrez31337/pyvs/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A web application for multi-engine text-to-speech synthesis, voice cloning, and speech-to-text transcription.

## What's New in v0.2

- **Chatterbox TTS Engine** - Multilingual TTS with voice cloning and emotion control (23 languages)
- **0.6B Fast TTS Mode** - Lightweight Qwen3-TTS models for faster generation
- **ClearerVoice Speech Enhancement** - AI-powered audio denoising with MossFormer2
- **Whisper large-v3-turbo** - Upgraded STT model for faster, more accurate transcription
- **WebM Audio Upload** - Automatic format conversion via ffmpeg

See [v0.1](https://github.com/jrtorrez31337/pyvs/releases/tag/v0.1) for the initial release.

## Features

### Text-to-Speech Engines
- **Qwen3-TTS** - Voice cloning, custom voices (9 presets with emotion/style control), and voice design from natural language descriptions
- **Qwen3-TTS 0.6B (Fast)** - Lightweight models for faster generation with lower VRAM usage
- **Chatterbox TTS** - Multilingual voice cloning with emotion exaggeration and temperature controls across 23 languages

### Speech-to-Text
- **Transcription** - Accurate speech recognition with Faster-Whisper (large-v3-turbo)
- **Speech Enhancement** - ClearerVoice (MossFormer2) denoising before transcription
- **Language Detection** - Auto-detect or specify source language

### Voice Profiles
- **Save & Load** - Persist voice samples for reuse
- **Export/Import** - Share profiles as ZIP files
- **Audio Trimming** - Trim reference samples to optimal length

### Interface
- **Streaming Playback** - Real-time audio generation with instant playback
- **Waveform Visualization** - Visual audio display with wavesurfer.js
- **Generation History** - Replay recent generations
- **GPU Monitoring** - Real-time GPU utilization display
- **HTTPS Support** - Self-signed certificates for secure LAN access

## Hardware Requirements

This is a multi-GPU application. Models are assigned to specific GPUs at startup:

| GPU | Models | VRAM Required |
|-----|--------|---------------|
| GPU 0 | Qwen3-TTS (Base, CustomVoice, VoiceDesign + 0.6B variants), Chatterbox TTS, ClearerVoice | ~16 GB+ |
| GPU 1 | Faster-Whisper large-v3-turbo | ~5 GB |

**Minimum:**
- 2x NVIDIA GPUs with CUDA support
- GPU 0: 16 GB VRAM (e.g. RTX 4080, A4000)
- GPU 1: 8 GB VRAM (e.g. RTX 3070, RTX 4060 Ti)
- 32 GB system RAM
- CUDA 12.4 compatible drivers

**Recommended:**
- 2x NVIDIA RTX A40 (48 GB) or equivalent
- 64 GB system RAM

> **Note:** Running with a single GPU is possible using `CUDA_VISIBLE_DEVICES` but TTS and STT cannot run concurrently and VRAM must accommodate all models.

## Software Requirements

- Python 3.12
- CUDA 12.4+ toolkit and compatible NVIDIA drivers
- ffmpeg (for WebM and audio format conversion)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/jrtorrez31337/pyvs.git
cd pyvs

# Install dependencies with uv
uv sync
```

### Expected Warnings

On startup you may see:

```
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
```

This is expected. Flash-attn requires compilation against the exact PyTorch ABI and the pre-built wheels don't match PyTorch from the cu124 index. The application works correctly without it — Qwen-TTS falls back to a manual attention implementation.

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
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Constants and configuration
│   ├── routes/
│   │   ├── tts.py               # Qwen3-TTS endpoints (clone, custom, design)
│   │   ├── chatterbox.py        # Chatterbox TTS endpoint
│   │   ├── stt.py               # Speech-to-text endpoint
│   │   ├── audio.py             # Audio upload, trim, stream
│   │   ├── profiles.py          # Voice profile CRUD, export/import
│   │   ├── history.py           # Generation history
│   │   └── system.py            # GPU status
│   ├── services/
│   │   ├── tts_service.py       # Qwen3-TTS model wrapper
│   │   ├── chatterbox_service.py # Chatterbox TTS wrapper
│   │   ├── stt_service.py       # Faster-Whisper wrapper
│   │   ├── clearvoice_service.py # ClearerVoice speech enhancement
│   │   ├── audio_utils.py       # Audio processing utilities
│   │   └── gpu_service.py       # GPU monitoring
│   └── static/
│       ├── index.html           # Single-page application
│       ├── css/style.css        # Dark theme styles
│       └── js/app.js            # Frontend logic
├── certs/                       # Auto-generated SSL certificates
├── profiles/                    # Saved voice profiles
├── uploads/                     # Temporary audio uploads
├── run.py                       # Application entry point
└── pyproject.toml               # Project dependencies
```

## Test Sentences

Good sentences for evaluating voice quality and testing TTS output:

| Purpose | Sentence |
|---------|----------|
| General (covers most phonemes) | "The quick brown fox jumps over the lazy dog, while the bright sun shines warmly across the meadow." |
| Emotion/expressiveness | "Oh wow, I can't believe it actually worked! This is absolutely incredible!" |
| Clarity and pacing | "Please remember to pick up milk, eggs, bread, and butter from the grocery store on your way home tonight." |
| Quick A/B comparison | "The rainbow appeared after the storm, painting the sky with brilliant colors." |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tts/clone/stream` | POST | Qwen3 voice cloning with streaming |
| `/api/tts/custom/stream` | POST | Qwen3 custom voice with streaming |
| `/api/tts/design/stream` | POST | Qwen3 voice design with streaming |
| `/api/tts/chatterbox/generate` | POST | Chatterbox multilingual TTS |
| `/api/tts/chatterbox/languages` | GET | List supported Chatterbox languages |
| `/api/stt` | POST | Speech-to-text transcription |
| `/api/audio/upload` | POST | Upload reference audio |
| `/api/audio/trim/<id>` | POST | Trim audio sample |
| `/api/profiles` | GET/POST | List/create voice profiles |
| `/api/profiles/<id>/export` | GET | Export profile as ZIP |
| `/api/profiles/import` | POST | Import profile from ZIP |
| `/api/history` | GET/POST | Generation history |
| `/api/system/gpu` | GET | GPU status |

## License

MIT License
