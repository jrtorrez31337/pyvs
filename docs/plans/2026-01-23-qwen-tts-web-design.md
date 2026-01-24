# Qwen3-TTS Web Application Design

## Overview

Flask-based web application providing a dashboard interface for Qwen3-TTS text-to-speech and Whisper speech-to-text, running locally on two NVIDIA A40 GPUs.

## Features

- **Speech-to-Text**: Whisper medium model for microphone transcription
- **Voice Clone**: Clone voice from 3-second audio sample
- **CustomVoice**: 9 preset voices with style/emotion control
- **VoiceDesign**: Create voices from natural language descriptions
- **Streaming**: Real-time audio playback during generation
- **Download**: Save generated audio files

## Project Structure

```
qwen-tts-web/
├── pyproject.toml          # uv project config
├── app/
│   ├── __init__.py         # Flask app factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── tts.py          # TTS endpoints (clone, custom, design)
│   │   ├── stt.py          # Speech-to-text endpoint
│   │   └── audio.py        # Audio upload/streaming
│   ├── services/
│   │   ├── __init__.py
│   │   ├── tts_service.py  # Qwen3-TTS model wrapper
│   │   └── stt_service.py  # Whisper model wrapper
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       └── index.html      # Single-page dashboard
├── uploads/                 # Temporary audio uploads
└── run.py                  # Entry point
```

## API Endpoints

### Speech-to-Text
```
POST /api/stt
  Body: audio file (webm/wav from microphone)
  Returns: { "text": "transcribed text", "language": "en" }
```

### Text-to-Speech

```
POST /api/tts/clone
  Body: { "text": "...", "language": "en", "ref_audio_id": "uuid" }
  Returns: streaming audio (wav)

POST /api/tts/custom
  Body: { "text": "...", "language": "en", "speaker": "Ryan", "instruct": "happy tone" }
  Returns: streaming audio

POST /api/tts/design
  Body: { "text": "...", "language": "en", "instruct": "young male, nervous" }
  Returns: streaming audio
```

### Audio Management
```
POST /api/audio/upload
  Body: audio file (for voice clone reference)
  Returns: { "id": "uuid", "duration": 3.2 }

GET /api/audio/download/<job_id>
  Returns: complete generated audio file
```

### Utility
```
GET /api/speakers
  Returns: list of CustomVoice speakers with descriptions

GET /api/languages
  Returns: supported languages
```

## UI Layout

Dashboard style with persistent sidebar and bottom audio player:

```
┌─────────────────────────────────────────────────────────────────┐
│  Qwen3-TTS Studio                                    [GPU Status]│
├────────────┬────────────────────────────────────────────────────┤
│            │                                                     │
│  MODES     │   MAIN WORKSPACE                                    │
│  ────────  │                                                     │
│            │   - Text input area                                 │
│  STT       │   - Reference audio player (clone mode)             │
│  Voice     │   - Speaker dropdown (custom mode)                  │
│    Clone   │   - Voice description field (design mode)           │
│  Custom    │   - Language selector                               │
│    Voice   │   - Instruction/style field                         │
│  Voice     │                                                     │
│    Design  │   [Generate Button]                                 │
│            │                                                     │
├────────────┴────────────────────────────────────────────────────┤
│  AUDIO PLAYER                                      [Download]    │
│  ▶ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  0:00 / 0:12  │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Allocation

- **GPU 0 (~46GB)**: All Qwen3-TTS models preloaded
  - Qwen3-TTS-Tokenizer-12Hz (~600MB)
  - Qwen3-TTS-12Hz-1.7B-Base (~4GB)
  - Qwen3-TTS-12Hz-1.7B-CustomVoice (~4GB)
  - Qwen3-TTS-12Hz-1.7B-VoiceDesign (~4GB)

- **GPU 1 (~49GB)**: Whisper medium (~5GB)

## Dependencies

- Flask, flask-cors
- qwen-tts
- openai-whisper
- torch, flash-attn
- soundfile, numpy

## Streaming Implementation

Server streams audio chunks via Flask Response generator. Client uses Web Audio API to decode and play chunks in real-time. Completed audio buffered server-side for download.

## Deployment

- HTTP on LAN (reverse proxy for HTTPS later)
- Models loaded from `/data/models/Qwen/`
- Thread locks per model for concurrent request safety
