# pyvs - Python Voice Synthesis

[![Release](https://img.shields.io/github/v/release/jrtorrez31337/pyvs)](https://github.com/jrtorrez31337/pyvs/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A web application for multi-engine text-to-speech synthesis, voice cloning, and speech-to-text transcription.

## What's New in v0.3

- **Audio Post-Processing** - Pitch shift, speed control, volume normalization (LUFS), and sample rate conversion on any generated audio
- **Advanced Inference Controls** - Temperature, top-k, top-p, and repetition penalty sliders for all Qwen3-TTS modes
- **Chatterbox Streaming & Batch** - Real-time streaming playback and ZIP batch export for Chatterbox TTS
- **Extended Chatterbox Params** - Repetition penalty, min-p, and top-p controls
- **STT Word Timestamps** - Word-level timing and confidence scores for transcription
- **Speaker Diarization** - Identify and label speakers in multi-speaker audio (requires pyannote-audio + HF_TOKEN)
- **SSML & Multi-Speaker Dialogue** - Parse SSML markup or structured segments to generate multi-speaker audio
- **Voice Similarity Scoring** - Cosine similarity badge comparing reference and generated voice embeddings
- **Speaker Blending** - Per-sample weight sliders for weighted speaker embedding interpolation across clone references
- **Mobile-Responsive Layout** - Hamburger menu, stacked controls, and touch-friendly targets at 480px/600px/768px breakpoints
- **Tooltips** - Hover tooltips on all parameter labels explaining each control

See [v0.2](https://github.com/jrtorrez31337/pyvs/releases/tag/v0.2) for Chatterbox TTS, fast mode, and ClearerVoice.
See [v0.1](https://github.com/jrtorrez31337/pyvs/releases/tag/v0.1) for the initial release.

## Features

### Text-to-Speech Engines
- **Qwen3-TTS** - Voice cloning, custom voices (9 presets with emotion/style control), and voice design from natural language descriptions
- **Qwen3-TTS 0.6B (Fast)** - Lightweight models for faster generation with lower VRAM usage
- **Chatterbox TTS** - Multilingual voice cloning with emotion exaggeration and temperature controls across 23 languages
- **SSML & Dialogue** - Multi-speaker dialogue generation from SSML markup or structured segment lists

### Audio Post-Processing
- **Pitch Shift** - Adjust pitch by -12 to +12 semitones
- **Speed Control** - Time-stretch from 0.5x to 2.0x without pitch change
- **Volume Normalization** - Peak normalization with configurable LUFS target (-24 to -6)
- **Sample Rate Conversion** - Resample output to 8/16/22.05/24/44.1/48 kHz

### Advanced Generation Controls
- **Qwen3 Inference Params** - Temperature, top-k, top-p, repetition penalty per mode
- **Chatterbox Extended Params** - Repetition penalty, min-p, top-p in addition to exaggeration/CFG
- **Speaker Blending** - Per-sample weight sliders for weighted embedding interpolation across multiple clone references
- **Voice Similarity** - Cosine similarity score badge comparing reference and generated speaker embeddings

### Speech-to-Text
- **Transcription** - Accurate speech recognition with Faster-Whisper (large-v3-turbo)
- **Word Timestamps** - Word-level timing with color-coded confidence highlighting
- **Speaker Diarization** - Multi-speaker identification with color-coded transcript segments (optional, requires pyannote-audio)
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
- **Mobile-Responsive** - Hamburger menu, stacked layouts, and 44px touch targets at phone/tablet breakpoints
- **Tooltips** - Hover tooltips on all parameter labels

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

### Optional Dependencies

| Package | Feature | Notes |
|---------|---------|-------|
| `pyannote-audio` | Speaker diarization | Requires `HF_TOKEN` env var for model access |
| `resemblyzer` | Voice similarity scoring | Falls back to librosa MFCCs if unavailable |
| `librosa` | Pitch shift, time stretch, MFCC fallback | Falls back to scipy if unavailable |

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
│   │   ├── tts.py               # Qwen3-TTS endpoints (clone, custom, design, dialogue, similarity)
│   │   ├── chatterbox.py        # Chatterbox TTS endpoints (generate, stream, batch)
│   │   ├── stt.py               # Speech-to-text with word timestamps and diarization
│   │   ├── audio.py             # Audio upload, trim, stream
│   │   ├── profiles.py          # Voice profile CRUD, export/import
│   │   ├── history.py           # Generation history
│   │   └── system.py            # GPU status
│   ├── services/
│   │   ├── tts_service.py       # Qwen3-TTS model wrapper
│   │   ├── chatterbox_service.py # Chatterbox TTS wrapper
│   │   ├── stt_service.py       # Faster-Whisper wrapper with word timestamps
│   │   ├── clearvoice_service.py # ClearerVoice speech enhancement
│   │   ├── audio_utils.py       # Post-processing (pitch, speed, normalize, resample)
│   │   ├── ssml_parser.py       # SSML subset parser for dialogue generation
│   │   ├── voice_similarity.py  # Speaker embedding similarity scoring
│   │   ├── diarization_service.py # Speaker diarization (optional, pyannote-audio)
│   │   ├── gpu_service.py       # GPU monitoring
│   │   └── gpu_lock.py          # Thread-safe GPU locking
│   └── static/
│       ├── index.html           # Single-page application
│       ├── css/style.css        # Dark theme styles (mobile-responsive)
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

### TTS Generation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tts/clone` | POST | Qwen3 voice cloning (non-streaming) |
| `/api/tts/clone/stream` | POST | Qwen3 voice cloning with streaming |
| `/api/tts/custom` | POST | Qwen3 custom voice (non-streaming) |
| `/api/tts/custom/stream` | POST | Qwen3 custom voice with streaming |
| `/api/tts/design` | POST | Qwen3 voice design (non-streaming) |
| `/api/tts/design/stream` | POST | Qwen3 voice design with streaming |
| `/api/tts/dialogue` | POST | Multi-speaker dialogue from SSML or segments |
| `/api/tts/similarity` | POST | Voice similarity score between reference and generated audio |
| `/api/tts/speakers` | GET | List available custom voice presets |
| `/api/tts/languages` | GET | List supported Qwen3 languages |
| `/api/tts/download/<job_id>` | GET | Download cached generated audio |

### Chatterbox TTS

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tts/chatterbox/generate` | POST | Chatterbox TTS generation |
| `/api/tts/chatterbox/stream` | POST | Chatterbox TTS with streaming |
| `/api/tts/chatterbox/batch` | POST | Batch generation, returns ZIP of WAVs |
| `/api/tts/chatterbox/languages` | GET | List supported Chatterbox languages |

### Speech-to-Text

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stt` | POST | Transcribe audio (supports word_timestamps, diarize params) |
| `/api/stt/capabilities` | GET | Check available STT features |

### Audio & Profiles

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/audio/upload` | POST | Upload reference audio |
| `/api/audio/trim/<id>` | POST | Trim audio sample |
| `/api/audio/stream/<id>` | GET | Stream audio file |
| `/api/audio/info/<id>` | GET | Audio metadata |
| `/api/audio/delete/<id>` | DELETE | Delete audio file |
| `/api/profiles` | GET/POST | List/create voice profiles |
| `/api/profiles/<id>/export` | GET | Export profile as ZIP |
| `/api/profiles/import` | POST | Import profile from ZIP |
| `/api/history` | GET/POST | Generation history |
| `/api/system/gpu` | GET | GPU status |

## License

MIT License
