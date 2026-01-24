# Qwen-TTS Web Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance the Qwen-TTS web application with streaming playback, GPU monitoring, generation history, audio visualization, sample trimming, batch generation, and profile export/import capabilities.

**Architecture:** Phased implementation with parallel workstreams. Phase 1 focuses on core UX improvements (streaming, GPU status). Phase 2 adds productivity features (history, batch). Phase 3 adds polish (visualization, trimming, export). Each phase has independent tasks that can run in parallel.

**Tech Stack:** Flask, Python 3.12, JavaScript (vanilla), Web Audio API, Server-Sent Events (SSE), pynvml (GPU monitoring), wavesurfer.js (visualization)

---

## Phase Overview

| Phase | Focus | Duration | Parallel Tasks |
|-------|-------|----------|----------------|
| **Phase 1** | Core UX | Priority | 1A, 1B can run in parallel |
| **Phase 2** | Productivity | After Phase 1 | 2A, 2B, 2C can run in parallel |
| **Phase 3** | Polish | After Phase 2 | 3A, 3B, 3C can run in parallel |

---

## Phase 1: Core UX Improvements

### Task 1A: GPU Status Monitoring

**Goal:** Display real-time GPU utilization, memory usage, and temperature in the header.

**Files:**
- Create: `app/services/gpu_service.py`
- Create: `app/routes/system.py`
- Modify: `app/__init__.py` (register blueprint)
- Modify: `app/static/js/app.js` (add GPU polling)
- Modify: `app/static/css/style.css` (GPU status styles)

**Dependencies:** `pynvml` (add to pyproject.toml)

**Step 1: Add pynvml dependency**

```bash
uv add pynvml
```

**Step 2: Create GPU service**

Create `app/services/gpu_service.py`:

```python
import threading
from typing import Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._nvml_initialized = False
        self._init_nvml()
        self._initialized = True

    def _init_nvml(self):
        if not PYNVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except pynvml.NVMLError:
            self._nvml_initialized = False

    def get_gpu_status(self) -> list:
        """Get status of all GPUs."""
        if not self._nvml_initialized:
            return []

        gpus = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                gpus.append({
                    'index': i,
                    'name': name,
                    'memory_used': memory.used // (1024 ** 2),  # MB
                    'memory_total': memory.total // (1024 ** 2),  # MB
                    'memory_percent': round(memory.used / memory.total * 100, 1),
                    'utilization': utilization.gpu,
                    'temperature': temperature,
                })
        except pynvml.NVMLError:
            pass

        return gpus


gpu_service = GPUService()
```

**Step 3: Create system routes**

Create `app/routes/system.py`:

```python
from flask import Blueprint, jsonify
from app.services.gpu_service import gpu_service

bp = Blueprint('system', __name__, url_prefix='/api/system')


@bp.route('/gpu', methods=['GET'])
def get_gpu_status():
    """Get GPU status for all devices."""
    gpus = gpu_service.get_gpu_status()
    return jsonify(gpus)
```

**Step 4: Register blueprint**

Modify `app/__init__.py`, add after other imports:

```python
from app.routes import tts, stt, audio, profiles, system
# ... existing code ...
app.register_blueprint(system.bp)
```

**Step 5: Add frontend GPU polling**

Add to `app/static/js/app.js` in the initialization section:

```javascript
// GPU Status Polling
function initGPUStatus() {
    const gpuStatus = document.getElementById('gpu-status');

    async function updateGPUStatus() {
        try {
            const response = await fetch('/api/system/gpu');
            const gpus = await response.json();

            if (gpus.length === 0) {
                gpuStatus.textContent = 'GPU: N/A';
                return;
            }

            const statusParts = gpus.map(gpu =>
                `GPU${gpu.index}: ${gpu.utilization}% | ${gpu.memory_used}/${gpu.memory_total}MB | ${gpu.temperature}°C`
            );
            gpuStatus.textContent = statusParts.join(' | ');
        } catch (err) {
            gpuStatus.textContent = 'GPU: Error';
        }
    }

    // Initial update and poll every 5 seconds
    updateGPUStatus();
    setInterval(updateGPUStatus, 5000);
}

// Add to DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    // ... existing init calls ...
    initGPUStatus();
});
```

**Step 6: Update CSS for GPU status**

Add to `app/static/css/style.css`:

```css
.gpu-status {
    font-size: 0.75rem;
    color: var(--text-secondary);
    padding: 0.5rem 1rem;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-family: monospace;
    max-width: 600px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add GPU status monitoring in header"
```

---

### Task 1B: Streaming Audio Playback

**Goal:** Stream TTS audio chunks to browser for real-time playback during generation.

**Files:**
- Modify: `app/services/tts_service.py` (add streaming generator)
- Modify: `app/routes/tts.py` (add streaming endpoints)
- Modify: `app/static/js/app.js` (Web Audio API streaming)

**Step 1: Add streaming support to TTS service**

Modify `app/services/tts_service.py`, add streaming method:

```python
def generate_clone_streaming(self, text: str, language: str, ref_audio_paths, ref_texts=None):
    """Generate speech using voice cloning with streaming output.

    Yields audio chunks as they are generated.
    """
    with self._model_lock:
        if isinstance(ref_audio_paths, str):
            ref_audio_paths = [ref_audio_paths]
        if ref_texts is None:
            ref_texts = [None] * len(ref_audio_paths)
        elif isinstance(ref_texts, str):
            ref_texts = [ref_texts]

        # Use streaming mode
        for chunk, sr in self.clone_model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_paths,
            ref_text=ref_texts,
            non_streaming_mode=False,  # Enable streaming
        ):
            yield chunk, sr

def generate_custom_streaming(self, text: str, language: str, speaker: str, instruct: str = None):
    """Generate speech using custom voice with streaming output."""
    with self._model_lock:
        kwargs = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "non_streaming_mode": False,
        }
        if instruct:
            kwargs["instruct"] = instruct

        for chunk, sr in self.custom_model.generate_custom_voice(**kwargs):
            yield chunk, sr

def generate_design_streaming(self, text: str, language: str, instruct: str):
    """Generate speech using voice design with streaming output."""
    with self._model_lock:
        for chunk, sr in self.design_model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            non_streaming_mode=False,
        ):
            yield chunk, sr
```

**Step 2: Add streaming endpoints**

Add to `app/routes/tts.py`:

```python
import struct

def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16):
    """Create a WAV header for streaming (size set to max)."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    # Use 0xFFFFFFFF for unknown size (streaming)
    data_size = 0xFFFFFFFF - 36

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        0xFFFFFFFF,  # File size (unknown for streaming)
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1 size
        1,   # Audio format (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )
    return header


@bp.route('/clone/stream', methods=['POST'])
def tts_clone_stream():
    """Stream speech generation using voice cloning."""
    data = request.get_json()

    text = data.get('text')
    language = data.get('language', 'English')
    ref_audio_ids = data.get('ref_audio_ids') or []
    ref_texts = data.get('ref_texts') or []

    if not text:
        return jsonify({'error': 'Text is required'}), 400
    if not ref_audio_ids:
        return jsonify({'error': 'At least one reference audio is required'}), 400

    ref_audio_paths = []
    for audio_id in ref_audio_ids:
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{audio_id}.wav")
        if not os.path.exists(path):
            return jsonify({'error': f'Reference audio not found: {audio_id}'}), 404
        ref_audio_paths.append(path)

    while len(ref_texts) < len(ref_audio_paths):
        ref_texts.append(None)

    def generate():
        header_sent = False
        all_chunks = []
        sample_rate = None

        try:
            for chunk, sr in tts_service.generate_clone_streaming(
                text, language, ref_audio_paths, ref_texts
            ):
                if not header_sent:
                    sample_rate = sr
                    yield create_wav_header(sr)
                    header_sent = True

                # Convert float32 to int16
                audio_int16 = (chunk * 32767).astype('int16')
                all_chunks.append(chunk)
                yield audio_int16.tobytes()

            # Store complete audio for download
            if all_chunks:
                import numpy as np
                full_audio = np.concatenate(all_chunks)
                job_id = str(uuid.uuid4())
                _generated_audio[job_id] = (full_audio, sample_rate)
                # Send job ID as final chunk marker (won't be played as audio)
                yield f"<!--JOB_ID:{job_id}-->".encode()

        except Exception as e:
            print(f"Streaming error: {e}")

    return Response(
        generate(),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
        }
    )
```

**Step 3: Add streaming playback to frontend**

Add to `app/static/js/app.js`:

```javascript
// Streaming audio player using Web Audio API
class StreamingAudioPlayer {
    constructor() {
        this.audioContext = null;
        this.sourceNodes = [];
        this.nextStartTime = 0;
        this.isPlaying = false;
        this.sampleRate = 24000; // Will be updated from stream
    }

    async init() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    reset() {
        this.sourceNodes.forEach(node => {
            try { node.stop(); } catch(e) {}
        });
        this.sourceNodes = [];
        this.nextStartTime = 0;
        this.isPlaying = false;
    }

    async playChunk(audioData) {
        await this.init();

        // Convert Int16 to Float32
        const float32Data = new Float32Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            float32Data[i] = audioData[i] / 32768;
        }

        const audioBuffer = this.audioContext.createBuffer(
            1, float32Data.length, this.sampleRate
        );
        audioBuffer.getChannelData(0).set(float32Data);

        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        const startTime = Math.max(
            this.audioContext.currentTime,
            this.nextStartTime
        );
        source.start(startTime);
        this.nextStartTime = startTime + audioBuffer.duration;

        this.sourceNodes.push(source);
        this.isPlaying = true;
    }
}

const streamingPlayer = new StreamingAudioPlayer();

async function generateWithStreaming(endpoint, body) {
    streamingPlayer.reset();
    showLoading();

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Generation failed');
        }

        const reader = response.body.getReader();
        let headerParsed = false;
        let buffer = new Uint8Array(0);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Append to buffer
            const newBuffer = new Uint8Array(buffer.length + value.length);
            newBuffer.set(buffer);
            newBuffer.set(value, buffer.length);
            buffer = newBuffer;

            // Skip WAV header (44 bytes)
            if (!headerParsed && buffer.length >= 44) {
                // Parse sample rate from header (bytes 24-27)
                const dataView = new DataView(buffer.buffer);
                streamingPlayer.sampleRate = dataView.getUint32(24, true);
                buffer = buffer.slice(44);
                headerParsed = true;
            }

            // Process complete samples (2 bytes per sample for int16)
            if (headerParsed && buffer.length >= 4800) { // ~100ms of audio at 24kHz
                const samplesToProcess = Math.floor(buffer.length / 2) * 2;
                const audioData = new Int16Array(
                    buffer.slice(0, samplesToProcess).buffer
                );
                await streamingPlayer.playChunk(audioData);
                buffer = buffer.slice(samplesToProcess);
            }
        }

        // Process remaining buffer
        if (headerParsed && buffer.length >= 2) {
            // Check for job ID marker
            const text = new TextDecoder().decode(buffer);
            const jobIdMatch = text.match(/<!--JOB_ID:([^>]+)-->/);
            if (jobIdMatch) {
                currentJobId = jobIdMatch[1];
                downloadBtn.disabled = false;
                buffer = buffer.slice(0, buffer.indexOf(60)); // Remove marker
            }

            if (buffer.length >= 2) {
                const samplesToProcess = Math.floor(buffer.length / 2) * 2;
                const audioData = new Int16Array(
                    buffer.slice(0, samplesToProcess).buffer
                );
                await streamingPlayer.playChunk(audioData);
            }
        }

        playBtn.disabled = false;
        playBtn.textContent = '⏸';

    } catch (err) {
        console.error('Streaming error:', err);
        alert('Generation failed: ' + err.message);
    } finally {
        hideLoading();
    }
}
```

**Step 4: Update generate functions to use streaming**

Modify the `generateClone`, `generateCustom`, and `generateDesign` functions to optionally use streaming:

```javascript
// In generateClone function, replace the fetch call:
async function generateClone() {
    const text = textInput.value.trim();
    const language = document.getElementById('clone-language').value;

    if (!text || samples.length === 0) return;

    const refAudioIds = samples.map(s => s.id);
    const refTexts = samples.map(s => s.transcript || null);

    await generateWithStreaming('/api/tts/clone/stream', {
        text,
        language,
        ref_audio_ids: refAudioIds,
        ref_texts: refTexts
    });
}
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add streaming audio playback for TTS generation"
```

---

## Phase 2: Productivity Features

### Task 2A: Generation History

**Goal:** Store and display recent TTS generations with replay capability.

**Files:**
- Create: `app/routes/history.py`
- Modify: `app/__init__.py` (register blueprint)
- Modify: `app/static/index.html` (history panel)
- Modify: `app/static/js/app.js` (history UI logic)
- Modify: `app/static/css/style.css` (history styles)

**Step 1: Create history route**

Create `app/routes/history.py`:

```python
import os
import json
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file
import soundfile as sf

bp = Blueprint('history', __name__, url_prefix='/api/history')

# In-memory history (could be persisted to file/db)
_history = []
MAX_HISTORY = 50


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
    # Check generated audio cache first
    from app.routes.tts import _generated_audio
    if audio_id in _generated_audio:
        import io
        wav, sr = _generated_audio[audio_id]
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        return send_file(buffer, mimetype='audio/wav')

    return jsonify({'error': 'Audio not found'}), 404
```

**Step 2: Register blueprint**

Add to `app/__init__.py`:

```python
from app.routes import tts, stt, audio, profiles, system, history
# ...
app.register_blueprint(history.bp)
```

**Step 3: Add history panel to HTML**

Add to `app/static/index.html` in the sidebar:

```html
<div class="nav-section history-section">
    <h3>HISTORY</h3>
    <div id="history-list" class="history-list"></div>
    <button id="clear-history-btn" class="secondary-btn small">Clear All</button>
</div>
```

**Step 4: Add history JavaScript**

Add to `app/static/js/app.js`:

```javascript
function initHistory() {
    const historyList = document.getElementById('history-list');
    const clearBtn = document.getElementById('clear-history-btn');

    loadHistory();

    clearBtn.addEventListener('click', async () => {
        if (!confirm('Clear all history?')) return;
        await fetch('/api/history/clear', { method: 'POST' });
        loadHistory();
    });

    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const history = await response.json();
            renderHistory(history.reverse()); // Most recent first
        } catch (err) {
            console.error('Error loading history:', err);
        }
    }

    function renderHistory(history) {
        historyList.innerHTML = history.slice(0, 10).map(item => `
            <div class="history-item" data-id="${item.id}" data-audio-id="${item.audio_id}">
                <div class="history-mode">${item.mode}</div>
                <div class="history-text">${item.text.substring(0, 50)}${item.text.length > 50 ? '...' : ''}</div>
                <button class="history-play" data-audio-id="${item.audio_id}">▶</button>
            </div>
        `).join('') || '<p class="empty">No history yet</p>';

        // Add click handlers
        historyList.querySelectorAll('.history-play').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const audioId = e.target.dataset.audioId;
                const response = await fetch(`/api/history/audio/${audioId}`);
                if (response.ok) {
                    const blob = await response.blob();
                    playGeneratedAudio(blob);
                }
            });
        });
    }

    // Export for use by generation functions
    window.addToHistory = async function(mode, text, language, params, audioId) {
        await fetch('/api/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, text, language, params, audio_id: audioId })
        });
        loadHistory();
    };
}

// Add to DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    // ... existing ...
    initHistory();
});
```

**Step 5: Add history styles**

Add to `app/static/css/style.css`:

```css
.history-section {
    margin-top: auto;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
}

.history-list {
    max-height: 200px;
    overflow-y: auto;
    margin-bottom: 0.5rem;
}

.history-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem;
    background: var(--bg-tertiary);
    border-radius: 4px;
    margin-bottom: 0.25rem;
    font-size: 0.75rem;
    cursor: pointer;
}

.history-item:hover {
    background: var(--accent);
}

.history-mode {
    background: var(--accent);
    color: white;
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
    font-size: 0.625rem;
    text-transform: uppercase;
}

.history-text {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.history-play {
    background: none;
    border: none;
    color: var(--text-primary);
    cursor: pointer;
    padding: 0.25rem;
}

.secondary-btn.small {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
}
```

**Step 6: Update generation functions to add to history**

After successful generation, call `addToHistory()`:

```javascript
// Example in generateClone after playGeneratedAudio:
if (currentJobId) {
    window.addToHistory('clone', text, language, { samples: samples.length }, currentJobId);
}
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add generation history with replay"
```

---

### Task 2B: Batch Text Generation

**Goal:** Generate multiple texts in sequence with queue management.

**Files:**
- Modify: `app/static/index.html` (batch input UI)
- Modify: `app/static/js/app.js` (batch processing logic)
- Modify: `app/static/css/style.css` (batch styles)

**Step 1: Add batch input toggle to HTML**

Add below each text input in the mode panels:

```html
<div class="batch-toggle">
    <label>
        <input type="checkbox" class="batch-mode-checkbox">
        Batch Mode (one text per line)
    </label>
</div>
```

**Step 2: Add batch processing JavaScript**

```javascript
function initBatchMode() {
    const batchCheckboxes = document.querySelectorAll('.batch-mode-checkbox');

    batchCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const panel = e.target.closest('.mode-panel');
            const textarea = panel.querySelector('textarea[id$="-text"]');
            if (textarea) {
                textarea.placeholder = e.target.checked
                    ? 'Enter multiple texts, one per line...'
                    : 'Enter the text you want to generate...';
            }
        });
    });
}

async function processBatch(texts, generateFn) {
    const results = [];
    const progressEl = document.createElement('div');
    progressEl.className = 'batch-progress';
    document.body.appendChild(progressEl);

    for (let i = 0; i < texts.length; i++) {
        progressEl.textContent = `Processing ${i + 1} of ${texts.length}...`;
        try {
            await generateFn(texts[i]);
            results.push({ text: texts[i], success: true });
        } catch (err) {
            results.push({ text: texts[i], success: false, error: err.message });
        }
    }

    progressEl.remove();
    return results;
}
```

**Step 3: Add batch styles**

```css
.batch-toggle {
    margin-top: 0.5rem;
    font-size: 0.875rem;
}

.batch-toggle label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    color: var(--text-secondary);
}

.batch-progress {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: var(--bg-secondary);
    padding: 2rem;
    border-radius: 8px;
    z-index: 1000;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add batch text generation mode"
```

---

### Task 2C: Character Count & Text Limits

**Goal:** Show character/word count and warn about length limits.

**Files:**
- Modify: `app/static/index.html` (add counter elements)
- Modify: `app/static/js/app.js` (counter logic)
- Modify: `app/static/css/style.css` (counter styles)

**Step 1: Add counter elements to HTML**

Add below each textarea:

```html
<div class="text-counter">
    <span class="char-count">0</span> characters | <span class="word-count">0</span> words
</div>
```

**Step 2: Add counter JavaScript**

```javascript
function initTextCounters() {
    const textareas = document.querySelectorAll('textarea[id$="-text"]');

    textareas.forEach(textarea => {
        const counter = textarea.parentElement.querySelector('.text-counter');
        if (!counter) return;

        const charCount = counter.querySelector('.char-count');
        const wordCount = counter.querySelector('.word-count');

        function updateCounts() {
            const text = textarea.value;
            charCount.textContent = text.length;
            wordCount.textContent = text.trim() ? text.trim().split(/\s+/).length : 0;

            // Warn if too long (adjust limit as needed)
            if (text.length > 1000) {
                counter.classList.add('warning');
            } else {
                counter.classList.remove('warning');
            }
        }

        textarea.addEventListener('input', updateCounts);
        updateCounts();
    });
}
```

**Step 3: Add counter styles**

```css
.text-counter {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-align: right;
    margin-top: 0.25rem;
}

.text-counter.warning {
    color: var(--error);
}
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add character and word count display"
```

---

## Phase 3: Polish Features

### Task 3A: Audio Waveform Visualization

**Goal:** Display waveform for reference audio and generated output.

**Files:**
- Modify: `app/static/index.html` (add wavesurfer container)
- Modify: `app/static/js/app.js` (wavesurfer integration)
- Modify: `app/static/css/style.css` (waveform styles)

**Dependencies:** wavesurfer.js (CDN)

**Step 1: Add wavesurfer.js CDN to HTML**

Add to `<head>`:

```html
<script src="https://unpkg.com/wavesurfer.js@7"></script>
```

**Step 2: Add waveform containers**

Replace audio player bar with:

```html
<footer class="audio-player-bar">
    <div class="player-controls">
        <button id="play-btn" class="play-btn" disabled>▶</button>
        <div class="waveform-container">
            <div id="waveform"></div>
            <span id="time-display">0:00 / 0:00</span>
        </div>
        <button id="download-btn" class="download-btn" disabled>⬇ Download</button>
    </div>
</footer>
```

**Step 3: Initialize wavesurfer**

```javascript
let wavesurfer = null;

function initWaveform() {
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#a0a0a0',
        progressColor: '#e94560',
        cursorColor: '#e94560',
        height: 48,
        barWidth: 2,
        barGap: 1,
        responsive: true,
    });

    wavesurfer.on('ready', () => {
        playBtn.disabled = false;
        updateTimeDisplay();
    });

    wavesurfer.on('audioprocess', updateTimeDisplay);
    wavesurfer.on('seeking', updateTimeDisplay);

    wavesurfer.on('play', () => {
        playBtn.textContent = '⏸';
    });

    wavesurfer.on('pause', () => {
        playBtn.textContent = '▶';
    });

    wavesurfer.on('finish', () => {
        playBtn.textContent = '▶';
    });

    playBtn.addEventListener('click', () => {
        wavesurfer.playPause();
    });
}

function updateTimeDisplay() {
    const current = formatTime(wavesurfer.getCurrentTime());
    const total = formatTime(wavesurfer.getDuration());
    timeDisplay.textContent = `${current} / ${total}`;
}

function playGeneratedAudio(blob) {
    const url = URL.createObjectURL(blob);
    wavesurfer.load(url);
    wavesurfer.on('ready', () => {
        wavesurfer.play();
    }, { once: true });
    downloadBtn.disabled = false;
}
```

**Step 4: Add waveform styles**

```css
.waveform-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

#waveform {
    width: 100%;
    height: 48px;
    background: var(--bg-tertiary);
    border-radius: 4px;
}
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add waveform visualization with wavesurfer.js"
```

---

### Task 3B: Audio Sample Trimming

**Goal:** Allow users to trim reference audio samples before use.

**Files:**
- Modify: `app/static/index.html` (trim controls)
- Modify: `app/static/js/app.js` (trim logic)
- Create: `app/routes/audio.py` additions (trim endpoint)

**Step 1: Add trim endpoint**

Add to `app/routes/audio.py`:

```python
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
```

**Step 2: Add trim UI to sample items**

Update the sample item rendering to include trim controls:

```javascript
function renderSamples() {
    samplesList.innerHTML = samples.map((sample, index) => `
        <div class="sample-item" data-index="${index}">
            <div class="sample-header">
                <span class="sample-number">Sample ${index + 1}</span>
                <button class="sample-trim" data-index="${index}">✂ Trim</button>
                <button class="sample-remove" data-index="${index}">✕</button>
            </div>
            <div class="sample-waveform" id="sample-waveform-${index}"></div>
            <input type="text" class="sample-transcript" placeholder="Transcript..."
                   value="${sample.transcript}" data-index="${index}">
        </div>
    `).join('');

    // Initialize mini waveforms for each sample
    samples.forEach((sample, index) => {
        if (sample.blobUrl) {
            const ws = WaveSurfer.create({
                container: `#sample-waveform-${index}`,
                waveColor: '#a0a0a0',
                progressColor: '#e94560',
                height: 32,
                barWidth: 1,
            });
            ws.load(sample.blobUrl);
            sample.wavesurfer = ws;
        }
    });
}
```

**Step 3: Add trim modal/dialog**

```javascript
async function showTrimDialog(sampleIndex) {
    const sample = samples[sampleIndex];
    // Implementation: Create modal with waveform and region selection
    // Use wavesurfer regions plugin for selection
    // On confirm, call /api/audio/trim endpoint
}
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add audio sample trimming"
```

---

### Task 3C: Profile Export/Import

**Goal:** Export voice profiles as downloadable files, import from files.

**Files:**
- Modify: `app/routes/profiles.py` (export/import endpoints)
- Modify: `app/static/index.html` (export/import buttons)
- Modify: `app/static/js/app.js` (export/import logic)

**Step 1: Add export endpoint**

Add to `app/routes/profiles.py`:

```python
import zipfile
from flask import send_file

@bp.route('/<profile_id>/export', methods=['GET'])
def export_profile(profile_id):
    """Export a profile as a ZIP file."""
    profiles_dir = get_profiles_dir()
    filepath = os.path.join(profiles_dir, f"{profile_id}.json")
    profile_audio_dir = os.path.join(profiles_dir, profile_id)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Profile not found'}), 404

    try:
        with open(filepath, 'r') as f:
            profile_data = json.load(f)

        # Create ZIP in memory
        import io
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add metadata
            zf.writestr('profile.json', json.dumps(profile_data, indent=2))

            # Add audio files
            for sample in profile_data.get('samples', []):
                audio_path = os.path.join(profile_audio_dir, f"{sample['id']}.wav")
                if os.path.exists(audio_path):
                    zf.write(audio_path, f"audio/{sample['id']}.wav")

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{profile_data['name']}.zip"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/import', methods=['POST'])
def import_profile():
    """Import a profile from a ZIP file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        profiles_dir = get_profiles_dir()

        with zipfile.ZipFile(file, 'r') as zf:
            # Read metadata
            profile_data = json.loads(zf.read('profile.json'))

            # Generate new ID for imported profile
            new_id = str(uuid.uuid4())
            profile_audio_dir = os.path.join(profiles_dir, new_id)
            os.makedirs(profile_audio_dir, exist_ok=True)

            # Extract and rename audio files
            new_samples = []
            for sample in profile_data.get('samples', []):
                old_audio_name = f"audio/{sample['id']}.wav"
                if old_audio_name in zf.namelist():
                    new_sample_id = str(uuid.uuid4())
                    audio_data = zf.read(old_audio_name)
                    new_audio_path = os.path.join(profile_audio_dir, f"{new_sample_id}.wav")
                    with open(new_audio_path, 'wb') as f:
                        f.write(audio_data)
                    new_samples.append({
                        'id': new_sample_id,
                        'transcript': sample.get('transcript', ''),
                    })

            # Save new profile metadata
            from datetime import datetime
            new_profile = {
                'id': new_id,
                'name': profile_data['name'] + ' (imported)',
                'samples': new_samples,
                'created_at': datetime.utcnow().isoformat(),
            }

            filepath = os.path.join(profiles_dir, f"{new_id}.json")
            with open(filepath, 'w') as f:
                json.dump(new_profile, f, indent=2)

            return jsonify({
                'id': new_id,
                'name': new_profile['name'],
                'sample_count': len(new_samples),
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

**Step 2: Add export/import buttons to HTML**

```html
<div class="profile-controls">
    <select id="profile-select">
        <option value="">-- New Profile --</option>
    </select>
    <button id="profile-load-btn" class="profile-btn" disabled>Load</button>
    <button id="profile-save-btn" class="profile-btn" disabled>Save</button>
    <button id="profile-export-btn" class="profile-btn" disabled>Export</button>
    <button id="profile-delete-btn" class="profile-btn danger" disabled>Delete</button>
</div>
<div class="profile-import">
    <input type="file" id="profile-import-input" accept=".zip" hidden>
    <button id="profile-import-btn" class="profile-btn">Import</button>
</div>
```

**Step 3: Add export/import JavaScript**

```javascript
profileExportBtn.addEventListener('click', () => {
    const profileId = profileSelect.value;
    if (profileId) {
        window.location.href = `/api/profiles/${profileId}/export`;
    }
});

profileImportBtn.addEventListener('click', () => {
    profileImportInput.click();
});

profileImportInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        showLoading();
        const response = await fetch('/api/profiles/import', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        alert(`Imported profile "${data.name}"`);
        await loadProfilesList();
        profileSelect.value = data.id;
    } catch (err) {
        alert('Import failed: ' + err.message);
    } finally {
        hideLoading();
        profileImportInput.value = '';
    }
});
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add profile export/import functionality"
```

---

## Summary: Parallel Execution Map

```
Phase 1 (Core UX)
├── Task 1A: GPU Status ──────────┐
│                                 ├── Both can run in parallel
└── Task 1B: Streaming Audio ─────┘

Phase 2 (Productivity) - Start after Phase 1
├── Task 2A: Generation History ──┐
├── Task 2B: Batch Generation ────┼── All three can run in parallel
└── Task 2C: Character Count ─────┘

Phase 3 (Polish) - Start after Phase 2
├── Task 3A: Waveform Visualization ──┐
├── Task 3B: Audio Trimming ──────────┼── All three can run in parallel
└── Task 3C: Profile Export/Import ───┘
```

**Estimated Total:** 9 tasks across 3 phases
