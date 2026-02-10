# Systematic Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all issues from code review: memory leaks, security, reliability, code quality, and minor bugs.

**Architecture:** Extract shared constants into `app/config.py`. Add TTL-based eviction for `_generated_audio`. Add input validation middleware. Consolidate duplicate noise reduction. Add upload cleanup. Fix ZIP import safety. Refactor STT to accept language parameter.

**Tech Stack:** Python/Flask, vanilla JS, existing dependencies only (no new packages).

---

### Task 1: Add `app/config.py` — shared constants and configuration

**Files:**
- Create: `app/config.py`

**Step 1: Create config module**

```python
"""Application constants and configuration."""

# Audio
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1
STREAMING_CHUNK_BYTES = 4800  # ~100ms at 24kHz int16 mono

# Limits
MAX_TEXT_LENGTH = 5000
MAX_INSTRUCT_LENGTH = 500
MAX_UPLOAD_SIZE_MB = 50
MAX_HISTORY_ITEMS = 50
MAX_CACHED_AUDIO = 100
AUDIO_CACHE_TTL_SECONDS = 3600  # 1 hour
UPLOAD_MAX_AGE_SECONDS = 86400  # 24 hours

# Noise reduction
NOISE_REDUCE_STATIONARY = True
NOISE_REDUCE_PROP_DECREASE = 0.75

# Model paths
TTS_MODEL_BASE_PATH = "/data/models/Qwen"
STT_MODEL_CACHE_PATH = "/data/models/whisper"
```

**Step 2: Commit**

```
feat: add centralized config module with shared constants
```

---

### Task 2: Fix memory leak — TTL cache for `_generated_audio`

**Files:**
- Modify: `app/routes/tts.py`
- Modify: `app/routes/history.py`

**Step 1: Replace `_generated_audio` dict with TTL-evicting OrderedDict in `tts.py`**

Replace line 39 `_generated_audio = {}` and add eviction logic:

```python
import time
from collections import OrderedDict
from app.config import MAX_CACHED_AUDIO, AUDIO_CACHE_TTL_SECONDS

# TTL-based audio cache
_generated_audio = OrderedDict()  # job_id -> (wav, sr, timestamp)
_audio_cache_lock = threading.Lock()


def _cache_audio(job_id, wav, sr):
    """Store audio with TTL eviction."""
    now = time.time()
    with _audio_cache_lock:
        _generated_audio[job_id] = (wav, sr, now)
        # Evict expired entries
        expired = [
            k for k, (_, _, ts) in _generated_audio.items()
            if now - ts > AUDIO_CACHE_TTL_SECONDS
        ]
        for k in expired:
            del _generated_audio[k]
        # Evict oldest if over capacity
        while len(_generated_audio) > MAX_CACHED_AUDIO:
            _generated_audio.popitem(last=False)


def get_cached_audio(job_id):
    """Retrieve cached audio, or None if expired/missing."""
    with _audio_cache_lock:
        entry = _generated_audio.get(job_id)
        if entry is None:
            return None
        wav, sr, ts = entry
        if time.time() - ts > AUDIO_CACHE_TTL_SECONDS:
            del _generated_audio[job_id]
            return None
        return wav, sr
```

Replace every `_generated_audio[job_id] = (wav, sr)` with `_cache_audio(job_id, wav, sr)`.

Replace every `_generated_audio[job_id]` read with `get_cached_audio(job_id)`.

Also add `import threading` at top if not present.

**Step 2: Update `history.py` to use the new accessor**

Replace the import `from app.routes.tts import _generated_audio` with `from app.routes.tts import get_cached_audio` and update `get_history_audio()`:

```python
def get_history_audio(audio_id):
    from app.routes.tts import get_cached_audio
    entry = get_cached_audio(audio_id)
    if entry:
        wav, sr = entry
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        return send_file(buffer, mimetype='audio/wav')
    return jsonify({'error': 'Audio not found or expired'}), 404
```

**Step 3: Commit**

```
fix: add TTL cache for generated audio to prevent memory leak
```

---

### Task 3: Add input validation to all TTS endpoints

**Files:**
- Modify: `app/routes/tts.py`

**Step 1: Add validation helper and apply to all endpoints**

Add at top of tts.py after imports:

```python
from app.config import MAX_TEXT_LENGTH, MAX_INSTRUCT_LENGTH


def _validate_text(text, field_name="text"):
    """Validate text input length."""
    if not text:
        return f"{field_name} is required"
    if len(text) > MAX_TEXT_LENGTH:
        return f"{field_name} exceeds maximum length of {MAX_TEXT_LENGTH} characters"
    return None


def _validate_instruct(instruct):
    """Validate instruct input length."""
    if instruct and len(instruct) > MAX_INSTRUCT_LENGTH:
        return f"Instruction exceeds maximum length of {MAX_INSTRUCT_LENGTH} characters"
    return None
```

Apply in each endpoint. For example, in `tts_clone()`:

```python
    text = data.get('text')
    err = _validate_text(text)
    if err:
        return jsonify({'error': err}), 400
```

Apply `_validate_instruct()` in custom and design endpoints similarly.

**Step 2: Commit**

```
fix: add input length validation to all TTS endpoints
```

---

### Task 4: Secure ZIP import against path traversal

**Files:**
- Modify: `app/routes/profiles.py`

**Step 1: Add safe extraction that validates member names**

Replace the ZIP extraction logic in `import_profile()` with validated reads. The current code already reads individual files by name (`zf.read(old_audio_name)`) rather than using `extractall()`, which is good. But we should validate the profile.json data and ensure no path components in sample IDs:

```python
@bp.route('/import', methods=['POST'])
def import_profile():
    """Import a profile from a ZIP file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        profiles_dir = get_profiles_dir()

        with zipfile.ZipFile(file, 'r') as zf:
            # Validate no path traversal in zip entries
            for member in zf.namelist():
                if os.path.isabs(member) or '..' in member:
                    return jsonify({'error': 'Invalid ZIP file: contains unsafe paths'}), 400

            if 'profile.json' not in zf.namelist():
                return jsonify({'error': 'Invalid profile ZIP: missing profile.json'}), 400

            profile_data = json.loads(zf.read('profile.json'))

            # Validate required fields
            if not isinstance(profile_data.get('name'), str) or not profile_data['name'].strip():
                return jsonify({'error': 'Invalid profile: missing or empty name'}), 400
            if not isinstance(profile_data.get('samples'), list):
                return jsonify({'error': 'Invalid profile: missing samples list'}), 400

            new_id = str(uuid.uuid4())
            profile_audio_dir = os.path.join(profiles_dir, new_id)
            os.makedirs(profile_audio_dir, exist_ok=True)

            new_samples = []
            for sample in profile_data.get('samples', []):
                sample_id = sample.get('id', '')
                # Validate sample ID is safe (no path components)
                if not sample_id or '/' in sample_id or '\\' in sample_id or '..' in sample_id:
                    continue
                old_audio_name = f"audio/{sample_id}.wav"
                if old_audio_name in zf.namelist():
                    new_sample_id = str(uuid.uuid4())
                    audio_data = zf.read(old_audio_name)
                    new_audio_path = os.path.join(profile_audio_dir, f"{new_sample_id}.wav")
                    with open(new_audio_path, 'wb') as f:
                        f.write(audio_data)
                    new_samples.append({
                        'id': new_sample_id,
                        'transcript': str(sample.get('transcript', ''))[:1000],
                    })

            if not new_samples:
                shutil.rmtree(profile_audio_dir, ignore_errors=True)
                return jsonify({'error': 'No valid samples found in ZIP'}), 400

            new_profile = {
                'id': new_id,
                'name': profile_data['name'].strip()[:200] + ' (imported)',
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

    except zipfile.BadZipFile:
        return jsonify({'error': 'Invalid ZIP file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

**Step 2: Commit**

```
fix: validate ZIP imports against path traversal and malformed data
```

---

### Task 5: Validate audio IDs to prevent path traversal in audio/tts routes

**Files:**
- Modify: `app/routes/audio.py`
- Modify: `app/routes/tts.py`

**Step 1: Add a shared audio ID validator**

In `app/config.py`, add:

```python
import re

# Audio ID validation (must be UUID format)
AUDIO_ID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def is_valid_audio_id(audio_id: str) -> bool:
    """Validate that an audio ID is a safe UUID string."""
    return bool(AUDIO_ID_PATTERN.match(audio_id))
```

**Step 2: Apply in `audio.py` — validate audio_id parameters**

Add to each endpoint that takes an `audio_id` parameter:

```python
from app.config import is_valid_audio_id

# At the top of delete_audio, stream_audio, trim_audio, get_audio_info:
    if not is_valid_audio_id(audio_id):
        return jsonify({'error': 'Invalid audio ID'}), 400
```

**Step 3: Apply in `tts.py` — validate ref_audio_ids**

In `tts_clone()` and `tts_clone_stream()`, validate each audio_id:

```python
from app.config import is_valid_audio_id

    for audio_id in ref_audio_ids:
        if not is_valid_audio_id(audio_id):
            return jsonify({'error': f'Invalid audio ID: {audio_id}'}), 400
```

Also validate `job_id` in `download_audio()`.

**Step 4: Commit**

```
fix: validate audio IDs to prevent path traversal
```

---

### Task 6: Consolidate duplicate noise reduction into shared utility

**Files:**
- Create: `app/services/audio_utils.py`
- Modify: `app/routes/stt.py`
- Modify: `app/routes/audio.py`

**Step 1: Create shared noise reduction module**

```python
"""Shared audio processing utilities."""
import numpy as np
import soundfile as sf
import noisereduce as nr
from app.config import NOISE_REDUCE_STATIONARY, NOISE_REDUCE_PROP_DECREASE


def reduce_noise(audio_data, sample_rate):
    """Apply noise reduction to audio data array.

    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate in Hz

    Returns:
        Noise-reduced audio as float32 numpy array
    """
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    reduced = nr.reduce_noise(
        y=audio_data,
        sr=sample_rate,
        stationary=NOISE_REDUCE_STATIONARY,
        prop_decrease=NOISE_REDUCE_PROP_DECREASE,
    )
    return reduced.astype(np.float32)


def reduce_noise_file(audio_path):
    """Apply noise reduction to an audio file, return path to cleaned file.

    Args:
        audio_path: path to input audio file

    Returns:
        Path to temporary file with noise-reduced audio
    """
    import tempfile

    data, sr = sf.read(audio_path)
    reduced = reduce_noise(data, sr)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        sf.write(tmp.name, reduced, sr)
        return tmp.name
```

**Step 2: Update `stt.py` to use shared utility**

Remove `apply_noise_reduction()` function, replace with:

```python
from app.services.audio_utils import reduce_noise_file
```

And change line 63 from `denoised_path = apply_noise_reduction(temp_path)` to `denoised_path = reduce_noise_file(temp_path)`.

**Step 3: Update `audio.py` to use shared utility**

Remove local `apply_noise_reduction()` function, replace with:

```python
from app.services.audio_utils import reduce_noise
```

The call site `data = apply_noise_reduction(data, sr)` becomes `data = reduce_noise(data, sr)`.

**Step 4: Commit**

```
refactor: consolidate duplicate noise reduction into shared utility
```

---

### Task 7: Fix STT hardcoded English — accept language parameter

**Files:**
- Modify: `app/services/stt_service.py`
- Modify: `app/routes/stt.py`

**Step 1: Merge transcribe methods in STTService**

Replace the two methods with a single one:

```python
    def transcribe(self, audio_path: str, language: str = None) -> dict:
        """Transcribe audio file to text.

        Args:
            audio_path: path to audio file
            language: language code (e.g., 'en'), or None for auto-detect
        """
        with self._model_lock:
            kwargs = {
                'beam_size': 5,
                'vad_filter': True,
            }
            if language:
                kwargs['language'] = language

            segments, info = self.model.transcribe(audio_path, **kwargs)
            text = " ".join([segment.text.strip() for segment in segments])

            return {
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
            }
```

Keep `transcribe_with_language_detect` as a thin wrapper for backward compatibility:

```python
    def transcribe_with_language_detect(self, audio_path: str) -> dict:
        """Transcribe with auto language detection. Alias for transcribe(path, None)."""
        return self.transcribe(audio_path, language=None)
```

**Step 2: Update STT route to pass language**

In `stt.py`, update the transcription call:

```python
        auto_detect = request.form.get('auto_detect', 'false').lower() == 'true'
        language = None if auto_detect else request.form.get('language', 'en')
        result = stt_service.transcribe(transcribe_path, language=language)
```

**Step 3: Commit**

```
fix: accept language parameter in STT instead of hardcoding English
```

---

### Task 8: Add upload directory cleanup on startup

**Files:**
- Modify: `app/__init__.py`

**Step 1: Add cleanup of stale uploads in `create_app()`**

After `os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)`, add:

```python
    # Clean up stale uploads older than 24 hours
    _cleanup_stale_uploads(app.config['UPLOAD_FOLDER'])
```

And define the helper before `create_app()`:

```python
import time
from app.config import UPLOAD_MAX_AGE_SECONDS


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
```

**Step 2: Commit**

```
feat: clean up stale upload files on startup
```

---

### Task 9: Use config constants in services

**Files:**
- Modify: `app/services/tts_service.py`
- Modify: `app/services/stt_service.py`

**Step 1: Update TTS service to use config path**

Replace `MODEL_BASE_PATH = "/data/models/Qwen"` with:

```python
from app.config import TTS_MODEL_BASE_PATH as MODEL_BASE_PATH
```

**Step 2: Update STT service to use config path**

Replace `MODEL_CACHE_PATH = "/data/models/whisper"` with:

```python
from app.config import STT_MODEL_CACHE_PATH as MODEL_CACHE_PATH
```

**Step 3: Commit**

```
refactor: use centralized config for model paths
```

---

### Task 10: Use config constants in `tts.py` route and `history.py`

**Files:**
- Modify: `app/routes/tts.py`
- Modify: `app/routes/history.py`

**Step 1: Use config in `tts.py`**

Import and use the constant in `create_wav_header` calls. The function already takes parameters, so just import for the streaming chunk size references in `app.js` (backend is fine, the magic number is in JS — see Task 12).

No direct changes needed in tts.py beyond what Task 2 and 3 already cover.

**Step 2: Use config in `history.py`**

Replace `MAX_HISTORY = 50` with:

```python
from app.config import MAX_HISTORY_ITEMS as MAX_HISTORY
```

**Step 3: Commit**

```
refactor: use centralized config constants in routes
```

---

### Task 11: Add error handler and consistent error responses

**Files:**
- Modify: `app/__init__.py`

**Step 1: Register global error handlers in `create_app()`**

Add after blueprint registration:

```python
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500
```

**Step 2: Commit**

```
feat: add global JSON error handlers for consistent API responses
```

---

### Task 12: Extract frontend magic numbers into constants

**Files:**
- Modify: `app/static/js/app.js`

**Step 1: Add constants block at top of `app.js`**

Add after line 1 (before `let currentMode`):

```javascript
// Constants
const DEFAULT_SAMPLE_RATE = 24000;
const WAV_HEADER_SIZE = 44;
const STREAMING_CHUNK_BYTES = 4800;  // ~100ms at 24kHz int16 mono
const INT16_MAX = 32768;
const MAX_TEXT_WARNING_LENGTH = 1000;
const GPU_POLL_INTERVAL_MS = 5000;
const HISTORY_DISPLAY_LIMIT = 10;
```

**Step 2: Replace magic numbers throughout the file**

- Line 15: `this.sampleRate = 24000;` → `this.sampleRate = DEFAULT_SAMPLE_RATE;`
- Line 43: `float32Data[i] = audioData[i] / 32768;` → `float32Data[i] = audioData[i] / INT16_MAX;`
- Line 99: `buffer.length >= 44` → `buffer.length >= WAV_HEADER_SIZE`
- Line 103: `buffer = buffer.slice(44);` → `buffer = buffer.slice(WAV_HEADER_SIZE);`
- Line 108: `buffer.length >= 4800` → `buffer.length >= STREAMING_CHUNK_BYTES`
- Line 203: `setInterval(updateGPUStatus, 5000);` → `setInterval(updateGPUStatus, GPU_POLL_INTERVAL_MS);`
- Line 230: `history.slice(0, 10)` → `history.slice(0, HISTORY_DISPLAY_LIMIT)`
- Line 335: `text.length > 1000` → `text.length > MAX_TEXT_WARNING_LENGTH`

**Step 3: Commit**

```
refactor: extract magic numbers into named constants in frontend
```

---

### Task 13: Fix cert regeneration — skip if cert exists and is valid

**Files:**
- Modify: `run.py`

**Step 1: The cert check already exists**

Looking at `run.py:26-27`:
```python
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file
```

This already skips regeneration. No fix needed — the review was incorrect on this point.

**No changes needed. Skip this task.**

---

### Task 14: Validate profile IDs in profile routes

**Files:**
- Modify: `app/routes/profiles.py`

**Step 1: Add validation at top of each endpoint taking `profile_id`**

```python
from app.config import is_valid_audio_id as is_valid_uuid

# In get_profile, delete_profile, load_profile, export_profile:
    if not is_valid_uuid(profile_id):
        return jsonify({'error': 'Invalid profile ID'}), 400
```

**Step 2: Commit**

```
fix: validate profile IDs to prevent path traversal
```

---

### Task 15: Add XSS protection to history rendering in frontend

**Files:**
- Modify: `app/static/js/app.js`

**Step 1: Add text escaping utility**

Add near the other utility functions:

```javascript
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

**Step 2: Apply in history rendering**

Change `renderHistory` to escape user text:

```javascript
    function renderHistory(history) {
        historyList.innerHTML = history.slice(0, HISTORY_DISPLAY_LIMIT).map(item => `
            <div class="history-item" data-id="${escapeHtml(item.id)}" data-audio-id="${escapeHtml(item.audio_id)}">
                <div class="history-mode">${escapeHtml(item.mode)}</div>
                <div class="history-text">${escapeHtml(item.text.substring(0, 50))}${item.text.length > 50 ? '...' : ''}</div>
                <button class="history-play" data-audio-id="${escapeHtml(item.audio_id)}">&#9654;</button>
            </div>
        `).join('') || '<p class="empty">No history yet</p>';
```

**Step 3: Apply in sample rendering**

Similarly, escape `sample.transcript` in `renderSamples()`:

```javascript
       value="${escapeHtml(sample.transcript)}"
```

**Step 4: Commit**

```
fix: escape HTML in dynamic content to prevent XSS
```

---

## Summary of changes by category

**Memory leaks:** Task 2 (TTL audio cache)
**Security:** Tasks 3 (input validation), 4 (ZIP traversal), 5 (audio ID validation), 14 (profile ID validation), 15 (XSS)
**Reliability:** Tasks 7 (STT language), 8 (upload cleanup), 11 (error handlers)
**Code quality:** Tasks 1 (config), 6 (noise dedup), 9-10 (config usage), 12 (JS constants)
**Minor:** Task 13 (cert — no change needed)
