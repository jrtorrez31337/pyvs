// Constants
const DEFAULT_SAMPLE_RATE = 24000;
const WAV_HEADER_SIZE = 44;
const STREAMING_CHUNK_BYTES = 4800;  // ~100ms at 24kHz int16 mono
const INT16_MAX = 32768;
const MAX_TEXT_WARNING_LENGTH = 1000;
const GPU_POLL_INTERVAL_MS = 5000;
const GPU_MAX_POLL_INTERVAL_MS = 60000;
const HISTORY_DISPLAY_LIMIT = 10;
const MAX_UPLOAD_SIZE_MB = 50;
const MAX_SAMPLES = 10;
const FETCH_TIMEOUT_MS = 300000; // 5 minutes for TTS generation
const MIN_SAMPLE_RATE = 3000;
const MAX_SAMPLE_RATE = 768000;

// State
let currentMode = 'stt';
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentJobId = null;
let currentAudioBlob = null;
let currentBlobUrl = null;
let isGenerating = false;

// Toast notification system
function showToast(message, type = 'error', duration = 5000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Fetch with timeout wrapper
async function fetchWithTimeout(url, options = {}, timeout = FETCH_TIMEOUT_MS) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(id);
        return response;
    } catch (err) {
        clearTimeout(id);
        if (err.name === 'AbortError') throw new Error('Request timed out');
        throw err;
    }
}

// Safe error extraction from response
async function getErrorMessage(response, fallback = 'Request failed') {
    try {
        const data = await response.json();
        return data.error || fallback;
    } catch {
        return fallback;
    }
}

function getCloneSamples() {
    if (typeof window._getCloneSamplesImpl !== 'function') {
        return [];
    }
    return window._getCloneSamplesImpl();
}

// Streaming audio player using Web Audio API
class StreamingAudioPlayer {
    constructor() {
        this.audioContext = null;
        this.sourceNodes = [];
        this.nextStartTime = 0;
        this.isPlaying = false;
        this.sampleRate = DEFAULT_SAMPLE_RATE; // Will be updated from stream
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
            float32Data[i] = audioData[i] / INT16_MAX;
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

async function generateWithStreaming(endpoint, body, onComplete) {
    if (isGenerating) {
        showToast('Generation already in progress', 'warning');
        return;
    }
    isGenerating = true;
    try {
        await _doGenerateStreaming(endpoint, body, onComplete);
    } finally {
        isGenerating = false;
    }
}

async function _doGenerateStreaming(endpoint, body, onComplete) {
    streamingPlayer.reset();

    // Clear stale audio state from previous generation
    currentJobId = null;
    currentAudioBlob = null;

    showLoading();

    const allPcmChunks = [];
    let streamSampleRate = DEFAULT_SAMPLE_RATE;

    try {
        const response = await fetchWithTimeout(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const msg = await getErrorMessage(response, 'Generation failed');
            throw new Error(msg);
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

            // If the backend emitted an early marker instead of audio, surface it clearly.
            if (!headerParsed && buffer.length >= 12) {
                const probeSize = Math.min(buffer.length, 4096);
                const probeText = new TextDecoder().decode(buffer.slice(0, probeSize));
                const errorMatch = probeText.match(/<!--ERROR:([\s\S]+?)-->/);
                if (errorMatch) {
                    throw new Error((errorMatch[1] || 'Generation failed').trim());
                }
            }

            // Parse and validate WAV header before decoding audio data.
            if (!headerParsed && buffer.length >= WAV_HEADER_SIZE) {
                const headerView = new DataView(buffer.buffer, buffer.byteOffset, WAV_HEADER_SIZE);
                const riff = readAscii(headerView, 0, 4);
                const wave = readAscii(headerView, 8, 4);
                if (riff !== 'RIFF' || wave !== 'WAVE') {
                    const probeSize = Math.min(buffer.length, 4096);
                    const probeText = new TextDecoder().decode(buffer.slice(0, probeSize));
                    const errorMatch = probeText.match(/<!--ERROR:([\s\S]+?)-->/);
                    if (errorMatch) {
                        throw new Error((errorMatch[1] || 'Generation failed').trim());
                    }

                    // Marker may be split across chunks; keep reading for a short window.
                    if (probeText.startsWith('<!--ERROR:') && !probeText.includes('-->') && buffer.length < 4096) {
                        continue;
                    }
                    throw new Error('Invalid audio stream header received from server');
                }

                streamSampleRate = headerView.getUint32(24, true);
                if (streamSampleRate < MIN_SAMPLE_RATE || streamSampleRate > MAX_SAMPLE_RATE) {
                    throw new Error(`Invalid stream sample rate in header: ${streamSampleRate}`);
                }

                streamingPlayer.sampleRate = streamSampleRate;
                buffer = buffer.slice(WAV_HEADER_SIZE);
                headerParsed = true;
            }

            // Process complete samples (2 bytes per sample for int16)
            if (headerParsed && buffer.length >= STREAMING_CHUNK_BYTES) {
                const samplesToProcess = Math.floor(buffer.length / 2) * 2;
                const pcmBytes = buffer.slice(0, samplesToProcess);
                allPcmChunks.push(pcmBytes);
                const audioData = new Int16Array(pcmBytes.buffer);
                await streamingPlayer.playChunk(audioData);
                buffer = buffer.slice(samplesToProcess);
            }
        }

        // Process remaining buffer — check tail for backend markers
        let jobId = null;
        if (headerParsed && buffer.length >= 2) {
            const checkSize = Math.min(4096, buffer.length);
            const tailText = new TextDecoder().decode(buffer.slice(buffer.length - checkSize));

            const errorMatch = tailText.match(/<!--ERROR:([\s\S]+?)-->/);
            if (errorMatch) throw new Error((errorMatch[1] || 'Generation failed').trim());

            const jobMatch = tailText.match(/<!--JOB_ID:([^>]+)-->/);
            let pcmEnd = buffer.length;
            if (jobMatch) {
                jobId = jobMatch[1];
                const markerPos = tailText.indexOf('<!--JOB_ID:');
                pcmEnd = buffer.length - checkSize + markerPos;
            }

            const samplesToProcess = Math.floor(pcmEnd / 2) * 2;
            if (samplesToProcess >= 2) {
                const pcmBytes = buffer.slice(0, samplesToProcess);
                allPcmChunks.push(pcmBytes);
                const audioData = new Int16Array(pcmBytes.buffer);
                await streamingPlayer.playChunk(audioData);
            }
        }

        if (!headerParsed) {
            throw new Error('No audio data received from server');
        }

        // Build complete WAV blob for WaveSurfer waveform and download
        currentJobId = jobId;
        const wavBlob = buildWavBlob(allPcmChunks, streamSampleRate);
        currentAudioBlob = wavBlob;

        // Revoke previous blob URL to prevent memory leak
        if (currentBlobUrl) {
            URL.revokeObjectURL(currentBlobUrl);
        }
        currentBlobUrl = URL.createObjectURL(wavBlob);

        // Load into WaveSurfer for waveform display and replay (don't auto-play — user already heard it live)
        playBtn.disabled = true;
        downloadBtn.disabled = true;
        wavesurfer.load(currentBlobUrl);
        wavesurfer.once('ready', () => {
            playBtn.disabled = false;
            downloadBtn.disabled = false;
        });

        if (onComplete) onComplete(jobId);

    } catch (err) {
        console.error('Streaming error:', err);
        showToast('Generation failed: ' + err.message);
    } finally {
        hideLoading();
    }
}

function buildWavBlob(pcmChunks, sampleRate) {
    let totalLength = 0;
    for (const chunk of pcmChunks) totalLength += chunk.length;

    const wavBuffer = new ArrayBuffer(44 + totalLength);
    const view = new DataView(wavBuffer);
    const bytes = new Uint8Array(wavBuffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + totalLength, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);  // PCM
    view.setUint16(22, 1, true);  // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);  // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(view, 36, 'data');
    view.setUint32(40, totalLength, true);

    let offset = 44;
    for (const chunk of pcmChunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }

    return new Blob([wavBuffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function readAscii(view, offset, length) {
    let result = '';
    for (let i = 0; i < length; i++) {
        result += String.fromCharCode(view.getUint8(offset + i));
    }
    return result;
}

// Elements
const navBtns = document.querySelectorAll('.nav-btn');
const modePanels = document.querySelectorAll('.mode-panel');
const outputAudio = document.getElementById('output-audio');
const playBtn = document.getElementById('play-btn');
const timeDisplay = document.getElementById('time-display');
const downloadBtn = document.getElementById('download-btn');
const loadingOverlay = document.getElementById('loading-overlay');

// Wavesurfer instance
let wavesurfer = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSTT();
    initVoiceClone();
    initCustomVoice();
    initVoiceDesign();
    initChatterbox();
    initAudioPlayer();
    initGPUStatus();
    initHistory();
    initBatchMode();
    initTextCounters();
    initTrimModal();
    loadLanguages();
    loadSpeakers();
});

// GPU Status Polling
function initGPUStatus() {
    const gpuStatus = document.getElementById('gpu-status');
    let pollInterval = GPU_POLL_INTERVAL_MS;
    let consecutiveErrors = 0;
    let timerId = null;

    async function updateGPUStatus() {
        try {
            const response = await fetchWithTimeout('/api/system/gpu', {}, 10000);
            const gpus = await response.json();

            if (gpus.length === 0) {
                gpuStatus.textContent = 'GPU: N/A';
                return;
            }

            const statusParts = gpus.map(gpu =>
                `GPU${gpu.index}: ${gpu.utilization}% | ${gpu.memory_used}/${gpu.memory_total}MB | ${gpu.temperature}°C`
            );
            gpuStatus.textContent = statusParts.join(' | ');
            consecutiveErrors = 0;
            pollInterval = GPU_POLL_INTERVAL_MS;
        } catch (err) {
            consecutiveErrors++;
            gpuStatus.textContent = 'GPU: Unavailable';
            pollInterval = Math.min(pollInterval * 2, GPU_MAX_POLL_INTERVAL_MS);
        }
        timerId = null;
        if (document.visibilityState === 'visible') {
            timerId = setTimeout(updateGPUStatus, pollInterval);
        }
    }

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            if (!timerId) updateGPUStatus();
        } else {
            if (timerId) {
                clearTimeout(timerId);
                timerId = null;
            }
        }
    });

    updateGPUStatus();
}

// History
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
        historyList.innerHTML = history.slice(0, HISTORY_DISPLAY_LIMIT).map(item => `
            <div class="history-item" data-id="${escapeHtml(item.id)}" data-audio-id="${escapeHtml(item.audio_id)}">
                <div class="history-mode">${escapeHtml(item.mode)}</div>
                <div class="history-text">${escapeHtml(item.text.substring(0, 50))}${item.text.length > 50 ? '...' : ''}</div>
                <button class="history-play" data-audio-id="${escapeHtml(item.audio_id)}">&#9654;</button>
            </div>
        `).join('') || '<p class="empty">No history yet</p>';

        // Add click handlers
        historyList.querySelectorAll('.history-play').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const audioId = e.target.dataset.audioId;
                try {
                    const response = await fetchWithTimeout(`/api/history/audio/${audioId}`, {}, 30000);
                    if (response.ok) {
                        const blob = await response.blob();
                        playGeneratedAudio(blob);
                    } else {
                        showToast('Audio expired or unavailable', 'warning');
                    }
                } catch (err) {
                    showToast('Failed to load audio: ' + err.message);
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

// Batch Mode
function initBatchMode() {
    const batchCheckboxes = document.querySelectorAll('.batch-mode-checkbox');

    batchCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const mode = e.target.dataset.mode;
            const textarea = document.getElementById(`${mode}-text`);
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

// Check if batch mode is enabled for a mode
function isBatchMode(mode) {
    const checkbox = document.querySelector(`.batch-mode-checkbox[data-mode="${mode}"]`);
    return checkbox && checkbox.checked;
}

// Get texts for batch or single mode
function getTextsForMode(mode) {
    const textarea = document.getElementById(`${mode}-text`);
    const text = textarea.value.trim();
    if (!text) return [];

    if (isBatchMode(mode)) {
        return text.split('\n').map(t => t.trim()).filter(t => t.length > 0);
    }
    return [text];
}

// Text Counters
function initTextCounters() {
    const textareas = document.querySelectorAll('#clone-text, #custom-text, #design-text, #chatterbox-text');

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
            if (text.length > MAX_TEXT_WARNING_LENGTH) {
                counter.classList.add('warning');
            } else {
                counter.classList.remove('warning');
            }
        }

        textarea.addEventListener('input', updateCounts);
        updateCounts();
    });
}

// Trim Modal
let trimWavesurfer = null;
let trimSampleIndex = null;
let trimAudioId = null;
let trimPreviewInterval = null;

function showTrimModal(sampleIndex, audioId, blobUrl) {
    const modal = document.getElementById('trim-modal');
    const trimStart = document.getElementById('trim-start');
    const trimEnd = document.getElementById('trim-end');

    trimSampleIndex = sampleIndex;
    trimAudioId = audioId;

    // Initialize or recreate wavesurfer for trim modal
    if (trimWavesurfer) {
        trimWavesurfer.destroy();
    }

    trimWavesurfer = WaveSurfer.create({
        container: '#trim-waveform',
        waveColor: '#a0a0a0',
        progressColor: '#e94560',
        cursorColor: '#e94560',
        height: 80,
        barWidth: 2,
        barGap: 1,
    });

    trimWavesurfer.load(blobUrl);

    trimWavesurfer.on('ready', () => {
        const duration = trimWavesurfer.getDuration();
        trimStart.value = 0;
        trimStart.max = duration;
        trimEnd.value = duration.toFixed(1);
        trimEnd.max = duration;
    });

    modal.classList.remove('hidden');
}

function initTrimModal() {
    const modal = document.getElementById('trim-modal');
    const trimStart = document.getElementById('trim-start');
    const trimEnd = document.getElementById('trim-end');
    const previewBtn = document.getElementById('trim-preview-btn');
    const saveBtn = document.getElementById('trim-save-btn');
    const cancelBtn = document.getElementById('trim-cancel-btn');

    previewBtn.addEventListener('click', () => {
        if (trimWavesurfer) {
            if (trimPreviewInterval) clearInterval(trimPreviewInterval);
            const start = parseFloat(trimStart.value) || 0;
            const end = parseFloat(trimEnd.value) || trimWavesurfer.getDuration();
            trimWavesurfer.setTime(start);
            trimWavesurfer.play();
            // Stop at end time
            trimPreviewInterval = setInterval(() => {
                if (!trimWavesurfer || trimWavesurfer.getCurrentTime() >= end) {
                    if (trimWavesurfer) trimWavesurfer.pause();
                    clearInterval(trimPreviewInterval);
                    trimPreviewInterval = null;
                }
            }, 100);
        }
    });

    saveBtn.addEventListener('click', async () => {
        if (trimPreviewInterval) {
            clearInterval(trimPreviewInterval);
            trimPreviewInterval = null;
        }
        const start = parseFloat(trimStart.value) || 0;
        const end = parseFloat(trimEnd.value);

        try {
            showLoading();
            const response = await fetch(`/api/audio/trim/${trimAudioId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start, end })
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // Update sample with new trimmed audio
            if (window.updateSampleWithTrimmed) {
                window.updateSampleWithTrimmed(trimSampleIndex, data.id);
            }

            closeTrimModal();
        } catch (err) {
            showToast('Trim failed: ' + err.message);
        } finally {
            hideLoading();
        }
    });

    cancelBtn.addEventListener('click', closeTrimModal);

    // Close on click outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeTrimModal();
        }
    });
}

function closeTrimModal() {
    const modal = document.getElementById('trim-modal');
    modal.classList.add('hidden');
    if (trimPreviewInterval) {
        clearInterval(trimPreviewInterval);
        trimPreviewInterval = null;
    }
    if (trimWavesurfer) {
        try { trimWavesurfer.destroy(); } catch (e) { /* ignore */ }
        trimWavesurfer = null;
    }
    trimSampleIndex = null;
    trimAudioId = null;
}

// Navigation
function initNavigation() {
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            switchMode(mode);
        });
    });
}

function switchMode(mode) {
    currentMode = mode;

    navBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    modePanels.forEach(panel => {
        panel.classList.toggle('active', panel.id === `${mode}-panel`);
    });
}

// STT Mode
function initSTT() {
    const micBtn = document.getElementById('mic-btn');
    const recordingIndicator = document.getElementById('recording-indicator');
    const sttResult = document.getElementById('stt-result');
    const useTextBtn = document.getElementById('use-text-btn');

    // Check if microphone is available in this context
    checkMicrophoneAvailability();

    micBtn.addEventListener('click', async () => {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    });

    function checkMicrophoneAvailability() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
            if (!isSecure) {
                sttResult.value = 'Microphone access requires HTTPS. Please access this page using https:// instead of http://';
                sttResult.style.color = '#f87171';
                micBtn.disabled = true;
                micBtn.style.opacity = '0.5';
                micBtn.style.cursor = 'not-allowed';
            }
        }
    }

    useTextBtn.addEventListener('click', () => {
        const text = sttResult.value;
        if (text) {
            // Copy text to active TTS mode
            document.getElementById('clone-text').value = text;
            document.getElementById('custom-text').value = text;
            document.getElementById('design-text').value = text;
            document.getElementById('chatterbox-text').value = text;
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    noiseSuppression: true,
                    echoCancellation: true,
                    autoGainControl: true,
                }
            });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                await transcribeAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add('recording');
            micBtn.querySelector('.mic-text').textContent = 'Click to Stop';
            recordingIndicator.classList.remove('hidden');
        } catch (err) {
            console.error('Error accessing microphone:', err);
            let message = 'Could not access microphone. ';
            if (err.name === 'NotAllowedError') {
                message += 'Please grant microphone permission in your browser settings.';
            } else if (err.name === 'NotFoundError') {
                message += 'No microphone found. Please connect a microphone.';
            } else if (err.name === 'NotSupportedError' || err.name === 'TypeError') {
                const isSecure = location.protocol === 'https:' || location.hostname === 'localhost';
                if (!isSecure) {
                    message = 'Microphone access requires HTTPS. Please access this page using https:// instead of http://';
                } else {
                    message += 'Your browser does not support microphone access.';
                }
            } else {
                message += err.message || 'Unknown error occurred.';
            }
            showToast(message);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        micBtn.classList.remove('recording');
        micBtn.querySelector('.mic-text').textContent = 'Click to Record';
        recordingIndicator.classList.add('hidden');
    }

    async function transcribeAudio(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        // Add denoise preference
        const denoise = document.getElementById('stt-denoise').checked;
        formData.append('denoise', denoise.toString());

        try {
            showLoading();
            const response = await fetchWithTimeout('/api/stt', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            sttResult.value = data.text;
            useTextBtn.disabled = !data.text;
        } catch (err) {
            console.error('Transcription error:', err);
            showToast('Transcription failed: ' + err.message);
        } finally {
            hideLoading();
        }
    }
}

// Voice Clone Mode
function initVoiceClone() {
    const uploadZone = document.getElementById('clone-upload-zone');
    const recordZone = document.getElementById('clone-record-zone');
    const audioInput = document.getElementById('clone-audio-input');
    const samplesList = document.getElementById('clone-samples-list');
    const generateBtn = document.getElementById('clone-generate-btn');
    const textInput = document.getElementById('clone-text');
    const uploadBtn = document.getElementById('clone-upload-btn');
    const recordBtn = document.getElementById('clone-record-btn');
    const micBtn = document.getElementById('clone-mic-btn');
    const recordingIndicator = document.getElementById('clone-recording-indicator');
    const cloneEngineSelect = document.getElementById('clone-engine');
    const qwenLanguageGroup = document.getElementById('clone-qwen-language-group');
    const chatterboxLanguageGroup = document.getElementById('clone-chatterbox-language-group');
    const qwenOptions = document.getElementById('clone-qwen-options');
    const chatterboxOptions = document.getElementById('clone-chatterbox-options');
    const cloneExaggeration = document.getElementById('clone-exaggeration');
    const cloneExaggerationValue = document.getElementById('clone-exaggeration-value');
    const cloneCfg = document.getElementById('clone-cfg');
    const cloneCfgValue = document.getElementById('clone-cfg-value');
    const cloneTemperature = document.getElementById('clone-temperature');
    const cloneTemperatureValue = document.getElementById('clone-temperature-value');

    // Profile elements
    const profileSelect = document.getElementById('profile-select');
    const profileLoadBtn = document.getElementById('profile-load-btn');
    const profileSaveBtn = document.getElementById('profile-save-btn');
    const profileDeleteBtn = document.getElementById('profile-delete-btn');

    // Multiple samples storage: [{id, blobUrl, transcript}, ...]
    let samples = [];
    window._getCloneSamplesImpl = () => samples.map(sample => ({
        id: sample.id,
        transcript: sample.transcript || null,
    }));

    // Initialize profiles
    loadProfilesList();
    loadChatterboxLanguages();
    let cloneMediaRecorder = null;
    let cloneAudioChunks = [];
    let isCloneRecording = false;
    let currentSource = 'upload';

    // Source toggle
    uploadBtn.addEventListener('click', () => switchSource('upload'));
    recordBtn.addEventListener('click', () => switchSource('record'));

    function switchSource(source) {
        currentSource = source;
        uploadBtn.classList.toggle('active', source === 'upload');
        recordBtn.classList.toggle('active', source === 'record');
        uploadZone.classList.toggle('hidden', source !== 'upload');
        recordZone.classList.toggle('hidden', source !== 'record');
    }

    function updateCloneEngineUI() {
        const useChatterbox = cloneEngineSelect.value === 'chatterbox';
        qwenLanguageGroup.classList.toggle('hidden', useChatterbox);
        chatterboxLanguageGroup.classList.toggle('hidden', !useChatterbox);
        qwenOptions.classList.toggle('hidden', useChatterbox);
        chatterboxOptions.classList.toggle('hidden', !useChatterbox);
    }

    cloneEngineSelect.addEventListener('change', updateCloneEngineUI);

    cloneExaggeration.addEventListener('input', () => {
        cloneExaggerationValue.textContent = cloneExaggeration.value;
    });
    cloneCfg.addEventListener('input', () => {
        cloneCfgValue.textContent = cloneCfg.value;
    });
    cloneTemperature.addEventListener('input', () => {
        cloneTemperatureValue.textContent = cloneTemperature.value;
    });

    updateCloneEngineUI();

    // Upload zone events
    uploadZone.addEventListener('click', () => audioInput.click());
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        if (!file.type.startsWith('audio/')) {
            showToast('Please drop an audio file (MP3, WAV, etc.)', 'warning');
            return;
        }
        if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
            showToast(`File too large. Maximum size is ${MAX_UPLOAD_SIZE_MB}MB.`, 'warning');
            return;
        }
        addSampleFromFile(file);
    });

    audioInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
                showToast(`File too large. Maximum size is ${MAX_UPLOAD_SIZE_MB}MB.`, 'warning');
                audioInput.value = '';
                return;
            }
            addSampleFromFile(file);
            audioInput.value = ''; // Reset for same file selection
        }
    });

    // Microphone recording events
    micBtn.addEventListener('click', async () => {
        if (!isCloneRecording) {
            await startCloneRecording();
        } else {
            stopCloneRecording();
        }
    });

    async function startCloneRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    noiseSuppression: true,
                    echoCancellation: true,
                    autoGainControl: true,
                }
            });
            cloneMediaRecorder = new MediaRecorder(stream);
            cloneAudioChunks = [];

            cloneMediaRecorder.ondataavailable = (e) => {
                cloneAudioChunks.push(e.data);
            };

            cloneMediaRecorder.onstop = async () => {
                micBtn.querySelector('.mic-status').textContent = 'Processing...';
                recordingIndicator.querySelector('.status-text').textContent = 'Uploading audio...';

                const audioBlob = new Blob(cloneAudioChunks, { type: 'audio/webm' });
                stream.getTracks().forEach(track => track.stop());

                const denoise = document.getElementById('clone-denoise').checked;
                recordingIndicator.querySelector('.status-text').textContent = denoise ? 'Denoising & processing...' : 'Processing audio...';

                // Upload and transcribe in parallel
                const [uploadResult, transcriptResult] = await Promise.all([
                    uploadAudio(audioBlob, denoise),
                    transcribeAudio(audioBlob, denoise)
                ]);

                if (uploadResult) {
                    const transcript = transcriptResult?.text || '';
                    addSample(uploadResult.id, URL.createObjectURL(audioBlob), transcript);
                }

                // Reset recording UI
                recordingIndicator.classList.add('hidden');
                recordingIndicator.querySelector('.status-text').textContent = 'Recording...';
                micBtn.querySelector('.mic-status').textContent = 'Click to Record';
            };

            cloneMediaRecorder.start();
            isCloneRecording = true;
            micBtn.classList.add('recording');
            micBtn.querySelector('.mic-status').textContent = 'Click to Stop';
            recordingIndicator.classList.remove('hidden');
        } catch (err) {
            console.error('Error accessing microphone:', err);
            let message = 'Could not access microphone. ';
            if (err.name === 'NotAllowedError') {
                message += 'Please grant microphone permission in your browser settings.';
            } else if (err.name === 'NotFoundError') {
                message += 'No microphone found. Please connect a microphone.';
            } else if (err.name === 'NotSupportedError' || err.name === 'TypeError') {
                const isSecure = location.protocol === 'https:' || location.hostname === 'localhost';
                if (!isSecure) {
                    message = 'Microphone access requires HTTPS. Please access this page using https:// instead of http://';
                } else {
                    message += 'Your browser does not support microphone access.';
                }
            } else {
                message += err.message || 'Unknown error occurred.';
            }
            showToast(message);
        }
    }

    function stopCloneRecording() {
        if (cloneMediaRecorder && cloneMediaRecorder.state !== 'inactive') {
            cloneMediaRecorder.stop();
        }
        isCloneRecording = false;
        micBtn.classList.remove('recording');
    }

    async function addSampleFromFile(file) {
        showLoading();
        const denoise = document.getElementById('clone-denoise')?.checked ?? true;

        try {
            const [uploadResult, transcriptResult] = await Promise.all([
                uploadAudio(file, denoise),
                transcribeAudio(file, denoise)
            ]);

            if (uploadResult) {
                const transcript = transcriptResult?.text || '';
                addSample(uploadResult.id, URL.createObjectURL(file), transcript);
            }
        } finally {
            hideLoading();
        }
    }

    function addSample(id, blobUrl, transcript) {
        if (samples.length >= MAX_SAMPLES) {
            showToast(`Maximum ${MAX_SAMPLES} samples allowed. Remove one to add more.`, 'warning');
            return;
        }
        samples.push({ id, blobUrl, transcript });
        renderSamples();
        updateGenerateButton();
        updateProfileButtons();
    }

    function removeSample(index) {
        if (samples[index] && samples[index].blobUrl) {
            URL.revokeObjectURL(samples[index].blobUrl);
        }
        samples.splice(index, 1);
        renderSamples();
        updateGenerateButton();
        updateProfileButtons();
    }

    function updateSampleTranscript(index, transcript) {
        if (samples[index]) {
            samples[index].transcript = transcript;
        }
    }

    // Export for trim modal
    window.updateSampleWithTrimmed = async function(index, newAudioId) {
        if (samples[index]) {
            // Fetch new audio blob URL
            const response = await fetch(`/api/audio/stream/${newAudioId}`);
            if (response.ok) {
                const blob = await response.blob();
                // Revoke old blob URL before replacing
                if (samples[index].blobUrl) {
                    URL.revokeObjectURL(samples[index].blobUrl);
                }
                samples[index].id = newAudioId;
                samples[index].blobUrl = URL.createObjectURL(blob);
                renderSamples();
            }
        }
    };

    function renderSamples() {
        samplesList.innerHTML = samples.map((sample, index) => `
            <div class="sample-item" data-index="${index}">
                <div class="sample-header">
                    <span class="sample-number">Sample ${index + 1}</span>
                    <div class="sample-audio">
                        <audio src="${sample.blobUrl}" controls></audio>
                    </div>
                    <button class="sample-trim" data-index="${index}" data-id="${escapeHtml(sample.id)}">✂</button>
                    <button class="sample-remove" data-index="${index}">✕</button>
                </div>
                <input type="text" class="sample-transcript" placeholder="Transcript of this sample..."
                       value="${escapeHtml(sample.transcript)}" data-index="${index}">
            </div>
        `).join('');

        // Add event listeners
        samplesList.querySelectorAll('.sample-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                removeSample(parseInt(e.target.dataset.index));
            });
        });

        samplesList.querySelectorAll('.sample-trim').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                const audioId = e.target.dataset.id;
                showTrimModal(index, audioId, samples[index].blobUrl);
            });
        });

        samplesList.querySelectorAll('.sample-transcript').forEach(input => {
            input.addEventListener('input', (e) => {
                updateSampleTranscript(parseInt(e.target.dataset.index), e.target.value);
            });
        });
    }

    function updateGenerateButton() {
        generateBtn.disabled = samples.length === 0 || !textInput.value.trim();
    }

    generateBtn.addEventListener('click', generateClone);
    textInput.addEventListener('input', updateGenerateButton);

    async function uploadAudio(fileOrBlob, denoise = true) {
        const formData = new FormData();
        const filename = fileOrBlob.name || 'recording.webm';
        formData.append('audio', fileOrBlob, filename);
        formData.append('denoise', denoise.toString());

        try {
            const response = await fetchWithTimeout('/api/audio/upload', {
                method: 'POST',
                body: formData
            }, 120000);
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            return data;
        } catch (err) {
            console.error('Upload error:', err);
            showToast('Upload failed: ' + err.message);
            return null;
        }
    }

    async function transcribeAudio(fileOrBlob, denoise = true) {
        const formData = new FormData();
        const filename = fileOrBlob.name || 'recording.webm';
        formData.append('audio', fileOrBlob, filename);
        formData.append('denoise', denoise.toString());

        try {
            const response = await fetchWithTimeout('/api/stt', {
                method: 'POST',
                body: formData
            }, 120000);
            const data = await response.json();
            if (data.error) {
                console.error('Transcription error:', data.error);
                return null;
            }
            return data;
        } catch (err) {
            console.error('Transcription error:', err);
            return null;
        }
    }

    async function generateClone() {
        if (isGenerating) {
            showToast('Generation already in progress', 'warning');
            return;
        }
        const texts = getTextsForMode('clone');
        const engine = cloneEngineSelect.value;

        if (texts.length === 0) {
            showToast('Please enter text to generate speech', 'warning');
            return;
        }
        if (samples.length === 0) {
            showToast('Please add at least one reference audio sample', 'warning');
            return;
        }

        const refAudioIds = samples.map(s => s.id);
        const refTexts = samples.map(s => s.transcript || null);
        const fast = document.getElementById('clone-fast-mode').checked;
        const qwenLanguage = document.getElementById('clone-language').value;
        const chatterboxLanguage = document.getElementById('clone-chatterbox-language').value || 'en';
        const exaggeration = parseFloat(cloneExaggeration.value);
        const cfgWeight = parseFloat(cloneCfg.value);
        const temperature = parseFloat(cloneTemperature.value);

        isGenerating = true;
        try {
            if (engine === 'chatterbox' && samples.length > 1) {
                showToast('Chatterbox clone uses Sample 1 as the reference voice.', 'info', 3500);
            }

            for (const text of texts) {
                if (engine === 'chatterbox') {
                    showLoading();
                    try {
                        const response = await fetchWithTimeout('/api/tts/chatterbox/generate', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                text,
                                language_id: chatterboxLanguage,
                                ref_audio_id: refAudioIds[0],
                                exaggeration,
                                cfg_weight: cfgWeight,
                                temperature,
                            })
                        });

                        if (!response.ok) {
                            const msg = await getErrorMessage(response, 'Generation failed');
                            throw new Error(msg);
                        }

                        currentJobId = response.headers.get('X-Job-Id');
                        const audioBlob = await response.blob();
                        playGeneratedAudio(audioBlob);

                        if (currentJobId) {
                            window.addToHistory(
                                'clone',
                                text,
                                chatterboxLanguage,
                                {
                                    engine: 'chatterbox',
                                    samples: samples.length,
                                    exaggeration,
                                    cfg_weight: cfgWeight,
                                    temperature,
                                    ref_sample_index: 0,
                                },
                                currentJobId
                            );
                        }
                    } catch (err) {
                        showToast('Generation failed: ' + err.message);
                    } finally {
                        hideLoading();
                    }
                } else {
                    await _doGenerateStreaming('/api/tts/clone/stream', {
                        text,
                        language: qwenLanguage,
                        ref_audio_ids: refAudioIds,
                        ref_texts: refTexts,
                        fast
                    }, (jobId) => {
                        if (jobId) {
                            window.addToHistory(
                                'clone',
                                text,
                                qwenLanguage,
                                { engine: 'qwen', samples: samples.length, fast },
                                jobId
                            );
                        }
                    });
                }
            }
            if (texts.length > 1) showToast(`Batch complete: ${texts.length} items generated`, 'success');
        } finally {
            isGenerating = false;
        }
    }

    // ===== Profile Management =====

    const profileExportBtn = document.getElementById('profile-export-btn');
    const profileImportBtn = document.getElementById('profile-import-btn');
    const profileImportInput = document.getElementById('profile-import-input');

    profileSelect.addEventListener('change', () => {
        const hasSelection = profileSelect.value !== '';
        profileLoadBtn.disabled = !hasSelection;
        profileExportBtn.disabled = !hasSelection;
        profileDeleteBtn.disabled = !hasSelection;
    });

    profileSaveBtn.addEventListener('click', saveProfile);
    profileLoadBtn.addEventListener('click', loadProfile);
    profileDeleteBtn.addEventListener('click', deleteProfile);

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

            showToast(`Imported profile "${data.name}"`, 'success');
            await loadProfilesList();
            profileSelect.value = data.id;
            profileLoadBtn.disabled = false;
            profileExportBtn.disabled = false;
            profileDeleteBtn.disabled = false;
        } catch (err) {
            showToast('Import failed: ' + err.message);
        } finally {
            hideLoading();
            profileImportInput.value = '';
        }
    });

    function updateProfileButtons() {
        profileSaveBtn.disabled = samples.length === 0;
    }

    async function loadProfilesList() {
        try {
            const response = await fetch('/api/profiles');
            const profiles = await response.json();

            // Clear existing options except the first one
            profileSelect.innerHTML = '<option value="">-- New Profile --</option>';

            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name} (${profile.sample_count} samples)`;
                profileSelect.appendChild(option);
            });
        } catch (err) {
            console.error('Error loading profiles:', err);
        }
    }

    async function saveProfile() {
        if (samples.length === 0) {
            showToast('Add at least one sample first', 'warning');
            return;
        }

        const name = prompt('Enter a name for this voice profile:');
        if (!name || !name.trim()) return;

        try {
            showLoading();
            const response = await fetch('/api/profiles', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name.trim(),
                    samples: samples.map(s => ({
                        id: s.id,
                        transcript: s.transcript
                    }))
                })
            });

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            showToast(`Profile "${name}" saved`, 'success');
            await loadProfilesList();

            // Select the newly created profile
            profileSelect.value = data.id;
            profileLoadBtn.disabled = false;
            profileDeleteBtn.disabled = false;
        } catch (err) {
            console.error('Error saving profile:', err);
            showToast('Failed to save profile: ' + err.message);
        } finally {
            hideLoading();
        }
    }

    async function loadProfile() {
        const profileId = profileSelect.value;
        if (!profileId) return;

        try {
            showLoading();
            const response = await fetch(`/api/profiles/${profileId}/load`, {
                method: 'POST'
            });

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            // Fetch all new sample blobs first, then swap
            const newSamples = [];
            for (const sample of data.samples) {
                const audioResponse = await fetch(`/api/audio/stream/${sample.id}`);
                if (audioResponse.ok) {
                    const blob = await audioResponse.blob();
                    newSamples.push({
                        id: sample.id,
                        blobUrl: URL.createObjectURL(blob),
                        transcript: sample.transcript || ''
                    });
                } else {
                    newSamples.push({
                        id: sample.id,
                        blobUrl: '',
                        transcript: sample.transcript || ''
                    });
                }
            }
            // Only revoke old URLs after all fetches succeed
            samples.forEach(s => { if (s.blobUrl) URL.revokeObjectURL(s.blobUrl); });
            samples = newSamples;

            renderSamples();
            updateGenerateButton();
            updateProfileButtons();
            showToast(`Loaded profile "${data.profile_name}" with ${samples.length} samples`, 'success');
        } catch (err) {
            console.error('Error loading profile:', err);
            showToast('Failed to load profile: ' + err.message);
        } finally {
            hideLoading();
        }
    }

    async function deleteProfile() {
        const profileId = profileSelect.value;
        if (!profileId) return;

        const selectedOption = profileSelect.options[profileSelect.selectedIndex];
        const profileName = selectedOption.textContent;

        if (!confirm(`Delete profile "${profileName}"? This cannot be undone.`)) {
            return;
        }

        try {
            showLoading();
            const response = await fetch(`/api/profiles/${profileId}`, {
                method: 'DELETE'
            });

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            await loadProfilesList();
            profileSelect.value = '';
            profileLoadBtn.disabled = true;
            profileDeleteBtn.disabled = true;
            showToast('Profile deleted', 'success');
        } catch (err) {
            console.error('Error deleting profile:', err);
            showToast('Failed to delete profile: ' + err.message);
        } finally {
            hideLoading();
        }
    }

}

// Custom Voice Mode
function initCustomVoice() {
    const generateBtn = document.getElementById('custom-generate-btn');
    const textInput = document.getElementById('custom-text');

    generateBtn.addEventListener('click', generateCustom);

    textInput.addEventListener('input', () => {
        generateBtn.disabled = !textInput.value.trim();
    });

    async function generateCustom() {
        if (isGenerating) {
            showToast('Generation already in progress', 'warning');
            return;
        }
        const texts = getTextsForMode('custom');
        const language = document.getElementById('custom-language').value;
        const speaker = document.getElementById('custom-speaker').value;
        const instruct = document.getElementById('custom-instruct').value.trim();
        const useClone = document.getElementById('custom-use-clone').checked;

        if (texts.length === 0) {
            showToast('Please enter text to generate speech', 'warning');
            return;
        }

        const fast = document.getElementById('custom-fast-mode').checked;
        const cloneSamples = useClone ? getCloneSamples() : [];
        if (useClone && cloneSamples.length === 0) {
            showToast('Add reference samples in Voice Clone tab first', 'warning');
            return;
        }

        isGenerating = true;
        try {
            if (useClone) {
                showToast('Applying clone reference conditioning with Custom Voice controls.', 'info', 3000);
            }
            for (const text of texts) {
                await _doGenerateStreaming('/api/tts/custom/stream', {
                    text,
                    language,
                    speaker,
                    instruct: instruct || null,
                    fast,
                    ref_audio_ids: useClone ? cloneSamples.map(s => s.id) : [],
                    ref_texts: useClone ? cloneSamples.map(s => s.transcript) : [],
                }, (jobId) => {
                    if (jobId) {
                        window.addToHistory('custom', text, language, {
                            speaker,
                            instruct: instruct || null,
                            via_clone: useClone
                        }, jobId);
                    }
                });
            }
            if (texts.length > 1) showToast(`Batch complete: ${texts.length} items generated`, 'success');
        } finally {
            isGenerating = false;
        }
    }
}

// Voice Design Mode
function initVoiceDesign() {
    const generateBtn = document.getElementById('design-generate-btn');
    const textInput = document.getElementById('design-text');
    const instructInput = document.getElementById('design-instruct');
    const useCloneToggle = document.getElementById('design-use-clone');

    generateBtn.addEventListener('click', generateDesign);

    function updateButtonState() {
        const hasText = !!textInput.value.trim();
        const hasInstruct = !!instructInput.value.trim();
        generateBtn.disabled = !hasText || !hasInstruct;
    }

    textInput.addEventListener('input', updateButtonState);
    instructInput.addEventListener('input', updateButtonState);
    useCloneToggle.addEventListener('change', updateButtonState);

    async function generateDesign() {
        if (isGenerating) {
            showToast('Generation already in progress', 'warning');
            return;
        }
        const texts = getTextsForMode('design');
        const language = document.getElementById('design-language').value;
        const instruct = instructInput.value.trim();
        const useClone = useCloneToggle.checked;
        const cloneSamples = useClone ? getCloneSamples() : [];

        if (!texts.length) {
            showToast('Please enter text to generate speech', 'warning');
            return;
        }
        if (useClone && cloneSamples.length === 0) {
            showToast('Add reference samples in Voice Clone tab first', 'warning');
            return;
        }
        if (!instruct) {
            showToast('Please enter a voice description', 'warning');
            return;
        }

        isGenerating = true;
        try {
            if (useClone) {
                showToast('Applying clone reference conditioning with Voice Design instruction.', 'info', 3000);
            }
            for (const text of texts) {
                await _doGenerateStreaming('/api/tts/design/stream', {
                    text,
                    language,
                    instruct,
                    ref_audio_ids: useClone ? cloneSamples.map(s => s.id) : [],
                    ref_texts: useClone ? cloneSamples.map(s => s.transcript) : [],
                }, (jobId) => {
                    if (jobId) window.addToHistory('design', text, language, { instruct, via_clone: useClone }, jobId);
                });
            }
            if (texts.length > 1) showToast(`Batch complete: ${texts.length} items generated`, 'success');
        } finally {
            isGenerating = false;
        }
    }

    updateButtonState();
}

// Chatterbox Mode
function initChatterbox() {
    const generateBtn = document.getElementById('chatterbox-generate-btn');
    const textInput = document.getElementById('chatterbox-text');
    const exaggerationSlider = document.getElementById('chatterbox-exaggeration');
    const exaggerationValue = document.getElementById('exaggeration-value');
    const cfgSlider = document.getElementById('chatterbox-cfg');
    const cfgValue = document.getElementById('cfg-value');
    const uploadZone = document.getElementById('chatterbox-upload-zone');
    const audioInput = document.getElementById('chatterbox-audio-input');
    const refPreview = document.getElementById('chatterbox-ref-preview');
    const refAudio = document.getElementById('chatterbox-ref-audio');
    const refRemove = document.getElementById('chatterbox-ref-remove');
    const useCloneRefToggle = document.getElementById('chatterbox-use-clone-ref');

    let refAudioId = null;

    // Load Chatterbox languages
    loadChatterboxLanguages();

    // Slider value display
    exaggerationSlider.addEventListener('input', () => {
        exaggerationValue.textContent = exaggerationSlider.value;
    });
    cfgSlider.addEventListener('input', () => {
        cfgValue.textContent = cfgSlider.value;
    });

    // Reference audio upload
    uploadZone.addEventListener('click', () => audioInput.click());
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        if (!file.type.startsWith('audio/')) {
            showToast('Please drop an audio file (MP3, WAV, etc.)', 'warning');
            return;
        }
        if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
            showToast(`File too large. Maximum size is ${MAX_UPLOAD_SIZE_MB}MB.`, 'warning');
            return;
        }
        uploadRefAudio(file);
    });
    audioInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
                showToast(`File too large. Maximum size is ${MAX_UPLOAD_SIZE_MB}MB.`, 'warning');
                audioInput.value = '';
                return;
            }
            uploadRefAudio(file);
            audioInput.value = '';
        }
    });
    refRemove.addEventListener('click', () => {
        if (refAudio.src && refAudio.src.startsWith('blob:')) {
            URL.revokeObjectURL(refAudio.src);
        }
        refAudio.src = '';
        refAudioId = null;
        refPreview.classList.add('hidden');
        uploadZone.classList.remove('hidden');
    });

    async function uploadRefAudio(file) {
        const formData = new FormData();
        formData.append('audio', file, file.name);
        formData.append('denoise', 'true');
        try {
            showLoading();
            const response = await fetchWithTimeout('/api/audio/upload', { method: 'POST', body: formData }, 120000);
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            refAudioId = data.id;
            if (refAudio.src && refAudio.src.startsWith('blob:')) {
                URL.revokeObjectURL(refAudio.src);
            }
            refAudio.src = URL.createObjectURL(file);
            refPreview.classList.remove('hidden');
            uploadZone.classList.add('hidden');
        } catch (err) {
            showToast('Upload failed: ' + err.message);
        } finally {
            hideLoading();
        }
    }

    // Generate
    textInput.addEventListener('input', () => { generateBtn.disabled = !textInput.value.trim(); });
    generateBtn.addEventListener('click', generateChatterbox);

    async function generateChatterbox() {
        if (isGenerating) {
            showToast('Generation already in progress', 'warning');
            return;
        }
        const text = textInput.value.trim();
        if (!text) {
            showToast('Please enter text to generate speech', 'warning');
            return;
        }

        const languageId = document.getElementById('chatterbox-language').value;
        const exaggeration = parseFloat(exaggerationSlider.value);
        const cfgWeight = parseFloat(cfgSlider.value);
        const useCloneRef = useCloneRefToggle.checked;
        const cloneSamples = useCloneRef ? getCloneSamples() : [];
        const effectiveRefAudioId = useCloneRef ? (cloneSamples[0]?.id || null) : refAudioId;
        if (useCloneRef && !effectiveRefAudioId) {
            showToast('Add reference samples in Voice Clone tab first', 'warning');
            return;
        }

        isGenerating = true;
        currentJobId = null;
        currentAudioBlob = null;

        try {
            showLoading();
            const response = await fetchWithTimeout('/api/tts/chatterbox/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text, language_id: languageId, exaggeration,
                    cfg_weight: cfgWeight, ref_audio_id: effectiveRefAudioId,
                })
            });

            if (!response.ok) {
                const msg = await getErrorMessage(response, 'Generation failed');
                throw new Error(msg);
            }

            currentJobId = response.headers.get('X-Job-Id');
            const audioBlob = await response.blob();
            playGeneratedAudio(audioBlob);

            if (currentJobId) {
                window.addToHistory('chatterbox', text, languageId,
                    { exaggeration, cfg_weight: cfgWeight, via_clone_ref: useCloneRef }, currentJobId);
            }
        } catch (err) {
            showToast('Generation failed: ' + err.message);
        } finally {
            isGenerating = false;
            hideLoading();
        }
    }
}

async function loadChatterboxLanguages() {
    try {
        const response = await fetch('/api/tts/chatterbox/languages');
        const languages = await response.json();
        const selectIds = ['chatterbox-language', 'clone-chatterbox-language'];

        selectIds.forEach((selectId) => {
            const select = document.getElementById(selectId);
            if (!select) return;

            const previous = select.value;
            select.innerHTML = '';

            languages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang.id;
                option.textContent = lang.name;
                if ((previous && previous === lang.id) || (!previous && lang.id === 'en')) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
        });
    } catch (err) {
        console.error('Error loading Chatterbox languages:', err);
    }
}

// Audio Player
function initAudioPlayer() {
    // Initialize wavesurfer
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
        downloadBtn.disabled = !currentAudioBlob;
        updateTimeDisplay();
    });

    wavesurfer.on('error', (err) => {
        console.error('WaveSurfer error:', err);
        playBtn.disabled = true;
        // Keep download enabled if we have audio data
        downloadBtn.disabled = !currentAudioBlob;
        showToast('Failed to load audio for playback. Try generating again.');
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
        if (wavesurfer) {
            wavesurfer.playPause();
        }
    });

    downloadBtn.addEventListener('click', async () => {
        // Prefer downloading via job ID from server (original quality)
        if (currentJobId) {
            downloadBtn.disabled = true;
            try {
                const response = await fetchWithTimeout(`/api/tts/download/${currentJobId}`, {}, 60000);
                if (!response.ok) {
                    // Job expired from cache — fall back to blob if available
                    if (currentAudioBlob) {
                        downloadBlob(currentAudioBlob);
                        return;
                    }
                    throw new Error('Audio expired. Please generate again.');
                }
                const blob = await response.blob();
                downloadBlob(blob);
            } catch (err) {
                showToast('Download failed: ' + err.message);
            } finally {
                downloadBtn.disabled = false;
            }
        } else if (currentAudioBlob) {
            downloadBlob(currentAudioBlob);
        } else {
            showToast('No audio to download', 'warning');
        }
    });
}

function downloadBlob(blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function updateTimeDisplay() {
    if (wavesurfer) {
        const current = formatTime(wavesurfer.getCurrentTime());
        const total = formatTime(wavesurfer.getDuration());
        timeDisplay.textContent = `${current} / ${total}`;
    }
}

function playGeneratedAudio(blob) {
    currentAudioBlob = blob;

    // Revoke previous blob URL to prevent memory leak
    if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
    }
    currentBlobUrl = URL.createObjectURL(blob);

    // Keep buttons disabled until WaveSurfer signals ready
    playBtn.disabled = true;
    downloadBtn.disabled = true;

    if (wavesurfer) {
        wavesurfer.load(currentBlobUrl);
        wavesurfer.once('ready', () => {
            playBtn.disabled = false;
            downloadBtn.disabled = false;
            wavesurfer.play();
        });
    }
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Load Data
async function loadLanguages() {
    try {
        const response = await fetch('/api/tts/languages');
        const languages = await response.json();

        const selects = [
            document.getElementById('clone-language'),
            document.getElementById('custom-language'),
            document.getElementById('design-language')
        ];

        selects.forEach(select => {
            languages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang;
                option.textContent = lang;
                if (lang === 'English') option.selected = true;
                select.appendChild(option);
            });
        });
    } catch (err) {
        console.error('Error loading languages:', err);
    }
}

async function loadSpeakers() {
    try {
        const response = await fetch('/api/tts/speakers');
        const speakers = await response.json();

        const select = document.getElementById('custom-speaker');
        speakers.forEach(speaker => {
            const option = document.createElement('option');
            option.value = speaker.name;
            option.textContent = `${speaker.name} - ${speaker.description}`;
            if (speaker.name === 'Ryan') option.selected = true;
            select.appendChild(option);
        });
    } catch (err) {
        console.error('Error loading speakers:', err);
    }
}

// Utility
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoading() {
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}
