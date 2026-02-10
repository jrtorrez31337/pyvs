// Constants
const DEFAULT_SAMPLE_RATE = 24000;
const WAV_HEADER_SIZE = 44;
const STREAMING_CHUNK_BYTES = 4800;  // ~100ms at 24kHz int16 mono
const INT16_MAX = 32768;
const MAX_TEXT_WARNING_LENGTH = 1000;
const GPU_POLL_INTERVAL_MS = 5000;
const HISTORY_DISPLAY_LIMIT = 10;

// State
let currentMode = 'stt';
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentJobId = null;

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
            if (!headerParsed && buffer.length >= WAV_HEADER_SIZE) {
                // Parse sample rate from header (bytes 24-27)
                const dataView = new DataView(buffer.buffer);
                streamingPlayer.sampleRate = dataView.getUint32(24, true);
                buffer = buffer.slice(WAV_HEADER_SIZE);
                headerParsed = true;
            }

            // Process complete samples (2 bytes per sample for int16)
            if (headerParsed && buffer.length >= STREAMING_CHUNK_BYTES) { // ~100ms of audio at 24kHz
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
    setInterval(updateGPUStatus, GPU_POLL_INTERVAL_MS);
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
    const textareas = document.querySelectorAll('#clone-text, #custom-text, #design-text');

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
            const start = parseFloat(trimStart.value) || 0;
            const end = parseFloat(trimEnd.value) || trimWavesurfer.getDuration();
            trimWavesurfer.setTime(start);
            trimWavesurfer.play();
            // Stop at end time
            const checkEnd = setInterval(() => {
                if (trimWavesurfer.getCurrentTime() >= end) {
                    trimWavesurfer.pause();
                    clearInterval(checkEnd);
                }
            }, 100);
        }
    });

    saveBtn.addEventListener('click', async () => {
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
            alert('Trim failed: ' + err.message);
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
    if (trimWavesurfer) {
        trimWavesurfer.pause();
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
            alert(message);
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
            const response = await fetch('/api/stt', {
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
            alert('Transcription failed: ' + err.message);
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

    // Profile elements
    const profileSelect = document.getElementById('profile-select');
    const profileLoadBtn = document.getElementById('profile-load-btn');
    const profileSaveBtn = document.getElementById('profile-save-btn');
    const profileDeleteBtn = document.getElementById('profile-delete-btn');

    // Multiple samples storage: [{id, blobUrl, transcript}, ...]
    let samples = [];

    // Initialize profiles
    loadProfilesList();
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
        if (file && file.type.startsWith('audio/')) {
            addSampleFromFile(file);
        }
    });

    audioInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
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
            alert(message);
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
        samples.push({ id, blobUrl, transcript });
        renderSamples();
        updateGenerateButton();
    }

    function removeSample(index) {
        // TODO: Could also delete from server
        samples.splice(index, 1);
        renderSamples();
        updateGenerateButton();
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
            const response = await fetch('/api/audio/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            return data;
        } catch (err) {
            console.error('Upload error:', err);
            alert('Upload failed: ' + err.message);
            return null;
        }
    }

    async function transcribeAudio(fileOrBlob, denoise = true) {
        const formData = new FormData();
        const filename = fileOrBlob.name || 'recording.webm';
        formData.append('audio', fileOrBlob, filename);
        formData.append('denoise', denoise.toString());

        try {
            const response = await fetch('/api/stt', {
                method: 'POST',
                body: formData
            });
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
        const text = textInput.value.trim();
        const language = document.getElementById('clone-language').value;

        if (!text || samples.length === 0) return;

        // Collect all sample IDs and transcripts
        const refAudioIds = samples.map(s => s.id);
        const refTexts = samples.map(s => s.transcript || null);

        try {
            showLoading();
            const response = await fetch('/api/tts/clone', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    language,
                    ref_audio_ids: refAudioIds,
                    ref_texts: refTexts
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Generation failed');
            }

            currentJobId = response.headers.get('X-Job-Id');
            const audioBlob = await response.blob();
            playGeneratedAudio(audioBlob);

            // Add to history
            if (currentJobId) {
                window.addToHistory('clone', text, language, { samples: samples.length }, currentJobId);
            }
        } catch (err) {
            console.error('Generation error:', err);
            alert('Generation failed: ' + err.message);
        } finally {
            hideLoading();
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

            alert(`Imported profile "${data.name}"`);
            await loadProfilesList();
            profileSelect.value = data.id;
            profileLoadBtn.disabled = false;
            profileExportBtn.disabled = false;
            profileDeleteBtn.disabled = false;
        } catch (err) {
            alert('Import failed: ' + err.message);
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
            alert('Add at least one sample first');
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

            alert(`Profile "${name}" saved successfully!`);
            await loadProfilesList();

            // Select the newly created profile
            profileSelect.value = data.id;
            profileLoadBtn.disabled = false;
            profileDeleteBtn.disabled = false;
        } catch (err) {
            console.error('Error saving profile:', err);
            alert('Failed to save profile: ' + err.message);
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

            // Clear current samples and load from profile
            samples = [];
            for (const sample of data.samples) {
                // Fetch the audio to create blob URL
                const audioResponse = await fetch(`/api/audio/stream/${sample.id}`);
                if (audioResponse.ok) {
                    const blob = await audioResponse.blob();
                    samples.push({
                        id: sample.id,
                        blobUrl: URL.createObjectURL(blob),
                        transcript: sample.transcript || ''
                    });
                } else {
                    // Fallback - add without blob URL (won't have preview)
                    samples.push({
                        id: sample.id,
                        blobUrl: '',
                        transcript: sample.transcript || ''
                    });
                }
            }

            renderSamples();
            updateGenerateButton();
            updateProfileButtons();
            alert(`Loaded profile "${data.profile_name}" with ${samples.length} samples`);
        } catch (err) {
            console.error('Error loading profile:', err);
            alert('Failed to load profile: ' + err.message);
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
            alert('Profile deleted');
        } catch (err) {
            console.error('Error deleting profile:', err);
            alert('Failed to delete profile: ' + err.message);
        } finally {
            hideLoading();
        }
    }

    // Update save button when samples change
    const originalAddSample = addSample;
    addSample = function(id, blobUrl, transcript) {
        originalAddSample(id, blobUrl, transcript);
        updateProfileButtons();
    };

    const originalRemoveSample = removeSample;
    removeSample = function(index) {
        originalRemoveSample(index);
        updateProfileButtons();
    };
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
        const text = textInput.value.trim();
        const language = document.getElementById('custom-language').value;
        const speaker = document.getElementById('custom-speaker').value;
        const instruct = document.getElementById('custom-instruct').value.trim();

        if (!text) return;

        try {
            showLoading();
            const response = await fetch('/api/tts/custom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    language,
                    speaker,
                    instruct: instruct || null
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Generation failed');
            }

            currentJobId = response.headers.get('X-Job-Id');
            const audioBlob = await response.blob();
            playGeneratedAudio(audioBlob);

            // Add to history
            if (currentJobId) {
                window.addToHistory('custom', text, language, { speaker }, currentJobId);
            }
        } catch (err) {
            console.error('Generation error:', err);
            alert('Generation failed: ' + err.message);
        } finally {
            hideLoading();
        }
    }
}

// Voice Design Mode
function initVoiceDesign() {
    const generateBtn = document.getElementById('design-generate-btn');
    const textInput = document.getElementById('design-text');
    const instructInput = document.getElementById('design-instruct');

    generateBtn.addEventListener('click', generateDesign);

    function updateButtonState() {
        generateBtn.disabled = !textInput.value.trim() || !instructInput.value.trim();
    }

    textInput.addEventListener('input', updateButtonState);
    instructInput.addEventListener('input', updateButtonState);

    async function generateDesign() {
        const text = textInput.value.trim();
        const language = document.getElementById('design-language').value;
        const instruct = instructInput.value.trim();

        if (!text || !instruct) return;

        try {
            showLoading();
            const response = await fetch('/api/tts/design', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    language,
                    instruct
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Generation failed');
            }

            currentJobId = response.headers.get('X-Job-Id');
            const audioBlob = await response.blob();
            playGeneratedAudio(audioBlob);

            // Add to history
            if (currentJobId) {
                window.addToHistory('design', text, language, { instruct }, currentJobId);
            }
        } catch (err) {
            console.error('Generation error:', err);
            alert('Generation failed: ' + err.message);
        } finally {
            hideLoading();
        }
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
        if (wavesurfer) {
            wavesurfer.playPause();
        }
    });

    downloadBtn.addEventListener('click', () => {
        if (currentJobId) {
            window.location.href = `/api/tts/download/${currentJobId}`;
        }
    });
}

function updateTimeDisplay() {
    if (wavesurfer) {
        const current = formatTime(wavesurfer.getCurrentTime());
        const total = formatTime(wavesurfer.getDuration());
        timeDisplay.textContent = `${current} / ${total}`;
    }
}

function playGeneratedAudio(blob) {
    const url = URL.createObjectURL(blob);
    if (wavesurfer) {
        wavesurfer.load(url);
        wavesurfer.once('ready', () => {
            wavesurfer.play();
        });
    }
    playBtn.disabled = false;
    downloadBtn.disabled = false;
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
