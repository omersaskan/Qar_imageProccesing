const API_BASE = window.MESHYSIZ_API_BASE || (window.location.origin !== "null" && window.location.protocol.startsWith("http") && window.location.hostname !== "localhost" ? window.location.origin + "/api" : "http://localhost:8001/api");

const state = {
    products: [],
    logs: [],
    searchQuery: ""
};

// DOM Elements
const productGrid = document.getElementById('product-grid');
const logStream = document.getElementById('log-stream');
const totalProductsEl = document.getElementById('total-products');
const activeVersionsEl = document.getElementById('active-versions');
const searchInput = document.getElementById('search-input');
const modalBackdrop = document.getElementById('modal-backdrop');
const closeModal = document.getElementById('close-modal');
const historyContainer = document.getElementById('history-container');

// Upload Elements
const openUploadBtn = document.getElementById('open-upload');
const uploadModal = document.getElementById('upload-modal-backdrop');
const closeUploadBtn = document.getElementById('close-upload');
const uploadForm = document.getElementById('upload-form');
const dropZone = document.getElementById('drop-zone');
const videoInput = document.getElementById('video-input');
const fileInfo = document.getElementById('file-info');
const fileNameEl = document.getElementById('file-name');
const progressContainer = document.getElementById('upload-progress-container');
const progressBar = document.getElementById('upload-progress');

// 3D Viewer Elements
const viewerModal = document.getElementById('viewer-modal-backdrop');
const closeViewerBtn = document.getElementById('close-viewer');
const mainViewer = document.getElementById('main-viewer');
const viewerTitle = document.getElementById('viewer-title');
const viewerStatus = document.getElementById('viewer-status');

// Guidance Elements
const guidanceSection = document.getElementById('guidance-section');
const guidanceStatusBadge = document.getElementById('guidance-status-badge');
const nextActionText = document.getElementById('next-action-text');
const guidanceMessages = document.getElementById('guidance-messages');
const refreshGuidanceBtn = document.getElementById('refresh-guidance-btn');

let activeSessionId = null;

// --- Initialization ---

async function init() {
    await fetchProducts();
    await fetchLogs();

    setInterval(fetchLogs, 5000);
    setInterval(pollGuidance, 10000); // Poll guidance every 10s
    setupUploadHandlers();

    refreshGuidanceBtn.onclick = () => pollGuidance();
}

async function pollGuidance() {
    // If we have an active session or a session needing recapture, track it
    const activeProducts = state.products.filter(p => p.has_active_session || p.status === 'processing');
    if (activeProducts.length > 0) {
        // For simplicity, take the first one or the one the user just uploaded
        // In a real app, this would be tied to a selection
        const productId = activeProducts[0].id;
        try {
            // Get history to find the session ID
            const resp = await fetch(`${API_BASE}/products/${productId}/history`);
            const history = await resp.json();
            if (history.length > 0) {
                const latestSession = history[0].asset_id;
                await fetchGuidance(latestSession);
            }
        } catch (err) {
            console.warn("Guidance polling failed:", err);
        }
    } else {
        guidanceSection.classList.add('hidden');
    }
}

async function fetchGuidance(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}/guidance`);
        const guidance = await response.json();
        renderGuidance(guidance);
    } catch (err) {
        console.error("Failed to fetch guidance:", err);
    }
}

function renderGuidance(guidance) {
    if (!guidance || guidance.status === 'CREATED' && !guidance.messages.length) {
        guidanceSection.classList.add('hidden');
        return;
    }

    guidanceSection.classList.remove('hidden');
    guidanceStatusBadge.textContent = guidance.status;
    guidanceStatusBadge.className = `badge ${guidance.status.toLowerCase()}`;

    nextActionText.textContent = guidance.next_action;

    guidanceMessages.innerHTML = (guidance.messages || []).map(msg => `
        <div class="guidance-item ${msg.severity}">
            <span class="symbol">${msg.severity === 'critical' ? '🔴' : msg.severity === 'warning' ? '🟡' : 'ℹ️'}</span>
            <span class="msg-text">${msg.message}</span>
        </div>
    `).join('');
}

// --- AR Capture Pure Logic ---

class MetricsProcessor {
    constructor(config = {}) {
        this.blurThreshold = config.blurThreshold || 100;
        this.lightingMin = config.lightingMin || 40;
        this.lightingMax = config.lightingMax || 220;
        this.highlightRatioThreshold = config.highlightRatioThreshold || 0.1;
    }

    // Pure function for blur detection (Laplacian Variance proxy)
    analyzeBlur(pixels, width, height) {
        let sum = 0;
        let sumSq = 0;
        const n = (width - 2) * (height - 2);
        
        // Simple 3x3 Laplacian kernel over grayscale
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                const center = (pixels[idx] + pixels[idx+1] + pixels[idx+2]) / 3;
                
                const neighbors = [
                    (pixels[((y-1)*width + x)*4] + pixels[((y-1)*width + x)*4+1] + pixels[((y-1)*width + x)*4+2]) / 3,
                    (pixels[((y+1)*width + x)*4] + pixels[((y+1)*width + x)*4+1] + pixels[((y+1)*width + x)*4+2]) / 3,
                    (pixels[(y*width + (x-1))*4] + pixels[(y*width + (x-1))*4+1] + pixels[(y*width + (x-1))*4+2]) / 3,
                    (pixels[(y*width + (x+1))*4] + pixels[(y*width + (x+1))*4+1] + pixels[(y*width + (x+1))*4+2]) / 3
                ];
                
                const laplacian = neighbors.reduce((a, b) => a + b, 0) - 4 * center;
                sum += laplacian;
                sumSq += laplacian * laplacian;
            }
        }
        
        const variance = (sumSq / n) - (sum / n) ** 2;
        return variance;
    }

    analyzeLighting(pixels) {
        let totalLum = 0;
        let highlights = 0;
        const n = pixels.length / 4;
        
        for (let i = 0; i < pixels.length; i += 4) {
            const lum = 0.2126 * pixels[i] + 0.7152 * pixels[i+1] + 0.0722 * pixels[i+2];
            totalLum += lum;
            if (lum > 240) highlights++;
        }
        
        return {
            avgLuminance: totalLum / n,
            highlightRatio: highlights / n
        };
    }

    checkQuality(metrics) {
        const reasons = [];
        if (metrics.blur < this.blurThreshold) reasons.push("Move slower (blur detected)");
        if (metrics.lighting.avgLuminance < this.lightingMin) reasons.push("Too dark");
        if (metrics.lighting.avgLuminance > this.lightingMax) reasons.push("Too bright");
        if (metrics.lighting.highlightRatio > this.highlightRatioThreshold) reasons.push("Reduce harsh highlights");
        
        return {
            isAccepted: reasons.length === 0,
            reasons
        };
    }
}

class CoverageTracker {
    constructor(numSectors = 36) {
        this.numSectors = numSectors;
        this.sectorSize = 360 / numSectors;
        this.sectors = new Array(numSectors).fill(false);
        this.lastAzimuth = null;
    }

    addFrame(azimuth, isAccepted) {
        if (!isAccepted) return;
        
        // Normalize azimuth to 0-360
        const normalized = ((azimuth % 360) + 360) % 360;
        const index = Math.floor(normalized / this.sectorSize);
        this.sectors[index] = true;
        this.lastAzimuth = normalized;
    }

    getPercent() {
        const covered = this.sectors.filter(s => s).length;
        return (covered / this.numSectors) * 100;
    }

    getMaxGap() {
        let maxGap = 0;
        let currentGap = 0;
        
        // Circular check (double the array to handle 359->0 wrap)
        const extendedSectors = [...this.sectors, ...this.sectors];
        for (const s of extendedSectors) {
            if (!s) {
                currentGap += this.sectorSize;
            } else {
                maxGap = Math.max(maxGap, currentGap);
                currentGap = 0;
            }
        }
        return Math.max(maxGap, currentGap);
    }

    getSummary() {
        return {
            percent: this.getPercent(),
            maxGap: this.getMaxGap(),
            sectors: [...this.sectors]
        };
    }
}

// --- AR Capture Logic ---

class ARCapture {
    constructor() {
        this.modal = document.getElementById('ar-capture-modal-backdrop');
        this.video = document.getElementById('ar-video-feed');
        this.samplingCanvas = document.getElementById('ar-sampling-canvas');
        this.samplingCtx = this.samplingCanvas.getContext('2d', { willReadFrequently: true });
        
        this.progressRing = document.getElementById('ar-progress-ring');
        this.statusLabel = document.getElementById('ar-status-label');
        this.captureBtn = document.getElementById('ar-capture-btn');
        this.coverageEl = document.getElementById('coverage-value');
        this.angleArrow = document.getElementById('angle-arrow');
        this.angleText = document.getElementById('current-angle');
        this.timerEl = document.getElementById('ar-timer');
        
        this.metrics = new MetricsProcessor();
        this.tracker = new CoverageTracker();
        
        this.stream = null;
        this.isRecording = false;
        this.isDemoMode = false;
        this.azimuth = 0;
        this.tilt = 0;
        
        this.stats = {
            acceptedCount: 0,
            rejectionReasons: {},
            selectedIndices: []
        };

        this.setupHandlers();
    }

    setupHandlers() {
        document.getElementById('close-ar').onclick = () => this.stop();
        this.captureBtn.onclick = () => this.toggleCapture();
        
        window.addEventListener('deviceorientation', (e) => {
            if (e.alpha !== null) {
                this.azimuth = e.alpha;
                this.tilt = e.beta;
            }
        });
    }

    async start() {
        this.modal.classList.remove('hidden');
        this.resetStats();
        
        // Detect desktop/no-sensor
        if (window.DeviceOrientationEvent && typeof DeviceOrientationEvent.requestPermission === 'function') {
            try {
                await DeviceOrientationEvent.requestPermission();
            } catch (e) { console.warn("Orientation permission denied"); }
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment' }, 
                audio: false 
            });
            this.video.srcObject = this.stream;
            this.isDemoMode = false;
            this.statusLabel.textContent = "ALIGNED & READY";
        } catch (err) {
            console.warn("Camera/Mobile sensors unavailable, entering DEMO MODE");
            this.isDemoMode = true;
            this.statusLabel.textContent = "DEMO MODE (Desktop)";
            this.statusLabel.style.color = "#fbbf24";
        }
        
        this.runMetricsLoop();
    }

    resetStats() {
        this.tracker = new CoverageTracker();
        this.stats = { acceptedCount: 0, rejectionReasons: {}, selectedIndices: [] };
        this.updateProgress(0);
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        this.modal.classList.add('hidden');
        this.isRecording = false;
        clearInterval(this.timerInterval);
    }

    toggleCapture() {
        if (!this.isRecording) {
            this.isRecording = true;
            this.captureBtn.classList.add('recording');
            this.startTime = Date.now();
        } else {
            this.finishRecording();
        }
    }

    runMetricsLoop() {
        if (this.modal.classList.contains('hidden')) return;

        // 1. Get current frame
        this.samplingCtx.drawImage(this.video, 0, 0, 160, 160);
        const imageData = this.samplingCtx.getImageData(0, 0, 160, 160);
        
        // 2. Analyze
        const blur = this.metrics.analyzeBlur(imageData.data, 160, 160);
        const lighting = this.metrics.analyzeLighting(imageData.data);
        const quality = this.metrics.checkQuality({ blur, lighting });
        
        // 3. Update Indicators
        this.updateUIIndicators(quality, blur, lighting);
        
        // 4. Update Coverage if recording
        if (this.isRecording) {
            const azimuth = this.isDemoMode ? (Date.now() / 50 % 360) : this.azimuth;
            this.tracker.addFrame(azimuth, quality.isAccepted);
            
            if (quality.isAccepted) {
                this.stats.acceptedCount++;
            } else {
                quality.reasons.forEach(r => {
                    this.stats.rejectionReasons[r] = (this.stats.rejectionReasons[r] || 0) + 1;
                });
            }
            
            const summary = this.tracker.getSummary();
            this.updateProgress(summary.percent);
            this.checkGate(summary);
        }

        requestAnimationFrame(() => this.runMetricsLoop());
    }

    updateUIIndicators(quality, blur, lighting) {
        document.getElementById('indicator-stability').querySelector('.dot').style.background = 
            blur > this.metrics.blurThreshold ? "var(--success)" : "var(--error)";
        
        document.getElementById('indicator-lighting').querySelector('.dot').style.background = 
            quality.reasons.some(r => r.includes("lighting") || r.includes("dark") || r.includes("bright")) ? "var(--error)" : "var(--success)";
            
        this.angleArrow.style.transform = `rotate(${this.azimuth}deg)`;
        this.angleText.textContent = `${Math.floor(this.azimuth)}°`;
    }

    updateProgress(percent) {
        const radius = 45;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (percent / 100) * circumference;
        this.progressRing.style.strokeDashoffset = offset;
        this.coverageEl.textContent = `${Math.floor(percent)}%`;
    }

    checkGate(summary) {
        const canFinish = summary.percent > 90 && summary.maxGap < 45;
        this.captureBtn.style.opacity = canFinish ? "1" : "0.5";
        
        if (!canFinish && this.isRecording) {
            let msg = "Rotate more";
            if (summary.maxGap > 45) msg = `Gap too large: ${Math.floor(summary.maxGap)}°`;
            this.statusLabel.textContent = msg;
        } else if (this.isRecording) {
            this.statusLabel.textContent = "READY TO FINISH";
            this.statusLabel.style.color = "var(--success)";
        }
    }

    async finishRecording() {
        const summary = this.tracker.getSummary();
        if (!this.isDemoMode && summary.maxGap > 60) {
            alert(`Capture failed quality gate. Max gap: ${Math.floor(summary.maxGap)}°. Please rotate more slowly and fully.`);
            return;
        }

        this.isRecording = false;
        this.captureBtn.classList.remove('recording');
        
        const manifest = {
            product_profile: document.querySelector('.profile-btn.active').dataset.profile,
            coverage_summary: summary,
            accepted_frame_count: this.stats.acceptedCount,
            rejection_stats: this.stats.rejectionReasons,
            is_demo: this.isDemoMode,
            timestamp: new Date().toISOString()
        };

        console.log("Quality Manifest Generated:", manifest);
        alert("Capture Finished! Quality Manifest created and attached to upload.");
        this.stop();
    }
}

const arCapture = new ARCapture();

// --- Upload Handlers ---

function setupUploadHandlers() {
    openUploadBtn.onclick = () => arCapture.start();
    closeUploadBtn.onclick = () => uploadModal.classList.add('hidden');

    // Drag and Drop
    dropZone.onclick = (e) => {
        videoInput.click();
    };

    dropZone.ondragover = (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    };

    dropZone.ondragleave = () => dropZone.classList.remove('active');

    dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    videoInput.onchange = (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    };

    uploadForm.onsubmit = async (e) => {
        e.preventDefault();
        const productId = document.getElementById('upload-product-id').value;
        const file = videoInput.files[0];

        if (!file || !productId) return;

        const formData = new FormData();
        formData.append('product_id', productId);
        formData.append('file', file);

        progressContainer.classList.remove('hidden');
        const xhr = new XMLHttpRequest();

        xhr.upload.onprogress = (event) => {
            if (event.lengthComputable) {
                const percent = (event.loaded / event.total) * 100;
                progressBar.style.width = percent + '%';
                document.getElementById('submit-upload').textContent = `Uploading... ${Math.round(percent)}%`;
            }
        };

        xhr.onload = async () => {
            if (xhr.status === 200) {
                alert("Upload successful! Session created.");
                uploadModal.classList.add('hidden');
                resetUploadForm();
                await fetchProducts();
            } else {
                alert("Upload failed: " + xhr.responseText);
            }
        };

        xhr.open('POST', `${API_BASE}/sessions/upload`);
        xhr.send(formData);
    };
}

function handleFileSelect(file) {
    fileNameEl.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    fileInfo.classList.remove('hidden');
}

function resetUploadForm() {
    uploadForm.reset();
    fileInfo.classList.add('hidden');
    progressContainer.classList.add('hidden');
    progressBar.style.width = '0%';
}

async function fetchProducts() {
    try {
        const response = await fetch(`${API_BASE}/products`);
        state.products = await response.json();
        renderProducts();
        updateStats();
    } catch (err) {
        console.error("Failed to fetch products:", err);
    }
}

async function fetchLogs() {
    try {
        const response = await fetch(`${API_BASE}/logs`);
        state.logs = await response.json();
        renderLogs();
    } catch (err) {
        console.error("Failed to fetch logs:", err);
    }
}

// --- 3D Viewer Actions ---

function open3DViewer(assetId, status) {
    if (!assetId || assetId === 'null' || assetId === 'undefined') {
        viewerStatus.innerHTML = '<span style="color: #f87171">❌ Error: Asset ID not found.</span>';
        return;
    }

    viewerModal.classList.remove('hidden');
    viewerTitle.textContent = `Asset Preview: ${assetId}`;
    mainViewer.src = "";
    viewerStatus.innerHTML = 'Checking availability...';

    const timestamp = Date.now();
    const modelUrl = `${API_BASE}/assets/blobs/${assetId}.glb?t=${timestamp}`;

    if (status === 'processing' || status === 'CREATED') {
        viewerStatus.innerHTML = '⚙️ <span style="color: #fbbf24">Processing...</span>';
        mainViewer.src = "https://modelviewer.dev/shared-assets/models/Astronaut.glb";
    } else {
        viewerStatus.textContent = "🚀 Fetching...";
        mainViewer.src = modelUrl;
    }
}

closeViewerBtn.onclick = () => {
    viewerModal.classList.add('hidden');
    mainViewer.src = "";
};

function renderProducts() {
    const filtered = state.products.filter(p =>
        p.id.toLowerCase().includes(state.searchQuery.toLowerCase())
    );

    productGrid.innerHTML = filtered.map(p => {
        const isProcessing = p.status === 'processing';
        return `
            <div class="product-card glass ${isProcessing ? 'processing' : ''}" onclick="showProductDetails('${p.id}')">
                <div class="product-id">${p.id}</div>
                <div class="badge">${isProcessing ? 'PROCESSING' : `v${p.active_id ? p.active_id.split('_').pop() : 'N/A'}`}</div>
                <div class="v-count">${p.asset_count} versions</div>
            </div>
        `;
    }).join('');
}

function renderLogs() {
    logStream.innerHTML = state.logs.map(log => `
        <div class="log-entry">
            <span class="log-time">${new Date(log.timestamp).toLocaleTimeString()}</span>
            <span class="log-level ${log.level}">${log.level}</span>
            <span class="log-msg">${log.message}</span>
        </div>
    `).join('');
}

function updateStats() {
    totalProductsEl.textContent = state.products.length;
    activeVersionsEl.textContent = state.products.filter(p => p.active_id).length;
}

searchInput.addEventListener('input', (e) => {
    state.searchQuery = e.target.value;
    renderProducts();
});

async function showProductDetails(productId) {
    modalBackdrop.classList.remove('hidden');
    document.getElementById('modal-title').textContent = `History: ${productId}`;
    historyContainer.innerHTML = '<div class="loader">Loading...</div>';

    try {
        const response = await fetch(`${API_BASE}/products/${productId}/history`);
        const history = await response.json();
        renderHistory(history);
    } catch (err) {
        historyContainer.innerHTML = `Error: ${err.message}`;
    }
}

function renderHistory(history) {
    historyContainer.innerHTML = `
        <table class="history-table">
            <thead>
                <tr><th>Asset ID</th><th>Version</th><th>Status</th><th>View</th></tr>
            </thead>
            <tbody>
                ${history.map(item => `
                    <tr>
                        <td>${item.asset_id}</td>
                        <td>${item.version}</td>
                        <td>${item.status}</td>
                        <td><button class="btn-view" onclick="open3DViewer('${item.asset_id}', '${item.status}')">View</button></td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

closeModal.addEventListener('click', () => {
    modalBackdrop.classList.add('hidden');
});

init();
