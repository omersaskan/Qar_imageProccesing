const API_BASE = window.MESHYSIZ_API_BASE || (window.location.origin !== "null" && window.location.protocol.startsWith("http") && window.location.hostname !== "localhost" ? window.location.origin + "/api" : "http://localhost:8001/api");
const IS_LOCAL_DEV = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

const REJECTION_LABELS = {
    "blur": "Daha yavaş hareket edin (bulanıklık)",
    "too_dark": "Ortam çok karanlık",
    "too_bright": "Ortam çok aydınlık",
    "highlight": "Sert yansımaları azaltın"
};

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
        if (metrics.blur < this.blurThreshold) reasons.push("blur");
        if (metrics.lighting.avgLuminance < this.lightingMin) reasons.push("too_dark");
        if (metrics.lighting.avgLuminance > this.lightingMax) reasons.push("too_bright");
        if (metrics.lighting.highlightRatio > this.highlightRatioThreshold) reasons.push("highlight");
        
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

    getLargestGapCenter() {
        let maxGap = 0;
        let startIdx = -1;
        let currentGap = 0;
        let currentStart = -1;
        
        const extendedSectors = [...this.sectors, ...this.sectors];
        for (let i = 0; i < extendedSectors.length; i++) {
            if (!extendedSectors[i]) {
                if (currentStart === -1) currentStart = i;
                currentGap += this.sectorSize;
            } else {
                if (currentGap > maxGap) {
                    maxGap = currentGap;
                    startIdx = currentStart;
                }
                currentGap = 0;
                currentStart = -1;
            }
        }
        if (startIdx === -1) return null;
        
        const centerIdx = (startIdx + (maxGap / this.sectorSize / 2)) % this.numSectors;
        return centerIdx * this.sectorSize;
    }

    isAngleCovered(azimuth) {
        const normalized = ((azimuth % 360) + 360) % 360;
        const index = Math.floor(normalized / this.sectorSize);
        return this.sectors[index];
    }

    getSummary() {
        return {
            percent: this.getPercent(),
            maxGap: this.getMaxGap(),
            sectors: [...this.sectors],
            gapCenter: this.getLargestGapCenter()
        };
    }
}

class GateValidator {
    constructor(config = {}) {
        this.minCoverage = config.minCoverage || 90;
        this.maxGap = config.maxGap || 45;
        this.minAcceptedFrames = config.minAcceptedFrames || 100;
        this.maxBlurRatio = config.maxBlurRatio || 0.3;
        this.minDuration = config.minDuration || 15;
    }

    validate(summary, stats, elapsedSec, profile, profileCompletion) {
        const blurRejections = stats.rejectionReasons["blur"] || 0;
        const blurRatio = stats.totalCount > 0 ? (blurRejections / stats.totalCount) : 0;
        
        const hasEnoughFrames = stats.acceptedCount >= this.minAcceptedFrames;
        const blurIsOk = blurRatio <= this.maxBlurRatio;
        const durationIsOk = elapsedSec >= this.minDuration;
        const coverageIsOk = summary.percent >= this.minCoverage && summary.maxGap <= this.maxGap;
        
        let profileIsOk = true;
        let missingReq = "";

        if (profile === 'box') {
            const completed = profileCompletion ? (profileCompletion.faces || []) : [];
            profileIsOk = completed.length >= 6;
            if (!profileIsOk) missingReq = "Kutu tamamlanmadı (6 yüzey taranmalı)";
        } else if (profile === 'bottle') {
            const cap = profileCompletion ? profileCompletion.cap : false;
            const base = profileCompletion ? profileCompletion.base : false;
            profileIsOk = cap && base;
            if (!profileIsOk) missingReq = "Şişe tamamlanmadı (Kapak ve Taban gerekli)";
        }

        const canFinish = coverageIsOk && hasEnoughFrames && blurIsOk && durationIsOk && profileIsOk;
        
        return {
            canFinish,
            reasons: {
                coverage: coverageIsOk,
                frames: hasEnoughFrames,
                blur: blurIsOk,
                duration: durationIsOk,
                profile: profileIsOk
            },
            missingReq: missingReq || (summary.maxGap > this.maxGap ? `Açı boşluğu çok büyük: ${Math.floor(summary.maxGap)}°` : "")
        };
    }
}

class BoxGhostGuide {
    constructor() {
        this.container = document.getElementById('ar-ghost-guide');
        this.ghost = this.container.querySelector('.box-ghost');
        this.faces = {
            front: this.container.querySelector('.front'),
            back: this.container.querySelector('.back'),
            left: this.container.querySelector('.left'),
            right: this.container.querySelector('.right'),
            top: this.container.querySelector('.top'),
            bottom: this.container.querySelector('.bottom')
        };
        this.completed = new Set();
    }

    show() { this.container.classList.remove('hidden'); }
    hide() { this.container.classList.add('hidden'); }

    reset() {
        this.completed.clear();
        Object.values(this.faces).forEach(el => {
            el.classList.add('missing');
            el.classList.remove('completed');
        });
    }

    // This is a HUD visualizer, NOT a 3D reconstruction.
    update(azimuth, tilt, isAccepted) {
        this.ghost.style.transform = `rotateX(${-tilt}deg) rotateY(${-azimuth}deg)`;
        if (!isAccepted) return;

        let faceKey = null;
        if (tilt > 45) faceKey = 'top';
        else if (tilt < -45) faceKey = 'bottom';
        else {
            const a = ((azimuth % 360) + 360) % 360;
            if (a >= 315 || a < 45) faceKey = 'front';
            else if (a >= 45 && a < 135) faceKey = 'right';
            else if (a >= 135 && a < 225) faceKey = 'back';
            else if (a >= 225 && a < 315) faceKey = 'left';
        }

        if (faceKey && !this.completed.has(faceKey)) {
            this.completed.add(faceKey);
            const el = this.faces[faceKey];
            el.classList.remove('missing');
            el.classList.add('completed');
        }
    }

    isFullyComplete() { return this.completed.size >= 6; }
    getMissingFaces() { return Object.keys(this.faces).filter(f => !this.completed.has(f)); }
}

class BottleGhostGuide {
    constructor() {
        this.container = document.getElementById('ar-bottle-guide');
        this.ghost = this.container.querySelector('.bottle-ghost');
        this.segments = Array.from(this.container.querySelectorAll('.seg-face'));
        this.cap = this.container.querySelector('.cap');
        this.base = this.container.querySelector('.base');
        this.isCapComplete = false;
        this.isBaseComplete = false;
    }

    show() { this.container.classList.remove('hidden'); }
    hide() { this.container.classList.add('hidden'); }

    reset() {
        this.segments.forEach(s => { s.classList.add('missing'); s.classList.remove('completed'); });
        this.cap.classList.add('missing'); this.cap.classList.remove('completed');
        this.base.classList.add('missing'); this.base.classList.remove('completed');
        this.isCapComplete = false;
        this.isBaseComplete = false;
    }

    update(azimuth, tilt, isAccepted, sectors = []) {
        this.ghost.style.transform = `rotateX(${-tilt}deg) rotateY(${-azimuth}deg)`;
        if (!isAccepted) return;

        if (tilt < -45) {
            this.isCapComplete = true;
            this.cap.classList.remove('missing');
            this.cap.classList.add('completed');
        }
        if (tilt > 30) {
            this.isBaseComplete = true;
            this.base.classList.remove('missing');
            this.base.classList.add('completed');
        }

        // Map 36 sectors to 12 segments
        if (sectors.length === 36) {
            for (let i = 0; i < 12; i++) {
                const covered = sectors[i*3] || sectors[i*3+1] || sectors[i*3+2];
                if (covered) {
                    this.segments[i].classList.remove('missing');
                    this.segments[i].classList.add('completed');
                }
            }
        }
    }

    getMissingRequirements() {
        const reqs = [];
        if (!this.isCapComplete) reqs.push("Cap/Top");
        if (!this.isBaseComplete) reqs.push("Base/Bottom");
        return reqs;
    }
}

class GenericGhostGuide {
    constructor() {
        this.container = document.getElementById('ar-generic-guide');
        this.ghost = this.container.querySelector('.bottle-ghost');
        this.segments = Array.from(this.container.querySelectorAll('.seg-face'));
    }

    show() { this.container.classList.remove('hidden'); }
    hide() { this.container.classList.add('hidden'); }

    reset() {
        this.segments.forEach(s => { s.classList.add('missing'); s.classList.remove('completed'); });
    }

    update(azimuth, tilt, isAccepted, sectors = []) {
        this.ghost.style.transform = `rotateX(${-tilt}deg) rotateY(${-azimuth}deg)`;
        if (!isAccepted) return;

        if (sectors.length === 36) {
            for (let i = 0; i < 12; i++) {
                const covered = sectors[i*3] || sectors[i*3+1] || sectors[i*3+2];
                if (covered) {
                    this.segments[i].classList.remove('missing');
                    this.segments[i].classList.add('completed');
                }
            }
        }
    }
}

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
        this.productIdInput = document.getElementById('ar-product-id');
        
        this.metrics = new MetricsProcessor();
        this.tracker = new CoverageTracker();
        this.boxGuide = new BoxGhostGuide();
        this.bottleGuide = new BottleGhostGuide();
        this.genericGuide = new GenericGhostGuide();
        this.demoBtn = document.getElementById('ar-demo-btn');
        this.gateValidator = new GateValidator();
        
        this.stream = null;
        this.mediaRecorder = null;
        this.chunks = [];
        this.isRecording = false;
        this.isDemoMode = false;
        this.profile = 'generic';
        this.azimuth = 0;
        this.tilt = 0;
        this.qualityManifest = null;
        this.canFinish = false;
        this.startTime = null;
        
        this.stats = {
            totalCount: 0,
            acceptedCount: 0,
            rejectionReasons: {},
            selectedIndices: []
        };

        this.toastTimeout = null;
        this.lastGuidanceTime = 0;
        this.lastToastMessage = "";
        
        // Hide demo btn outside local dev
        if (!IS_LOCAL_DEV) {
            this.demoBtn.style.display = 'none';
        }
        
        this.setupHandlers();
    }

    showGuidanceToast(message, type = 'error') {
        const toast = document.getElementById('ar-guidance-toast');
        if (!toast) return;
        
        if (this.lastToastMessage === message && !toast.classList.contains('hidden')) return;
        
        toast.textContent = message;
        toast.className = `ar-toast ${type}`;
        toast.classList.remove('hidden');
        this.lastToastMessage = message;
        
        clearTimeout(this.toastTimeout);
        this.toastTimeout = setTimeout(() => {
            toast.classList.add('hidden');
            this.lastToastMessage = "";
        }, 2000);
    }

    setupHandlers() {
        document.getElementById('close-ar').onclick = () => this.stop();
        this.captureBtn.onclick = () => this.toggleCapture();
        this.demoBtn.onclick = () => {
            if (this.isRecording) return; // Disable during recording
            this.isDemoMode = !this.isDemoMode;
            this.demoBtn.classList.toggle('active', this.isDemoMode);
            if (this.isDemoMode) {
                this.statusLabel.textContent = "DEMO MODE AKTİF";
                this.resetStats();
            } else {
                this.statusLabel.textContent = "HAZIR";
            }
        };
        
        window.addEventListener('deviceorientation', (e) => {
            if (e.alpha !== null) {
                this.azimuth = e.alpha; 
                this.tilt = e.beta;
            }
        });

        document.querySelectorAll('.profile-btn').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('.profile-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.profile = btn.dataset.profile;
                this.updateGuideVisibility();
            };
        });
    }

    updateGuideVisibility() {
        this.boxGuide.hide();
        this.bottleGuide.hide();
        this.genericGuide.hide();
        if (this.profile === 'box') {
            this.boxGuide.show();
        } else if (this.profile === 'bottle') {
            this.bottleGuide.show();
        } else if (this.profile === 'generic') {
            this.genericGuide.show();
        }
    }

    async start() {
        this.modal.classList.remove('hidden');
        this.resetStats();
        this.qualityManifest = null;
        this.canFinish = false;
        this.updateGuideVisibility();
        this.productIdInput.value = "";
        
        this.captureBtn.disabled = true;
        this.statusLabel.textContent = "BAŞLATILIYOR...";

        // 1. Secure Context Check
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        if (isMobile && !window.isSecureContext && window.location.hostname !== "localhost") {
            this.statusLabel.textContent = "HTTPS GEREKLİ";
            this.statusLabel.style.color = "var(--error)";
            alert("Mobil çekim için HTTPS (güvenli bağlantı) gereklidir.");
            return;
        }

        // 2. Permission request for iOS
        if (window.DeviceOrientationEvent && typeof DeviceOrientationEvent.requestPermission === 'function') {
            try {
                await DeviceOrientationEvent.requestPermission();
            } catch (e) { console.warn("Orientation permission denied"); }
        }

        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("MediaDevices API not found.");
            }
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }, 
                audio: false 
            });
            this.video.srcObject = this.stream;
            this.isDemoMode = false;
            this.statusLabel.textContent = "ALIGNED & READY";
            this.statusLabel.style.color = "var(--success)";
            document.getElementById('ar-debug-log').textContent = "";
        } catch (err) {
            console.warn("Camera failed, entering DEMO MODE:", err);
            this.isDemoMode = true;
            this.statusLabel.textContent = "DEMO MODE (No Camera)";
            this.statusLabel.style.color = "#fbbf24";
            document.getElementById('ar-debug-log').textContent = `ERR: ${err.message || err}`;
        }
        
        this.captureBtn.disabled = false;
        this.runMetricsLoop();
    }

    resetStats() {
        this.tracker = new CoverageTracker();
        this.boxGuide.reset();
        this.bottleGuide.reset();
        this.genericGuide.reset();
        this.stats = { totalCount: 0, acceptedCount: 0, rejectionReasons: {}, selectedIndices: [] };
        this.updateProgress(0, []);
        this.chunks = [];
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        this.modal.classList.add('hidden');
        this.isRecording = false;
        this.timerEl.classList.add('hidden');
        clearInterval(this.timerInterval);
    }

    getMimeType() {
        const types = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4'
        ];
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) return type;
        }
        return '';
    }

    toggleCapture() {
        if (!this.isRecording) {
            let productId = this.productIdInput.value.trim();
            if (!productId) {
                if (IS_LOCAL_DEV || this.isDemoMode) {
                    productId = `prod_${Date.now().toString().slice(-6)}`;
                    this.productIdInput.value = productId;
                } else {
                    alert("Product ID gereklidir!");
                    return;
                }
            }

            this.isRecording = true;
            this.demoBtn.disabled = true; // Disable toggle while recording
            this.captureBtn.classList.add('recording');
            this.startTime = Date.now();
            this.chunks = [];
            this.timerEl.classList.remove('hidden');
            this.startTimer();

            if (!this.isDemoMode) {
                const mimeType = this.getMimeType();
                this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
                this.mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) this.chunks.push(e.data);
                };
                this.mediaRecorder.onstop = () => this.uploadResult();
                this.mediaRecorder.start(1000); // Collect chunks every second
            }
        } else {
            this.finishRecording();
        }
    }

    startTimer() {
        clearInterval(this.timerInterval);
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const m = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const s = (elapsed % 60).toString().padStart(2, '0');
            this.timerEl.textContent = `${m}:${s}`;
        }, 1000);
    }

    runMetricsLoop() {
        if (this.modal.classList.contains('hidden')) return;

        this.samplingCtx.drawImage(this.video, 0, 0, 160, 160);
        const imageData = this.samplingCtx.getImageData(0, 0, 160, 160);
        
        const blur = this.metrics.analyzeBlur(imageData.data, 160, 160);
        const lighting = this.metrics.analyzeLighting(imageData.data);
        const quality = this.metrics.checkQuality({ blur, lighting });
        
        const curAzimuth = this.isDemoMode ? (Date.now() / 50 % 360) : this.azimuth;
        const curTilt = this.isDemoMode ? (Math.sin(Date.now() / 1000) * 20) : this.tilt;
        
        this.updateUIIndicators(quality, blur, lighting, curAzimuth);
        
        const summary = this.tracker.getSummary();
        if (this.profile === 'box') this.boxGuide.update(curAzimuth, curTilt, quality.isAccepted);
        if (this.profile === 'bottle') this.bottleGuide.update(curAzimuth, curTilt, quality.isAccepted, summary.sectors);
        if (this.profile === 'generic') this.genericGuide.update(curAzimuth, curTilt, quality.isAccepted, summary.sectors);
        
        if (this.isRecording) {
            this.stats.totalCount++;
            const isRedundant = this.tracker.isAngleCovered(curAzimuth);
            this.tracker.addFrame(curAzimuth, quality.isAccepted);
            
            const now = Date.now();
            if (now - this.lastGuidanceTime > 1500) {
                if (!quality.isAccepted) {
                    const label = REJECTION_LABELS[quality.reasons[0]] || quality.reasons[0];
                    this.showGuidanceToast(label);
                    this.lastGuidanceTime = now;
                } else if (isRedundant) {
                    this.showGuidanceToast("Bu açı zaten tarandı, ilerlemeye devam edin!", "info");
                    this.lastGuidanceTime = now;
                }
            }

            if (quality.isAccepted) {
                this.stats.acceptedCount++;
                if (this.stats.acceptedCount % 10 === 0) {
                    this.stats.selectedIndices.push({ t: Date.now() - this.startTime, a: curAzimuth });
                }
            } else {
                quality.reasons.forEach(r => {
                    this.stats.rejectionReasons[r] = (this.stats.rejectionReasons[r] || 0) + 1;
                });
            }
            
            const updatedSummary = this.tracker.getSummary();
            this.updateProgress(updatedSummary.percent, updatedSummary.sectors);
            
            // Update shell again with latest sectors if recording
            if (this.profile === 'bottle') this.bottleGuide.update(curAzimuth, curTilt, quality.isAccepted, updatedSummary.sectors);
            if (this.profile === 'generic') this.genericGuide.update(curAzimuth, curTilt, quality.isAccepted, updatedSummary.sectors);

            this.checkGate(updatedSummary);

            if (now - this.lastGuidanceTime > 3000 && updatedSummary.percent < 90 && updatedSummary.maxGap > 30) {
                const target = Math.floor(updatedSummary.gapCenter);
                this.showGuidanceToast(`Boşluğu doldurmak için ${target}° açısına doğru dönün`, "info");
                this.lastGuidanceTime = now;
            }
        }

        requestAnimationFrame(() => this.runMetricsLoop());
    }

    updateUIIndicators(quality, blur, lighting, azimuth) {
        document.getElementById('indicator-stability').querySelector('.dot').style.background = 
            blur > this.metrics.blurThreshold ? "var(--accent-color)" : "var(--error)";
        
        document.getElementById('indicator-lighting').querySelector('.dot').style.background = 
            quality.reasons.some(r => r === "too_dark" || r === "too_bright" || r === "highlight") ? "var(--error)" : "var(--accent-color)";
            
        this.angleArrow.style.transform = `rotate(${azimuth}deg)`;
        this.angleText.textContent = `${Math.floor(azimuth)}°`;

        if (this.isRecording) {
            this.statusLabel.textContent = quality.isAccepted ? "KAYDEDİLİYOR..." : "KALİTE UYARISI";
            this.statusLabel.style.color = quality.isAccepted ? "var(--accent-color)" : "var(--error)";
        } else {
            this.statusLabel.textContent = "HAZIR";
            this.statusLabel.style.color = "var(--accent-color)";
        }
    }

    updateProgress(percent, sectors = []) {
        this.coverageEl.textContent = `${Math.floor(percent)}%`;
        if (sectors.length > 0) {
            this.drawSectors(sectors);
        }
    }

    drawSectors(sectors) {
        const container = document.getElementById('ar-sector-ring');
        if (!container) return;
        
        if (container.children.length === 0) {
            const numSectors = sectors.length;
            const sectorSize = 360 / numSectors;
            for (let i = 0; i < numSectors; i++) {
                const arc = document.createElementNS("http://www.w3.org/2000/svg", "path");
                const startAngle = i * sectorSize - 90;
                const endAngle = (i + 1) * sectorSize - 90;
                const d = this.describeArc(50, 50, 45, startAngle, endAngle);
                arc.setAttribute("d", d);
                arc.setAttribute("fill", "none");
                arc.setAttribute("stroke", "rgba(255, 255, 255, 0.05)");
                arc.setAttribute("stroke-width", "3");
                container.appendChild(arc);
            }
        }
        
        Array.from(container.children).forEach((arc, i) => {
            if (sectors[i]) {
                arc.setAttribute("stroke", "var(--accent-color)");
                arc.setAttribute("stroke-width", "5");
                arc.setAttribute("opacity", "0.8");
            }
        });
    }

    describeArc(x, y, radius, startAngle, endAngle) {
        const start = this.polarToCartesian(x, y, radius, endAngle);
        const end = this.polarToCartesian(x, y, radius, startAngle);
        const arcSweep = endAngle - startAngle <= 180 ? "0" : "1";
        return ["M", start.x, start.y, "A", radius, radius, 0, arcSweep, 0, end.x, end.y].join(" ");
    }

    polarToCartesian(centerX, centerY, radius, angleInDegrees) {
        const angleInRadians = (angleInDegrees * Math.PI) / 180.0;
        return {
            x: centerX + radius * Math.cos(angleInRadians),
            y: centerY + radius * Math.sin(angleInRadians)
        };
    }

    checkGate(summary) {
        const elapsedSec = (Date.now() - this.startTime) / 1000;
        const profileCompletion = this.profile === 'box' ? {
            faces: Array.from(this.boxGuide.completed)
        } : this.profile === 'bottle' ? {
            cap: this.bottleGuide.isCapComplete,
            base: this.bottleGuide.isBaseComplete
        } : null;

        const validation = this.gateValidator.validate(summary, this.stats, elapsedSec, this.profile, profileCompletion);
        
        const newlyPassed = validation.canFinish && !this.canFinish;
        this.canFinish = validation.canFinish || this.isDemoMode;
        this.captureBtn.style.opacity = this.canFinish ? "1" : "0.5";
        
        if (newlyPassed && this.isRecording) {
            this.showGuidanceToast("Kalite kriterleri sağlandı! Bitirebilirsiniz.", "success");
        }

        if (!this.canFinish && this.isRecording) {
            let msg = validation.missingReq || "Dönmeye devam edin";
            
            if (!validation.reasons.duration) msg = `Çekiliyor... (${Math.max(0, Math.floor(this.gateValidator.minDuration - elapsedSec))}s kaldı)`;
            else if (!validation.reasons.frames) msg = "Detay ekleniyor...";
            else if (!validation.reasons.blur) msg = "Çok hızlı!";
            
            this.statusLabel.textContent = msg;
            this.statusLabel.style.color = "var(--error)";
        } else if (this.isRecording) {
            this.statusLabel.textContent = this.isDemoMode ? "DEMO MODU: HAZIR" : "BİTİRMEYE HAZIR";
            this.statusLabel.style.color = "var(--accent-color)";
        }
    }

    async finishRecording() {
        if (!this.canFinish) {
            alert("Capture quality gate not met. Please follow the guidance before finishing.");
            return;
        }

        this.isRecording = false;
        this.captureBtn.classList.remove('recording');
        this.demoBtn.disabled = false;
        this.timerEl.classList.add('hidden');
        clearInterval(this.timerInterval);

        const summary = this.tracker.getSummary();
        this.qualityManifest = {
            product_profile: this.profile,
            coverage_summary: summary,
            accepted_frame_count: this.stats.acceptedCount,
            total_frame_count: this.stats.totalCount,
            rejection_stats: this.stats.rejectionReasons,
            is_demo: this.isDemoMode,
            selected_frames: this.stats.selectedIndices,
            profile_completion: this.profile === 'box' ? {
                faces: Array.from(this.boxGuide.completed)
            } : this.profile === 'bottle' ? {
                cap: this.bottleGuide.isCapComplete,
                base: this.bottleGuide.isBaseComplete
            } : null,
            object_metrics: {
                object_bbox_area_ratio: 0.0,
                object_center_distance: 0.0,
                border_touch: false,
                clipping_risk: false,
                foreground_background_contrast: 0.0,
                _note: "Framing metrics placeholder. Not yet validated."
            },
            timestamp: new Date().toISOString()
        };

        if (this.isDemoMode) {
            // Demo mode upload with fake video - skip production upload if desired, 
            // but user says "Must not upload to production as a real capture"
            // For now, we allow the local flow but we could block it if needed.
            // The user said: "Demo captures must not be sent as production uploads."
            console.log("Demo capture - skipping production upload");
            alert("Demo capture completed (Not uploaded to production)");
            this.stop();
            return;
        } else if (this.mediaRecorder) {
            this.mediaRecorder.stop();
        }
    }

    async uploadResult(blob) {
        const videoBlob = blob || new Blob(this.chunks, { type: this.mediaRecorder.mimeType });
        const productId = this.productIdInput.value.trim();
        
        const formData = new FormData();
        formData.append('product_id', productId);
        formData.append('file', videoBlob, `capture_${productId}.webm`);
        formData.append('quality_manifest', JSON.stringify(this.qualityManifest));

        this.statusLabel.textContent = "UPLOADING...";
        this.statusLabel.style.color = "var(--accent-color)";
        this.captureBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/sessions/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (response.ok) {
                alert("Upload successful! Session created.");
                this.stop();
                await fetchProducts();
            } else {
                let errorMsg = result.detail;
                if (typeof result.detail === 'object') {
                    errorMsg = `${result.detail.message}\n${result.detail.reasons.join('\n')}`;
                }
                alert("Upload failed: " + errorMsg);
                this.statusLabel.textContent = "UPLOAD FAILED";
                this.captureBtn.disabled = false;
            }
        } catch (err) {
            alert("Upload error: " + err.message);
            this.statusLabel.textContent = "UPLOAD ERROR";
            this.captureBtn.disabled = false;
        }
    }
}

const arCapture = new ARCapture();

// --- Upload Handlers ---

function setupUploadHandlers() {
    openUploadBtn.onclick = () => arCapture.start();
    
    const manualUploadBtn = document.getElementById('open-manual-upload');
    if (manualUploadBtn) {
        manualUploadBtn.onclick = () => uploadModal.classList.remove('hidden');
    }

    closeUploadBtn.onclick = () => uploadModal.classList.add('hidden');

    uploadForm.onsubmit = async (e) => {
        e.preventDefault();
        const productId = document.getElementById('upload-product-id').value;
        const file = videoInput.files[0];

        if (!file || !productId) {
            alert("Please select a file and enter Product ID");
            return;
        }

        const formData = new FormData();
        formData.append('product_id', productId);
        formData.append('file', file);
        
        if (arCapture.qualityManifest) {
            formData.append('quality_manifest', JSON.stringify(arCapture.qualityManifest));
        }

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

async function cancelSession(sessionId) {
    if (!confirm(`Are you sure you want to cancel session ${sessionId}?`)) return;
    
    try {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}/cancel`, {
            method: 'POST'
        });
        const result = await response.json();
        if (response.ok) {
            alert("Session cancelled successfully.");
            // Refresh details
            const productId = document.getElementById('modal-title').textContent.split(': ')[1];
            showProductDetails(productId);
            fetchProducts();
        } else {
            alert("Failed to cancel: " + result.detail);
        }
    } catch (err) {
        alert("Error: " + err.message);
    }
}

function renderHistory(history) {
    historyContainer.innerHTML = `
        <table class="history-table">
            <thead>
                <tr><th>Asset ID</th><th>Version</th><th>Status</th><th>Actions</th></tr>
            </thead>
            <tbody>
                ${history.map(item => {
                    const isProcessing = item.status === 'processing' || item.status === 'CREATED' || item.status === 'uploaded';
                    return `
                        <tr>
                            <td>${item.asset_id}</td>
                            <td>${item.version}</td>
                            <td><span class="badge ${item.status.toLowerCase()}">${item.status}</span></td>
                            <td>
                                <div class="btn-group">
                                    <button class="btn-view" onclick="open3DViewer('${item.asset_id}', '${item.status}')">View</button>
                                    ${isProcessing ? `<button class="btn-cancel" onclick="cancelSession('${item.asset_id}')">Stop</button>` : ''}
                                </div>
                            </td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;
}

closeModal.addEventListener('click', () => {
    modalBackdrop.classList.add('hidden');
});

init();
