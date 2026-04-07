const API_BASE = "http://localhost:8001/api";

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

// --- Initialization ---

async function init() {
    await fetchProducts();
    await fetchLogs();
    
    setInterval(fetchLogs, 5000);
    setupUploadHandlers();
}

// --- Upload Handlers ---

function setupUploadHandlers() {
    openUploadBtn.onclick = () => uploadModal.classList.remove('hidden');
    closeUploadBtn.onclick = () => uploadModal.classList.add('hidden');

    // Drag and Drop
    dropZone.onclick = (e) => {
        console.log('Dropzone clicked');
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
                // Show 'Uploading' text
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
        console.error("View3D called without valid Asset ID");
        viewerStatus.innerHTML = '<span style="color: #f87171">❌ Error: Asset ID not found. Still processing?</span>';
        return;
    }

    viewerModal.classList.remove('hidden');
    viewerTitle.textContent = `Asset Preview: ${assetId}`;
    
    // Reset viewer state
    console.log(`Attempting to load asset: ${assetId} (Status: ${status})`);
    mainViewer.src = "";
    viewerStatus.innerHTML = '<span class="status-pulse-yellow"></span> Checking availability...';
    
    // Use timestamp to break any cache during repeated tests
    const timestamp = Date.now();
    const modelUrl = `${API_BASE}/assets/blobs/${assetId}.glb?t=${timestamp}`;
    
    if (status === 'processing' || status === 'CREATED') {
        viewerStatus.innerHTML = '⚙️ <span style="color: #fbbf24">Processing: Reconstruction in progress. Showing demo model.</span>';
        mainViewer.src = "https://modelviewer.dev/shared-assets/models/Astronaut.glb";
    } else {
        viewerStatus.textContent = "🚀 Fetching produced asset...";
        mainViewer.src = modelUrl;

        // More robust load detection
        const onLoad = () => {
             console.log("Model-viewer: Success loading", modelUrl);
             viewerStatus.innerHTML = '✅ <span style="color: #4ade80">Asset Produced Successfully.</span>';
             mainViewer.removeEventListener('load', onLoad);
        };

        const onError = (error) => {
             console.error("Model-viewer error:", error);
             viewerStatus.innerHTML = '⚠️ <span style="color: #fbbf24">Asset was registered but file loading failed. Showing fallback.</span>';
             mainViewer.src = "https://modelviewer.dev/shared-assets/models/Astronaut.glb";
             mainViewer.removeEventListener('error', onError);
        };

        mainViewer.addEventListener('load', onLoad);
        mainViewer.addEventListener('error', onError);
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
        const hasActiveUpdate = p.has_active_session ? '<span class="pulse-dot"></span>' : '';
        
        return `
            <div class="product-card glass ${isProcessing ? 'processing' : ''}" onclick="showProductDetails('${p.id}')">
                <div class="product-id">
                    ${p.id}
                    ${hasActiveUpdate}
                </div>
                <div class="badge ${isProcessing ? 'badge-warning' : ''}">
                    ${isProcessing ? 'PROCESSING' : `v${p.active_id ? p.active_id.split('_').pop() : 'N/A'}`}
                </div>
                <div class="v-count">
                    ${isProcessing ? 'New session in progress' : `${p.asset_count} versions total`}
                </div>
                <div class="last_seen" style="font-size: 0.7rem; color: #666; margin-top: 10px;">
                    Last activity: ${new Date(p.last_updated * 1000).toLocaleString()}
                </div>
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
            ${log.component ? `<br/><small style="color:#555">[${log.component}]</small>` : ''}
        </div>
    `).join('');
}

function updateStats() {
    totalProductsEl.textContent = state.products.length;
    activeVersionsEl.textContent = state.products.filter(p => p.active_id).length;
}

// Search Handler
searchInput.addEventListener('input', (e) => {
    state.searchQuery = e.target.value;
    renderProducts();
});

// Modal Actions
async function showProductDetails(productId) {
    modalBackdrop.classList.remove('hidden');
    document.getElementById('modal-title').textContent = `History: ${productId}`;
    historyContainer.innerHTML = '<div class="loader">Loading history...</div>';

    try {
        const response = await fetch(`${API_BASE}/products/${productId}/history`);
        const history = await response.json();
        renderHistory(history);
    } catch (err) {
        historyContainer.innerHTML = `<div class="error">Error loading history: ${err.message}</div>`;
    }
}

function renderHistory(history) {
    historyContainer.innerHTML = `
        <div class="history-table-container">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Asset ID</th>
                        <th>Version</th>
                        <th>Status</th>
                        <th>Active</th>
                        <th>Approved</th>
                        <th>View</th>
                    </tr>
                </thead>
                <tbody>
                    ${history.map(item => `
                        <tr>
                            <td class="id-cell">${item.asset_id}</td>
                            <td>${item.version}</td>
                            <td><span class="status-tag">${item.status}</span></td>
                            <td>${item.is_active ? '✅' : '-'}</td>
                            <td>${item.approved ? '💎' : '⏳'}</td>
                            <td>
                                <button class="btn-view" onclick="open3DViewer('${item.asset_id}', '${item.status}')">
                                    View 3D
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        <div class="audit-section">
            <h3>Recent Audit Events</h3>
            ${history.map(item => item.audit.map(a => `
                <div class="audit-log">
                    [${new Date(item.created_at || Date.now()).toLocaleDateString()}] 
                    <b>${a.action}</b>: ${a.asset_id}
                </div>
            `).join('')).join('')}
        </div>
    `;
}

closeModal.addEventListener('click', () => {
    modalBackdrop.classList.add('hidden');
});

// Initialize
init();
