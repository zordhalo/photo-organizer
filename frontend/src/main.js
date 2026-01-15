/**
 * Construction Photo Analyzer - Main Entry Point
 * Initializes the application and demonstrates configuration loading
 */

import './styles/main.css';
import { API_CONFIG, APP_CONFIG, checkApiHealth } from './config/index.js';
import { connectionManager, apiClient, analysisService } from './services/index.js';
import { ConnectionStatus, ProgressTracker, UploadQueue } from './components/index.js';
import { notifications } from './utils/notifications.js';
import VirtualPhotoGrid from './components/VirtualPhotoGrid.js';
import { stateManager } from './services/stateManager.js';

// Global component instances
let connectionStatus = null;
let progressTracker = null;
let uploadQueue = null;
let photoGrid = null;
let sessionId = crypto.randomUUID(); // Session management
let queuedFiles = [];
let errorStates = {};
let filterCategory = 'all';
let sortBy = 'confidence';
let analysisPaused = false;
const MAX_FEEDBACK_LENGTH = 500;

// Drag & Drop and Upload Queue UI
function setupUploadUI() {
  const uploadSection = document.getElementById('upload-section');
  if (!uploadSection) return;

  uploadSection.innerHTML = `
    <div class="section-header">
      <div>
        <h2>Upload Photos</h2>
        <p class="section-subtitle">Drag files here or browse to start an analysis run.</p>
      </div>
    </div>
    <div id="drop-zone" class="drop-zone" tabindex="0" role="button" aria-label="Upload photos">
      <input id="file-input" class="file-input" type="file" accept="image/*" multiple />
      <div class="drop-zone-content">
        <div class="drop-zone-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" role="presentation">
            <path d="M12 3l4 4h-3v6h-2V7H8l4-4zm-6 14h12v2H6v-2z"></path>
          </svg>
        </div>
        <div class="drop-zone-text">
          <p id="drop-zone-title" class="drop-zone-title">Drag & drop files or folders here</p>
          <p class="drop-zone-subtitle">Images only. Max file size ${formatBytes(APP_CONFIG.MAX_FILE_SIZE)}.</p>
        </div>
        <div class="drop-zone-actions">
          <button id="browse-files-btn" class="btn btn-secondary" type="button">Browse files</button>
        </div>
      </div>
    </div>
    <div id="upload-queue-container"></div>
  `;

  const dropZone = document.getElementById('drop-zone');
  const dropZoneTitle = document.getElementById('drop-zone-title');
  const browseButton = document.getElementById('browse-files-btn');
  const fileInput = document.getElementById('file-input');
  const uploadQueueContainer = document.getElementById('upload-queue-container');

  const openFileDialog = () => fileInput.click();

  browseButton.addEventListener('click', (event) => {
    event.stopPropagation();
    openFileDialog();
  });

  dropZone.addEventListener('click', () => {
    openFileDialog();
  });

  dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openFileDialog();
    }
  });

  fileInput.addEventListener('change', (event) => {
    if (event.target.files?.length) {
      handleFiles(event.target.files);
    }
    fileInput.value = '';
  });

  // Drag & drop events
  dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('drag-over');
    const count = event.dataTransfer.items?.length || event.dataTransfer.files?.length || 0;
    dropZoneTitle.textContent = count
      ? `Release to upload ${count} file${count === 1 ? '' : 's'}`
      : 'Release to upload files';
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
    dropZoneTitle.textContent = 'Drag & drop files or folders here';
  });
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    dropZoneTitle.textContent = 'Drag & drop files or folders here';
    handleFiles(event.dataTransfer.files);
  });

  // Initial render
  renderUploadQueue(uploadQueueContainer);
}

function handleFiles(fileList) {
  for (const file of fileList) {
    if (!file.type.startsWith('image/')) {
      errorStates[file.name] = 'Invalid file type';
      continue;
    }
    if (file.size > APP_CONFIG.MAX_FILE_SIZE) {
      errorStates[file.name] = 'File too large';
      continue;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      queuedFiles.push({ file, previewUrl: e.target.result, size: file.size, type: file.type });
      renderUploadQueue(document.getElementById('upload-queue-container'));
    };
    reader.readAsDataURL(file);
  }
  renderUploadQueue(document.getElementById('upload-queue-container'));
}

function renderUploadQueue(container) {
  if (!container) return;
  container.innerHTML = '';
  uploadQueue = UploadQueue({
    files: queuedFiles,
    onRemove: (idx) => {
      queuedFiles.splice(idx, 1);
      renderUploadQueue(container);
    },
    onReorder: (fromIndex, toIndex) => {
      if (fromIndex === toIndex) return;
      const [moved] = queuedFiles.splice(fromIndex, 1);
      queuedFiles.splice(toIndex, 0, moved);
      renderUploadQueue(container);
    },
    onStartAnalysis: startBatchAnalysis,
    errorStates
  });
  container.appendChild(uploadQueue);
}

function setupPauseResumeUI() {
  const pauseBtn = document.getElementById('pause-analysis-btn');
  const resumeBtn = document.getElementById('resume-analysis-btn');
  if (!pauseBtn || !resumeBtn) return;

  pauseBtn.addEventListener('click', () => {
    analysisPaused = true;
    pauseBtn.style.display = 'none';
    resumeBtn.style.display = 'inline-block';
    notifications.info('Analysis paused');
  });
  resumeBtn.addEventListener('click', () => {
    analysisPaused = false;
    pauseBtn.style.display = 'inline-block';
    resumeBtn.style.display = 'none';
    notifications.info('Analysis resumed');
  });
}

// In startBatchAnalysis, check for pause
async function startBatchAnalysis() {
  // Send files to backend with sessionId
  try {
    const total = queuedFiles.length;
    const startedAt = Date.now();
    if (progressTracker) {
      progressTracker.show();
    }
    for (let i = 0; i < total; i++) {
      while (analysisPaused) {
        await new Promise(res => setTimeout(res, 500));
      }
      const result = await apiClient.analyze(queuedFiles[i].file, sessionId);
      handleAnalysisResults([result]);

      const current = i + 1;
      const percentage = total ? Math.round((current / total) * 100) : 0;
      const elapsedSeconds = Math.max((Date.now() - startedAt) / 1000, 1);
      const speed = (current / elapsedSeconds).toFixed(1);
      const eta = total > current ? Math.ceil((total - current) / (current / elapsedSeconds)) : 0;
      if (progressTracker) {
        progressTracker.update({ current, total, percentage, speed, eta });
      }
    }
    notifications.success('Batch upload and analysis complete');
    queuedFiles = [];
    errorStates = {};
    renderUploadQueue(document.getElementById('upload-queue-container'));
  } catch (err) {
    notifications.error('Upload failed: ' + err.message);
    // Keep files in queue
  }
}

// Update stateManager when new analysis results arrive
function handleAnalysisResults(results) {
  results.forEach(result => {
    stateManager.addPhoto({
      id: crypto.randomUUID(),
      filename: result.filename,
      thumbnail: result.thumbnail || result.base64 || '',
      category: result.category || 'unknown',
      confidence: result.confidence || 0,
      analyzedAt: Date.now(),
      sessionId,
      error: result.error || null
    });
  });
}

// Retry failed files UI
function setupRetryFailedUI() {
  const retryButton = document.getElementById('retry-failed-btn');
  if (!retryButton) return;

  retryButton.addEventListener('click', async () => {
    const failedPhotos = stateManager.getState().photos.filter(p => p.error);
    if (!failedPhotos.length) {
      notifications.info('No failed files to retry');
      return;
    }
    notifications.info(`Retrying ${failedPhotos.length} failed files...`);
    for (const photo of failedPhotos) {
      try {
        const result = await apiClient.analyzeSingle(photo.thumbnail, photo.filename);
        stateManager.updatePhoto(photo.id, {
          category: result.category || 'unknown',
          confidence: result.confidence || 0,
          error: null
        });
      } catch (err) {
        stateManager.updatePhoto(photo.id, { error: err.message });
      }
    }
  });
}

function setupCSVExportUI() {
  const exportButton = document.getElementById('export-csv-btn');
  if (!exportButton) return;

  exportButton.addEventListener('click', () => {
    const photos = stateManager.getState().photos;
    if (!photos.length) {
      notifications.info('No results to export');
      return;
    }
    const csvRows = [
      'Session ID,Filename,Category,Confidence,Analyzed At,Error',
      ...photos.map(p => `"${p.sessionId}","${p.filename}","${p.category}",${Math.round(p.confidence * 100)}%,${new Date(p.analyzedAt).toISOString()},"${p.error || ''}"`)
    ];
    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `photo-analysis-results-${sessionId}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    notifications.success('CSV exported successfully');
  });
}

/**
 * Initialize the application
 */
async function initApp() {
  const app = document.querySelector('#app');
  
  // Check API health status initially
  const apiHealthy = await checkApiHealth();
  const statusClass = apiHealthy ? 'status-online' : 'status-offline';
  const statusText = apiHealthy ? 'Online' : 'Offline';
  
  app.innerHTML = `
    <div class="container">
      <header class="header">
        <div class="header-title">
          <h1>${APP_CONFIG.APP_NAME}</h1>
          <p class="subtitle">Construction Photo Analyzer UI</p>
        </div>
        <div class="header-actions">
          <button id="toggle-dev-panel" class="btn btn-secondary" type="button" aria-expanded="false">
            Show developer tools
          </button>
        </div>
      </header>
      
      <main class="main">
        <section class="upload-section" id="upload-section"></section>

        <section class="photo-grid-section" id="photo-grid-section"></section>

        <section class="status-section" id="status-section">
          <div class="status-header">
            <div>
              <h2>System Status</h2>
              <p class="section-subtitle">Live backend health and configuration snapshot.</p>
            </div>
            <div id="connection-status-container"></div>
          </div>
          <div class="status-card">
            <div class="status-item">
              <span class="status-label">API Server</span>
              <span class="status-value ${statusClass}" id="api-status">${statusText}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Backend Info</span>
              <span class="status-value" id="backend-info">Checking...</span>
            </div>
            <div class="status-item">
              <span class="status-label">API URL</span>
              <span class="status-value">${API_CONFIG.BASE_URL}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Max File Size</span>
              <span class="status-value">${formatBytes(APP_CONFIG.MAX_FILE_SIZE)}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Batch Size</span>
              <span class="status-value">${API_CONFIG.BATCH_SIZE} images</span>
            </div>
            <div class="status-item">
              <span class="status-label">Last Check</span>
              <span class="status-value" id="last-check">Never</span>
            </div>
          </div>
        </section>

        <section class="feedback-section" id="feedback-section"></section>

        <section class="developer-section" id="developer-section">
          <div class="developer-header">
            <h2>Developer Tools</h2>
            <p class="section-subtitle">Diagnostics and API testing utilities.</p>
          </div>
          <div class="developer-content" id="developer-content" hidden>
            <div class="developer-grid">
              <div class="developer-card endpoints-section">
                <h3>Available Endpoints</h3>
                <ul class="endpoints-list">
                  ${Object.entries(API_CONFIG.ENDPOINTS).map(([name, path]) => `
                    <li class="endpoint-item">
                      <code>${name}</code>
                      <span>${API_CONFIG.BASE_URL}${path}</span>
                    </li>
                  `).join('')}
                </ul>
              </div>

              <div class="developer-card config-section">
                <h3>Configuration Loaded</h3>
                <p>Phase 2: API Client Integration complete. Connection monitoring is active.</p>
                <div class="features-list">
                  <div class="feature-item ${APP_CONFIG.ENABLE_FALLBACK ? 'enabled' : 'disabled'}">
                    Fallback Mode: ${APP_CONFIG.ENABLE_FALLBACK ? 'Enabled' : 'Disabled'}
                  </div>
                  <div class="feature-item ${APP_CONFIG.ENABLE_BATCH_UPLOAD ? 'enabled' : 'disabled'}">
                    Batch Upload: ${APP_CONFIG.ENABLE_BATCH_UPLOAD ? 'Enabled' : 'Disabled'}
                  </div>
                  <div class="feature-item ${APP_CONFIG.ENABLE_DRAG_DROP ? 'enabled' : 'disabled'}">
                    Drag & Drop: ${APP_CONFIG.ENABLE_DRAG_DROP ? 'Enabled' : 'Disabled'}
                  </div>
                </div>
              </div>

              <div class="developer-card api-test-section">
                <h3>API Client Test</h3>
                <div class="button-group">
                  <button id="test-health-btn" class="btn btn-primary">Test Health Check</button>
                  <button id="test-stats-btn" class="btn btn-secondary">Get Stats</button>
                  <button id="test-analysis-btn" class="btn btn-secondary">Test Analysis</button>
                  <button id="test-notifications-btn" class="btn btn-secondary">Test Notifications</button>
                </div>
                <div id="api-test-result" class="api-result"></div>
              </div>
            </div>
          </div>
        </section>
      </main>
      
      <footer class="footer">
        <p>Construction Photo Analyzer - API Client Ready</p>
        <p class="version">v${APP_CONFIG.VERSION}</p>
      </footer>
    </div>
  `;
  
  // Set up connection monitoring with real-time updates
  setupConnectionMonitoring();

  // Build primary UI sections
  setupUploadUI();
  setupPhotoGridUI();
  setupInteractiveFeedbackUI();

  // Initialize components that depend on DOM nodes
  initializeComponents();

  // Set up API test buttons
  setupApiTestButtons();

  // Hook up action controls
  setupPauseResumeUI();
  setupRetryFailedUI();
  setupCSVExportUI();
  setupDeveloperPanel();
}

/**
 * Set up connection monitoring and UI updates
 */
function setupConnectionMonitoring() {
  // Listen for connection status changes
  connectionManager.onConnectionChange((isConnected, backendInfo) => {
    updateConnectionStatus(isConnected, backendInfo);
  });
  
  // Start monitoring
  connectionManager.startMonitoring();
}

/**
 * Initialize Phase 3 components
 */
function initializeComponents() {
  // Initialize ConnectionStatus component
  const connectionStatusContainer = document.getElementById('connection-status-container');
  if (connectionStatusContainer) {
    connectionStatus = new ConnectionStatus();
    connectionStatus.mount(connectionStatusContainer);
  }
  
  // Initialize ProgressTracker component
  const progressTrackerContainer = document.getElementById('progress-tracker-container');
  if (progressTrackerContainer) {
    progressTracker = new ProgressTracker();
    progressTracker.mount(progressTrackerContainer);
    
    // Set up cancel handler
    progressTracker.onCancel(() => {
      analysisService.cancel();
      notifications.warning('Analysis cancelled');
    });
  }
  
  // Show welcome notification
  setTimeout(() => {
    notifications.info('Phase 3 components loaded successfully');
  }, 500);
}

/**
 * Update the connection status in the UI
 * @param {boolean} isConnected - Connection status
 * @param {Object} backendInfo - Backend health info
 */
function updateConnectionStatus(isConnected, backendInfo) {
  const statusElement = document.getElementById('api-status');
  const backendInfoElement = document.getElementById('backend-info');
  const lastCheckElement = document.getElementById('last-check');
  
  if (statusElement) {
    const statusText = isConnected ? 'Online' : 'Offline';
    statusElement.textContent = statusText;
    statusElement.className = `status-value ${isConnected ? 'status-online' : 'status-offline'}`;
  }
  
  if (backendInfoElement) {
    if (backendInfo) {
      const cudaStatus = backendInfo.cuda_available ? '✓ CUDA' : '✗ CPU only';
      backendInfoElement.textContent = `${backendInfo.status} | ${cudaStatus}`;
    } else {
      backendInfoElement.textContent = 'Unavailable';
    }
  }
  
  if (lastCheckElement) {
    lastCheckElement.textContent = new Date().toLocaleTimeString();
  }
}

/**
 * Set up API test buttons
 */
function setupApiTestButtons() {
  const healthBtn = document.getElementById('test-health-btn');
  const statsBtn = document.getElementById('test-stats-btn');
  const analysisBtn = document.getElementById('test-analysis-btn');
  const notificationsBtn = document.getElementById('test-notifications-btn');
  const resultDiv = document.getElementById('api-test-result');
  
  if (healthBtn) {
    healthBtn.addEventListener('click', async () => {
      resultDiv.textContent = 'Testing health endpoint...';
      resultDiv.className = 'api-result loading';
      try {
        const result = await apiClient.checkHealth();
        resultDiv.textContent = JSON.stringify(result, null, 2);
        resultDiv.className = 'api-result success';
        notifications.success('Health check passed');
      } catch (error) {
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.className = 'api-result error';
        notifications.error(`Health check failed: ${error.message}`);
      }
    });
  }
  
  if (statsBtn) {
    statsBtn.addEventListener('click', async () => {
      resultDiv.textContent = 'Fetching stats...';
      resultDiv.className = 'api-result loading';
      try {
        const result = await apiClient.getStats();
        resultDiv.textContent = JSON.stringify(result, null, 2);
        resultDiv.className = 'api-result success';
        notifications.success('Stats retrieved successfully');
      } catch (error) {
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.className = 'api-result error';
        notifications.error(`Failed to get stats: ${error.message}`);
      }
    });
  }
  
  if (analysisBtn) {
    analysisBtn.addEventListener('click', async () => {
      // Demo analysis with mock photos
      await runDemoAnalysis(resultDiv);
    });
  }
  
  if (notificationsBtn) {
    notificationsBtn.addEventListener('click', () => {
      // Demo all notification types
      notifications.success('This is a success notification');
      setTimeout(() => notifications.error('This is an error notification'), 500);
      setTimeout(() => notifications.warning('This is a warning notification'), 1000);
      setTimeout(() => notifications.info('This is an info notification'), 1500);
    });
  }
}

/**
 * Run demo analysis with mock photos to test progress tracking
 * @param {HTMLElement} resultDiv - Element to show results
 */
async function runDemoAnalysis(resultDiv) {
  // Generate mock photos for demo
  const mockPhotos = [];
  const totalPhotos = 20; // Simulate 20 photos
  
  for (let i = 0; i < totalPhotos; i++) {
    mockPhotos.push({
      filename: `demo_photo_${i + 1}_${getRandomCategory()}.jpg`,
      base64: 'mock_base64_data'
    });
  }
  
  notifications.info(`Starting analysis of ${totalPhotos} demo photos...`);
  
  // Show progress tracker
  if (progressTracker) {
    progressTracker.show();
  }
  
  resultDiv.textContent = 'Analyzing demo photos...';
  resultDiv.className = 'api-result loading';
  
  try {
    const results = await analysisService.analyzePhotos(mockPhotos, (progress) => {
      if (progressTracker) {
        progressTracker.update(progress);
      }
    });
    
    resultDiv.textContent = JSON.stringify(results, null, 2);
    resultDiv.className = 'api-result success';
    notifications.success(`Analysis complete! Processed ${results.length} photos`);
  } catch (error) {
    resultDiv.textContent = `Error: ${error.message}`;
    resultDiv.className = 'api-result error';
    notifications.error(`Analysis failed: ${error.message}`);
    
    if (progressTracker) {
      progressTracker.showError(error.message);
    }
  }
}

/**
 * Get a random category name for demo photos
 * @returns {string} Random category
 */
function getRandomCategory() {
  const categories = ['interior', 'exterior', 'mep', 'structure', 'roofing', 'flooring', 'plumbing'];
  return categories[Math.floor(Math.random() * categories.length)];
}

/**
 * Format bytes to human-readable string
 * @param {number} bytes - Number of bytes
 * @returns {string} Formatted string
 */
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function setupPhotoGridUI() {
  const gridSection = document.getElementById('photo-grid-section');
  if (!gridSection) return;

  gridSection.innerHTML = `
    <div class="results-header">
      <div>
        <h2>Photo Analysis Results</h2>
        <p class="section-subtitle">Review detections, confidence, and outputs.</p>
      </div>
      <div class="results-actions">
        <button id="pause-analysis-btn" class="btn btn-secondary" type="button">Pause Analysis</button>
        <button id="resume-analysis-btn" class="btn btn-secondary" type="button" style="display:none;">Resume Analysis</button>
        <button id="retry-failed-btn" class="btn btn-secondary" type="button">Retry Failed</button>
        <button id="export-csv-btn" class="btn btn-secondary" type="button">Export Results (CSV)</button>
      </div>
    </div>
    <div id="progress-tracker-container" class="results-progress"></div>
    <div id="photo-filter-bar" class="photo-filter-bar">
      <label for="category-filter">Filter by Category</label>
      <select id="category-filter">
        <option value="all">All</option>
        <option value="interior">Interior</option>
        <option value="exterior">Exterior</option>
        <option value="mep">MEP</option>
        <option value="structure">Structure</option>
        <option value="roofing">Roofing</option>
        <option value="flooring">Flooring</option>
        <option value="insulation">Insulation</option>
        <option value="drywall">Drywall</option>
        <option value="windows">Windows</option>
        <option value="doors">Doors</option>
        <option value="unknown">Unknown</option>
      </select>
      <label for="sort-by">Sort by</label>
      <select id="sort-by">
        <option value="confidence">Confidence</option>
        <option value="filename">Filename</option>
        <option value="analyzedAt">Analyzed Time</option>
      </select>
    </div>
    <div id="photo-grid-container"></div>
  `;

  const categoryFilter = document.getElementById('category-filter');
  const sortBySelect = document.getElementById('sort-by');
  const photoGridContainer = document.getElementById('photo-grid-container');

  // Render initial photo grid
  renderPhotoGrid(photoGridContainer, stateManager.getState().photos);

  categoryFilter.addEventListener('change', () => {
    filterCategory = categoryFilter.value;
    renderPhotoGrid(photoGridContainer, stateManager.getState().photos);
  });

  sortBySelect.addEventListener('change', () => {
    sortBy = sortBySelect.value;
    renderPhotoGrid(photoGridContainer, stateManager.getState().photos);
  });

  stateManager.subscribe((state) => {
    renderPhotoGrid(photoGridContainer, state.photos);
    updateResultsActions(state.photos);
  });

  updateResultsActions(stateManager.getState().photos);
}

function renderPhotoGrid(container, photos) {
  if (!container) return;
  container.innerHTML = '';

  // Filter and sort photos
  let filteredPhotos = photos.filter(p => filterCategory === 'all' || p.category === filterCategory);
  if (sortBy === 'confidence') {
    filteredPhotos.sort((a, b) => b.confidence - a.confidence);
  } else if (sortBy === 'filename') {
    filteredPhotos.sort((a, b) => a.filename.localeCompare(b.filename));
  } else if (sortBy === 'analyzedAt') {
    filteredPhotos.sort((a, b) => b.analyzedAt - a.analyzedAt);
  }

  if (!filteredPhotos.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" role="presentation">
            <path d="M4 5h16a1 1 0 011 1v12a1 1 0 01-1 1H4a1 1 0 01-1-1V6a1 1 0 011-1zm0 2v10h16V7H4zm4 2l2 2 3-3 4 4H6l2-3z"></path>
          </svg>
        </div>
        <h3>No analysis results yet</h3>
        <p>Upload photos to kick off analysis and see thumbnails, labels, and confidence scores here.</p>
        <button class="btn btn-primary" type="button" id="empty-upload-cta">Upload photos</button>
      </div>
    `;
    const emptyCta = container.querySelector('#empty-upload-cta');
    if (emptyCta) {
      emptyCta.addEventListener('click', () => {
        document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' });
      });
    }
    return;
  }

  // Paginate photos
  const pageSize = 10;
  const pageCount = Math.ceil(filteredPhotos.length / pageSize);
  let currentPage = 1;

  function renderPage(page) {
    const start = (page - 1) * pageSize;
    const end = start + pageSize;
    const pagePhotos = filteredPhotos.slice(start, end);

    container.innerHTML = `
      <div class="photo-grid">
        ${pagePhotos.map(p => `
          <div class="photo-card">
            <img src="${p.thumbnail}" alt="${p.filename}" class="photo-thumbnail">
            <div class="photo-info">
              <div class="photo-meta">
                <span class="photo-category">${p.category || 'Unknown'}</span>
                <span class="photo-confidence">Confidence: ${Math.round(p.confidence * 100)}%</span>
              </div>
              <div class="photo-actions">
                <button class="btn btn-secondary btn-analyze-again" data-id="${p.id}">Analyze Again</button>
                <button class="btn btn-danger btn-remove" data-id="${p.id}">Remove</button>
              </div>
            </div>
          </div>
        `).join('')}
      </div>
      <div class="pagination">
        <button class="btn btn-primary btn-prev" ${currentPage === 1 ? 'disabled' : ''}>« Previous</button>
        <span class="page-info">Page ${currentPage} of ${pageCount}</span>
        <button class="btn btn-primary btn-next" ${currentPage === pageCount ? 'disabled' : ''}>Next »</button>
      </div>
    `;

    // Attach event listeners for pagination
    container.querySelector('.btn-prev').addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        renderPage(currentPage);
      }
    });
    container.querySelector('.btn-next').addEventListener('click', () => {
      if (currentPage < pageCount) {
        currentPage++;
        renderPage(currentPage);
      }
    });

    // Attach event listeners for analyze again and remove buttons
    pagePhotos.forEach(p => {
      container.querySelector(`.btn-analyze-again[data-id="${p.id}"]`).addEventListener('click', async () => {
        try {
          const result = await apiClient.analyzeSingle(p.thumbnail, p.filename);
          stateManager.updatePhoto(p.id, {
            category: result.category || 'unknown',
            confidence: result.confidence || 0,
            error: null
          });
          notifications.success(`Re-analysis complete for ${p.filename}`);
        } catch (err) {
          notifications.error(`Re-analysis failed for ${p.filename}: ${err.message}`);
        }
      });
      container.querySelector(`.btn-remove[data-id="${p.id}"]`).addEventListener('click', () => {
        stateManager.removePhoto(p.id);
        renderPhotoGrid(container, stateManager.getState().photos);
        notifications.info(`Removed ${p.filename} from results`);
      });
    });
  }

  renderPage(currentPage);
}

/**
 * Set up interactive feedback UI for demo
 */
function setupInteractiveFeedbackUI() {
  const feedbackSection = document.getElementById('feedback-section');
  if (!feedbackSection) return;

  feedbackSection.innerHTML = `
    <div class="section-header">
      <div>
        <h2>Interactive Feedback</h2>
        <p class="section-subtitle">Share quick notes, issues, and improvement ideas.</p>
      </div>
    </div>
    <div id="feedback-form" class="feedback-form">
      <label for="feedback-input">Leave your feedback</label>
      <textarea id="feedback-input" rows="6" maxlength="${MAX_FEEDBACK_LENGTH}" placeholder="What did you notice or need help with?"></textarea>
      <div class="feedback-meta-row">
        <span id="feedback-count" class="feedback-count">0 / ${MAX_FEEDBACK_LENGTH}</span>
        <span id="feedback-status" class="feedback-status" role="status" aria-live="polite"></span>
      </div>
      <div class="button-group">
        <button id="submit-feedback-btn" class="btn btn-primary">Submit Feedback</button>
        <button id="clear-feedback-btn" class="btn btn-secondary">Clear</button>
      </div>
    </div>
    <div id="feedback-list" class="feedback-list"></div>
  `;

  const feedbackInput = document.getElementById('feedback-input');
  const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
  const clearFeedbackBtn = document.getElementById('clear-feedback-btn');
  const feedbackList = document.getElementById('feedback-list');
  const feedbackCount = document.getElementById('feedback-count');
  const feedbackStatus = document.getElementById('feedback-status');

  const autoResize = () => {
    feedbackInput.style.height = 'auto';
    feedbackInput.style.height = `${feedbackInput.scrollHeight}px`;
  };

  const updateCount = () => {
    feedbackCount.textContent = `${feedbackInput.value.length} / ${MAX_FEEDBACK_LENGTH}`;
  };

  // Load saved feedback from stateManager
  const savedFeedback = stateManager.getState().feedback || [];
  savedFeedback.forEach(f => addFeedbackItem(f));

  updateCount();
  autoResize();
  feedbackInput.addEventListener('input', () => {
    updateCount();
    autoResize();
    feedbackStatus.textContent = '';
  });

  submitFeedbackBtn.addEventListener('click', () => {
    const text = feedbackInput.value.trim();
    if (!text) {
      notifications.warning('Feedback cannot be empty');
      return;
    }
    // Save feedback to stateManager
    const feedbackItem = {
      id: crypto.randomUUID(),
      text,
      timestamp: Date.now()
    };
    stateManager.addFeedback(feedbackItem);
    addFeedbackItem(feedbackItem);
    feedbackInput.value = '';
    updateCount();
    feedbackStatus.textContent = 'Thanks! Your feedback was recorded.';
    notifications.success('Feedback submitted successfully');
  });

  clearFeedbackBtn.addEventListener('click', () => {
    feedbackInput.value = '';
    updateCount();
    autoResize();
  });

  function addFeedbackItem(item) {
    const div = document.createElement('div');
    div.className = 'feedback-item';
    div.innerHTML = `
      <div class="feedback-text">${item.text}</div>
      <div class="feedback-meta">
        <span class="feedback-timestamp">${new Date(item.timestamp).toLocaleString()}</span>
        <button class="btn btn-danger btn-delete-feedback" data-id="${item.id}">Delete</button>
      </div>
    `;
    feedbackList.prepend(div);

    // Attach delete event
    div.querySelector('.btn-delete-feedback').addEventListener('click', () => {
      stateManager.removeFeedback(item.id);
      div.remove();
      notifications.info('Feedback deleted');
    });
  }
}

function setupDeveloperPanel() {
  const toggleButton = document.getElementById('toggle-dev-panel');
  const content = document.getElementById('developer-content');
  if (!toggleButton || !content) return;

  const setExpanded = (isExpanded) => {
    content.hidden = !isExpanded;
    toggleButton.setAttribute('aria-expanded', String(isExpanded));
    toggleButton.textContent = isExpanded ? 'Hide developer tools' : 'Show developer tools';
  };

  toggleButton.addEventListener('click', () => {
    setExpanded(content.hidden);
  });

  setExpanded(false);
}

function updateResultsActions(photos) {
  const retryButton = document.getElementById('retry-failed-btn');
  const exportButton = document.getElementById('export-csv-btn');
  if (retryButton) {
    const hasFailures = photos.some(photo => photo.error);
    retryButton.disabled = !hasFailures;
  }
  if (exportButton) {
    exportButton.disabled = photos.length === 0;
  }
}

// Initialize the app
initApp().catch(err => {
  console.error('Error initializing app:', err);
  notifications.error('Failed to initialize app: ' + err.message);
});
