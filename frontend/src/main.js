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

// Drag & Drop and Upload Queue UI
function setupUploadUI() {
  const uploadSection = document.createElement('section');
  uploadSection.className = 'upload-section';
  uploadSection.innerHTML = `
    <h2>Upload Photos</h2>
    <div id="drop-zone" class="drop-zone">Drag & drop files or folders here</div>
    <div id="upload-queue-container"></div>
  `;
  document.querySelector('.main').prepend(uploadSection);

  const dropZone = document.getElementById('drop-zone');
  const uploadQueueContainer = document.getElementById('upload-queue-container');

  // Drag & drop events
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
    dropZone.textContent = `Release to upload (${e.dataTransfer.items.length} files)`;
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
    dropZone.textContent = 'Drag & drop files or folders here';
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    dropZone.textContent = 'Drag & drop files or folders here';
    handleFiles(e.dataTransfer.files);
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
    onStartAnalysis: startBatchAnalysis,
    errorStates
  });
  container.appendChild(uploadQueue);
}

function setupPauseResumeUI() {
  const pauseResumeBar = document.createElement('div');
  pauseResumeBar.className = 'pause-resume-bar';
  pauseResumeBar.innerHTML = `
    <button id="pause-analysis-btn">Pause Analysis</button>
    <button id="resume-analysis-btn" style="display:none;">Resume Analysis</button>
  `;
  document.querySelector('.main').prepend(pauseResumeBar);

  const pauseBtn = document.getElementById('pause-analysis-btn');
  const resumeBtn = document.getElementById('resume-analysis-btn');

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
    for (let i = 0; i < queuedFiles.length; i++) {
      while (analysisPaused) {
        await new Promise(res => setTimeout(res, 500));
      }
      const result = await apiClient.analyze(queuedFiles[i].file, sessionId);
      handleAnalysisResults([result]);
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
  const retryBar = document.createElement('div');
  retryBar.className = 'retry-failed-bar';
  retryBar.innerHTML = `<button id="retry-failed-btn">Retry Failed Files</button>`;
  document.querySelector('.main').prepend(retryBar);
  document.getElementById('retry-failed-btn').addEventListener('click', async () => {
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
  const exportBar = document.createElement('div');
  exportBar.className = 'csv-export-bar';
  exportBar.innerHTML = `<button id="export-csv-btn">Export Results (CSV)</button>`;
  document.querySelector('.main').prepend(exportBar);
  document.getElementById('export-csv-btn').addEventListener('click', () => {
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
        <h1>${APP_CONFIG.APP_NAME}</h1>
        <p class="version">v${APP_CONFIG.VERSION}</p>
      </header>
      
      <main class="main">
        <section class="status-section">
          <h2>System Status</h2>
          <div class="status-card">
            <div class="status-item">
              <span class="status-label">API Server:</span>
              <span class="status-value ${statusClass}" id="api-status">${statusText}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Backend Info:</span>
              <span class="status-value" id="backend-info">Checking...</span>
            </div>
            <div class="status-item">
              <span class="status-label">API URL:</span>
              <span class="status-value">${API_CONFIG.BASE_URL}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Max File Size:</span>
              <span class="status-value">${formatBytes(APP_CONFIG.MAX_FILE_SIZE)}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Batch Size:</span>
              <span class="status-value">${API_CONFIG.BATCH_SIZE} images</span>
            </div>
            <div class="status-item">
              <span class="status-label">Last Check:</span>
              <span class="status-value" id="last-check">Never</span>
            </div>
          </div>
        </section>
        
        <section class="endpoints-section">
          <h2>Available Endpoints</h2>
          <ul class="endpoints-list">
            ${Object.entries(API_CONFIG.ENDPOINTS).map(([name, path]) => `
              <li class="endpoint-item">
                <code>${name}</code>
                <span>${API_CONFIG.BASE_URL}${path}</span>
              </li>
            `).join('')}
          </ul>
        </section>
        
        <section class="config-section">
          <h2>Configuration Loaded ✓</h2>
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
        </section>
        
        <section class="api-test-section">
          <h2>API Client Test</h2>
          <div id="connection-status-container"></div>
          <div id="progress-tracker-container"></div>
          <div class="button-group">
            <button id="test-health-btn" class="btn btn-primary">Test Health Check</button>
            <button id="test-stats-btn" class="btn btn-secondary">Get Stats</button>
            <button id="test-analysis-btn" class="btn btn-secondary">Test Analysis</button>
            <button id="test-notifications-btn" class="btn btn-secondary">Test Notifications</button>
          </div>
          <div id="api-test-result" class="api-result"></div>
        </section>
      </main>
      
      <footer class="footer">
        <p>Construction Photo Analyzer - API Client Ready</p>
      </footer>
    </div>
  `;
  
  // Set up connection monitoring with real-time updates
  setupConnectionMonitoring();
  
  // Initialize Phase 3 components
  initializeComponents();
  
  // Set up API test buttons
  setupApiTestButtons();
  
  // Set up Upload UI
  setupUploadUI();
  setupPauseResumeUI();
  setupRetryFailedUI();
  setupPhotoGridUI();
  setupInteractiveFeedbackUI();
  setupCSVExportUI();
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
    statusElement.textContent = isConnected ? 'Online' : 'Offline';
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
  const gridSection = document.createElement('section');
  gridSection.className = 'photo-grid-section';
  gridSection.innerHTML = `
    <h2>Photo Analysis Results</h2>
    <div id="photo-filter-bar" class="photo-filter-bar">
      <label>Filter by Category:</label>
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
      <label>Sort by:</label>
      <select id="sort-by">
        <option value="confidence">Confidence</option>
        <option value="filename">Filename</option>
        <option value="analyzedAt">Analyzed Time</option>
      </select>
    </div>
    <div id="photo-grid-container"></div>
  `;
  document.querySelector('.main').prepend(gridSection);

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
  const feedbackSection = document.createElement('section');
  feedbackSection.className = 'feedback-section';
  feedbackSection.innerHTML = `
    <h2>Interactive Feedback</h2>
    <div id="feedback-form" class="feedback-form">
      <label for="feedback-input">Leave your feedback:</label>
      <textarea id="feedback-input" rows="4" placeholder="Enter your feedback here..."></textarea>
      <div class="button-group">
        <button id="submit-feedback-btn" class="btn btn-primary">Submit Feedback</button>
        <button id="clear-feedback-btn" class="btn btn-secondary">Clear</button>
      </div>
    </div>
    <div id="feedback-list" class="feedback-list"></div>
  `;
  document.querySelector('.main').prepend(feedbackSection);

  const feedbackInput = document.getElementById('feedback-input');
  const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
  const clearFeedbackBtn = document.getElementById('clear-feedback-btn');
  const feedbackList = document.getElementById('feedback-list');

  // Load saved feedback from stateManager
  const savedFeedback = stateManager.getState().feedback || [];
  savedFeedback.forEach(f => addFeedbackItem(f));

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
    notifications.success('Feedback submitted successfully');
  });

  clearFeedbackBtn.addEventListener('click', () => {
    feedbackInput.value = '';
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

// Initialize the app
initApp().catch(err => {
  console.error('Error initializing app:', err);
  notifications.error('Failed to initialize app: ' + err.message);
});
