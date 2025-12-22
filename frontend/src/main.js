/**
 * Construction Photo Analyzer - Main Entry Point
 * Initializes the application and demonstrates configuration loading
 */

import './styles/main.css';
import { API_CONFIG, APP_CONFIG, checkApiHealth } from './config/index.js';
import { connectionManager, apiClient, analysisService } from './services/index.js';
import { ConnectionStatus, ProgressTracker } from './components/index.js';
import { notifications } from './utils/notifications.js';

// Global component instances
let connectionStatus = null;
let progressTracker = null;

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

// Initialize the application
initApp();
