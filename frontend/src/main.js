/**
 * Construction Photo Analyzer - Main Entry Point
 * Initializes the application and demonstrates configuration loading
 */

import './styles/main.css';
import { API_CONFIG, APP_CONFIG, checkApiHealth } from './config/index.js';

/**
 * Initialize the application
 */
async function initApp() {
  const app = document.querySelector('#app');
  
  // Check API health status
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
              <span class="status-value ${statusClass}">${statusText}</span>
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
          <p>The frontend is successfully configured and ready for Phase 2 (API Client Integration).</p>
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
      </main>
      
      <footer class="footer">
        <p>Construction Photo Analyzer - Frontend Ready</p>
      </footer>
    </div>
  `;
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
