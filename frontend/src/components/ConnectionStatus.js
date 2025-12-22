/**
 * Connection Status Component
 * Displays real-time backend connection status with GPU availability
 */

import { connectionManager } from '../services/connectionManager';
import { apiClient } from '../services/apiClient';

class ConnectionStatus {
  constructor() {
    this.element = this.createStatusIndicator();
    this.unsubscribe = null;
    this.setupListeners();
  }

  /**
   * Create the status indicator DOM element
   * @returns {HTMLElement} The status indicator element
   */
  createStatusIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'connection-status';
    indicator.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Checking...</span>
      <span class="backend-info"></span>
    `;
    return indicator;
  }

  /**
   * Set up connection change listeners
   */
  setupListeners() {
    this.unsubscribe = connectionManager.onConnectionChange((isConnected, backendInfo) => {
      this.updateStatus(isConnected, backendInfo);
    });
  }

  /**
   * Update the status display based on connection state
   * @param {boolean} isConnected - Whether backend is connected
   * @param {Object} backendInfo - Backend health information
   */
  updateStatus(isConnected, backendInfo) {
    const dot = this.element.querySelector('.status-dot');
    const text = this.element.querySelector('.status-text');
    const info = this.element.querySelector('.backend-info');

    if (isConnected) {
      dot.className = 'status-dot connected';
      text.textContent = '✅ Backend Connected';
      
      // Display backend info if available from health check
      if (backendInfo) {
        const gpuStatus = backendInfo.cuda_available ? '✅ GPU Available' : '❌ CPU Only';
        info.textContent = gpuStatus;
        info.title = backendInfo.model_loaded ? 'Model loaded and ready' : 'Model loading...';
      } else {
        // Fetch additional stats if not provided
        this.fetchBackendInfo();
      }
    } else {
      dot.className = 'status-dot disconnected';
      text.textContent = '❌ Backend Offline (Fallback Mode)';
      info.textContent = '';
      info.title = '';
    }
  }

  /**
   * Fetch backend statistics for GPU info
   */
  async fetchBackendInfo() {
    try {
      const stats = await apiClient.getStats();
      const info = this.element.querySelector('.backend-info');
      
      if (stats.cuda_available) {
        info.textContent = `✅ GPU: ${stats.gpu_name || 'Available'}`;
      } else {
        info.textContent = '❌ GPU Not Available';
      }
      
      if (stats.images_analyzed !== undefined) {
        info.title = `Images analyzed: ${stats.images_analyzed}`;
      }
    } catch (error) {
      console.error('Failed to fetch backend info:', error);
      // Don't update info on error - keep previous state
    }
  }

  /**
   * Mount the component to a parent element
   * @param {HTMLElement} parentElement - The element to mount to
   */
  mount(parentElement) {
    parentElement.appendChild(this.element);
    connectionManager.startMonitoring();
  }

  /**
   * Unmount and cleanup the component
   */
  unmount() {
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = null;
    }
    connectionManager.stopMonitoring();
    if (this.element.parentElement) {
      this.element.parentElement.removeChild(this.element);
    }
  }

  /**
   * Get the underlying DOM element
   * @returns {HTMLElement}
   */
  getElement() {
    return this.element;
  }
}

export default ConnectionStatus;
