/**
 * Connection Manager for Construction Photo Analyzer
 * Monitors backend connectivity and notifies listeners of status changes
 */

import { apiClient } from './apiClient';

class ConnectionManager {
  constructor() {
    this.isConnected = false;
    this.lastCheck = null;
    this.lastError = null;
    this.checkInterval = 5000; // 5 seconds
    this.intervalId = null;
    this.listeners = [];
    this.backendInfo = null;
  }

  /**
   * Check connection to the backend
   * @returns {Promise<boolean>} Connection status
   */
  async checkConnection() {
    try {
      const health = await apiClient.checkHealth();
      this.backendInfo = health;
      this.lastError = null;
      this.setConnected(true);
      return true;
    } catch (error) {
      this.lastError = error.message || 'Unknown error';
      this.backendInfo = null;
      this.setConnected(false);
      return false;
    }
  }

  /**
   * Update connection status and notify listeners if changed
   * @param {boolean} status - New connection status
   */
  setConnected(status) {
    const previousStatus = this.isConnected;
    this.isConnected = status;
    this.lastCheck = Date.now();

    // Only notify if status changed
    if (previousStatus !== status) {
      console.log(`Connection status changed: ${status ? 'Connected' : 'Disconnected'}`);
      this.notifyListeners(status);
    }
  }

  /**
   * Start periodic connection monitoring
   */
  startMonitoring() {
    if (this.intervalId) {
      console.warn('Connection monitoring already started');
      return;
    }

    console.log('Starting connection monitoring...');
    
    // Check immediately
    this.checkConnection();

    // Then check periodically
    this.intervalId = setInterval(() => {
      this.checkConnection();
    }, this.checkInterval);
  }

  /**
   * Stop connection monitoring
   */
  stopMonitoring() {
    if (this.intervalId) {
      console.log('Stopping connection monitoring');
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Register a callback for connection status changes
   * @param {Function} callback - Function to call with (isConnected, backendInfo)
   * @returns {Function} Unsubscribe function
   */
  onConnectionChange(callback) {
    this.listeners.push(callback);
    
    // Immediately call with current status
    callback(this.isConnected, this.backendInfo);

    // Return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(callback);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  /**
   * Notify all listeners of status change
   * @param {boolean} status - New connection status
   */
  notifyListeners(status) {
    this.listeners.forEach(callback => {
      try {
        callback(status, this.backendInfo);
      } catch (error) {
        console.error('Error in connection listener:', error);
      }
    });
  }

  /**
   * Get current connection state
   * @returns {Object} Connection state object
   */
  getState() {
    return {
      isConnected: this.isConnected,
      lastCheck: this.lastCheck,
      lastError: this.lastError,
      backendInfo: this.backendInfo,
      isMonitoring: this.intervalId !== null
    };
  }

  /**
   * Set the check interval for monitoring
   * @param {number} interval - Interval in milliseconds
   */
  setCheckInterval(interval) {
    this.checkInterval = interval;
    
    // Restart monitoring with new interval if currently monitoring
    if (this.intervalId) {
      this.stopMonitoring();
      this.startMonitoring();
    }
  }

  /**
   * Wait for connection to be established
   * @param {number} timeout - Maximum wait time in milliseconds
   * @returns {Promise<boolean>} True if connected, false if timeout
   */
  async waitForConnection(timeout = 30000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      if (await this.checkConnection()) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    return false;
  }
}

// Export singleton instance
export const connectionManager = new ConnectionManager();

// Also export class for testing
export { ConnectionManager };
