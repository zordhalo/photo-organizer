/**
 * API Client for Construction Photo Analyzer
 * Handles all HTTP communication with the FastAPI backend
 * Includes retry logic, timeout handling, and error management
 */

import axios from 'axios';
import { API_CONFIG } from '../config/api.config';

class APIClient {
  constructor() {
    // Create axios instance with default configuration
    this.client = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      config => {
        console.log(`API Request: ${config.method.toUpperCase()} ${config.url}`);
        return config;
      },
      error => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      response => response,
      error => this.handleError(error)
    );
  }

  /**
   * Retry a request with exponential backoff
   * @param {Function} requestFn - Function that returns a promise
   * @param {number} retries - Number of retry attempts
   * @returns {Promise} - Resolved response or rejected error
   */
  async retryRequest(requestFn, retries = API_CONFIG.RETRY_ATTEMPTS) {
    let lastError;
    
    for (let i = 0; i < retries; i++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error;
        
        // Don't retry on client errors (4xx)
        if (error.response && error.response.status >= 400 && error.response.status < 500) {
          throw error;
        }
        
        if (i === retries - 1) {
          throw error;
        }
        
        // Exponential backoff: 1s, 2s, 4s...
        const delay = Math.pow(2, i) * 1000;
        console.log(`Retry attempt ${i + 1}/${retries} after ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError;
  }

  /**
   * Handle API errors with user-friendly messages
   * @param {Error} error - The error object
   * @throws {Error} - Enhanced error with user-friendly message
   */
  handleError(error) {
    // Timeout error
    if (error.code === 'ECONNABORTED') {
      const timeoutError = new Error('Request timeout - the server is taking too long to respond. Please try again.');
      timeoutError.code = 'TIMEOUT';
      timeoutError.originalError = error;
      throw timeoutError;
    }

    // Network error (no response)
    if (!error.response) {
      const networkError = new Error('Network error - unable to connect to the server. Please check if the backend is running.');
      networkError.code = 'NETWORK_ERROR';
      networkError.originalError = error;
      throw networkError;
    }

    // Server errors with response
    const status = error.response.status;
    const data = error.response.data;

    if (status === 400) {
      error.message = data?.detail || 'Invalid request - please check the image format';
    } else if (status === 413) {
      error.message = 'Image too large - please use a smaller image';
    } else if (status === 422) {
      error.message = data?.detail || 'Invalid data format';
    } else if (status === 500) {
      error.message = 'Server error - please try again later';
    } else if (status === 503) {
      error.message = 'Server unavailable - the service is temporarily down';
    }

    error.code = `HTTP_${status}`;
    throw error;
  }

  // ===================
  // API Endpoint Methods
  // ===================

  /**
   * Check backend health status
   * @returns {Promise<Object>} Health status object
   */
  async checkHealth() {
    const response = await this.client.get(API_CONFIG.ENDPOINTS.HEALTH);
    return response.data;
  }

  /**
   * Analyze a single image using multipart file upload
   * @param {File|Blob} file - The image file to analyze
   * @param {string} filename - Original filename (used if file is a Blob)
   * @returns {Promise<Object>} Analysis result
   */
  async analyzeSingle(file, filename = null) {
    return this.retryRequest(async () => {
      const formData = new FormData();
      
      // Handle both File objects and Blobs
      if (file instanceof File) {
        formData.append('file', file);
      } else if (file instanceof Blob) {
        // If it's a Blob, we need to give it a filename
        formData.append('file', file, filename || 'image.jpg');
      } else if (typeof file === 'string' && file.startsWith('data:')) {
        // Handle base64 data URLs - convert to Blob
        const blob = await this.dataURLtoBlob(file);
        formData.append('file', blob, filename || 'image.jpg');
      } else {
        throw new Error('Invalid file format. Expected File, Blob, or base64 data URL.');
      }
      
      const response = await this.client.post(API_CONFIG.ENDPOINTS.ANALYZE, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    });
  }

  /**
   * Analyze multiple images in a batch using multipart file upload (optimized for GPU)
   * @param {Array<{file: File|Blob, filename: string}>} files - Array of file objects
   * @returns {Promise<Object>} Batch analysis results
   */
  async analyzeBatch(files) {
    return this.retryRequest(async () => {
      const formData = new FormData();
      
      for (const item of files) {
        const file = item.file || item;
        const filename = item.filename || (file instanceof File ? file.name : 'image.jpg');
        
        if (file instanceof File || file instanceof Blob) {
          formData.append('files', file, filename);
        } else if (typeof file === 'string' && file.startsWith('data:')) {
          // Handle base64 data URLs - convert to Blob
          const blob = await this.dataURLtoBlob(file);
          formData.append('files', blob, filename);
        }
      }
      
      const response = await this.client.post(API_CONFIG.ENDPOINTS.ANALYZE_BATCH, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    });
  }

  /**
   * Convert a data URL to a Blob
   * @param {string} dataURL - Base64 data URL
   * @returns {Promise<Blob>} Blob object
   */
  async dataURLtoBlob(dataURL) {
    const response = await fetch(dataURL);
    return response.blob();
  }

  /**
   * Analyze a single image file (multipart upload) - alias for analyzeSingle
   * @param {File} file - The image file to analyze
   * @param {string} sessionId - Optional session ID for tracking (not used by backend)
   * @returns {Promise<Object>} Analysis result
   */
  async analyze(file, sessionId = null) {
    return this.analyzeSingle(file, file.name);
  }

  /**
   * Get system statistics
   * @returns {Promise<Object>} System stats including GPU info
   */
  async getStats() {
    const response = await this.client.get(API_CONFIG.ENDPOINTS.STATS);
    return response.data;
  }

  /**
   * Set custom timeout for a specific request type
   * Useful for batch processing which may take longer
   * @param {number} timeout - Timeout in milliseconds
   */
  setRequestTimeout(timeout) {
    this.client.defaults.timeout = timeout;
  }

  /**
   * Reset timeout to default value
   */
  resetRequestTimeout() {
    this.client.defaults.timeout = API_CONFIG.TIMEOUT;
  }

  /**
   * Submit user feedback to create a GitHub issue
   * @param {string} feedback - The feedback text
   * @param {Object} systemStatus - Optional system status info
   * @returns {Promise<Object>} Response with issue URL and number
   */
  async submitFeedback(feedback, systemStatus = null) {
    return this.retryRequest(async () => {
      const response = await this.client.post(API_CONFIG.ENDPOINTS.FEEDBACK, {
        feedback,
        system_status: systemStatus
      });
      return response.data;
    }, 1); // Only 1 retry for feedback submission
  }
}

// Export singleton instance
export const apiClient = new APIClient();

// Also export class for testing
export { APIClient };
