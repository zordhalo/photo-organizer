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
   * Analyze a single image
   * @param {string} imageBase64 - Base64 encoded image data
   * @param {string} filename - Original filename of the image
   * @returns {Promise<Object>} Analysis result
   */
  async analyzeSingle(imageBase64, filename) {
    return this.retryRequest(async () => {
      const response = await this.client.post(API_CONFIG.ENDPOINTS.ANALYZE, {
        image: imageBase64,
        filename: filename
      });
      return response.data;
    });
  }

  /**
   * Analyze multiple images in a batch (optimized for GPU)
   * @param {Array<Object>} images - Array of {image: base64, filename: string}
   * @returns {Promise<Object>} Batch analysis results
   */
  async analyzeBatch(images) {
    return this.retryRequest(async () => {
      const response = await this.client.post(API_CONFIG.ENDPOINTS.ANALYZE_BATCH, {
        images: images,
        batch_size: API_CONFIG.BATCH_SIZE
      });
      return response.data;
    });
  }

  /**
   * Analyze a single image file (multipart upload)
   * @param {File} file - The image file to analyze
   * @param {string} sessionId - Optional session ID for tracking
   * @returns {Promise<Object>} Analysis result
   */
  async analyze(file, sessionId = null) {
    return this.retryRequest(async () => {
      const formData = new FormData();
      formData.append('file', file);
      if (sessionId) {
        formData.append('session_id', sessionId);
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
}

// Export singleton instance
export const apiClient = new APIClient();

// Also export class for testing
export { APIClient };
