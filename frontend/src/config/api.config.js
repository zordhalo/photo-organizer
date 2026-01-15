/**
 * API Configuration for Construction Photo Analyzer
 * Manages backend connection settings and endpoint definitions
 */

export const API_CONFIG = {
  // Base URL for the backend API
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  
  // API Endpoints (backend uses /api/v1 prefix for most endpoints)
  ENDPOINTS: {
    ANALYZE: '/api/v1/analyze',
    ANALYZE_BATCH: '/api/v1/batch-analyze',  // Note: backend uses batch-analyze not analyze-batch
    HEALTH: '/health',  // Health endpoint is at root level
    STATS: '/api/v1/stats',
    FEEDBACK: '/api/v1/feedback'  // Submit feedback -> GitHub issue
  },
  
  // Request Configuration
  TIMEOUT: 30000,       // 30 seconds timeout
  RETRY_ATTEMPTS: 3,    // Number of retry attempts on failure
  BATCH_SIZE: 16        // Optimal batch size for GPU processing
};

/**
 * Get the full URL for an endpoint
 * @param {string} endpoint - The endpoint key from ENDPOINTS
 * @returns {string} Full URL
 */
export function getEndpointUrl(endpoint) {
  const path = API_CONFIG.ENDPOINTS[endpoint];
  if (!path) {
    throw new Error(`Unknown endpoint: ${endpoint}`);
  }
  return `${API_CONFIG.BASE_URL}${path}`;
}

/**
 * Check if the API is available
 * @returns {Promise<boolean>}
 */
export async function checkApiHealth() {
  try {
    const response = await fetch(getEndpointUrl('HEALTH'), {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    return response.ok;
  } catch {
    return false;
  }
}

export default API_CONFIG;
