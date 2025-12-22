/**
 * Configuration Module Entry Point
 * Centralizes all configuration exports
 */

export { API_CONFIG, getEndpointUrl, checkApiHealth } from './api.config.js';
export { APP_CONFIG, validateFile } from './app.config.js';

// Log configuration in development mode
if (import.meta.env.DEV) {
  console.log('🔧 Configuration loaded');
  console.log('📡 API URL:', import.meta.env.VITE_API_URL || 'http://localhost:8000');
}
