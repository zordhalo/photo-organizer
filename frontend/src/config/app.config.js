/**
 * Application Configuration
 * General application settings and feature flags
 */

export const APP_CONFIG = {
  // Application Info
  APP_NAME: 'Construction Photo Analyzer',
  VERSION: '1.0.0',
  
  // File Upload Settings
  MAX_FILE_SIZE: parseInt(import.meta.env.VITE_MAX_FILE_SIZE) || 10485760, // 10MB default
  ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'],
  ALLOWED_MIME_TYPES: [
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/gif',
    'image/bmp'
  ],
  
  // Feature Flags
  ENABLE_FALLBACK: import.meta.env.VITE_ENABLE_FALLBACK === 'true',
  ENABLE_BATCH_UPLOAD: true,
  ENABLE_DRAG_DROP: true,
  
  // UI Settings
  TOAST_DURATION: 3000,
  LOADING_DELAY: 200,   // Delay before showing loading indicator
  
  // Analysis Settings
  MIN_CONFIDENCE_THRESHOLD: 0.1,
  MAX_BATCH_FILES: 16
};

/**
 * Validate a file against the configuration
 * @param {File} file - The file to validate
 * @returns {{valid: boolean, error?: string}}
 */
export function validateFile(file) {
  if (file.size > APP_CONFIG.MAX_FILE_SIZE) {
    const maxSizeMB = Math.round(APP_CONFIG.MAX_FILE_SIZE / 1024 / 1024);
    return { valid: false, error: `File size exceeds ${maxSizeMB}MB limit` };
  }
  
  if (!APP_CONFIG.ALLOWED_MIME_TYPES.includes(file.type)) {
    return { valid: false, error: 'File type not supported' };
  }
  
  return { valid: true };
}

export default APP_CONFIG;
