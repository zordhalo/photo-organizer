import { notifications } from '../utils/notifications';

class ErrorHandler {
  constructor() {
    this.errorLog = [];
    this.maxLogSize = 100;
  }
  
  handle(error, context = '') {
    const errorInfo = {
      message: error.message || 'Unknown error',
      context: context,
      timestamp: new Date().toISOString(),
      stack: error.stack,
      type: this.categorizeError(error)
    };
    
    this.log(errorInfo);
    this.notify(errorInfo);
    
    return this.getRecoveryAction(errorInfo.type);
  }
  
  categorizeError(error) {
    if (error.message?.includes('Network')) return 'NETWORK_ERROR';
    if (error.message?.includes('timeout')) return 'TIMEOUT_ERROR';
    if (error.code === 'ECONNREFUSED') return 'BACKEND_OFFLINE';
    if (error.response?.status === 500) return 'SERVER_ERROR';
    if (error.response?.status === 429) return 'RATE_LIMIT';
    return 'UNKNOWN_ERROR';
  }
  
  getRecoveryAction(errorType) {
    const actions = {
      NETWORK_ERROR: 'retry',
      TIMEOUT_ERROR: 'retry',
      BACKEND_OFFLINE: 'fallback',
      SERVER_ERROR: 'retry',
      RATE_LIMIT: 'wait',
      UNKNOWN_ERROR: 'fallback'
    };
    return actions[errorType] || 'fallback';
  }
  
  log(errorInfo) {
    this.errorLog.push(errorInfo);
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog.shift();
    }
    console.error(`[${errorInfo.type}] ${errorInfo.context}:`, errorInfo.message);
  }
  
  notify(errorInfo) {
    const userMessages = {
      NETWORK_ERROR: 'Network connection lost. Retrying...',
      TIMEOUT_ERROR: 'Request timed out. Trying again...',
      BACKEND_OFFLINE: 'Backend unavailable. Using fallback mode.',
      SERVER_ERROR: 'Server error occurred. Retrying...',
      RATE_LIMIT: 'Too many requests. Please wait.',
      UNKNOWN_ERROR: 'An error occurred. Using fallback mode.'
    };
    
    const message = userMessages[errorInfo.type] || 'Something went wrong.';
    notifications.warning(message);
  }
  
  getErrorLog() {
    return this.errorLog;
  }
  
  clearLog() {
    this.errorLog = [];
  }
}

export const errorHandler = new ErrorHandler();
