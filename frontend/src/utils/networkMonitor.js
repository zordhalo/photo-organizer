import { notifications } from '../utils/notifications';
import { connectionManager } from '../services/connectionManager';
import { fallbackService } from '../services/fallbackService';

class NetworkMonitor {
  constructor() {
    this.isOnline = navigator.onLine;
    this.setupListeners();
  }
  
  setupListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true;
      notifications.success('Internet connection restored');
      connectionManager.checkConnection();
    });
    
    window.addEventListener('offline', () => {
      this.isOnline = false;
      notifications.error('Internet connection lost');
      fallbackService.enable();
    });
  }
  
  getStatus() {
    return this.isOnline;
  }
}

export const networkMonitor = new NetworkMonitor();
