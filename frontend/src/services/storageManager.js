import { notifications } from '../utils/notifications';

class StorageManager {
  constructor() {
    this.storageKey = 'photo_organizer_state';
    this.autoSaveInterval = 5000; // 5 seconds
    this.autoSaveTimer = null;
  }
  
  save(data) {
    try {
      const serialized = JSON.stringify(data);
      localStorage.setItem(this.storageKey, serialized);
      console.log('State saved to localStorage');
      return true;
    } catch (error) {
      console.error('Failed to save state:', error);
      if (error.name === 'QuotaExceededError') {
        notifications.error('Storage quota exceeded. Consider exporting data.');
      }
      return false;
    }
  }
  
  load() {
    try {
      const serialized = localStorage.getItem(this.storageKey);
      if (serialized === null) {
        return null;
      }
      return JSON.parse(serialized);
    } catch (error) {
      console.error('Failed to load state:', error);
      return null;
    }
  }
  
  clear() {
    localStorage.removeItem(this.storageKey);
    console.log('State cleared from localStorage');
  }
  
  getSize() {
    const data = localStorage.getItem(this.storageKey);
    if (!data) return 0;
    // Size in bytes
    return new Blob([data]).size;
  }
  
  getSizeFormatted() {
    const bytes = this.getSize();
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }
}

export const storageManager = new StorageManager();
