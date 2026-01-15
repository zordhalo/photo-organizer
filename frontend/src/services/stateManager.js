import { storageManager } from './storageManager';
import { notifications } from '../utils/notifications';

class StateManager {
  constructor() {
    this.state = {
      photos: [],
      categories: {},
      lastAnalyzed: null,
      feedback: [],
      settings: {
        batchSize: 16,
        autoSave: true,
        fallbackEnabled: true
      },
      session: {
        startTime: Date.now(),
        photosAnalyzed: 0,
        lastSaved: null
      }
    };
    
    this.listeners = [];
    this.autoSaveEnabled = true;
  }
  
  initialize() {
    const savedState = storageManager.load();
    if (savedState) {
      this.state = { ...this.state, ...savedState };
      this.state.session.startTime = Date.now(); // Reset session time
      console.log('State recovered from localStorage');
      notifications.success('Previous session recovered!');
    }
    
    if (this.state.settings.autoSave) {
      this.startAutoSave();
    }
  }
  
  getState() {
    return this.state;
  }
  
  setState(updates) {
    this.state = { ...this.state, ...updates };
    this.notifyListeners();
    if (this.autoSaveEnabled) {
      this.saveState();
    }
  }
  
  addPhoto(photo) {
    this.state.photos.push(photo);
    this.state.session.photosAnalyzed++;
    this.notifyListeners();
  }

  addFeedback(item) {
    this.state.feedback.unshift(item);
    this.notifyListeners();
  }

  removeFeedback(id) {
    this.state.feedback = this.state.feedback.filter(item => item.id !== id);
    this.notifyListeners();
  }
  
  updatePhoto(id, updates) {
    const index = this.state.photos.findIndex(p => p.id === id);
    if (index !== -1) {
      this.state.photos[index] = { ...this.state.photos[index], ...updates };
      this.notifyListeners();
    }
  }
  
  getPhotosByCategory(category) {
    return this.state.photos.filter(p => p.category === category);
  }
  
  saveState() {
    this.state.session.lastSaved = Date.now();
    const success = storageManager.save(this.state);
    if (success) {
      console.log(`State saved (${storageManager.getSizeFormatted()})`);
    }
  }
  
  startAutoSave() {
    this.autoSaveTimer = setInterval(() => {
      this.saveState();
    }, 5000);
    console.log('Auto-save enabled (every 5 seconds)');
  }
  
  stopAutoSave() {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
      console.log('Auto-save disabled');
    }
  }
  
  clearState() {
    this.state.photos = [];
    this.state.categories = {};
    storageManager.clear();
    this.notifyListeners();
    notifications.success('All data cleared');
  }
  
  exportData() {
    const dataStr = JSON.stringify(this.state, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `photo-organizer-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    notifications.success('Data exported successfully');
  }
  
  subscribe(listener) {
    this.listeners.push(listener);
  }
  
  notifyListeners() {
    this.listeners.forEach(listener => listener(this.state));
  }
}

export const stateManager = new StateManager();
