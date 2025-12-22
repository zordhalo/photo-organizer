import { stateManager } from '../services/stateManager';

class AutoSaveIndicator {
  constructor() {
    this.element = this.createIndicator();
    this.setupMonitoring();
  }
  
  createIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'autosave-indicator';
    indicator.innerHTML = `
      <span class="save-icon">💾</span>
      <span class="save-text">Saved</span>
    `;
    return indicator;
  }
  
  setupMonitoring() {
    stateManager.subscribe((state) => {
      this.updateLastSaved(state.session.lastSaved);
    });
  }
  
  updateLastSaved(timestamp) {
    const text = this.element.querySelector('.save-text');
    if (!timestamp) {
      text.textContent = 'Not saved';
      return;
    }
    
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 10) {
      text.textContent = 'Saved just now';
    } else if (seconds < 60) {
      text.textContent = `Saved ${seconds}s ago`;
    } else {
      const minutes = Math.floor(seconds / 60);
      text.textContent = `Saved ${minutes}m ago`;
    }
  }
  
  mount(parentElement) {
    parentElement.appendChild(this.element);
  }
}

export default AutoSaveIndicator;
