import { stateManager } from '../services/stateManager';
import { notifications } from '../utils/notifications';

class SessionRecovery {
  constructor() {
    this.element = this.createRecoveryBanner();
  }
  
  createRecoveryBanner() {
    const banner = document.createElement('div');
    banner.className = 'session-recovery-banner';
    banner.innerHTML = `
      <div class="recovery-content">
        <span class="recovery-icon">💾</span>
        <div class="recovery-text">
          <strong>Previous session found!</strong>
          <span class="recovery-details"></span>
        </div>
        <div class="recovery-actions">
          <button class="btn-recover">Recover</button>
          <button class="btn-dismiss">Start Fresh</button>
        </div>
      </div>
    `;
    return banner;
  }
  
  show(sessionData) {
    const details = this.element.querySelector('.recovery-details');
    details.textContent = `${sessionData.photos.length} photos from previous session`;
    
    const recoverBtn = this.element.querySelector('.btn-recover');
    const dismissBtn = this.element.querySelector('.btn-dismiss');
    
    recoverBtn.onclick = () => {
      this.recover();
      this.hide();
    };
    
    dismissBtn.onclick = () => {
      this.dismiss();
      this.hide();
    };
    
    document.body.prepend(this.element);
  }
  
  recover() {
    notifications.success('Session recovered!');
    console.log('User chose to recover session');
  }
  
  dismiss() {
    stateManager.clearState();
    console.log('User chose to start fresh');
  }
  
  hide() {
    this.element.remove();
  }
}

export default SessionRecovery;
