/**
 * Progress Tracker Component
 * Displays analysis progress with smooth animations and ETA
 */

class ProgressTracker {
  constructor() {
    this.element = this.createProgressBar();
    this.autoHideTimeout = null;
  }

  /**
   * Create the progress bar DOM structure
   * @returns {HTMLElement} The progress container element
   */
  createProgressBar() {
    const container = document.createElement('div');
    container.className = 'progress-container';
    container.innerHTML = `
      <div class="progress-header">
        <span class="progress-label">Analyzing photos...</span>
        <span class="progress-percentage">0%</span>
      </div>
      <div class="progress-bar-wrapper">
        <div class="progress-bar-fill"></div>
      </div>
      <div class="progress-details">
        <span class="progress-count">0 / 0</span>
        <span class="progress-speed"></span>
        <span class="progress-eta"></span>
      </div>
      <button class="progress-cancel-btn" title="Cancel analysis">✕</button>
    `;
    return container;
  }

  /**
   * Update the progress display
   * @param {Object} progress - Progress data
   * @param {number} progress.current - Current processed count
   * @param {number} progress.total - Total items
   * @param {number} progress.percentage - Percentage complete
   * @param {string} progress.speed - Processing speed (items/sec)
   * @param {number} progress.eta - Estimated time remaining in seconds
   */
  update({ current, total, percentage, speed, eta }) {
    const fill = this.element.querySelector('.progress-bar-fill');
    const percentText = this.element.querySelector('.progress-percentage');
    const countText = this.element.querySelector('.progress-count');
    const speedText = this.element.querySelector('.progress-speed');
    const etaText = this.element.querySelector('.progress-eta');

    // Update progress bar width with smooth transition
    fill.style.width = `${percentage}%`;
    percentText.textContent = `${percentage}%`;
    countText.textContent = `${current} / ${total}`;

    // Update speed and ETA
    if (speed !== undefined && parseFloat(speed) > 0) {
      speedText.textContent = `${speed} photos/sec`;
    }

    if (eta !== undefined && eta > 0) {
      etaText.textContent = `ETA: ${this.formatTime(eta)}`;
    } else if (percentage === 100) {
      etaText.textContent = '';
    }

    // Handle completion
    if (percentage === 100) {
      this.showComplete();
    }
  }

  /**
   * Format seconds to human readable time
   * @param {number} seconds - Time in seconds
   * @returns {string} Formatted time string
   */
  formatTime(seconds) {
    if (seconds < 60) {
      return `${seconds}s`;
    } else if (seconds < 3600) {
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      return `${mins}m ${secs}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const mins = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${mins}m`;
    }
  }

  /**
   * Show completion state
   */
  showComplete() {
    const label = this.element.querySelector('.progress-label');
    const fill = this.element.querySelector('.progress-bar-fill');
    const cancelBtn = this.element.querySelector('.progress-cancel-btn');

    label.textContent = '✅ Analysis Complete!';
    fill.classList.add('complete');
    cancelBtn.style.display = 'none';

    // Clear any existing timeout
    if (this.autoHideTimeout) {
      clearTimeout(this.autoHideTimeout);
    }

    // Auto-hide after 3 seconds
    this.autoHideTimeout = setTimeout(() => {
      this.hide();
    }, 3000);
  }

  /**
   * Show error state
   * @param {string} message - Error message to display
   */
  showError(message) {
    const label = this.element.querySelector('.progress-label');
    const fill = this.element.querySelector('.progress-bar-fill');

    label.textContent = `❌ ${message || 'Analysis failed'}`;
    fill.classList.add('error');
  }

  /**
   * Reset the progress tracker to initial state
   */
  reset() {
    const fill = this.element.querySelector('.progress-bar-fill');
    const label = this.element.querySelector('.progress-label');
    const percentText = this.element.querySelector('.progress-percentage');
    const countText = this.element.querySelector('.progress-count');
    const speedText = this.element.querySelector('.progress-speed');
    const etaText = this.element.querySelector('.progress-eta');
    const cancelBtn = this.element.querySelector('.progress-cancel-btn');

    fill.style.width = '0%';
    fill.classList.remove('complete', 'error');
    label.textContent = 'Analyzing photos...';
    percentText.textContent = '0%';
    countText.textContent = '0 / 0';
    speedText.textContent = '';
    etaText.textContent = '';
    cancelBtn.style.display = 'block';

    if (this.autoHideTimeout) {
      clearTimeout(this.autoHideTimeout);
      this.autoHideTimeout = null;
    }
  }

  /**
   * Show the progress tracker
   */
  show() {
    this.reset();
    this.element.style.display = 'block';
    this.element.classList.add('visible');
  }

  /**
   * Hide the progress tracker
   */
  hide() {
    this.element.classList.remove('visible');
    this.element.classList.add('hiding');

    setTimeout(() => {
      this.element.style.display = 'none';
      this.element.classList.remove('hiding');
    }, 300);
  }

  /**
   * Set the cancel callback
   * @param {Function} callback - Function to call when cancel is clicked
   */
  onCancel(callback) {
    const cancelBtn = this.element.querySelector('.progress-cancel-btn');
    cancelBtn.addEventListener('click', () => {
      callback();
      this.showError('Analysis cancelled');
    });
  }

  /**
   * Mount the component to a parent element
   * @param {HTMLElement} parentElement - The element to mount to
   */
  mount(parentElement) {
    parentElement.appendChild(this.element);
    this.hide(); // Hidden by default
  }

  /**
   * Unmount and cleanup the component
   */
  unmount() {
    if (this.autoHideTimeout) {
      clearTimeout(this.autoHideTimeout);
    }
    if (this.element.parentElement) {
      this.element.parentElement.removeChild(this.element);
    }
  }

  /**
   * Get the underlying DOM element
   * @returns {HTMLElement}
   */
  getElement() {
    return this.element;
  }
}

export default ProgressTracker;
