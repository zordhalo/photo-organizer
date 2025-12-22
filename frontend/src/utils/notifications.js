/**
 * Notification System for Construction Photo Analyzer
 * Provides toast-style notifications with different severity levels
 */

class NotificationSystem {
  constructor() {
    this.container = null;
    this.notifications = [];
    this.maxNotifications = 5;
    this.ensureContainer();
  }

  /**
   * Ensure the notification container exists in the DOM
   */
  ensureContainer() {
    if (!this.container) {
      this.container = document.getElementById('notification-container');
      
      if (!this.container) {
        this.container = document.createElement('div');
        this.container.id = 'notification-container';
        this.container.className = 'notification-container';
        document.body.appendChild(this.container);
      }
    }
    return this.container;
  }

  /**
   * Show a notification
   * @param {string} message - The notification message
   * @param {string} type - Notification type: 'info', 'success', 'warning', 'error'
   * @param {number} duration - Duration in milliseconds before auto-dismiss
   * @returns {HTMLElement} The notification element
   */
  show(message, type = 'info', duration = 3000) {
    this.ensureContainer();

    // Limit the number of visible notifications
    while (this.notifications.length >= this.maxNotifications) {
      const oldest = this.notifications.shift();
      this.removeNotification(oldest);
    }

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Add icon based on type
    const icon = this.getIcon(type);
    
    notification.innerHTML = `
      <span class="notification-icon">${icon}</span>
      <span class="notification-message">${this.escapeHtml(message)}</span>
      <button class="notification-close" aria-label="Close notification">×</button>
    `;

    // Add close button handler
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
      this.removeNotification(notification);
    });

    // Add to container
    this.container.appendChild(notification);
    this.notifications.push(notification);

    // Trigger animation
    requestAnimationFrame(() => {
      notification.classList.add('notification-visible');
    });

    // Auto-dismiss after duration
    if (duration > 0) {
      setTimeout(() => {
        this.removeNotification(notification);
      }, duration);
    }

    return notification;
  }

  /**
   * Get icon for notification type
   * @param {string} type - Notification type
   * @returns {string} Icon character
   */
  getIcon(type) {
    const icons = {
      success: '✓',
      error: '✕',
      warning: '⚠',
      info: 'ℹ'
    };
    return icons[type] || icons.info;
  }

  /**
   * Escape HTML to prevent XSS
   * @param {string} str - String to escape
   * @returns {string} Escaped string
   */
  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Remove a notification with fade-out animation
   * @param {HTMLElement} notification - The notification element to remove
   */
  removeNotification(notification) {
    if (!notification || !notification.parentElement) return;

    notification.classList.add('notification-fade-out');
    notification.classList.remove('notification-visible');

    setTimeout(() => {
      if (notification.parentElement) {
        notification.parentElement.removeChild(notification);
      }
      const index = this.notifications.indexOf(notification);
      if (index > -1) {
        this.notifications.splice(index, 1);
      }
    }, 300);
  }

  /**
   * Show a success notification
   * @param {string} message - The notification message
   * @param {number} duration - Duration in milliseconds
   */
  success(message, duration = 3000) {
    return this.show(message, 'success', duration);
  }

  /**
   * Show an error notification
   * @param {string} message - The notification message
   * @param {number} duration - Duration in milliseconds
   */
  error(message, duration = 5000) {
    return this.show(message, 'error', duration);
  }

  /**
   * Show a warning notification
   * @param {string} message - The notification message
   * @param {number} duration - Duration in milliseconds
   */
  warning(message, duration = 4000) {
    return this.show(message, 'warning', duration);
  }

  /**
   * Show an info notification
   * @param {string} message - The notification message
   * @param {number} duration - Duration in milliseconds
   */
  info(message, duration = 3000) {
    return this.show(message, 'info', duration);
  }

  /**
   * Clear all notifications
   */
  clearAll() {
    [...this.notifications].forEach(notification => {
      this.removeNotification(notification);
    });
  }
}

// Export singleton instance
export const notifications = new NotificationSystem();

// Also export class for testing
export { NotificationSystem };
