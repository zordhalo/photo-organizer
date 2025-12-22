/**
 * Tests for Notification System
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NotificationSystem } from '../src/utils/notifications';

describe('NotificationSystem', () => {
  let notificationSystem;

  beforeEach(() => {
    // Clear any existing notification containers
    document.body.innerHTML = '';
    vi.useFakeTimers();
    notificationSystem = new NotificationSystem();
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  describe('constructor', () => {
    it('should create a notification container', () => {
      expect(notificationSystem.container).toBeDefined();
      expect(document.getElementById('notification-container')).toBeTruthy();
    });
  });

  describe('show', () => {
    it('should create a notification element', () => {
      notificationSystem.show('Test message', 'info');

      const notifications = document.querySelectorAll('.notification');
      expect(notifications.length).toBe(1);
    });

    it('should apply correct type class', () => {
      notificationSystem.show('Test message', 'success');

      const notification = document.querySelector('.notification');
      expect(notification.classList.contains('notification-success')).toBe(true);
    });

    it('should display the message', () => {
      notificationSystem.show('Hello World', 'info');

      const message = document.querySelector('.notification-message');
      expect(message.textContent).toBe('Hello World');
    });

    it('should auto-dismiss after duration', () => {
      notificationSystem.show('Test message', 'info', 3000);

      expect(document.querySelectorAll('.notification').length).toBe(1);

      // Advance timers
      vi.advanceTimersByTime(3300);

      expect(document.querySelectorAll('.notification').length).toBe(0);
    });

    it('should not auto-dismiss if duration is 0', () => {
      notificationSystem.show('Persistent message', 'info', 0);

      vi.advanceTimersByTime(10000);

      expect(document.querySelectorAll('.notification').length).toBe(1);
    });

    it('should limit max notifications', () => {
      for (let i = 0; i < 7; i++) {
        notificationSystem.show(`Message ${i}`, 'info', 0);
      }

      // Wait for removal animations to complete
      vi.advanceTimersByTime(400);

      expect(document.querySelectorAll('.notification').length).toBeLessThanOrEqual(5);
    });
  });

  describe('getIcon', () => {
    it('should return correct icon for each type', () => {
      expect(notificationSystem.getIcon('success')).toBe('✓');
      expect(notificationSystem.getIcon('error')).toBe('✕');
      expect(notificationSystem.getIcon('warning')).toBe('⚠');
      expect(notificationSystem.getIcon('info')).toBe('ℹ');
    });

    it('should return info icon for unknown types', () => {
      expect(notificationSystem.getIcon('unknown')).toBe('ℹ');
    });
  });

  describe('escapeHtml', () => {
    it('should escape HTML entities', () => {
      const escaped = notificationSystem.escapeHtml('<script>alert("xss")</script>');
      expect(escaped).not.toContain('<script>');
      expect(escaped).toContain('&lt;');
    });
  });

  describe('success', () => {
    it('should show success notification', () => {
      notificationSystem.success('Success!');

      const notification = document.querySelector('.notification-success');
      expect(notification).toBeTruthy();
    });
  });

  describe('error', () => {
    it('should show error notification with longer duration', () => {
      notificationSystem.error('Error!');

      const notification = document.querySelector('.notification-error');
      expect(notification).toBeTruthy();
    });
  });

  describe('warning', () => {
    it('should show warning notification', () => {
      notificationSystem.warning('Warning!');

      const notification = document.querySelector('.notification-warning');
      expect(notification).toBeTruthy();
    });
  });

  describe('info', () => {
    it('should show info notification', () => {
      notificationSystem.info('Info!');

      const notification = document.querySelector('.notification-info');
      expect(notification).toBeTruthy();
    });
  });

  describe('removeNotification', () => {
    it('should remove notification from DOM', () => {
      notificationSystem.show('Test', 'info', 0);
      const notification = document.querySelector('.notification');

      notificationSystem.removeNotification(notification);

      vi.advanceTimersByTime(400);

      expect(document.querySelectorAll('.notification').length).toBe(0);
    });

    it('should handle null notification gracefully', () => {
      expect(() => {
        notificationSystem.removeNotification(null);
      }).not.toThrow();
    });
  });

  describe('clearAll', () => {
    it('should remove all notifications', () => {
      for (let i = 0; i < 3; i++) {
        notificationSystem.show(`Message ${i}`, 'info', 0);
      }

      expect(document.querySelectorAll('.notification').length).toBe(3);

      notificationSystem.clearAll();

      vi.advanceTimersByTime(400);

      expect(document.querySelectorAll('.notification').length).toBe(0);
    });
  });

  describe('close button', () => {
    it('should remove notification when close button clicked', () => {
      notificationSystem.show('Test', 'info', 0);
      const closeBtn = document.querySelector('.notification-close');

      closeBtn.click();

      vi.advanceTimersByTime(400);

      expect(document.querySelectorAll('.notification').length).toBe(0);
    });
  });
});
