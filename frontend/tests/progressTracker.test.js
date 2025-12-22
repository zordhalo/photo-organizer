/**
 * Tests for Progress Tracker Component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import ProgressTracker from '../src/components/ProgressTracker';

describe('ProgressTracker', () => {
  let progressTracker;

  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
    vi.useFakeTimers();
    progressTracker = new ProgressTracker();
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  describe('createProgressBar', () => {
    it('should create progress bar DOM structure', () => {
      const element = progressTracker.getElement();

      expect(element.classList.contains('progress-container')).toBe(true);
      expect(element.querySelector('.progress-header')).toBeTruthy();
      expect(element.querySelector('.progress-label')).toBeTruthy();
      expect(element.querySelector('.progress-percentage')).toBeTruthy();
      expect(element.querySelector('.progress-bar-wrapper')).toBeTruthy();
      expect(element.querySelector('.progress-bar-fill')).toBeTruthy();
      expect(element.querySelector('.progress-count')).toBeTruthy();
      expect(element.querySelector('.progress-cancel-btn')).toBeTruthy();
    });
  });

  describe('update', () => {
    it('should update progress bar width', () => {
      progressTracker.update({ current: 50, total: 100, percentage: 50 });

      const fill = progressTracker.getElement().querySelector('.progress-bar-fill');
      expect(fill.style.width).toBe('50%');
    });

    it('should update percentage text', () => {
      progressTracker.update({ current: 75, total: 100, percentage: 75 });

      const percentText = progressTracker.getElement().querySelector('.progress-percentage');
      expect(percentText.textContent).toBe('75%');
    });

    it('should update count text', () => {
      progressTracker.update({ current: 10, total: 50, percentage: 20 });

      const countText = progressTracker.getElement().querySelector('.progress-count');
      expect(countText.textContent).toBe('10 / 50');
    });

    it('should update speed display', () => {
      progressTracker.update({ current: 10, total: 50, percentage: 20, speed: '2.5' });

      const speedText = progressTracker.getElement().querySelector('.progress-speed');
      expect(speedText.textContent).toBe('2.5 photos/sec');
    });

    it('should update ETA display', () => {
      progressTracker.update({ current: 10, total: 50, percentage: 20, eta: 45 });

      const etaText = progressTracker.getElement().querySelector('.progress-eta');
      expect(etaText.textContent).toBe('ETA: 45s');
    });

    it('should show complete state at 100%', () => {
      progressTracker.update({ current: 100, total: 100, percentage: 100 });

      const label = progressTracker.getElement().querySelector('.progress-label');
      expect(label.textContent).toBe('✅ Analysis Complete!');
    });
  });

  describe('formatTime', () => {
    it('should format seconds correctly', () => {
      expect(progressTracker.formatTime(30)).toBe('30s');
    });

    it('should format minutes correctly', () => {
      expect(progressTracker.formatTime(90)).toBe('1m 30s');
    });

    it('should format hours correctly', () => {
      expect(progressTracker.formatTime(3720)).toBe('1h 2m');
    });
  });

  describe('show and hide', () => {
    it('should show the progress tracker', () => {
      progressTracker.mount(document.getElementById('container'));
      progressTracker.show();

      expect(progressTracker.getElement().style.display).toBe('block');
      expect(progressTracker.getElement().classList.contains('visible')).toBe(true);
    });

    it('should hide the progress tracker', () => {
      progressTracker.mount(document.getElementById('container'));
      progressTracker.show();
      progressTracker.hide();

      vi.advanceTimersByTime(400);

      expect(progressTracker.getElement().style.display).toBe('none');
    });
  });

  describe('reset', () => {
    it('should reset all values to initial state', () => {
      progressTracker.update({ current: 50, total: 100, percentage: 50, speed: '2.0', eta: 30 });
      progressTracker.reset();

      const element = progressTracker.getElement();
      expect(element.querySelector('.progress-bar-fill').style.width).toBe('0%');
      expect(element.querySelector('.progress-percentage').textContent).toBe('0%');
      expect(element.querySelector('.progress-count').textContent).toBe('0 / 0');
      expect(element.querySelector('.progress-label').textContent).toBe('Analyzing photos...');
    });
  });

  describe('showError', () => {
    it('should display error state', () => {
      progressTracker.showError('Something went wrong');

      const label = progressTracker.getElement().querySelector('.progress-label');
      const fill = progressTracker.getElement().querySelector('.progress-bar-fill');

      expect(label.textContent).toBe('❌ Something went wrong');
      expect(fill.classList.contains('error')).toBe(true);
    });
  });

  describe('onCancel', () => {
    it('should call callback when cancel button clicked', () => {
      const callback = vi.fn();
      progressTracker.onCancel(callback);

      const cancelBtn = progressTracker.getElement().querySelector('.progress-cancel-btn');
      cancelBtn.click();

      expect(callback).toHaveBeenCalled();
    });
  });

  describe('mount and unmount', () => {
    it('should mount to parent element', () => {
      const container = document.getElementById('container');
      progressTracker.mount(container);

      expect(container.contains(progressTracker.getElement())).toBe(true);
    });

    it('should be hidden by default after mount', () => {
      const container = document.getElementById('container');
      progressTracker.mount(container);

      // After hide() is called, the element is hidden after the timeout
      vi.advanceTimersByTime(400);
      expect(progressTracker.getElement().style.display).toBe('none');
    });

    it('should unmount and cleanup', () => {
      const container = document.getElementById('container');
      progressTracker.mount(container);
      progressTracker.unmount();

      expect(container.contains(progressTracker.getElement())).toBe(false);
    });
  });
});
