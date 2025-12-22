/**
 * Tests for Connection Status Component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import ConnectionStatus from '../src/components/ConnectionStatus';

// Mock the connectionManager
vi.mock('../src/services/connectionManager', () => ({
  connectionManager: {
    onConnectionChange: vi.fn((callback) => {
      // Store callback for testing
      connectionManager._callback = callback;
      // Return unsubscribe function
      return vi.fn();
    }),
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn(),
    isConnected: false,
    _callback: null
  }
}));

// Mock the apiClient
vi.mock('../src/services/apiClient', () => ({
  apiClient: {
    getStats: vi.fn()
  }
}));

import { connectionManager } from '../src/services/connectionManager';
import { apiClient } from '../src/services/apiClient';

describe('ConnectionStatus', () => {
  let connectionStatus;

  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
    vi.clearAllMocks();
    connectionStatus = new ConnectionStatus();
  });

  afterEach(() => {
    connectionStatus.unmount();
    document.body.innerHTML = '';
  });

  describe('createStatusIndicator', () => {
    it('should create status indicator DOM structure', () => {
      const element = connectionStatus.getElement();

      expect(element.classList.contains('connection-status')).toBe(true);
      expect(element.querySelector('.status-dot')).toBeTruthy();
      expect(element.querySelector('.status-text')).toBeTruthy();
      expect(element.querySelector('.backend-info')).toBeTruthy();
    });

    it('should show checking state initially', () => {
      const text = connectionStatus.getElement().querySelector('.status-text');
      expect(text.textContent).toBe('Checking...');
    });
  });

  describe('updateStatus', () => {
    it('should show connected state when connected', () => {
      connectionStatus.updateStatus(true, { cuda_available: true });

      const dot = connectionStatus.getElement().querySelector('.status-dot');
      const text = connectionStatus.getElement().querySelector('.status-text');

      expect(dot.classList.contains('connected')).toBe(true);
      expect(text.textContent).toBe('✅ Backend Connected');
    });

    it('should show disconnected state when disconnected', () => {
      connectionStatus.updateStatus(false, null);

      const dot = connectionStatus.getElement().querySelector('.status-dot');
      const text = connectionStatus.getElement().querySelector('.status-text');

      expect(dot.classList.contains('disconnected')).toBe(true);
      expect(text.textContent).toBe('❌ Backend Offline (Fallback Mode)');
    });

    it('should display GPU status from backend info', () => {
      connectionStatus.updateStatus(true, { cuda_available: true, model_loaded: true });

      const info = connectionStatus.getElement().querySelector('.backend-info');
      expect(info.textContent).toBe('✅ GPU Available');
    });

    it('should display CPU only when no CUDA', () => {
      connectionStatus.updateStatus(true, { cuda_available: false, model_loaded: true });

      const info = connectionStatus.getElement().querySelector('.backend-info');
      expect(info.textContent).toBe('❌ CPU Only');
    });
  });

  describe('fetchBackendInfo', () => {
    it('should fetch and display stats', async () => {
      apiClient.getStats.mockResolvedValue({
        cuda_available: true,
        gpu_name: 'RTX 3080',
        images_analyzed: 100
      });

      await connectionStatus.fetchBackendInfo();

      const info = connectionStatus.getElement().querySelector('.backend-info');
      expect(info.textContent).toBe('✅ GPU: RTX 3080');
    });

    it('should handle fetch errors gracefully', async () => {
      apiClient.getStats.mockRejectedValue(new Error('Network error'));

      // Should not throw
      await expect(connectionStatus.fetchBackendInfo()).resolves.not.toThrow();
    });
  });

  describe('mount', () => {
    it('should mount to parent element', () => {
      const container = document.getElementById('container');
      connectionStatus.mount(container);

      expect(container.contains(connectionStatus.getElement())).toBe(true);
    });

    it('should start monitoring on mount', () => {
      const container = document.getElementById('container');
      connectionStatus.mount(container);

      expect(connectionManager.startMonitoring).toHaveBeenCalled();
    });
  });

  describe('unmount', () => {
    it('should unmount from parent element', () => {
      const container = document.getElementById('container');
      connectionStatus.mount(container);
      connectionStatus.unmount();

      expect(container.contains(connectionStatus.getElement())).toBe(false);
    });

    it('should stop monitoring on unmount', () => {
      const container = document.getElementById('container');
      connectionStatus.mount(container);
      connectionStatus.unmount();

      expect(connectionManager.stopMonitoring).toHaveBeenCalled();
    });
  });

  describe('setupListeners', () => {
    it('should register connection change listener', () => {
      expect(connectionManager.onConnectionChange).toHaveBeenCalled();
    });
  });
});
