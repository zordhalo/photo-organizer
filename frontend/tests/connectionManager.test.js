/**
 * Tests for Connection Manager
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ConnectionManager } from '../src/services/connectionManager';

// Mock the apiClient
vi.mock('../src/services/apiClient', () => ({
  apiClient: {
    checkHealth: vi.fn()
  }
}));

import { apiClient } from '../src/services/apiClient';

describe('ConnectionManager', () => {
  let connectionManager;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    connectionManager = new ConnectionManager();
  });

  afterEach(() => {
    connectionManager.stopMonitoring();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  describe('checkConnection', () => {
    it('should return true when health check succeeds', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      const result = await connectionManager.checkConnection();

      expect(result).toBe(true);
      expect(connectionManager.isConnected).toBe(true);
      expect(connectionManager.backendInfo).toEqual({ status: 'healthy' });
    });

    it('should return false when health check fails', async () => {
      apiClient.checkHealth.mockRejectedValue(new Error('Connection refused'));

      const result = await connectionManager.checkConnection();

      expect(result).toBe(false);
      expect(connectionManager.isConnected).toBe(false);
      expect(connectionManager.lastError).toBe('Connection refused');
    });

    it('should update lastCheck timestamp', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      await connectionManager.checkConnection();

      expect(connectionManager.lastCheck).toBeDefined();
      expect(typeof connectionManager.lastCheck).toBe('number');
    });
  });

  describe('setConnected', () => {
    it('should notify listeners when status changes', () => {
      const listener = vi.fn();
      connectionManager.listeners.push(listener);

      connectionManager.setConnected(true);

      expect(listener).toHaveBeenCalledWith(true, null);
    });

    it('should not notify when status unchanged', () => {
      const listener = vi.fn();
      connectionManager.isConnected = true;
      connectionManager.listeners.push(listener);

      connectionManager.setConnected(true);

      expect(listener).not.toHaveBeenCalled();
    });
  });

  describe('startMonitoring', () => {
    it('should check connection immediately', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      connectionManager.startMonitoring();
      await vi.runOnlyPendingTimersAsync();

      expect(apiClient.checkHealth).toHaveBeenCalled();
    });

    it('should check periodically', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      connectionManager.startMonitoring();
      await vi.runOnlyPendingTimersAsync();

      // Get initial call count after startup
      const initialCalls = apiClient.checkHealth.mock.calls.length;

      // Advance time by check interval
      await vi.advanceTimersByTimeAsync(5000);
      expect(apiClient.checkHealth.mock.calls.length).toBe(initialCalls + 1);

      // Another interval
      await vi.advanceTimersByTimeAsync(5000);
      expect(apiClient.checkHealth.mock.calls.length).toBe(initialCalls + 2);
    });

    it('should warn if already monitoring', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      connectionManager.startMonitoring();
      connectionManager.startMonitoring();

      expect(warnSpy).toHaveBeenCalledWith('Connection monitoring already started');
    });
  });

  describe('stopMonitoring', () => {
    it('should stop periodic checks', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      connectionManager.startMonitoring();
      await vi.runOnlyPendingTimersAsync();

      connectionManager.stopMonitoring();

      const callCount = apiClient.checkHealth.mock.calls.length;
      await vi.advanceTimersByTimeAsync(10000);

      expect(apiClient.checkHealth.mock.calls.length).toBe(callCount);
    });

    it('should set intervalId to null', () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      connectionManager.startMonitoring();
      connectionManager.stopMonitoring();

      expect(connectionManager.intervalId).toBeNull();
    });
  });

  describe('onConnectionChange', () => {
    it('should register listener and call immediately', () => {
      const callback = vi.fn();

      connectionManager.onConnectionChange(callback);

      expect(callback).toHaveBeenCalledWith(false, null);
      expect(connectionManager.listeners).toContain(callback);
    });

    it('should return unsubscribe function', () => {
      const callback = vi.fn();

      const unsubscribe = connectionManager.onConnectionChange(callback);
      expect(connectionManager.listeners).toContain(callback);

      unsubscribe();
      expect(connectionManager.listeners).not.toContain(callback);
    });
  });

  describe('getState', () => {
    it('should return current state', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy', cuda_available: true });
      await connectionManager.checkConnection();

      const state = connectionManager.getState();

      expect(state).toEqual({
        isConnected: true,
        lastCheck: expect.any(Number),
        lastError: null,
        backendInfo: { status: 'healthy', cuda_available: true },
        isMonitoring: false
      });
    });

    it('should indicate monitoring status', () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      connectionManager.startMonitoring();
      const state = connectionManager.getState();

      expect(state.isMonitoring).toBe(true);
    });
  });

  describe('setCheckInterval', () => {
    it('should update check interval', () => {
      connectionManager.setCheckInterval(10000);

      expect(connectionManager.checkInterval).toBe(10000);
    });

    it('should restart monitoring with new interval', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });

      connectionManager.startMonitoring();
      await vi.runOnlyPendingTimersAsync();

      connectionManager.setCheckInterval(10000);

      // Old interval should not trigger
      await vi.advanceTimersByTimeAsync(5000);
      const callCountAt5s = apiClient.checkHealth.mock.calls.length;

      // New interval should trigger
      await vi.advanceTimersByTimeAsync(5000);
      expect(apiClient.checkHealth.mock.calls.length).toBeGreaterThan(callCountAt5s);
    });
  });

  describe('waitForConnection', () => {
    it('should return true when connected', async () => {
      apiClient.checkHealth.mockResolvedValue({ status: 'healthy' });
      vi.useRealTimers();

      const result = await connectionManager.waitForConnection(5000);

      expect(result).toBe(true);
    });

    it('should return true after retrying', async () => {
      vi.useRealTimers();
      
      // Fail first 2 times, then succeed
      apiClient.checkHealth
        .mockRejectedValueOnce(new Error('Fail'))
        .mockRejectedValueOnce(new Error('Fail'))
        .mockResolvedValue({ status: 'healthy' });

      const result = await connectionManager.waitForConnection(5000);

      expect(result).toBe(true);
      expect(apiClient.checkHealth.mock.calls.length).toBeGreaterThanOrEqual(3);
    });

    it('should return false on timeout', async () => {
      vi.useRealTimers();
      apiClient.checkHealth.mockRejectedValue(new Error('Always fail'));

      const result = await connectionManager.waitForConnection(2000);

      expect(result).toBe(false);
    });
  });

  describe('notifyListeners', () => {
    it('should catch errors in listeners', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const badListener = vi.fn().mockImplementation(() => {
        throw new Error('Listener error');
      });
      const goodListener = vi.fn();

      connectionManager.listeners.push(badListener);
      connectionManager.listeners.push(goodListener);

      // Should not throw
      connectionManager.notifyListeners(true);

      expect(errorSpy).toHaveBeenCalledWith('Error in connection listener:', expect.any(Error));
      expect(goodListener).toHaveBeenCalled();
    });
  });
});
