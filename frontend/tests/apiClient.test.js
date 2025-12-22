/**
 * Tests for API Client
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { APIClient } from '../src/services/apiClient';

// Mock axios
vi.mock('axios', () => {
  const mockAxiosInstance = {
    get: vi.fn(),
    post: vi.fn(),
    defaults: { timeout: 30000 },
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() }
    }
  };
  
  return {
    default: {
      create: vi.fn(() => mockAxiosInstance)
    }
  };
});

import axios from 'axios';

describe('APIClient', () => {
  let apiClient;
  let mockClient;

  beforeEach(() => {
    vi.clearAllMocks();
    apiClient = new APIClient();
    mockClient = axios.create();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should create axios instance with correct config', () => {
      expect(axios.create).toHaveBeenCalledWith({
        baseURL: expect.any(String),
        timeout: expect.any(Number),
        headers: {
          'Content-Type': 'application/json'
        }
      });
    });

    it('should set up request interceptor', () => {
      expect(mockClient.interceptors.request.use).toHaveBeenCalled();
    });

    it('should set up response interceptor', () => {
      expect(mockClient.interceptors.response.use).toHaveBeenCalled();
    });
  });

  describe('checkHealth', () => {
    it('should call health endpoint', async () => {
      const healthData = { status: 'healthy', cuda_available: true };
      mockClient.get.mockResolvedValue({ data: healthData });

      const result = await apiClient.checkHealth();

      expect(mockClient.get).toHaveBeenCalledWith('/health');
      expect(result).toEqual(healthData);
    });

    it('should throw on error', async () => {
      mockClient.get.mockRejectedValue(new Error('Network error'));

      await expect(apiClient.checkHealth()).rejects.toThrow();
    });
  });

  describe('analyzeSingle', () => {
    it('should send image for analysis', async () => {
      const analysisResult = {
        category: 'Interior',
        confidence: 0.95,
        method: 'resnet50',
        processing_time: 0.024
      };
      mockClient.post.mockResolvedValue({ data: analysisResult });

      const result = await apiClient.analyzeSingle('base64data', 'test.jpg');

      expect(mockClient.post).toHaveBeenCalledWith('/analyze', {
        image: 'base64data',
        filename: 'test.jpg'
      });
      expect(result).toEqual(analysisResult);
    });
  });

  describe('analyzeBatch', () => {
    it('should send batch of images for analysis', async () => {
      const batchResult = {
        results: [{ category: 'Interior', confidence: 0.95 }],
        total_time: 0.856,
        images_processed: 16
      };
      mockClient.post.mockResolvedValue({ data: batchResult });

      const images = [{ image: 'base64data', filename: 'test.jpg' }];
      const result = await apiClient.analyzeBatch(images);

      expect(mockClient.post).toHaveBeenCalledWith('/analyze-batch', {
        images: images,
        batch_size: expect.any(Number)
      });
      expect(result).toEqual(batchResult);
    });
  });

  describe('getStats', () => {
    it('should fetch system stats', async () => {
      const statsData = { gpu: 'available', memory: '8GB' };
      mockClient.get.mockResolvedValue({ data: statsData });

      const result = await apiClient.getStats();

      expect(mockClient.get).toHaveBeenCalledWith('/stats');
      expect(result).toEqual(statsData);
    });
  });

  describe('retryRequest', () => {
    it('should return result on first success', async () => {
      const requestFn = vi.fn().mockResolvedValue('success');

      const result = await apiClient.retryRequest(requestFn, 3);

      expect(result).toBe('success');
      expect(requestFn).toHaveBeenCalledTimes(1);
    });

    it('should retry on failure and succeed', async () => {
      const requestFn = vi.fn()
        .mockRejectedValueOnce(new Error('Fail 1'))
        .mockResolvedValueOnce('success');

      const result = await apiClient.retryRequest(requestFn, 3);

      expect(result).toBe('success');
      expect(requestFn).toHaveBeenCalledTimes(2);
    });

    it('should throw after all retries exhausted', async () => {
      const requestFn = vi.fn().mockRejectedValue(new Error('Always fail'));

      await expect(apiClient.retryRequest(requestFn, 2)).rejects.toThrow('Always fail');
      expect(requestFn).toHaveBeenCalledTimes(2);
    });

    it('should not retry on 4xx errors', async () => {
      const error = new Error('Bad request');
      error.response = { status: 400 };
      const requestFn = vi.fn().mockRejectedValue(error);

      await expect(apiClient.retryRequest(requestFn, 3)).rejects.toThrow();
      expect(requestFn).toHaveBeenCalledTimes(1);
    });
  });

  describe('handleError', () => {
    it('should handle timeout errors', () => {
      const error = { code: 'ECONNABORTED' };

      expect(() => apiClient.handleError(error)).toThrow('Request timeout');
    });

    it('should handle network errors', () => {
      const error = { response: undefined };

      expect(() => apiClient.handleError(error)).toThrow('Network error');
    });

    it('should handle 400 errors', () => {
      const error = {
        response: {
          status: 400,
          data: { detail: 'Bad request' }
        }
      };

      expect(() => apiClient.handleError(error)).toThrow();
    });

    it('should handle 500 errors', () => {
      const error = {
        response: {
          status: 500,
          data: {}
        }
      };

      expect(() => apiClient.handleError(error)).toThrow();
    });
  });

  describe('timeout configuration', () => {
    it('should set custom timeout', () => {
      apiClient.setRequestTimeout(60000);
      expect(mockClient.defaults.timeout).toBe(60000);
    });

    it('should reset timeout to default', () => {
      apiClient.setRequestTimeout(60000);
      apiClient.resetRequestTimeout();
      expect(mockClient.defaults.timeout).toBe(30000);
    });
  });
});
