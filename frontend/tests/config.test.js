/**
 * API Configuration Tests
 */
import { describe, it, expect, vi } from 'vitest';
import { API_CONFIG, getEndpointUrl } from '../src/config/api.config.js';

describe('API Configuration', () => {
  it('should have a valid BASE_URL', () => {
    expect(API_CONFIG.BASE_URL).toBeDefined();
    expect(typeof API_CONFIG.BASE_URL).toBe('string');
  });

  it('should have all required endpoints', () => {
    expect(API_CONFIG.ENDPOINTS).toBeDefined();
    expect(API_CONFIG.ENDPOINTS.ANALYZE).toBe('/analyze');
    expect(API_CONFIG.ENDPOINTS.ANALYZE_BATCH).toBe('/analyze-batch');
    expect(API_CONFIG.ENDPOINTS.HEALTH).toBe('/health');
    expect(API_CONFIG.ENDPOINTS.STATS).toBe('/stats');
  });

  it('should have valid timeout value', () => {
    expect(API_CONFIG.TIMEOUT).toBe(30000);
  });

  it('should have valid retry attempts', () => {
    expect(API_CONFIG.RETRY_ATTEMPTS).toBe(3);
  });

  it('should have valid batch size', () => {
    expect(API_CONFIG.BATCH_SIZE).toBe(16);
  });
});

describe('getEndpointUrl', () => {
  it('should return full URL for valid endpoint', () => {
    const url = getEndpointUrl('ANALYZE');
    expect(url).toContain('/analyze');
  });

  it('should throw error for invalid endpoint', () => {
    expect(() => getEndpointUrl('INVALID')).toThrow('Unknown endpoint: INVALID');
  });
});
