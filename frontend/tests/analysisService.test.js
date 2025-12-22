/**
 * Tests for Analysis Service
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AnalysisService } from '../src/services/analysisService';

// Mock the apiClient
vi.mock('../src/services/apiClient', () => ({
  apiClient: {
    analyzeBatch: vi.fn(),
    analyzeSingle: vi.fn()
  }
}));

// Mock the connectionManager
vi.mock('../src/services/connectionManager', () => ({
  connectionManager: {
    isConnected: true
  }
}));

import { apiClient } from '../src/services/apiClient';
import { connectionManager } from '../src/services/connectionManager';

describe('AnalysisService', () => {
  let analysisService;

  beforeEach(() => {
    vi.clearAllMocks();
    analysisService = new AnalysisService();
  });

  afterEach(() => {
    analysisService.cancel();
  });

  describe('createBatches', () => {
    it('should create correct number of batches', () => {
      const items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const batches = analysisService.createBatches(items, 3);

      expect(batches.length).toBe(4);
      expect(batches[0]).toEqual([1, 2, 3]);
      expect(batches[1]).toEqual([4, 5, 6]);
      expect(batches[2]).toEqual([7, 8, 9]);
      expect(batches[3]).toEqual([10]);
    });

    it('should handle empty array', () => {
      const batches = analysisService.createBatches([], 3);
      expect(batches).toEqual([]);
    });

    it('should handle array smaller than batch size', () => {
      const items = [1, 2];
      const batches = analysisService.createBatches(items, 5);

      expect(batches.length).toBe(1);
      expect(batches[0]).toEqual([1, 2]);
    });
  });

  describe('fallbackAnalysis', () => {
    it('should detect interior category from filename', () => {
      const result = analysisService.fallbackAnalysis('photo_interior_living.jpg');

      expect(result.category).toBe('Interior');
      expect(result.method).toBe('keyword_fallback');
      expect(result.confidence).toBe(0.5);
    });

    it('should detect exterior category from filename', () => {
      const result = analysisService.fallbackAnalysis('house_exterior_front.jpg');

      expect(result.category).toBe('Exterior');
      expect(result.method).toBe('keyword_fallback');
    });

    it('should detect mep category from filename', () => {
      const result = analysisService.fallbackAnalysis('electrical_panel.jpg');

      expect(result.category).toBe('Mep');
      expect(result.method).toBe('keyword_fallback');
    });

    it('should detect structure category from filename', () => {
      const result = analysisService.fallbackAnalysis('foundation_beam.jpg');

      expect(result.category).toBe('Structure');
      expect(result.method).toBe('keyword_fallback');
    });

    it('should return Uncategorized for unknown filenames', () => {
      const result = analysisService.fallbackAnalysis('IMG_20231201.jpg');

      expect(result.category).toBe('Uncategorized');
      expect(result.confidence).toBe(0);
      expect(result.method).toBe('keyword_fallback');
    });

    it('should be case insensitive', () => {
      const result = analysisService.fallbackAnalysis('KITCHEN_INTERIOR.JPG');

      expect(result.category).toBe('Interior');
    });
  });

  describe('analyzePhotos', () => {
    const mockPhotos = [
      { filename: 'photo1.jpg', base64: 'data1' },
      { filename: 'photo2.jpg', base64: 'data2' },
      { filename: 'photo3.jpg', base64: 'data3' }
    ];

    it('should use batch processing when backend is connected', async () => {
      connectionManager.isConnected = true;
      apiClient.analyzeBatch.mockResolvedValue({
        results: [
          { category: 'Interior', confidence: 0.9 },
          { category: 'Exterior', confidence: 0.8 },
          { category: 'Structure', confidence: 0.7 }
        ]
      });

      const progressCallback = vi.fn();
      const results = await analysisService.analyzePhotos(mockPhotos, progressCallback);

      expect(apiClient.analyzeBatch).toHaveBeenCalled();
      expect(results.length).toBe(3);
      expect(progressCallback).toHaveBeenCalled();
    });

    it('should use fallback when backend is offline', async () => {
      connectionManager.isConnected = false;

      const photos = [
        { filename: 'interior_kitchen.jpg', base64: 'data1' },
        { filename: 'exterior_front.jpg', base64: 'data2' }
      ];

      const results = await analysisService.analyzePhotos(photos, vi.fn());

      expect(apiClient.analyzeBatch).not.toHaveBeenCalled();
      expect(results.length).toBe(2);
      expect(results[0].method).toBe('keyword_fallback');
    });

    it('should fall back to sequential on batch error', async () => {
      connectionManager.isConnected = true;
      apiClient.analyzeBatch.mockRejectedValue(new Error('Batch failed'));
      apiClient.analyzeSingle.mockResolvedValue({ category: 'Interior', confidence: 0.8 });

      const results = await analysisService.analyzePhotos(mockPhotos, vi.fn());

      expect(apiClient.analyzeSingle).toHaveBeenCalledTimes(3);
      expect(results.length).toBe(3);
    });

    it('should throw if analysis already in progress', async () => {
      connectionManager.isConnected = false;
      
      // Start first analysis
      const firstPromise = analysisService.analyzePhotos(mockPhotos, vi.fn());
      
      // Try to start second analysis
      await expect(analysisService.analyzePhotos(mockPhotos, vi.fn()))
        .rejects.toThrow('Analysis already in progress');
      
      // Wait for first to complete
      await firstPromise;
    });

    it('should update progress correctly', async () => {
      connectionManager.isConnected = false;
      const progressCallback = vi.fn();

      await analysisService.analyzePhotos(mockPhotos, progressCallback);

      // Check that progress was updated for each photo
      expect(progressCallback).toHaveBeenCalledTimes(3);
      
      // Check final progress
      const lastCall = progressCallback.mock.calls[2][0];
      expect(lastCall.current).toBe(3);
      expect(lastCall.total).toBe(3);
      expect(lastCall.percentage).toBe(100);
    });
  });

  describe('cancel', () => {
    it('should set cancelled flag', () => {
      analysisService.cancel();
      expect(analysisService.cancelled).toBe(true);
    });
  });

  describe('isProcessing', () => {
    it('should return processing state', () => {
      expect(analysisService.isProcessing()).toBe(false);
    });
  });

  describe('capitalizeFirst', () => {
    it('should capitalize first letter', () => {
      expect(analysisService.capitalizeFirst('interior')).toBe('Interior');
      expect(analysisService.capitalizeFirst('MEP')).toBe('MEP');
      expect(analysisService.capitalizeFirst('a')).toBe('A');
    });
  });
});
