/**
 * Analysis Service for Construction Photo Analyzer
 * Handles batch processing of photos with progress tracking and fallback support
 */

import { apiClient } from './apiClient';
import { API_CONFIG } from '../config/api.config';
import { connectionManager } from './connectionManager';

class AnalysisService {
    /**
     * Analyze a single photo with graceful degradation and retry/fallback logic
     * @param {Object} photo - Photo object with base64 and filename
     * @returns {Promise<Object>} Analysis result
     */
    async analyzeWithGracefulDegradation(photo) {
      // Import dependencies here to avoid circular imports
      const { errorHandler } = await import('../utils/errorHandler');
      const { fallbackService } = await import('./fallbackService');
      const { connectionManager } = await import('./connectionManager');
      const { apiClient } = await import('./apiClient');

      let attempts = 0;
      let lastError = null;
      while (attempts < 3) {
        try {
          if (connectionManager.isConnected && !fallbackService.isEnabled()) {
            const result = await apiClient.analyzeSingle(photo.base64, photo.filename);
            return {
              ...result,
              filename: photo.filename,
              method: 'ai_analysis'
            };
          }
          break; // If not connected, skip retry
        } catch (error) {
          lastError = error;
          const recovery = errorHandler.handle(error, 'Photo Analysis');
          if (recovery === 'retry') {
            attempts++;
            continue;
          } else if (recovery === 'wait') {
            // Wait 2 seconds before retrying
            await new Promise(res => setTimeout(res, 2000));
            attempts++;
            continue;
          } else {
            break;
          }
        }
      }
      // Fallback to keyword detection
      fallbackService.enable();
      const fallbackResult = fallbackService.detectCategoryFallback(photo.filename);
      return {
        ...fallbackResult,
        filename: photo.filename
      };
    }
  constructor() {
    this.queue = [];
    this.processing = false;
    this.progressCallback = null;
    this.cancelled = false;
    this.startTime = null;
  }

  /**
   * Analyze multiple photos with batch processing
   * @param {Array<Object>} photos - Array of photo objects with base64 and filename
   * @param {Function} onProgress - Progress callback function
   * @returns {Promise<Array>} Analysis results
   */
  async analyzePhotos(photos, onProgress) {
    if (this.processing) {
      throw new Error('Analysis already in progress');
    }

    this.processing = true;
    this.cancelled = false;
    this.progressCallback = onProgress;
    this.startTime = Date.now();

    const results = [];
    const total = photos.length;

    try {
      // Check if backend is available
      const isConnected = connectionManager.isConnected;

      if (isConnected) {
        // Use batch processing when backend is available
        const batches = this.createBatches(photos, API_CONFIG.BATCH_SIZE);
        let processed = 0;

        for (const batch of batches) {
          if (this.cancelled) {
            break;
          }

          try {
            // Format batch for API
            const formattedBatch = batch.map(photo => ({
              image: photo.base64,
              filename: photo.filename
            }));

            const batchResults = await apiClient.analyzeBatch(formattedBatch);
            
            // Handle results
            if (batchResults.results) {
              results.push(...batchResults.results);
            } else if (Array.isArray(batchResults)) {
              results.push(...batchResults);
            }

            processed += batch.length;
            this.updateProgress(processed, total);

          } catch (error) {
            console.error('Batch analysis failed:', error);
            
            // Fallback to sequential processing for this batch
            for (const photo of batch) {
              if (this.cancelled) break;

              const result = await this.analyzeSingleWithFallback(photo);
              results.push(result);
              processed++;
              this.updateProgress(processed, total);
            }
          }
        }
      } else {
        // Backend offline - use fallback for all photos
        console.log('Backend offline, using fallback analysis');
        let processed = 0;

        for (const photo of photos) {
          if (this.cancelled) break;

          const result = this.fallbackAnalysis(photo.filename);
          results.push({ ...result, filename: photo.filename });
          processed++;
          this.updateProgress(processed, total);

          // Small delay to allow UI updates
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }

      return results;

    } finally {
      this.processing = false;
      this.progressCallback = null;
    }
  }

  /**
   * Create batches from an array
   * @param {Array} array - Array to batch
   * @param {number} batchSize - Size of each batch
   * @returns {Array<Array>} Array of batches
   */
  createBatches(array, batchSize) {
    const batches = [];
    for (let i = 0; i < array.length; i += batchSize) {
      batches.push(array.slice(i, i + batchSize));
    }
    return batches;
  }

  /**
   * Analyze a single photo with fallback support
   * @param {Object} photo - Photo object with base64 and filename
   * @returns {Promise<Object>} Analysis result
   */
  async analyzeSingleWithFallback(photo) {
    try {
      const result = await apiClient.analyzeSingle(photo.base64, photo.filename);
      return {
        ...result,
        filename: photo.filename,
        method: 'ai_analysis'
      };
    } catch (error) {
      console.error('Single analysis failed, using fallback:', error);
      return {
        ...this.fallbackAnalysis(photo.filename),
        filename: photo.filename
      };
    }
  }

  /**
   * Fallback analysis using keyword detection
   * @param {string} filename - The filename to analyze
   * @returns {Object} Analysis result
   */
  fallbackAnalysis(filename) {
    // Keyword mappings for construction photo categories
    const keywords = {
      interior: ['interior', 'inside', 'room', 'kitchen', 'bathroom', 'bedroom', 'living', 'dining', 'hallway', 'closet', 'indoor'],
      exterior: ['exterior', 'outside', 'facade', 'front', 'rear', 'side', 'outdoor', 'yard', 'garden', 'driveway', 'entrance'],
      mep: ['mep', 'hvac', 'electrical', 'plumbing', 'duct', 'pipe', 'wire', 'conduit', 'panel', 'meter', 'vent', 'sprinkler'],
      structure: ['structure', 'framing', 'foundation', 'beam', 'column', 'joist', 'truss', 'rebar', 'concrete', 'steel', 'lumber'],
      roofing: ['roof', 'roofing', 'shingle', 'flashing', 'gutter', 'soffit', 'fascia', 'attic'],
      flooring: ['floor', 'flooring', 'tile', 'hardwood', 'carpet', 'laminate', 'subfloor'],
      insulation: ['insulation', 'insulate', 'vapor', 'barrier', 'foam', 'fiberglass'],
      drywall: ['drywall', 'sheetrock', 'gypsum', 'wall', 'ceiling', 'plaster'],
      windows: ['window', 'glass', 'glazing', 'frame', 'sill', 'casing'],
      doors: ['door', 'doorway', 'entry', 'exit', 'threshold', 'jamb']
    };

    const lowerFilename = filename.toLowerCase();
    
    // Check each category for keyword matches
    for (const [category, words] of Object.entries(keywords)) {
      if (words.some(word => lowerFilename.includes(word))) {
        return {
          category: this.capitalizeFirst(category),
          confidence: 0.5,
          method: 'keyword_fallback',
          matched_keyword: words.find(word => lowerFilename.includes(word))
        };
      }
    }

    // No match found
    return {
      category: 'Uncategorized',
      confidence: 0,
      method: 'keyword_fallback',
      matched_keyword: null
    };
  }

  /**
   * Capitalize first letter of a string
   * @param {string} str - String to capitalize
   * @returns {string} Capitalized string
   */
  capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  /**
   * Update progress and notify callback
   * @param {number} current - Current processed count
   * @param {number} total - Total items count
   */
  updateProgress(current, total) {
    const percentage = Math.round((current / total) * 100);
    const elapsed = Date.now() - this.startTime;
    const speed = current / (elapsed / 1000); // items per second
    const remaining = total - current;
    const eta = remaining > 0 ? Math.round(remaining / speed) : 0;

    if (this.progressCallback) {
      this.progressCallback({
        current,
        total,
        percentage,
        speed: speed.toFixed(1),
        eta,
        elapsed: Math.round(elapsed / 1000)
      });
    }
  }

  /**
   * Cancel the current analysis
   */
  cancel() {
    this.cancelled = true;
  }

  /**
   * Check if analysis is currently in progress
   * @returns {boolean}
   */
  isProcessing() {
    return this.processing;
  }
}

// Export singleton instance
export const analysisService = new AnalysisService();

// Also export class for testing
export { AnalysisService };
