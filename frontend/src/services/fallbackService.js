import { notifications } from '../utils/notifications';

class FallbackService {
  constructor() {
    this.enabled = false;
    this.categoryKeywords = {
      interior: ['interior', 'inside', 'room', 'kitchen', 'bath', 'bedroom', 'living', 'ceiling', 'floor', 'wall'],
      exterior: ['exterior', 'outside', 'facade', 'front', 'rear', 'roof', 'siding', 'entrance', 'building'],
      mep: ['mep', 'hvac', 'electrical', 'plumbing', 'mechanical', 'duct', 'pipe', 'conduit', 'panel'],
      structure: ['structure', 'framing', 'foundation', 'beam', 'column', 'steel', 'concrete', 'rebar'],
      finishing: ['finishing', 'paint', 'tile', 'flooring', 'drywall', 'trim', 'fixture'],
      safety: ['safety', 'fire', 'exit', 'alarm', 'extinguisher', 'sign']
    };
  }
  
  enable() {
    this.enabled = true;
    console.log('Fallback mode ENABLED');
    notifications.warning('Backend unavailable. Using keyword-based classification.');
  }
  
  disable() {
    this.enabled = false;
    console.log('Fallback mode DISABLED');
    notifications.success('Backend reconnected. Using AI classification.');
  }
  
  detectCategoryFallback(filename) {
    const lowerFilename = filename.toLowerCase();
    
    // Score each category
    const scores = {};
    for (const [category, keywords] of Object.entries(this.categoryKeywords)) {
      const matches = keywords.filter(keyword => lowerFilename.includes(keyword));
      scores[category] = matches.length;
    }
    
    // Find highest score
    let bestCategory = 'Unknown';
    let bestScore = 0;
    
    for (const [category, score] of Object.entries(scores)) {
      if (score > bestScore) {
        bestScore = score;
        bestCategory = category;
      }
    }
    
    // Calculate confidence (0.3 to 0.7 range for fallback)
    const confidence = bestScore > 0 ? 0.3 + (Math.min(bestScore, 3) * 0.15) : 0;
    
    return {
      category: bestCategory.charAt(0).toUpperCase() + bestCategory.slice(1),
      confidence: confidence,
      method: 'keyword_fallback',
      keywords_matched: bestScore
    };
  }
  
  isEnabled() {
    return this.enabled;
  }
}

export const fallbackService = new FallbackService();
