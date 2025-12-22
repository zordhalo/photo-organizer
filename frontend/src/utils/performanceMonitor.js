class PerformanceMonitor {
  constructor() {
    this.metrics = {
      fps: 60,
      memoryUsage: 0,
      renderTime: 0
    };
    this.startMonitoring();
  }
  
  startMonitoring() {
    // Monitor FPS
    let lastTime = performance.now();
    let frames = 0;
    
    const measureFPS = () => {
      frames++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        this.metrics.fps = Math.round((frames * 1000) / (currentTime - lastTime));
        frames = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    
    measureFPS();
    
    // Monitor memory (if available)
    if (performance.memory) {
      setInterval(() => {
        this.metrics.memoryUsage = Math.round(
          performance.memory.usedJSHeapSize / 1048576
        ); // MB
      }, 1000);
    }
  }
  
  getMetrics() {
    return this.metrics;
  }
  
  logMetrics() {
    console.log('Performance Metrics:', this.metrics);
  }
}

export const performanceMonitor = new PerformanceMonitor();
