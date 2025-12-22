import { defineConfig } from 'vite';

export default defineConfig({
  // Server configuration
  server: {
    port: 5173,
    host: true,
    open: false,
    // Proxy API requests to backend during development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  
  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['axios']
        }
      }
    }
  },
  
  // Test configuration (vitest)
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['tests/**/*.test.js', 'src/**/*.test.js'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'tests/']
    }
  },
  
  // Resolve configuration
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@services': '/src/services',
      '@utils': '/src/utils',
      '@config': '/src/config'
    }
  }
});
