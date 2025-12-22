class ImageCompressor {
  constructor() {
    this.maxWidth = 1920;
    this.maxHeight = 1080;
    this.quality = 0.85;
    this.maxSizeMB = 10;
  }
  
  async compressImage(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const compressed = this.resizeAndCompress(img);
          resolve(compressed);
        };
        img.onerror = reject;
        img.src = e.target.result;
      };
      
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
  
  resizeAndCompress(img) {
    const canvas = document.createElement('canvas');
    let width = img.width;
    let height = img.height;
    
    // Calculate new dimensions
    if (width > this.maxWidth || height > this.maxHeight) {
      const ratio = Math.min(this.maxWidth / width, this.maxHeight / height);
      width = width * ratio;
      height = height * ratio;
    }
    
    canvas.width = width;
    canvas.height = height;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, width, height);
    
    // Convert to base64 with compression
    const compressed = canvas.toDataURL('image/jpeg', this.quality);
    
    console.log(`Image compressed: ${this.getBase64Size(compressed)} KB`);
    return compressed;
  }
  
  getBase64Size(base64) {
    const sizeInBytes = (base64.length * 3) / 4;
    return Math.round(sizeInBytes / 1024);
  }
  
  async compressBatch(files) {
    const promises = Array.from(files).map(file => this.compressImage(file));
    return Promise.all(promises);
  }
}

export const imageCompressor = new ImageCompressor();
