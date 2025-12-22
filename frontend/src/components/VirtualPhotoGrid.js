class VirtualPhotoGrid {
  constructor(container, photos) {
    this.container = container;
    this.photos = photos;
    this.itemHeight = 200; // Height of each photo card
    this.itemsPerRow = 4;
    this.visibleItems = [];
    this.scrollTop = 0;
    
    this.setupContainer();
    this.setupScrollListener();
    this.render();
  }
  
  setupContainer() {
    this.container.style.position = 'relative';
    this.container.style.overflow = 'auto';
    this.container.style.height = '600px'; // Viewport height
    
    // Create spacer to maintain scroll height
    this.spacer = document.createElement('div');
    const totalRows = Math.ceil(this.photos.length / this.itemsPerRow);
    this.spacer.style.height = `${totalRows * this.itemHeight}px`;
    this.container.appendChild(this.spacer);
  }
  
  setupScrollListener() {
    this.container.addEventListener('scroll', () => {
      this.scrollTop = this.container.scrollTop;
      this.render();
    });
  }
  
  render() {
    const viewportHeight = this.container.clientHeight;
    const startRow = Math.floor(this.scrollTop / this.itemHeight);
    const endRow = Math.ceil((this.scrollTop + viewportHeight) / this.itemHeight);
    
    const startIndex = startRow * this.itemsPerRow;
    const endIndex = Math.min(endRow * this.itemsPerRow, this.photos.length);
    
    // Clear existing items
    this.clearVisibleItems();
    
    // Render visible items
    for (let i = startIndex; i < endIndex; i++) {
      const photo = this.photos[i];
      if (!photo) continue;
      
      const item = this.createPhotoCard(photo, i);
      this.container.appendChild(item);
      this.visibleItems.push(item);
    }
  }
  
  createPhotoCard(photo, index) {
    const card = document.createElement('div');
    card.className = 'photo-card';
    const row = Math.floor(index / this.itemsPerRow);
    const col = index % this.itemsPerRow;
    card.style.position = 'absolute';
    card.style.top = `${row * this.itemHeight}px`;
    card.style.left = `${col * (100 / this.itemsPerRow)}%`;
    card.style.width = `${100 / this.itemsPerRow}%`;
    card.style.height = `${this.itemHeight}px`;

    // Highlight based on confidence
    let highlightClass = '';
    if (photo.confidence >= 0.9) highlightClass = 'high-confidence';
    else if (photo.confidence >= 0.7) highlightClass = 'medium-confidence';
    else highlightClass = 'low-confidence';
    card.classList.add(highlightClass);

    // Overlay for category
    const overlay = `<div class="category-overlay ${photo.category}">${photo.category}</div>`;

    card.innerHTML = `
      <div class="photo-thumb-wrapper">
        <img src="${photo.thumbnail}" loading="lazy" alt="${photo.filename}">
        ${overlay}
      </div>
      <div class="photo-info">
        <span class="category">${photo.category}</span>
        <span class="confidence">${Math.round(photo.confidence * 100)}%</span>
      </div>
    `;

    // Tooltip for details
    card.title = `File: ${photo.filename}\nCategory: ${photo.category}\nConfidence: ${Math.round(photo.confidence * 100)}%`;

    return card;
  }
  
  clearVisibleItems() {
    this.visibleItems.forEach(item => item.remove());
    this.visibleItems = [];
  }
  
  updatePhotos(photos) {
    this.photos = photos;
    const totalRows = Math.ceil(this.photos.length / this.itemsPerRow);
    this.spacer.style.height = `${totalRows * this.itemHeight}px`;
    this.render();
  }
}

export default VirtualPhotoGrid;
