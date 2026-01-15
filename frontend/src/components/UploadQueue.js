/**
 * UploadQueue Component
 * UI component for managing the pre-upload queue, thumbnails, file info, removal, batch stats, and Start Analysis button
 * 
 * @param {Object} props - Component properties
 * @param {Array} props.files - Array of file objects with { file, previewUrl, size, type }
 * @param {Function} props.onRemove - Callback when removing a file (receives index)
 * @param {Function} props.onStartAnalysis - Callback when starting analysis
 * @param {Object} props.errorStates - Object mapping filename to error message
 * @param {number} props.processingIndex - Current index being processed (-1 if not processing)
 * @returns {HTMLElement} The upload queue DOM element
 */
function UploadQueue({ files, onRemove, onReorder, onStartAnalysis, errorStates, processingIndex = -1 }) {
  const totalSize = files.reduce((sum, f) => sum + f.size, 0);
  let draggingIndex = null;
  const isProcessing = processingIndex >= 0;

  const container = document.createElement('div');
  container.className = 'upload-queue';

  // Header
  const header = document.createElement('h3');
  header.textContent = isProcessing ? 'Processing...' : 'Upload Queue';
  container.appendChild(header);

  // Queue grid
  const queueGrid = document.createElement('div');
  queueGrid.className = 'queue-grid';

  files.forEach((f, idx) => {
    const queueItem = document.createElement('div');
    queueItem.className = 'queue-item';
    
    // Add processing state classes
    if (isProcessing) {
      if (idx < processingIndex) {
        queueItem.classList.add('completed');
      } else if (idx === processingIndex) {
        queueItem.classList.add('processing');
      } else {
        queueItem.classList.add('pending');
      }
    }
    
    // Only allow drag when not processing
    queueItem.draggable = !isProcessing;
    queueItem.dataset.index = idx;

    // Only add drag events when not processing
    if (!isProcessing) {
      queueItem.addEventListener('dragstart', (event) => {
        draggingIndex = idx;
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', String(idx));
        queueItem.classList.add('dragging');
      });

      queueItem.addEventListener('dragend', () => {
        queueItem.classList.remove('dragging');
      });

      queueItem.addEventListener('dragover', (event) => {
        event.preventDefault();
        queueItem.classList.add('drag-over');
      });

      queueItem.addEventListener('dragleave', () => {
        queueItem.classList.remove('drag-over');
      });

      queueItem.addEventListener('drop', (event) => {
        event.preventDefault();
        queueItem.classList.remove('drag-over');
        const fromIndex = draggingIndex ?? Number(event.dataTransfer.getData('text/plain'));
        const toIndex = Number(queueItem.dataset.index);
        draggingIndex = null;
        if (Number.isInteger(fromIndex) && Number.isInteger(toIndex) && onReorder) {
          onReorder(fromIndex, toIndex);
        }
      });
    }

    // Thumbnail
    const thumb = document.createElement('img');
    thumb.src = f.previewUrl;
    thumb.alt = f.file.name;
    thumb.className = 'thumb';
    queueItem.appendChild(thumb);

    // File info
    const fileInfo = document.createElement('div');
    fileInfo.className = 'file-info';

    const fileName = document.createElement('span');
    fileName.className = 'file-name';
    fileName.textContent = f.file.name;
    fileInfo.appendChild(fileName);

    const fileSize = document.createElement('span');
    fileSize.className = 'file-size';
    fileSize.textContent = `${(f.size / 1024).toFixed(1)} KB`;
    fileInfo.appendChild(fileSize);

    const fileType = document.createElement('span');
    fileType.className = 'file-type';
    fileType.textContent = f.type;
    fileInfo.appendChild(fileType);

    // Error message if any
    if (errorStates && errorStates[f.file.name]) {
      const errorSpan = document.createElement('span');
      errorSpan.className = 'error';
      errorSpan.textContent = errorStates[f.file.name];
      fileInfo.appendChild(errorSpan);
    }

    queueItem.appendChild(fileInfo);

    // Remove button (only show when not processing)
    if (!isProcessing && onRemove) {
      const removeBtn = document.createElement('button');
      removeBtn.className = 'btn btn-danger btn-remove';
      removeBtn.textContent = 'Remove';
      removeBtn.addEventListener('click', () => onRemove(idx));
      queueItem.appendChild(removeBtn);
    }
    
    // Status indicator when processing
    if (isProcessing) {
      const statusIndicator = document.createElement('div');
      statusIndicator.className = 'queue-item-status';
      if (idx < processingIndex) {
        statusIndicator.innerHTML = '<span class="status-icon status-complete">✓</span>';
      } else if (idx === processingIndex) {
        statusIndicator.innerHTML = '<span class="status-icon status-active"><span class="spinner"></span></span>';
      } else {
        statusIndicator.innerHTML = '<span class="status-icon status-waiting">○</span>';
      }
      queueItem.appendChild(statusIndicator);
    }

    queueGrid.appendChild(queueItem);
  });

  container.appendChild(queueGrid);

  // Queue stats
  const queueStats = document.createElement('div');
  queueStats.className = 'queue-stats';

  const fileCount = document.createElement('span');
  fileCount.textContent = `Files: ${files.length}`;
  queueStats.appendChild(fileCount);

  const totalSizeSpan = document.createElement('span');
  totalSizeSpan.textContent = `Total Size: ${(totalSize / 1024).toFixed(1)} KB`;
  queueStats.appendChild(totalSizeSpan);

  container.appendChild(queueStats);

  // Start Analysis button (hide when processing)
  if (!isProcessing) {
    const startBtn = document.createElement('button');
    startBtn.className = 'btn btn-primary start-analysis';
    startBtn.textContent = 'Start Analysis';
    startBtn.disabled = !files.length;
    startBtn.addEventListener('click', onStartAnalysis);
    container.appendChild(startBtn);
  }

  return container;
}

export default UploadQueue;
