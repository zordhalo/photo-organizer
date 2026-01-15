/**
 * UploadQueue Component
 * UI component for managing the pre-upload queue, thumbnails, file info, removal, batch stats, and Start Analysis button
 * 
 * @param {Object} props - Component properties
 * @param {Array} props.files - Array of file objects with { file, previewUrl, size, type }
 * @param {Function} props.onRemove - Callback when removing a file (receives index)
 * @param {Function} props.onStartAnalysis - Callback when starting analysis
 * @param {Object} props.errorStates - Object mapping filename to error message
 * @returns {HTMLElement} The upload queue DOM element
 */
function UploadQueue({ files, onRemove, onStartAnalysis, errorStates }) {
  const totalSize = files.reduce((sum, f) => sum + f.size, 0);

  const container = document.createElement('div');
  container.className = 'upload-queue';

  // Header
  const header = document.createElement('h3');
  header.textContent = 'Upload Queue';
  container.appendChild(header);

  // Queue grid
  const queueGrid = document.createElement('div');
  queueGrid.className = 'queue-grid';

  files.forEach((f, idx) => {
    const queueItem = document.createElement('div');
    queueItem.className = 'queue-item';

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

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn btn-danger btn-remove';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => onRemove(idx));
    queueItem.appendChild(removeBtn);

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

  // Start Analysis button
  const startBtn = document.createElement('button');
  startBtn.className = 'btn btn-primary start-analysis';
  startBtn.textContent = 'Start Analysis';
  startBtn.disabled = !files.length;
  startBtn.addEventListener('click', onStartAnalysis);
  container.appendChild(startBtn);

  return container;
}

export default UploadQueue;
