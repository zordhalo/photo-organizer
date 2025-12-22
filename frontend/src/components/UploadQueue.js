// UploadQueue.js
// UI component for managing the pre-upload queue, thumbnails, file info, removal, batch stats, and Start Analysis button

import { useState } from 'react';

function UploadQueue({ files, onRemove, onStartAnalysis, errorStates }) {
  // files: [{ file, previewUrl, size, type, error }]
  // errorStates: { [filename]: errorMsg }
  const totalSize = files.reduce((sum, f) => sum + f.size, 0);

  return (
    <div className="upload-queue">
      <h3>Upload Queue</h3>
      <div className="queue-grid">
        {files.map((f, idx) => (
          <div key={f.file.name} className="queue-item">
            <img src={f.previewUrl} alt={f.file.name} className="thumb" />
            <div className="file-info">
              <span>{f.file.name}</span>
              <span>{(f.size / 1024).toFixed(1)} KB</span>
              <span>{f.type}</span>
              {errorStates[f.file.name] && (
                <span className="error">{errorStates[f.file.name]}</span>
              )}
            </div>
            <button onClick={() => onRemove(idx)}>Remove</button>
          </div>
        ))}
      </div>
      <div className="queue-stats">
        <span>Files: {files.length}</span>
        <span>Total Size: {(totalSize / 1024).toFixed(1)} KB</span>
      </div>
      <button className="start-analysis" onClick={onStartAnalysis} disabled={!files.length}>
        Start Analysis
      </button>
    </div>
  );
}

export default UploadQueue;
