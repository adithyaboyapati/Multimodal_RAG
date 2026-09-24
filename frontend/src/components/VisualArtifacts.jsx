import React, { useState } from 'react';
import { Image as ImageIcon, ZoomIn, X, Info } from 'lucide-react';

export default function VisualArtifacts({ artifacts }) {
  const [selectedArtifact, setSelectedArtifact] = useState(null);

  if (!artifacts || artifacts.length === 0) {
    return null;
  }

  return (
    <div className="glass-panel visuals-container" id="visual-artifacts-section">
      <div className="section-label">
        <ImageIcon size={14} color="var(--cyan)" />
        <span>Retrieved Visual Evidence ({artifacts.length})</span>
      </div>

      <div className="visuals-grid">
        {artifacts.map((art, idx) => (
          <div
            key={art.asset_id || idx}
            className="visual-card"
            onClick={() => setSelectedArtifact(art)}
            id={`visual-card-${idx}`}
          >
            <div className="visual-img-wrapper">
              {art.data_uri ? (
                <img
                  src={art.data_uri}
                  alt={`Evidence on page ${art.page_number}`}
                  className="visual-img"
                />
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                  [Visual on Page {art.page_number}]
                </div>
              )}
              <span className="visual-overlay">Page {art.page_number}</span>
            </div>

            <div className="visual-info">
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--cyan)', fontSize: '12px', fontWeight: '600' }}>
                <ZoomIn size={12} />
                <span>Click to Expand</span>
              </div>
              {art.summary && (
                <p className="visual-summary-preview">{art.summary}</p>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox Expander Modal */}
      {selectedArtifact && (
        <div className="modal-overlay" onClick={() => setSelectedArtifact(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} id="visual-lightbox-modal">
            <button className="modal-close" onClick={() => setSelectedArtifact(null)}>
              <X size={16} />
            </button>

            <h3 style={{ fontSize: '16px', fontWeight: '600' }}>
              Visual Evidence — Page {selectedArtifact.page_number}
            </h3>

            {selectedArtifact.data_uri && (
              <img
                src={selectedArtifact.data_uri}
                alt="Expanded chart"
                className="modal-img"
              />
            )}

            {selectedArtifact.summary && (
              <div style={{ background: 'rgba(0,0,0,0.4)', padding: '16px', borderRadius: 'var(--radius-md)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--cyan)', fontSize: '13px', fontWeight: '600', marginBottom: '8px' }}>
                  <Info size={14} />
                  <span>VLM Factual Summary</span>
                </div>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                  {selectedArtifact.summary}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
