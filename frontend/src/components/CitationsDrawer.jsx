import React from 'react';
import { Bookmark, FileText, Table, Image as ImageIcon } from 'lucide-react';

export default function CitationsDrawer({ citations, highlightedPage }) {
  if (!citations || citations.length === 0) {
    return null;
  }

  const getModalityIcon = (modality) => {
    switch (modality) {
      case 'table': return <Table size={13} color="var(--emerald)" />;
      case 'visual': return <ImageIcon size={13} color="var(--cyan)" />;
      default: return <FileText size={13} color="#94a3b8" />;
    }
  };

  return (
    <div className="glass-panel" id="citations-section">
      <div className="section-label">
        <Bookmark size={14} color="var(--cyan)" />
        <span>Grounded Citations ({citations.length})</span>
      </div>

      <div className="citations-grid">
        {citations.map((cite, idx) => {
          const isHighlighted = highlightedPage === cite.page_number;

          return (
            <div
              key={idx}
              className="citation-card"
              style={{
                borderColor: isHighlighted ? 'var(--cyan)' : 'var(--border-subtle)',
                background: isHighlighted ? 'rgba(6, 182, 212, 0.08)' : 'rgba(255, 255, 255, 0.02)',
              }}
              id={`citation-card-${idx}`}
            >
              <div className="citation-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {getModalityIcon(cite.modality)}
                  <span>Page {cite.page_number} ({cite.modality.toUpperCase()})</span>
                </div>
                {cite.confidence_score && (
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
                    Score: {cite.confidence_score.toFixed(2)}
                  </span>
                )}
              </div>

              <p className="citation-excerpt">"{cite.excerpt}"</p>

              {cite.bounding_box && (
                <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  Box: [{cite.bounding_box.x0.toFixed(0)}, {cite.bounding_box.y0.toFixed(0)}, {cite.bounding_box.x1.toFixed(0)}, {cite.bounding_box.y1.toFixed(0)}]
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
