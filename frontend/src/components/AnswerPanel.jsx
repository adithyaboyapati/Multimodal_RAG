import React from 'react';
import { FileText, Image as ImageIcon, Cpu } from 'lucide-react';

export default function AnswerPanel({ response, loading, onSelectCitation }) {
  if (loading) {
    return (
      <div className="glass-panel answer-card" id="answer-loading-card">
        <div className="answer-header">
          <div className="intent-badge intent-cross">
            <Cpu size={14} />
            <span>Analyzing Multimodal Evidence...</span>
          </div>
        </div>
        <div className="answer-body">
          <div className="skeleton-line" style={{ width: '85%' }}></div>
          <div className="skeleton-line" style={{ width: '95%' }}></div>
          <div className="skeleton-line" style={{ width: '70%' }}></div>
          <div className="skeleton-line" style={{ width: '80%' }}></div>
        </div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="glass-panel answer-card" id="answer-placeholder">
        <div className="answer-header">
          <span style={{ color: 'var(--text-muted)' }}>Ready for input</span>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: '1.6' }}>
          Select one of the benchmark queries on the left or enter a custom question to inspect
          multimodal retrieval, cross-modal reranking, and strictly grounded answers.
        </p>
      </div>
    );
  }

  const getIntentBadge = (modality) => {
    switch (modality) {
      case 'multimodal_vlm':
        return (
          <span className="intent-badge intent-visual">
            <ImageIcon size={14} />
            <span>Multimodal Vision Route</span>
          </span>
        );
      default:
        return (
          <span className="intent-badge intent-text">
            <FileText size={14} />
            <span>Text / Tabular Route</span>
          </span>
        );
    }
  };

  // Convert inline citations "[Page X | MODALITY]" into interactive badges
  const renderFormattedAnswer = (text) => {
    if (!text) return null;

    const citationRegex = /\[Page\s+(\d+)\s*\|\s*([^\]]+)\]/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = citationRegex.exec(text)) !== null) {
      // Push preceding text
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      const pageNum = match[1];
      const modType = match[2];

      parts.push(
        <button
          key={`cite-${match.index}`}
          className="citation-badge"
          onClick={() => onSelectCitation && onSelectCitation(Number(pageNum))}
          title={`Click to view Page ${pageNum} evidence`}
        >
          <span>P.{pageNum}</span>
          <span style={{ opacity: 0.7 }}>{modType}</span>
        </button>
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts;
  };

  return (
    <div className="glass-panel answer-card" id="answer-result-card">
      <div className="answer-header">
        {getIntentBadge(response.modality_used)}
        <div className="model-tag" title="Answer synthesized via Groq LPU inference using GROQ_API_KEY">
          <span>Inference: </span>
          <span style={{ color: 'var(--cyan)' }}>Groq LPU ({response.model})</span>
        </div>
      </div>

      <div className="answer-body" id="answer-text-content">
        <p>{renderFormattedAnswer(response.answer)}</p>
      </div>
    </div>
  );
}
