import React from 'react';
import { Send, Sliders, Sparkles } from 'lucide-react';
import { BENCHMARK_PRESETS } from '../data/mockData';

export default function QueryPanel({
  query,
  setQuery,
  topK,
  setTopK,
  alpha,
  setAlpha,
  rerank,
  setRerank,
  returnVisuals,
  setReturnVisuals,
  onSubmit,
  loading,
}) {
  const handleSelectPreset = (presetQuestion) => {
    setQuery(presetQuestion);
  };

  const getTagClass = (type) => {
    switch (type) {
      case 'text': return 'tag-text';
      case 'table': return 'tag-table';
      case 'visual': return 'tag-visual';
      case 'cross': return 'tag-cross';
      default: return 'tag-text';
    }
  };

  return (
    <aside className="sidebar-panel">
      {/* Benchmark Presets Section */}
      <div className="glass-panel">
        <div className="section-label">
          <Sparkles size={14} color="var(--cyan)" />
          <span>Benchmark Presets</span>
        </div>
        <div className="presets-grid">
          {BENCHMARK_PRESETS.map((preset) => (
            <button
              key={preset.id}
              className="preset-chip"
              onClick={() => handleSelectPreset(preset.question)}
              id={`preset-${preset.id}`}
            >
              <span>{preset.label}</span>
              <span className={`chip-tag ${getTagClass(preset.type)}`}>
                {preset.tag}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Query Input Section */}
      <div className="glass-panel query-box">
        <div className="section-label">
          <span>Query Input</span>
        </div>
        <textarea
          id="query-input"
          className="query-textarea"
          placeholder="Ask any question about NovaCore FY2026 performance, charts, or tables..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <button
          id="btn-submit-query"
          className="btn-submit"
          onClick={onSubmit}
          disabled={loading || !query.trim()}
        >
          <Send size={16} />
          <span>{loading ? 'Synthesizing Answers...' : 'Query Multimodal Engine'}</span>
        </button>
      </div>

      {/* Tunable Pipeline Controls */}
      <div className="glass-panel param-group">
        <div className="section-label">
          <Sliders size={14} color="var(--violet)" />
          <span>Retrieval Parameters</span>
        </div>

        {/* Top-K Slider */}
        <div className="slider-item">
          <div className="slider-header">
            <span>Candidates (Top-K)</span>
            <span className="slider-value">{topK}</span>
          </div>
          <input
            id="slider-top-k"
            type="range"
            min="1"
            max="10"
            step="1"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </div>

        {/* Hybrid Alpha Slider */}
        <div className="slider-item">
          <div className="slider-header">
            <span>Hybrid Alpha: {alpha < 0.5 ? 'Sparse (BM25)' : 'Dense (Neural)'}</span>
            <span className="slider-value">{alpha.toFixed(2)}</span>
          </div>
          <input
            id="slider-alpha"
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          />
        </div>

        {/* Reranker Toggle */}
        <label className="toggle-item" htmlFor="toggle-rerank">
          <span>Cross-Encoder Reranking</span>
          <input
            id="toggle-rerank"
            type="checkbox"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
          />
        </label>

        {/* Visual Artifacts Toggle */}
        <label className="toggle-item" htmlFor="toggle-visuals">
          <span>Attach Visual Evidence</span>
          <input
            id="toggle-visuals"
            type="checkbox"
            checked={returnVisuals}
            onChange={(e) => setReturnVisuals(e.target.checked)}
          />
        </label>
      </div>
    </aside>
  );
}
