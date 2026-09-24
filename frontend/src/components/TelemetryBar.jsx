import React from 'react';
import { Activity, Zap } from 'lucide-react';

export default function TelemetryBar({ latency }) {
  if (!latency) return null;

  return (
    <div className="telemetry-bar" id="telemetry-bar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)' }}>
        <Activity size={14} color="var(--cyan)" />
        <span style={{ fontWeight: '600' }}>Pipeline Telemetry</span>
      </div>

      <div className="telemetry-metrics">
        <div className="metric-item">
          <span className="metric-label">Intent:</span>
          <span className="metric-val">{latency.intent_ms || 0}ms</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Retrieval:</span>
          <span className="metric-val">{latency.retrieval_ms || 0}ms</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Reranking:</span>
          <span className="metric-val">{latency.rerank_ms || 0}ms</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Generation:</span>
          <span className="metric-val">{latency.generation_ms || 0}ms</span>
        </div>

        <div className="metric-item" style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: '16px' }}>
          <Zap size={13} color="var(--emerald)" />
          <span className="metric-label">Total:</span>
          <span className="metric-val" style={{ color: 'var(--emerald)' }}>{latency.total_ms || 0}ms</span>
        </div>
      </div>
    </div>
  );
}
