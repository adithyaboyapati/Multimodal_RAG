import React from 'react';
import { Layers, UploadCloud, AlertCircle } from 'lucide-react';

export default function Navbar({ onOpenIngest, healthStatus }) {
  const isHealthy = healthStatus?.status === 'ready' || healthStatus?.status === 'live';

  return (
    <header className="navbar">
      <div className="brand">
        <div className="brand-icon">
          <Layers size={20} color="#fff" />
        </div>
        <div>
          <span className="brand-title">NovaCore Intelligence</span>
        </div>
        <span className="brand-badge">Multimodal RAG</span>
      </div>

      <div className="nav-actions">
        <div className="status-pill" title="Pinecone & Groq Hybrid Connectivity">
          {isHealthy ? (
            <>
              <span className="status-dot"></span>
              <span>Services Online</span>
            </>
          ) : (
            <>
              <AlertCircle size={14} color="#f59e0b" />
              <span>Checking Connectivity</span>
            </>
          )}
        </div>

        <button 
          id="btn-open-ingest"
          className="btn-primary" 
          onClick={onOpenIngest}
        >
          <UploadCloud size={16} />
          <span>Ingest Document</span>
        </button>
      </div>
    </header>
  );
}
