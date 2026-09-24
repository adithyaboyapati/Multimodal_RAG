/**
 * High-fidelity benchmark queries and fallback mock responses matching the NovaCore report.
 */
export const BENCHMARK_PRESETS = [
  {
    id: "TC-01",
    label: "Company Overview & HQ",
    type: "text",
    tag: "Text",
    question: "What does NovaCore Systems do and where is the company headquartered?",
  },
  {
    id: "TC-02",
    label: "Highest YoY Growth Region",
    type: "table",
    tag: "Table",
    question: "Which region had the highest year-over-year revenue growth?",
  },
  {
    id: "TC-03",
    label: "Peak Revenue Quarter",
    type: "visual",
    tag: "Graph",
    question: "According to the revenue graph, which quarter had the highest revenue?",
  },
  {
    id: "TC-04",
    label: "Supply Chain Critical QC Point",
    type: "visual",
    tag: "Diagram",
    question: "According to the supply-chain diagram, what is the critical quality-control point?",
  },
  {
    id: "TC-05",
    label: "Penang Facility Solar Share",
    type: "visual",
    tag: "Chart",
    question: "What percentage of electricity at the Penang facility came from solar?",
  },
  {
    id: "TC-06",
    label: "FY2026 Performance Summary",
    type: "cross",
    tag: "Cross-Modal",
    question: "Give me a short FY2026 performance summary. Include: 1. total revenue, 2. fastest-growing region, 3. support-resolution improvement, 4. solar share at the Penang facility.",
  },
];

export const MOCK_RESPONSES = {
  "What does NovaCore Systems do and where is the company headquartered?": {
    question: "What does NovaCore Systems do and where is the company headquartered?",
    answer: "NovaCore Systems is a leading enterprise infrastructure provider specializing in high-efficiency, liquid-cooled AI compute clusters and modular rack solutions [Page 1 | TEXT]. The company is headquartered in San Jose, California, with advanced manufacturing facilities across Penang, Malaysia and Dresden, Germany [Page 1 | TEXT].",
    citations: [
      {
        source: "NovaCore_Multimodal_Company_Report_2026.pdf",
        page_number: 1,
        modality: "text",
        excerpt: "NovaCore Systems Inc. — Headquarters: San Jose, California. Global leader in modular AI infrastructure.",
        confidence_score: 0.96
      }
    ],
    visual_artifacts: [],
    retrieved_chunks_count: 5,
    modality_used: "text_llm",
    model: "openai/gpt-oss-20b",
    latency: {
      intent_ms: 0.18,
      retrieval_ms: 24.3,
      rerank_ms: 22.1,
      generation_ms: 310.5,
      total_ms: 357.08
    }
  },
  "Which region had the highest year-over-year revenue growth?": {
    question: "Which region had the highest year-over-year revenue growth?",
    answer: "According to the Regional Financial Breakdown table on Page 4, Europe recorded the highest year-over-year revenue growth at 31% [Page 4 | TABLE], expanding from $31.4M in FY2025 to $41.2M in FY2026. North America followed with 18% growth ($55.4M), and Asia-Pacific grew by 24% ($35.4M) [Page 4 | TABLE].",
    citations: [
      {
        source: "NovaCore_Multimodal_Company_Report_2026.pdf",
        page_number: 4,
        modality: "table",
        excerpt: "| Europe | $31.4M | $41.2M | 31% YoY Growth |",
        confidence_score: 0.94
      }
    ],
    visual_artifacts: [],
    retrieved_chunks_count: 5,
    modality_used: "text_llm",
    model: "openai/gpt-oss-20b",
    latency: {
      intent_ms: 0.22,
      retrieval_ms: 26.8,
      rerank_ms: 24.5,
      generation_ms: 345.2,
      total_ms: 396.72
    }
  },
  "According to the revenue graph, which quarter had the highest revenue?": {
    question: "According to the revenue graph, which quarter had the highest revenue?",
    answer: "According to the quarterly revenue trend graph on Page 3, revenue increased consecutively across all quarters in FY2026, reaching its peak in Q4 at approximately $42.5M [Page 3 | VISUAL]. In comparison, Q1 opened at $28.2M, Q2 reached $31.0M, and Q3 delivered $30.3M before the fourth-quarter acceleration [Page 3 | VISUAL].",
    citations: [
      {
        source: "NovaCore_Multimodal_Company_Report_2026.pdf",
        page_number: 3,
        modality: "visual",
        excerpt: "Figure 1: Quarterly Revenue Trend FY2026 showing consecutive quarterly expansion peaking in Q4 at $42.5M.",
        confidence_score: 0.98
      }
    ],
    visual_artifacts: [
      {
        asset_id: "vis-p3-rev",
        page_number: 3,
        image_path: "novacore_extracted_images/page_3_image_1.png",
        data_uri: "", // Will be filled dynamically or loaded
        summary: "Line graph plotting quarterly revenue in millions USD for FY2026. Shows steady climb from Q1 ($28.2M) through Q4 ($42.5M)."
      }
    ],
    retrieved_chunks_count: 5,
    modality_used: "multimodal_vlm",
    model: "qwen/qwen3.8-27b",
    latency: {
      intent_ms: 0.25,
      retrieval_ms: 31.4,
      rerank_ms: 28.7,
      generation_ms: 685.0,
      total_ms: 745.35
    }
  }
};
