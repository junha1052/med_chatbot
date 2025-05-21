### Features

- **Agent-Driven Pipeline**  
  Routes queries intelligently between local and remote knowledge sources, ensuring fast, context-aware responses for common medication questions.

- **Fallback to Vector-Based Retrieval**  
  - Pre-indexes reference documents (e.g., drug monographs, guidelines) in a Chroma vector store  
  - Leverages Google Generative AI embeddings to fetch the most relevant passages when local data is insufficient

- **Google Generative AI (Gemini) Integration**  
  Uses Gemini for both embedding lookups **and** chat completions, delivering high-quality, up-to-date natural language answers.

- **Conditional Retrieval Optimization**  
  Checks local data coverage first to minimize embedding API calls; falls back only when needed, reducing latency and cost while preserving answer completeness.
