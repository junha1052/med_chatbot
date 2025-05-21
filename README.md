Agent-Driven Pipeline

Routes user queries intelligently between local and remote knowledge sources.

Ensures fast, context-aware responses for common medication questions.

Fallback to Vector-Based Retrieval

Pre-indexes reference documents (e.g., drug monographs, guidelines) in a Chroma vector store.

Uses Google Generative AI embeddings to find the most relevant passages when local data is insufficient.

Google Generative AI (Gemini) Integration

Leverages Gemini for both embedding lookups and chat completions.

Delivers high-quality, natural language responses enriched by up-to-date external content.

Conditional Retrieval Optimization

Minimizes embedding API calls by first checking local data coverage.

Falls back only when needed, reducing latency and cost while maintaining answer completeness.
