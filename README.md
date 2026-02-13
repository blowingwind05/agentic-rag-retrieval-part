<div align="center">
  <p align="right">
    <span> 🌎English </span> | <a href="README_CN.md"> 🇨🇳中文 </a>
  </p>
</div>

# IR Papers Retrieval System

An information retrieval system for academic papers, supporting hybrid search (BM25 + Dense Embeddings) and reranking. Built with Python, utilizing Sentence Transformers, BM25, and API integrations.

### Features
- **Hybrid Search**: Combines sparse (BM25) and dense (embedding-based) retrieval for better accuracy.
- **Reranking**: Uses CrossEncoder or API-based reranker to refine results.
- **API Support**: Integrates with external APIs for embeddings and reranking (e.g., DMXAPI).
- **Configurable**: YAML-based configuration for easy setup.
- **Interactive CLI**: Command-line interface for querying.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/blowingwind05/agentic-rag-retrieval-part.git
   cd agentic-rag-retrieval-part
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (If no requirements.txt, install manually: pandas, numpy, torch, sentence-transformers, bm25s, pyyaml, requests, openai)

3. Ensure PyYAML is installed:
   ```bash
   pip install PyYAML
   ```

### Configuration
Edit `config.yaml` to set parameters:
- `data_path`: Path to data directory (e.g., "./").
- `json_file`: JSON file containing papers data (e.g., "ir_papers.json").
- API keys and URLs for embeddings/reranking.
- `top_k_retrieval`, `top_k_final`, `batch_size`, etc.

### Usage
1. Prepare data: Place `ir_papers.json` in the data path.
2. Run the system:
   ```bash
   python ir_papers_retrieval.py
   ```
3. Enter queries interactively. Type 'quit' to exit.

### Dependencies
- Python 3.8+
- pandas, numpy, torch
- sentence-transformers, bm25s
- PyYAML, requests, openai

### License
MIT License.
