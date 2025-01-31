# arXiv CVPR RAG System

A Retrieval-Augmented Generation (RAG) system for interacting with CVPR research papers from arXiv. Built with LangChain and Ollama.

## Features

- Automated paper retrieval from arXiv CVPR category
- Document processing and embedding with FAISS vector store
- Intelligent search and retrieval using semantic similarity
- Custom tools for research paper analysis:
  - Paper summarization with key contributions
  - Methodology comparison between papers
  - Technical concept explanation at different expertise levels
- Local LLM integration via Ollama 

## Requirements

- Python 3.9+
- Ollama service running locally
- Sufficient disk space for paper storage and embeddings
- RAM for running embeddings and LLM (recommended: 8GB+)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/arxiv-cvpr-rag.git
cd arxiv-cvpr-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Ollama:
```bash
# Start Ollama service
ollama serve

# In separate terminal, pull model:
ollama pull llama2
```

## Configuration

Create `config.yaml` in the project root:

```yaml
# Storage directories
data_dir: "./data/arxiv_papers"
model_dir: "./models"
vector_db_path: "./vector_store/cvpr_db"

# ArXiv query parameters
arxiv_query: "cat:cs.CV AND submittedDate:[2023 TO 2024]"

# Text processing parameters
chunk_size: 512
chunk_overlap: 100
embedding_model: "all-MiniLM-L6-v2"
```

## Initial Setup

Download papers and create vector store:

```bash
python main.py --setup
```

## Usage

1. Start the interactive agent:
```bash
python main.py
```

2. Example queries:
```
> "Summarize the key contributions of 'Mask R-CNN'"
> "Compare methodologies between Paper A and Paper B"
> "Explain transformer architectures for computer vision to a beginner"
```

## Project Structure

```
.
├── main.py             # Main application logic
├── config.yaml         # Configuration file
├── requirements.txt    # Dependency list
├── data/              # Downloaded arXiv papers
│   └── arxiv_papers/  # PDF storage
├── models/            # Local model storage
└── vector_store/      # FAISS index files
    └── cvpr_db/      # Vector embeddings
```

