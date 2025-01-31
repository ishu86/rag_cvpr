# arXiv CVPR RAG System

A Retrieval-Augmented Generation (RAG) system for interacting with CVPR research papers from arXiv. Built with LangChain and Ollama.

## Features
- Paper retrieval from arXiv CVPR category
- Document embedding with FAISS
- Custom tools for:
  - Paper summarization
  - Methodology comparison
  - Technical concept explanation
- Local LLM integration via Ollama

## Requirements
- Python 3.9+
- Ollama service running locally

## Installation

1. Clone repository:
```bash
git clone https://github.com/<your-username>/arxiv-cvpr-rag.git
cd arxiv-cvpr-rag

2. Install dependencies:
pip install -r requirements.txt
