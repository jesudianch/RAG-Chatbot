---
title: RAG with MMR + PDF Upload
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🧠 Retrieval-Augmented Generation with MMR and PDF Upload

A powerful RAG (Retrieval-Augmented Generation) system that combines PDF document processing with advanced retrieval techniques and question answering capabilities.

## ✨ Features

- **PDF Document Processing**
  - Upload and process PDF documents
  - Automatic text extraction and chunking
  - Smart content segmentation

- **Advanced Retrieval System**
  - Uses FAISS for efficient vector storage and retrieval
  - Implements Maximal Marginal Relevance (MMR) for diverse and relevant results
  - Leverages MiniLM embeddings for semantic search

- **Question Answering**
  - Powered by FLAN-T5 for accurate responses
  - Context-aware answers based on document content
  - Natural language understanding

## 🛠️ Tech Stack

- **Frameworks & Libraries**
  - LangChain: For building the RAG pipeline
  - HuggingFace: For models and embeddings
  - Gradio: For the web interface
  - FAISS: For vector similarity search

- **Models**
  - MiniLM: For text embeddings
  - FLAN-T5: For question answering

## 🚀 Getting Started

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Running the Application**
   ```bash
   python app.py
   ```

3. **Usage**
   - Upload a PDF document through the interface
   - Wait for the document to be processed
   - Ask questions about the document content
   - Receive accurate, context-aware answers

## 📚 How It Works

1. **Document Processing**
   - PDF is uploaded and processed
   - Text is extracted and split into chunks
   - Chunks are embedded using MiniLM

2. **Vector Storage**
   - Embeddings are stored in FAISS index
   - MMR is used to ensure diverse and relevant retrieval

3. **Question Answering**
   - User questions are processed
   - Relevant context is retrieved using MMR
   - FLAN-T5 generates answers based on the context

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- HuggingFace for the models and tools
- FAISS team for the efficient similarity search
- Gradio for the beautiful interface