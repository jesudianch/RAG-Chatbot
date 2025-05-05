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

This Gradio demo allows you to:

- Upload a PDF document
- Chunk the content and embed using `MiniLM`
- Store and search chunks using FAISS with **Maximal Marginal Relevance (MMR)**
- Answer questions using `FLAN-T5`

> Powered by LangChain + HuggingFace + Gradio + FAISS