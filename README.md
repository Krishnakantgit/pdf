---
title: Pdf Assistant
emoji: 🔥
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


An AI-powered PDF Assistant that allows users to upload PDF documents and ask questions about the content using natural language. Built using Streamlit for the front-end and hosted on Hugging Face Spaces.

👉 Live Demo: https://gamerclub-pdf-assistant-33329c2.hf.space/

🚀 Features
📄 Upload and process any PDF document

❓ Ask natural language questions based on the content

⚡ Fast, interactive, and user-friendly interface

💬 AI generates relevant answers from PDF content

🌐 No installation required – works directly in your browser

🛠️ Tech Stack
Python

Streamlit – Interactive UI

Hugging Face Spaces – App hosting

PyMuPDF / pdfplumber – PDF text extraction

LangChain / Transformers / OpenAI / LLM (optional) – For semantic question answering

GitHub Actions – Continuous Deployment (CI/CD)

📁 Project Structure

pdf-assistant/
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
├── utils.py            # Helper functions for PDF parsing and QA (optional)
🔧 Installation



🧠 How It Works
The PDF file is uploaded and processed using PyMuPDF or pdfplumber.

Extracted text is chunked and indexed.

A natural language query is run against the content using simple keyword search or LLM-powered semantic search.

Relevant context is retrieved and passed to the model for answer generation.
