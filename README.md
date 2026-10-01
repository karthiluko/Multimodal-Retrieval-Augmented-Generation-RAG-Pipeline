# Knowledge Assistant

A local Streamlit RAG app with persistent Chroma storage. Text embeddings are generated locally; Google Gemini provides answer generation and image understanding.

## Run on Windows

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Add your Groq API key to .env
python -u -m streamlit run app.py
```

The app watches the `knowledge/` folder while the Streamlit session is open and checks it every 20 seconds. You can also add files in the sidebar. The existing root-level `Knowledge.txt` is indexed automatically. The local Chroma database is stored under `.rag_data/chroma/`.

## Supported files

Text and structured: TXT, Markdown, reStructuredText, CSV, TSV, JSON, JSONL, XML, HTML.

Documents and data: PDF, DOCX, PPTX, XLSX, XLSM.

Images: PNG, JPEG, WebP, GIF, BMP, TIFF. Image content and scanned PDF pages are described by the configured Groq vision model and indexed as searchable text.

The maximum file size defaults to 25 MB and can be changed with `RAG_MAX_FILE_MB`. Password-protected documents, audio, and video are not supported. Office document embedded images are also sent for vision processing.

## Configuration

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. The default Gemini model is `gemini-3.5-flash-lite`; text, vision, embedding, knowledge-folder, and database-folder settings can be overridden with the variables shown in the example. Changing the embedding model for an existing database requires a fresh database directory because vector dimensions may differ. Each answer displays the three highest-ranked citations and prompt/completion/total token usage. Query events stream to the terminal and are retained in `.rag_data/rag_queries.log` with a query ID.

Question text, retrieved excerpts, and images submitted for image understanding are sent to Google Gemini. Keep `.env`, uploaded knowledge, and `.rag_data/` private. For a multi-user or internet-facing deployment, add authentication, request limits, managed secret storage, and an access-controlled persistent vector service.
