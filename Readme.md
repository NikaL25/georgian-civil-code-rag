# RAG Agent for Georgian Civil Code

A Retrieval-Augmented Generation (RAG) system that answers legal questions in Georgian based exclusively on the Georgian Civil Code (from the official Matsne PDF). The agent uses ReAct reasoning to search the code, generate precise answers, and always cite sources (articles and pages). It supports conversational memory for follow-up questions and ensures responses are accurate, concise, and in Georgian only.

## Project Overview

This is an Agentic RAG implementation for legal Q&A:

- **Input**: Georgian legal questions (e.g., "რა არის საკუთრება?" — "What is property?").
- **Process**: The ReAct agent retrieves relevant articles from the vectorized Civil Code, reasons step-by-step, and generates a response.
- **Output**: Concise Georgian answer with citations (e.g., "საკუთრებაა... (მუხლი 147, გვერდი 25)" — "Property is... (Article 147, page 25)").
- **Key Features**:
  - Strict adherence to the Civil Code (no hallucinations; "არ ვიცი" if insufficient data).
  - Conversational memory for multi-turn queries.
  - Source citation for legal accuracy.
  - Supports local (FAISS) or cloud (AstraDB) vector storage.

## Technologies Used

- **LangChain**: Core framework for RAG agent, ReAct reasoning, tools, memory, and chains.
- **Groq**: LLM backend (`meta-llama/llama-4-maverick-17b-128e-instruct`) for fast, multilingual generation.
- **Hugging Face**: Embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) for semantic search in Georgian text.
- **FAISS/AstraDB**: Vector database for storing and querying code chunks (local or cloud).
- **PyPDF**: For extracting text from the Matsne PDF.
- **Streamlit**: Web UI for interactive Q&A.
- **Python Libraries**: `dotenv` for config, `re` for text cleaning, `langchain-community` for loaders/splitters.

The system is modular: preprocessing creates the vector store, while the agent handles queries.

## Local Setup

To run locally:

1. **Clone the Repo**:
   git clone https://github.com/NikaL25/georgian-civil-code-rag.git
   cd georgian-civil-code-rag

2. **Create virtual environmet**
   py -3.11 venv venv

venv/scripts/activate

3. **Install Dependencies**:
   pip install -r requirements.txt

(Create `requirements.txt` with: `langchain==0.2.0 langchain-community==0.2.0 langchain-huggingface==0.0.3 langchain-groq==0.1.0 langchain-astradb==0.3.0 streamlit==1.31.0 pypdf==4.2.0 python-dotenv==1.0.0 sentence-transformers==2.2.2`).

<!-- 3. **Download the PDF**:
- Get the official Georgian Civil Code PDF: [Matsne Download](https://www.matsne.gov.ge/ka/document/download/31702/134/ka/pdf).
- Save as `matsne_civil_code.pdf` in the root directory. -->

4. **Configure Environment**:

- Copy `.env.example` to `.env` and fill in your keys:

GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key # Fallback for embeddings if needed
MATSNE_PDF_PATH=./matsne_civil_code.pdf
Optional: AstraDB for cloud storage
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
ASTRA_DB_API_ENDPOINT=your_astra_endpoint
ASTRA_DB_COLLECTION=agentragcoll

- Get keys:
- Groq: [Groq Console](https://console.groq.com/keys).
- OpenAI: [OpenAI Platform](https://platform.openai.com/api-keys) (for fallback).
- AstraDB: [Astra Console](https://astra.datastax.com) (optional; use FAISS for local).

5. **Preprocess and Index Data**:

python preprocess_and_index.py

- This loads the PDF, cleans text, splits into chunks (articles/subchunks), creates embeddings, and saves to FAISS (`./faiss_index`) or AstraDB.
- Output: ~2000 chunks; takes 5-10 minutes on first run.

6. **Run the Agent**:

- **Console Mode** (for testing):

python rag_groq.py

- Enter Georgian questions (e.g., "რა არის საკუთრება?").
- Type "exit" to quit.
- **Web UI** (Streamlit):

streamlit run app_streamlit.py

- Open http://localhost:8501.
- Enter questions and click "უპასუხეთ" (Answer).

## Usage Example

- Question: "რა არის საკუთრება?"
- Response: "საკუთრებაა... [brief explanation]. \n\nწყარო(ები): - მუხლი 147 (გვერდი 25)"

## Live Demo

Try the agent without cloning: [Georgian Civil Code RAG Demo](https://-----.streamlit.app) (deployed on Streamlit Cloud).

## Contributing

- Fork the repo and create a pull request.
- Run tests: `python preprocess_and_index.py` and query in console.
- Issues? Open a GitHub issue.
