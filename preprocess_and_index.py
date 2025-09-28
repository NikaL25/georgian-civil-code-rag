import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb.vectorstores import AstraDBVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

MATSNE_PDF = os.getenv("MATSNE_PDF_PATH", "matsne_civil_code.pdf")

ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "agentragcoll")

article_re = re.compile(r"(მუხლი\s*[0-9]{1,5})", re.IGNORECASE)

assert os.path.exists(MATSNE_PDF), f"PDF not found: {MATSNE_PDF}"

print("Loading PDF...")
loader = PyPDFLoader(MATSNE_PDF)
pages = loader.load()

for i, doc in enumerate(pages):
    text = re.sub(r'\n\s*\n', '\n', doc.page_content.strip())  
    text = re.sub(r'გვერდი \d+', '', text)  
    doc.page_content = text  

print(f"Loaded {len(pages)} pages.")

article_chunks = []

print("Splitting PDF by articles...")
for i, doc in enumerate(pages):
    text = doc.page_content
    parts = article_re.split(text)
    for j in range(1, len(parts), 2):
        article_title = parts[j].strip()
        article_text = parts[j+1].strip() if (j+1) < len(parts) else ""
        if article_text:
            metadata = {
                "article": article_title,
                "page": i + 1,
                "source": f"matsne://matsne_civil_code.pdf#page={i+1}"
            }
            article_chunks.append(Document(page_content=article_text, metadata=metadata))

print(f"Split into {len(article_chunks)} article chunks.")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
final_chunks = []

for doc in article_chunks:
    if len(doc.page_content) > 600:
        sub_chunks = splitter.split_documents([doc])
        final_chunks.extend(sub_chunks)
    else:
        final_chunks.append(doc)

chunks = final_chunks
print(f"Final chunks after splitting long articles: {len(chunks)}")

print("Creating embeddings (HuggingFace paraphrase-multilingual-MiniLM-L12-v2)...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
if ASTRA_TOKEN and ASTRA_ENDPOINT:
    print("Using AstraDB vectorstore...")
    vstore = AstraDBVectorStore(
        api_endpoint=ASTRA_ENDPOINT,
        token=ASTRA_TOKEN,
        collection_name=ASTRA_COLLECTION,
        embedding=emb
    )
    print("Upserting documents to AstraDB (this may take a while)...")
    vstore.add_documents(chunks)
    print("Done — documents upserted to AstraDB collection:", ASTRA_COLLECTION)
else:
    print("AstraDB not configured — using local FAISS fallback.")
    from langchain.vectorstores import FAISS
    faiss_store = FAISS.from_documents(chunks, embedding=emb)
    faiss_store.save_local("./faiss_index")
    print("FAISS index saved to ./faiss_index")

print("Preprocessing and indexing finished.")