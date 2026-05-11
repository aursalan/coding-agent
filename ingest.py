import os
import pickle
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# 1. Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./bindu_vector_store")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
collection = chroma_client.get_or_create_collection(name="bindu_codebase", embedding_function=sentence_transformer_ef)

# 2. Define Splitters & Blacklists
python_splitter = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=1000, chunk_overlap=200)
ts_splitter = RecursiveCharacterTextSplitter.from_language(Language.TS, chunk_size=1000, chunk_overlap=200)
md_splitter = RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, chunk_size=1000, chunk_overlap=200)
html_splitter = RecursiveCharacterTextSplitter.from_language(Language.HTML, chunk_size=1000, chunk_overlap=200)
config_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

IGNORE_EXTENSIONS = {'.lock', '.png', '.jpeg', '.svg', '.ttf', '.mp3', '.pem', '.pyc', '.log', '.baseline'}
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'dist', 'build', '.pytest_cache'}

# Tokenization helper for BM25
def tokenize(text: str):
    return text.lower().split()

def ingest_repository(repo_path="."):
    documents = []
    metadatas = []
    ids = []
    
    # BM25 specific structures
    bm25_corpus = []
    bm25_docs = [] # To store the actual content mapped to the BM25 index
    doc_id_counter = 0

    print("Chunking repository for Chroma & BM25...")
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            filepath = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            ext = ext.lower()
            
            if ext in IGNORE_EXTENSIONS:
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                if not content.strip():
                    continue

                if ext == '.py' or ext == '.pyi':
                    chunks = python_splitter.create_documents([content])
                    lang_type = "python"
                elif ext in ['.ts', '.js', '.cjs']:
                    chunks = ts_splitter.create_documents([content])
                    lang_type = "typescript"
                elif ext == '.md':
                    chunks = md_splitter.create_documents([content])
                    lang_type = "markdown"
                elif ext in ['.svelte', '.html']:
                    chunks = html_splitter.create_documents([content])
                    lang_type = "html/svelte"
                else:
                    chunks = config_splitter.create_documents([content])
                    lang_type = "config/other"

                for chunk in chunks:
                    # Append to Chroma lists
                    documents.append(chunk.page_content)
                    metadatas.append({"filepath": filepath, "language": lang_type})
                    ids.append(f"doc_{doc_id_counter}")
                    
                    # Append to BM25 lists
                    bm25_corpus.append(tokenize(chunk.page_content))
                    bm25_docs.append({
                        "id": f"doc_{doc_id_counter}",
                        "filepath": filepath,
                        "content": chunk.page_content
                    })
                    
                    doc_id_counter += 1

            except Exception as e:
                print(f"Failed to read {filepath}: {e}")

    # 3. Insert into ChromaDB
    print(f"\nInserting {len(documents)} chunks into ChromaDB...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        print(f"Embedding and inserting chunks {i} to {end_idx} of {len(documents)}...")
        
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

    # 4. Build and Save BM25 Index
    print("Building BM25 Index...")
    bm25_index = BM25Okapi(bm25_corpus)
    
    with open("bindu_bm25_index.pkl", "wb") as f:
        pickle.dump({"index": bm25_index, "docs": bm25_docs}, f)

    print("✅ RAG ingestion complete! (ChromaDB + BM25)")

if __name__ == "__main__":
    ingest_repository()