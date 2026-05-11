import os
import re
import pickle
import subprocess
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
# ==========================================
# 1. Vector Store & BM25 Initialization
# ==========================================

# Load the Databases 
chroma_client = chromadb.PersistentClient(path="./bindu_vector_store")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
collection = chroma_client.get_collection(name="bindu_codebase", embedding_function=sentence_transformer_ef)

# Update the file path and dictionary keys to match your ingestion script
with open("./bindu_bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["index"]
    bm25_docs_list = bm25_data["docs"]

# Reranker - runs locally, zero API calls
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)

def tokenize_code(text):
    return re.findall(r'\w+', text.lower())


# ==========================================
# 2. Reranker
# ==========================================

def rerank_chunks(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Takes RRF-fused chunks, scores each against the query using a 
    cross-encoder, returns only the top_k most relevant.
    Runs locally — no API calls, no tokens consumed.
    """
    if not chunks:
        return []

    # Cross-encoder scores query against each chunk together (not independently)
    pairs = [(query, chunk["doc"][:512]) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]


# ==========================================
# 3. Search Tool
# ==========================================

def search_codebase(query: str, top_k: int = 3) -> str:
    """
    Searches the Bindu codebase using Hybrid BM25 + Dense Vector RRF Fusion,
    then reranks results with a cross-encoder to return only the most 
    relevant chunks. Keeps LLM context tight and token usage low.
    """

    # --- A. Dense Search (ChromaDB) ---
    chroma_results = collection.query(
        query_texts=[query],
        n_results=top_k * 4  # Cast wide net before reranking
    )
    dense_hits = []
    if chroma_results['documents'] and chroma_results['metadatas']:
        for doc, meta in zip(
            chroma_results['documents'][0], 
            chroma_results['metadatas'][0]
        ):
            dense_hits.append({"doc": doc, "meta": meta})

    # --- B. Sparse Search (BM25) ---
    tokenized_query = tokenize_code(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)), 
        key=lambda i: bm25_scores[i], 
        reverse=True
    )[:top_k * 4]

    sparse_hits = []
    for idx in top_bm25_indices:
        doc_dict = bm25_docs_list[idx]
        sparse_hits.append({
            "doc": doc_dict["content"],
            "meta": {"filepath": doc_dict["filepath"]}
        })

    # --- C. RRF Fusion ---
    k_constant = 60
    rrf_scores = {}
    fused_results = {}

    for rank, hit in enumerate(dense_hits):
        doc_content = hit["doc"]
        rrf_scores[doc_content] = rrf_scores.get(doc_content, 0) + (
            1 / (rank + 1 + k_constant)
        )
        fused_results[doc_content] = hit

    for rank, hit in enumerate(sparse_hits):
        doc_content = hit["doc"]
        rrf_scores[doc_content] = rrf_scores.get(doc_content, 0) + (
            1 / (rank + 1 + k_constant)
        )
        fused_results[doc_content] = hit

    # Get top candidates from RRF (wider pool for reranker to work with)
    sorted_fused = sorted(
        rrf_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k * 2]

    rrf_candidates = [
        fused_results[doc_content] 
        for doc_content, _ in sorted_fused
    ]

    # --- D. Rerank (cross-encoder, local, zero tokens) ---
    reranked = rerank_chunks(query, rrf_candidates, top_k=top_k)

    # --- E. Format output — only what the LLM needs ---
    if not reranked:
        return f"No relevant results found for query: '{query}'"

    output = f"=== Results for: '{query}' ===\n\n"
    for i, hit in enumerate(reranked, 1):
        filepath = hit["meta"].get("filepath", "Unknown")
        # Truncate each chunk to 400 chars max
        snippet = hit['doc'][:400]
        if len(hit['doc']) > 400:
            snippet += "\n...[truncated, use read_file for full content]"
        output += f"[{i}] FILE: {filepath}\n```\n{snippet}\n```\n\n"
    
    return output


# ==========================================
# 3. File Execution & Validation Tools
# ==========================================

def run_tests(target: str = "") -> str:
    """Executes the pytest suite on a specific file or directory."""
    cmd = ["uv", "run", "pytest", "-v", "--tb=short"]
    if target:
        cmd.append(target)
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            encoding="utf-8",
            errors="replace"
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        return stdout + ("\n--- STDERR ---\n" + stderr if stderr.strip() else "")
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out after 30 seconds."
    except Exception as e:
        return f"System Error running tests: {e}"

def run_all_checks(target: str = "") -> str:
    """Runs the linter, type checker, and test suite. Returns the combined output."""
    output_log = ""

    def run_cmd(cmd: list[str]) -> tuple[int, str]:
        """Helper that handles encoding safely on Windows."""
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                encoding="utf-8",
                errors="replace"  # replaces undecodable bytes with ? instead of crashing
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return result.returncode, stdout + stderr
        except subprocess.TimeoutExpired:
            return 1, "Error: Command timed out after 60 seconds."
        except Exception as e:
            return 1, f"Error running command: {e}"

    # 1. Linter
    output_log += "--- LINTER (Ruff) ---\n"
    code, out = run_cmd(["uv", "run", "ruff", "check", target or "."])
    output_log += out + "\n"
    lint_failed = code != 0

    # 2. Type Checker
    output_log += "--- TYPE CHECKER (Mypy) ---\n"
    code, out = run_cmd(["uv", "run", "mypy", target or "."])
    output_log += out + "\n"
    type_failed = code != 0

    # 3. Tests
    output_log += "--- TESTS (Pytest) ---\n"
    test_cmd = ["uv", "run", "pytest", "-v", "--tb=short"]
    if target:
        test_cmd.append(target)
    code, out = run_cmd(test_cmd)
    output_log += out + "\n"
    test_failed = code != 0

    if lint_failed or type_failed or test_failed:
        output_log += "\nSTATUS: FAILED CHECKS ENCOUNTERED."
    else:
        output_log += "\nSTATUS: ALL CHECKS PASSED."

    return output_log


# ==========================================
# 4. File Editing Tools
# ==========================================

def read_file(filepath: str) -> str:
    """Reads a file and returns its content with line numbers."""
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' not found."
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        numbered_lines = [f"{i+1:04d} | {line}" for i, line in enumerate(lines)]
        return "".join(numbered_lines)
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(filepath: str, content: str) -> str:
    """Creates a new file or completely overwrites an existing one with new content."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote new code to {filepath}."
    except Exception as e:
        return f"Error writing file: {e}"

def patch_file(filepath: str, find_str: str, replace_str: str) -> str:
    """Replaces a specific string block in a file with new code."""
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' not found."

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if find_str not in content:
            return "Error: The exact `find_str` was not found in the file. Make sure you copied it perfectly, including whitespace."

        updated_content = content.replace(find_str, replace_str, 1)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        return f"Successfully patched {filepath}."
    except Exception as e:
        return f"Error patching file: {e}"


# ==========================================
# 5. Agent Logic Tools
# ==========================================

def handoff_to_coder(filepath: str, exact_instructions: str) -> str:
    """
    Use this tool when you have diagnosed the bug and are ready to write code. 
    Pass the exact file path and highly specific instructions to the Coder model.
    """
    return f"Task delegated to Coder model for file: {filepath}. Instructions: {exact_instructions}"

def delegate_to_coder(instructions: str) -> str:
    """Call this tool ONLY when you have retrieved enough context and are ready to generate code."""
    return f"SUCCESS: Context gathered. Instructions sent to Coder: {instructions}"