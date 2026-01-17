from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import os

# -----------------------------
# Utility: Context Compressor
# -----------------------------
def compress_context(docs, max_words=220):
    text = " ".join(doc.page_content for doc in docs)
    return " ".join(text.split()[:max_words])

# -----------------------------
# 1. Load PDF
# -----------------------------
loader = PyPDFLoader("doc/JAVID ALI.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# -----------------------------
# 2. Embeddings
# -----------------------------
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# -----------------------------
# 3. SAFE PRE-CHUNKING (CRITICAL)
# -----------------------------
pre_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # < 512 tokens
    chunk_overlap=50
)

pre_chunks = pre_splitter.split_documents(documents)

print(f"Pre-chunks created: {len(pre_chunks)}")

# -----------------------------
# 4. SEMANTIC CHUNKING (SAFE NOW)
# -----------------------------
semantic_splitter = SemanticChunker(embeddings=embeddings)

chunks = []
for doc in pre_chunks:
    chunks.extend(semantic_splitter.split_documents([doc]))

print(f"Final semantic chunks: {len(chunks)}")

###API Connection

vector_store = FAISS.from_documents(chunks, embeddings)

# -----------------------------
# 4. User Query
# -----------------------------
q=input()
query = q

docs = vector_store.similarity_search(query, k=5)

context = "\n\n".join(doc.page_content for doc in docs)

# -----------------------------
# 5. NVIDIA LLM
# -----------------------------
llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key="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",
    temperature=0.4
)

prompt = f"""
You are an resume evaluator assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""
response = llm.invoke(prompt)
print("\nANSWER:\n")
print(response.content)

