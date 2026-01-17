from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# -----------------------------
# 1. Load PDF
# -----------------------------
loader = PyPDFLoader("doc/JAVID ALI.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# -----------------------------
# 2. Chunking
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "•", "-", " "]
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# -----------------------------
# 3. NVIDIA Embeddings
# -----------------------------
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key="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"
)

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
