from networkx import hits
from torch import mode
import typesense
import os
import time
from dotenv import load_dotenv

from r1 import rerank_hits
from r3 import neural_rerank

load_dotenv()

latency_stats = {}
leaderboard = []

from sentence_transformers import CrossEncoder

# ⭐ R2 Neural Re-ranker
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


from langchain_nvidia_ai_endpoints import ChatNVIDIA
client = typesense.Client({
  "nodes": [{
    "host": "localhost",
    "port": "8108",
    "protocol": "http"
  }],
  "api_key": "xyz",
  "connection_timeout_seconds": 2
}) 



schema = {
  "name": "docs",
  "fields": [
    {"name": "id", "type": "string"},
    {"name": "text", "type": "string"},
    {"name": "page", "type": "int32"},   # metadata
    {"name": "embedding", "type": "float[]", "num_dim": 384}
  ]
}

if "docs" not in [c["name"] for c in client.collections.retrieve()]:
    client.collections.create(schema)



from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    #print("Embeded Query :",model.encode(text).tolist())
    return model.encode(text).tolist()




text = load_pdf("doc/st.pdf")
chunks = chunk_text(text)

docs = []
for i, chunk in enumerate(chunks):
    docs.append({
        "id": str(i),
        "text": chunk,
        "page": i // 2,   # example metadata
        "embedding": embed(chunk)
    })


try:
    result = client.collections["docs"].documents.search({
        "q": "*",
        "query_by": "text",
        "per_page": 1
    })

    if result["found"] == 0:
        print("Indexing documents...")
        client.collections["docs"].documents.import_(docs)

except Exception:
    # collection empty or search fails
    print("Indexing documents...")
    client.collections["docs"].documents.import_(docs)

    
def search(query, k=3, mode="r2"):

    retrieval_start = time.time()

    query_embedding = embed(query)

    vector_query = (
        f"embedding:([{','.join(map(str, query_embedding))}], k:20)"
    )

    # ⭐ HYBRID SEARCH ENABLED
    searches = [{
        "collection": "docs",
        "q": query,              # enables BM25
        "query_by": "text",
        "vector_query": vector_query
    }]

    result = client.multi_search.perform(
        {"searches": searches},
        {}
    )

    hits = result["results"][0]["hits"]

    # ⭐ CUSTOM RERANKING
   # ---------- R1 ----------
    r1_start = time.time()
    r1_hits = rerank_hits(hits)
    latency_stats["r1"] = time.time() - r1_start

# take top candidates for neural rerank
    top_candidates = r1_hits[:20]

# ---------- R2 ----------
    r2_start = time.time()

    if mode == "baseline":
        final_hits = hits

    elif mode == "r1":
        final_hits = rerank_hits(hits)

    elif mode == "r2":
        r1_hits = rerank_hits(hits)
    
    final_hits = neural_rerank(query, r1_hits[:20])
    r2_hits = neural_rerank(query, top_candidates)
    latency_stats["r2"] = time.time() - r2_start

    latency_stats["retrieval"] = time.time() - retrieval_start
    
    # return top-k chunks
    

    return [
    hit["document"]["text"]
    for hit in r2_hits[:k]
]

   

def normalize(values):
    min_v = min(values)
    max_v = max(values)

    if max_v - min_v == 0:
        return [1 for _ in values]

    return [(v - min_v) / (max_v - min_v) for v in values]

def rerank_hits(hits):

    bm25_scores = [hit["text_match"] for hit in hits]

    vector_distances = [
        hit.get("_vector_distance", hit.get("vector_distance"))
        for hit in hits
    ]

    # distance → similarity
    semantic_scores = [1 - d for d in vector_distances]

    # normalize
    bm25_norm = normalize(bm25_scores)
    semantic_norm = normalize(semantic_scores)

    # metadata example (page boost)
    pages = [
        hit["document"].get("page", 0)
        for hit in hits
    ]

    metadata_scores = [
        1.0 if p < 5 else 0.5
        for p in pages
    ]

    reranked = []

    print("\n========== SCORE VISUALIZATION ==========")
    print("Rank | Final | Semantic | BM25 | Meta | Preview")
    print("------------------------------------------------")

    for i, hit in enumerate(hits):

        final_score = (
            0.5 * semantic_norm[i] +
            0.3 * bm25_norm[i] +
            0.2 * metadata_scores[i]
        )

        hit["final_score"] = final_score
        reranked.append(hit)

    # sort using custom score
    reranked.sort(key=lambda x: x["final_score"], reverse=True)

    # ⭐ visualization after sorting
    for rank, hit in enumerate(reranked[:10], start=1):

        preview = hit["document"]["text"][:40].replace("\n", " ")

        print(
            f"{rank:<4} | "
            f"{hit['final_score']:.3f} | "
            f"{semantic_norm[hits.index(hit)]:.3f} | "
            f"{bm25_norm[hits.index(hit)]:.3f} | "
            f"{metadata_scores[hits.index(hit)]:.1f} | "
            f"{preview}..."
        )

    print("==========================================\n")

    return reranked

def neural_rerank(query, hits):

    print("\n========== R2 NEURAL RERANK ==========")

    pairs = [
        (query, hit["document"]["text"])
        for hit in hits
    ]

    # model judges relevance
    scores = reranker_model.predict(pairs)

    for hit, score in zip(hits, scores):
        hit["r2_score"] = float(score)

    # sort by neural relevance
    hits.sort(key=lambda x: x["r2_score"], reverse=True)

    # visualization
    print("Rank | R2 Score | Preview")
    print("-------------------------------------")

    for rank, hit in enumerate(hits[:10], start=1):
        preview = hit["document"]["text"][:40].replace("\n", " ")
        print(f"{rank:<4} | {hit['r2_score']:.3f} | {preview}...")

    print("======================================\n")

    return hits

def build_prompt(context, question):
    return f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2
)

def generate_answer(context, query):

    start_llm = time.time()

    

    prompt = build_prompt(context, query)

    response = llm.invoke(prompt)

    answer = response.content

    latency_stats["llm"] = llm_latency = time.time() - start_llm

    # ⭐ TOKEN TRACKING (R3)
    usage = response.response_metadata.get("token_usage", {})

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    latency_stats["tokens"] = total_tokens

    print("\n========== TOKEN USAGE (R3) ==========")
    print(f"Prompt Tokens:     {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Tokens:      {total_tokens}")

    # Optional cost estimate (example placeholder rate)
    cost_per_1k = 0.002  # adjust if needed
    estimated_cost = (total_tokens / 1000) * cost_per_1k

    print(f"Estimated Cost:    ${estimated_cost:.6f}")
    print("======================================\n")

    return answer

def show_leaderboard():

    print("\n========== RAG LEADERBOARD ==========")
    print("System        Latency   Tokens")
    print("------------------------------------")

    for row in leaderboard:
        print(
            f"{row['system']:<13}"
            f"{row['latency']:.2f}s     "
            f"{row['tokens']}"
        )

    print("=====================================\n")

def rag(query, system_name, mode):
    
    total_start = time.time()

    chunks = search(query, mode=mode) 
    context = "\n\n".join(chunks)

    answer = generate_answer(context, query)

    total_time = time.time() - total_start

    print("\n========== LATENCY BENCHMARK (R4) ==========")
    print(f"Retrieval Time : {latency_stats.get('retrieval',0):.2f}s")
    print(f"R1 Ranking     : {latency_stats.get('r1',0):.2f}s")
    print(f"R2 Reranking   : {latency_stats.get('r2',0):.2f}s")
    print(f"LLM Generation : {latency_stats.get('llm',0):.2f}s")
    print(f"TOTAL TIME     : {total_time:.2f}s")
    print("=============================================\n")

    total_time = time.time() - total_start

    total_tokens = latency_stats.get("tokens", 0)

    leaderboard.append({
    "system": system_name,
    "latency": total_time,
    "tokens": total_tokens
    })
    

    return answer

if __name__ == "__main__":

    query = input("Query: ")

    rag(query, "Baseline", "baseline")
    rag(query, "R1", "r1")
    rag(query, "R1 + R2", "r2")

    show_leaderboard()
