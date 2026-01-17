import typesense
import os
from dotenv import load_dotenv
load_dotenv()

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
    {
      "name": "embedding",
      "type": "float[]",
      "num_dim": 384
    }
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
    return model.encode(text).tolist()




text = load_pdf("doc/st.pdf")
chunks = chunk_text(text)

docs = []
for i, chunk in enumerate(chunks):
    docs.append({
        "id": str(i),
        "text": chunk,
        "embedding": embed(chunk)
    })

client.collections["docs"].documents.import_(docs)


def search(query, k=3):
    query_embedding = embed(query)

    vector_query = (
        f"embedding:([{','.join(map(str, query_embedding))}], k:{k})"
    )

    searches = [{
        "collection": "docs",
        "q": "*",
        "vector_query": vector_query
    }]

    result = client.multi_search.perform(
        {"searches": searches},
        {}
    )

    hits = result["results"][0]["hits"]
    return [hit["document"]["text"] for hit in hits]





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
    temperature=0.5
)

def generate_answer(context, query):
    prompt = build_prompt(context, query)
    response = llm.invoke(prompt)
    return response.content

def rag(query):
    chunks = search(query)
    context = "\n\n".join(chunks)
    return generate_answer(context, query)

if __name__ == "__main__":
    query = input("Query :")
    answer = rag(query)
    print("RESPONSE :\n")
    print(answer)


