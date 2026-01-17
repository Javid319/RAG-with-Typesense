documents = [
    "Javid is a 16 year old boy",
    "Javid likes to play cricket",
    "Javid has 8 friends in which three of them are below the age of 18"
]
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key="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"
)

from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_texts(documents, embeddings)
query = "What sport does javid play?"
docs = vector_store.similarity_search(query, k=2)

llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key="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",
    temperature=0.7
)
context = "\n".join([doc.page_content for doc in docs])
prompt = f"""
Use the following context to answer the question.
Context:
{context}
Question:
{query}
"""
response = llm.invoke(prompt)
print(response.content)