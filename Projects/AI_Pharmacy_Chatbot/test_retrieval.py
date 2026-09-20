from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading vector database...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    "data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "Augmentin 625 Duo Tablet"

print("\nSearching for:", query)

results = vector_db.similarity_search(query, k=3)

print("\n========== RESULTS ==========")

for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(result.page_content)

print("\n==============================")