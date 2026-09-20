from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. LOAD NEW MEDICINE DATABASE
# ============================================================

print("Loading medicine vector database...")

vector_db = FAISS.load_local(
    "data/faiss_medicine_records",
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================================
# 3. TEST QUESTIONS
# ============================================================

test_questions = [
    "What is the manufacturer of Augmentin 625 Duo Tablet?",
    "What is the composition of Augmentin 625 Duo Tablet?",
    "What is the pack size of Augmentin 625 Duo Tablet?",
    "What is the price of Augmentin 625 Duo Tablet?",
    "Is Augmentin 625 Duo Tablet discontinued?"
]


# ============================================================
# 4. RETRIEVAL TEST
# ============================================================

print("\n========================================")
print("MEDICINE RETRIEVAL TEST")
print("========================================")


for i, question in enumerate(test_questions, 1):

    print("\n----------------------------------------")
    print("Test", i)
    print("Question:", question)

    results = vector_db.similarity_search(
        question,
        k=3
    )

    for rank, result in enumerate(results, 1):

        print(f"\n----- Result {rank} -----")
        print(result.page_content)


print("\n========================================")
print("TEST COMPLETED")
print("========================================")