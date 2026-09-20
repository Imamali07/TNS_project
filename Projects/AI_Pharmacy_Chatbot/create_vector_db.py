import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. LOAD TEXT DATA
# ============================================================

print("Loading medicine text data...")

loader = TextLoader(
    "data/cleaned_medicines.txt",
    encoding="utf-8"
)

documents = loader.load()

text = documents[0].page_content


# ============================================================
# 2. SPLIT BY MEDICINE RECORD
# ============================================================

print("Splitting into complete medicine records...")

records = text.split(
    "MEDICINE INFORMATION"
)


medicine_documents = []


for record in records:

    record = record.strip()

    if not record:
        continue

    # Add the heading back
    record = "MEDICINE INFORMATION\n\n" + record

    medicine_documents.append(
        Document(page_content=record)
    )


print(
    "Total medicine records:",
    len(medicine_documents)
)


# ============================================================
# 3. USE ONLY 5,000 RECORDS FOR TESTING
# ============================================================

test_documents = medicine_documents[:5000]

print(
    "Records used for testing:",
    len(test_documents)
)


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 5. CREATE FAISS DATABASE
# ============================================================

print("Creating FAISS vector database...")

vector_db = FAISS.from_documents(
    test_documents,
    embeddings
)


# ============================================================
# 6. SAVE NEW DATABASE
# ============================================================

output_path = "data/faiss_medicine_records"

print("Saving vector database...")

vector_db.save_local(output_path)


# ============================================================
# 7. FINISHED
# ============================================================

print("\n========================================")
print("MEDICINE VECTOR DATABASE CREATED")
print("========================================")

print(
    "Medicine records:",
    len(test_documents)
)

print(
    "Location:",
    output_path
)

print("========================================")