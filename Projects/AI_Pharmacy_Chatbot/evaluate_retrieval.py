import warnings
import os

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# TEST QUESTIONS
# ============================================================

tests = [
    {
        "question": "What is the manufacturer of Augmentin 625 Duo Tablet?",
        "expected": "Augmentin 625 Duo Tablet"
    },
    {
        "question": "What is the composition of Augmentin 625 Duo Tablet?",
        "expected": "Augmentin 625 Duo Tablet"
    },
    {
        "question": "What is the price of Augmentin 625 Duo Tablet?",
        "expected": "Augmentin 625 Duo Tablet"
    },
    {
        "question": "Is Augmentin 625 Duo Tablet discontinued?",
        "expected": "Augmentin 625 Duo Tablet"
    },
    {
        "question": "What is the pack size of Augmentin 625 Duo Tablet?",
        "expected": "Augmentin 625 Duo Tablet"
    }
]


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD FAISS DATABASE
# ============================================================

print("Loading FAISS vector database...")

vector_db = FAISS.load_local(
    "data/faiss_medicine_records",
    embeddings,
    allow_dangerous_deserialization=True
)


print()
print("========================================")
print("RAG RETRIEVAL EVALUATION")
print("========================================")
print()


# ============================================================
# RUN TESTS
# ============================================================

passed = 0
failed = 0


for index, test in enumerate(tests, start=1):

    print(f"Test {index}/{len(tests)}")

    print(
        "Question:",
        test["question"]
    )


    # Retrieve top 3 documents
    results = vector_db.similarity_search(
        test["question"],
        k=3
    )


    expected = test["expected"].lower()

    found = False


    # Check whether expected medicine
    # appears in the retrieved documents

    for result in results:

        document = result.page_content.lower()

        if expected in document:

            found = True

            break


    if found:

        print(
            "Result: PASS"
        )

        passed += 1

    else:

        print(
            "Result: FAIL"
        )

        failed += 1


    print(
        "----------------------------------------"
    )


# ============================================================
# FINAL RESULT
# ============================================================

total = len(tests)

accuracy = (
    passed / total
) * 100


print()

print(
    "========================================"
)

print(
    "FINAL RETRIEVAL RESULT"
)

print(
    "========================================"
)

print(
    f"Retrieval Accuracy : {accuracy:.2f}%"
)

print(
    f"Passed             : {passed}/{total}"
)

print(
    f"Failed             : {failed}/{total}"
)

print(
    "========================================"
)
