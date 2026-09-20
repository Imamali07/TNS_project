import warnings
import os
import re

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================
# IMPORTS
# ============================================

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================
# LOAD EMBEDDING MODEL
# ============================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================
# LOAD VECTOR DATABASE
# ============================================

print("Loading medicine vector database...")

vector_db = FAISS.load_local(
    "data/faiss_medicine_records",
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================
# TEST CASES
# ============================================

test_cases = [

    {
        "question":
        "What is the manufacturer of Augmentin 625 Duo Tablet?",
        "medicine":
        "Augmentin 625 Duo Tablet"
    },

    {
        "question":
        "What is the composition of Augmentin 625 Duo Tablet?",
        "medicine":
        "Augmentin 625 Duo Tablet"
    },

    {
        "question":
        "What is the pack size of Augmentin 625 Duo Tablet?",
        "medicine":
        "Augmentin 625 Duo Tablet"
    },

    {
        "question":
        "What is the price of Augmentin 625 Duo Tablet?",
        "medicine":
        "Augmentin 625 Duo Tablet"
    },

    {
        "question":
        "Is Augmentin 625 Duo Tablet discontinued?",
        "medicine":
        "Augmentin 625 Duo Tablet"
    },

    {
        "question":
        "What is the manufacturer of Azithral 500 Tablet?",
        "medicine":
        "Azithral 500 Tablet"
    },

    {
        "question":
        "What is the composition of Azithral 500 Tablet?",
        "medicine":
        "Azithral 500 Tablet"
    },

    {
        "question":
        "What is the pack size of Azithral 500 Tablet?",
        "medicine":
        "Azithral 500 Tablet"
    },

    {
        "question":
        "What is the price of Azithral 500 Tablet?",
        "medicine":
        "Azithral 500 Tablet"
    },

    {
        "question":
        "Is Azithral 500 Tablet discontinued?",
        "medicine":
        "Azithral 500 Tablet"
    },

    {
        "question":
        "What is the manufacturer of Azee 500 Tablet?",
        "medicine":
        "Azee 500 Tablet"
    },

    {
        "question":
        "What is the composition of Azee 500 Tablet?",
        "medicine":
        "Azee 500 Tablet"
    },

    {
        "question":
        "What is the price of Azee 500 Tablet?",
        "medicine":
        "Azee 500 Tablet"
    },

    {
        "question":
        "What is the manufacturer of Augpen LB 625 Tablet?",
        "medicine":
        "Augpen LB 625 Tablet"
    },

    {
        "question":
        "What is the price of Augpen LB 625 Tablet?",
        "medicine":
        "Augpen LB 625 Tablet"
    },

    {
        "question":
        "What is the composition of Augwin-LB Tablet?",
        "medicine":
        "Augwin-LB Tablet"
    },

    {
        "question":
        "What is the manufacturer of Augwin-LB Tablet?",
        "medicine":
        "Augwin-LB Tablet"
    },

    {
        "question":
        "What is the price of Augwin-LB Tablet?",
        "medicine":
        "Augwin-LB Tablet"
    },

    {
        "question":
        "What is the pack size of Augwin-LB Tablet?",
        "medicine":
        "Augwin-LB Tablet"
    },

    {
        "question":
        "Is Augwin-LB Tablet discontinued?",
        "medicine":
        "Augwin-LB Tablet"
    }
]


# ============================================
# EXTRACT MEDICINE NAME
# ============================================

def extract_medicine_name(question):

    patterns = [
        r"of\s+(.+?)(?:\?|$)",
        r"for\s+(.+?)(?:\?|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:

            return match.group(1).strip()

    return None


# ============================================
# EVALUATION
# ============================================

print("\n========================================")
print("RAG RETRIEVAL EVALUATION")
print("========================================")


correct = 0
total = len(test_cases)


for i, test in enumerate(test_cases, 1):

    question = test["question"]

    expected_medicine = test["medicine"]


    print("\n----------------------------------------")

    print("Test", i)

    print("Question:", question)


    # ========================================
    # EXTRACT MEDICINE NAME
    # ========================================

    medicine_name = extract_medicine_name(
        question
    )


    # ========================================
    # CREATE SEARCH QUERY
    # ========================================

    if medicine_name:

        search_query = (
            medicine_name
            + " "
            + question
        )

    else:

        search_query = question


    # ========================================
    # RETRIEVE TOP 20
    # ========================================

    results = vector_db.similarity_search(
        search_query,
        k=20
    )


    found = False


    # ========================================
    # CHECK RESULTS
    # ========================================

    for rank, result in enumerate(
        results,
        1
    ):

        context = result.page_content


        match = re.search(
            r"Medicine Name:\s*(.+)",
            context,
            re.IGNORECASE
        )


        if match:

            medicine = (
                match.group(1)
                .strip()
            )

        else:

            medicine = "Medicine name not found"


        print(
            f"Result {rank}: {medicine}"
        )


        # ====================================
        # EXACT MEDICINE MATCH
        # ====================================

        if (
            medicine.lower()
            == expected_medicine.lower()
        ):

            found = True


    # ========================================
    # TEST RESULT
    # ========================================

    if found:

        print("Result: PASS")

        correct += 1

    else:

        print("Result: FAIL")


# ============================================
# FINAL SCORE
# ============================================

accuracy = (
    correct / total
) * 100


print("\n========================================")

print("FINAL EVALUATION RESULT")

print("========================================")

print(
    "Correct retrievals:",
    correct
)

print(
    "Total tests:",
    total
)

print(
    f"Retrieval Accuracy: {accuracy:.2f}%"
)

print("========================================")