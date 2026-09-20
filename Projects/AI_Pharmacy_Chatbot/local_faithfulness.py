import re
import warnings

warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


# ============================================================
# 1. LOAD MODELS
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS database...")

vector_db = FAISS.load_local(
    "data/faiss_medicine_records",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Loading Qwen...")

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=30,
    do_sample=False
)


# ============================================================
# 2. TEST QUESTIONS
# ============================================================

tests = [
    {
        "question": "What is the manufacturer of Augmentin 625 Duo Tablet?",
        "expected": "Glaxo SmithKline Pharmaceuticals Ltd"
    },
    {
        "question": "What is the composition of Augmentin 625 Duo Tablet?",
        "expected": "Amoxycillin (500mg), Clavulanic Acid (125mg)"
    },
    {
        "question": "What is the price of Augmentin 625 Duo Tablet?",
        "expected": "₹223.42"
    },
    {
        "question": "Is Augmentin 625 Duo Tablet discontinued?",
        "expected": "False"
    }
]


# ============================================================
# 3. NORMALIZE TEXT
# ============================================================

def normalize(text):
    text = text.lower()

    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("₹", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. EVALUATION
# ============================================================

correct = 0
total = len(tests)


for i, test in enumerate(tests, start=1):

    question = test["question"]
    expected = test["expected"]

    print("\n----------------------------------------")
    print("Test:", i)
    print("Question:", question)

    # Retrieve context
    results = vector_db.similarity_search(
        question,
        k=3
    )

    context = results[0].page_content

    # Generate answer
    prompt = f"""
You are a pharmacy database assistant.

Use ONLY the CONTEXT.

Do not invent information.

CONTEXT:
{context}

QUESTION:
{question}

Give a short direct answer.

ANSWER:
"""

    response = llm(
        prompt,
        return_full_text=False
    )

    answer = response[0]["generated_text"].strip()

    print("Expected:", expected)
    print("Generated:", answer)

    # Check whether expected fact appears
    expected_normalized = normalize(expected)
    answer_normalized = normalize(answer)

    if expected_normalized in answer_normalized:

        print("Result: PASS")
        correct += 1

    else:

        print("Result: FAIL")


# ============================================================
# 5. FINAL SCORE
# ============================================================

accuracy = (correct / total) * 100


print("\n========================================")
print("LOCAL FAITHFULNESS EVALUATION")
print("========================================")

print("Correct:", correct)
print("Total:", total)

print(f"Faithfulness Score: {accuracy:.2f}%")

print("========================================")