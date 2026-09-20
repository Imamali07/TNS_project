import warnings
import os
import re

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# ========================================
# LOAD EMBEDDING MODEL
# ========================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ========================================
# LOAD VECTOR DATABASE
# ========================================

print("Loading medicine vector database...")

vector_db = FAISS.load_local(
    "data/faiss_medicine_records",
    embeddings,
    allow_dangerous_deserialization=True
)


# ========================================
# LOAD QWEN MODEL
# ========================================

print("Loading Qwen language model...")

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)


# ========================================
# TEST QUESTIONS
# ========================================

test_cases = [
    {
        "question": "What is the manufacturer of Augmentin 625 Duo Tablet?",
        "medicine": "Augmentin 625 Duo Tablet",
        "field": "Manufacturer"
    },
    {
        "question": "What is the composition of Augmentin 625 Duo Tablet?",
        "medicine": "Augmentin 625 Duo Tablet",
        "field": "Composition"
    },
    {
        "question": "What is the price of Augmentin 625 Duo Tablet?",
        "medicine": "Augmentin 625 Duo Tablet",
        "field": "Price"
    },
    {
        "question": "What is the pack size of Augmentin 625 Duo Tablet?",
        "medicine": "Augmentin 625 Duo Tablet",
        "field": "Pack Size"
    },
    {
        "question": "Is Augmentin 625 Duo Tablet discontinued?",
        "medicine": "Augmentin 625 Duo Tablet",
        "field": "Discontinued"
    },
    {
        "question": "What is the manufacturer of Azithral 500 Tablet?",
        "medicine": "Azithral 500 Tablet",
        "field": "Manufacturer"
    },
    {
        "question": "What is the composition of Azithral 500 Tablet?",
        "medicine": "Azithral 500 Tablet",
        "field": "Composition"
    },
    {
        "question": "What is the pack size of Azithral 500 Tablet?",
        "medicine": "Azithral 500 Tablet",
        "field": "Pack Size"
    },
    {
        "question": "What is the price of Azithral 500 Tablet?",
        "medicine": "Azithral 500 Tablet",
        "field": "Price"
    },
    {
        "question": "Is Azithral 500 Tablet discontinued?",
        "medicine": "Azithral 500 Tablet",
        "field": "Discontinued"
    }
]


# ========================================
# EXTRACT FIELD
# ========================================

def extract_field(context, field_name):

    pattern = rf"{re.escape(field_name)}:\s*(.+)"

    match = re.search(
        pattern,
        context,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return None


# ========================================
# FIND EXACT MEDICINE
# ========================================

def find_medicine(question, expected_medicine):

    search_query = expected_medicine + " " + question

    results = vector_db.similarity_search(
        search_query,
        k=20
    )

    for result in results:

        context = result.page_content

        match = re.search(
            r"Medicine Name:\s*(.+)",
            context,
            re.IGNORECASE
        )

        if match:

            medicine_name = match.group(1).strip()

            if medicine_name.lower() == expected_medicine.lower():

                return context

    return None


# ========================================
# GENERATE ANSWER USING QWEN
# ========================================

def generate_answer(context, question):

    prompt = f"""
You are a pharmacy database assistant.

Use ONLY the information in the CONTEXT.

Answer the QUESTION briefly and directly.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. Do not recommend medicines.
4. Do not prescribe medicines.
5. Do not provide dosage instructions.
6. Do not give personalized medical advice.
7. If the information is missing, say:
Information not available in the database.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    response = llm(
        prompt,
        max_new_tokens=30,
        do_sample=False,
        return_full_text=False
    )

    answer = response[0]["generated_text"].strip()

    return answer


# ========================================
# NORMALIZE ANSWER
# ========================================

def normalize_answer(answer):

    answer = answer.lower().strip()

    # Remove common punctuation
    answer = answer.replace(".", "")
    answer = answer.replace(",", "")
    answer = answer.replace(":", "")

    return answer


# ========================================
# START EVALUATION
# ========================================

print("\n========================================")
print("ACTUAL QWEN ANSWER EVALUATION")
print("========================================")


correct = 0
total = len(test_cases)


# ========================================
# RUN TESTS
# ========================================

for i, test in enumerate(test_cases, 1):

    question = test["question"]
    medicine = test["medicine"]
    field = test["field"]

    print("\n----------------------------------------")
    print("Test:", i)
    print("Question:", question)

    # ------------------------------------
    # Retrieve medicine
    # ------------------------------------

    context = find_medicine(
        question,
        medicine
    )

    if context is None:

        print("Result: FAIL")
        print("Reason: Medicine not found")

        continue


    # ------------------------------------
    # Get expected answer
    # ------------------------------------

    expected_answer = extract_field(
        context,
        field
    )

    if expected_answer is None:

        print("Result: FAIL")
        print("Reason: Field not found")

        continue


    print("Expected Answer:", expected_answer)


    # ------------------------------------
    # Generate actual Qwen answer
    # ------------------------------------

    chatbot_answer = generate_answer(
        context,
        question
    )

    print("Qwen Answer:", chatbot_answer)


    # ------------------------------------
    # Compare
    # ------------------------------------

    expected_normalized = normalize_answer(
        expected_answer
    )

    chatbot_normalized = normalize_answer(
        chatbot_answer
    )


    if expected_normalized in chatbot_normalized:

        print("Result: PASS")

        correct += 1

    else:

        print("Result: FAIL")


# ========================================
# FINAL RESULT
# ========================================

accuracy = (correct / total) * 100


print("\n========================================")
print("FINAL QWEN ANSWER EVALUATION")
print("========================================")

print("Correct answers:", correct)
print("Total questions:", total)

print(f"Answer Accuracy: {accuracy:.2f}%")

print("========================================")