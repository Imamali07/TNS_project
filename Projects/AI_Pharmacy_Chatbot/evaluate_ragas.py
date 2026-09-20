import warnings
import os

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from transformers import pipeline

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline
)

from langchain_community.vectorstores import FAISS

from ragas import evaluate
from ragas.metrics import faithfulness

from ragas.llms import LangchainLLMWrapper

from datasets import Dataset


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. LOAD FAISS VECTOR DATABASE
# ============================================================

print("Loading FAISS vector database...")

vector_db = FAISS.load_local(
    "data/faiss_medicine_records",
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================================
# 3. LOAD QWEN MODEL
# ============================================================

print("Loading Qwen model...")

qwen_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=30,
    do_sample=False
)

llm = HuggingFacePipeline(
    pipeline=qwen_pipeline
)

evaluator_llm = LangchainLLMWrapper(llm)


# ============================================================
# 4. TEST QUESTIONS
# ============================================================

test_questions = [

    "What is the manufacturer of Augmentin 625 Duo Tablet?",

    "What is the composition of Augmentin 625 Duo Tablet?"

]


questions = []
answers = []
contexts = []


# ============================================================
# 5. GENERATE ANSWERS
# ============================================================

for question in test_questions:

    print("\nQuestion:", question)

    results = vector_db.similarity_search(
        question,
        k=3
    )

    if not results:
        print("No context found.")
        continue

    context = results[0].page_content

    prompt = f"""
You are a pharmacy database assistant.

Use ONLY the information in the CONTEXT.

Do not invent information.

CONTEXT:

{context}

QUESTION:

{question}

Give a short direct answer.

ANSWER:
"""

    response = qwen_pipeline(
        prompt,
        return_full_text=False
    )

    answer = response[0]["generated_text"].strip()

    print("Answer:", answer)

    questions.append(question)
    answers.append(answer)
    contexts.append([context])


# ============================================================
# 6. CREATE RAGAS DATASET
# ============================================================

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
)


# ============================================================
# 7. RUN RAGAS
# ============================================================

print("\n========================================")
print("Running RAGAS Faithfulness evaluation...")
print("========================================")

result = evaluate(
    dataset,
    metrics=[
        faithfulness
    ],
    llm=evaluator_llm,
    embeddings=embeddings
)


# ============================================================
# 8. DISPLAY RESULT
# ============================================================

print("\n========================================")
print("RAGAS RESULT")
print("========================================")

print(result)