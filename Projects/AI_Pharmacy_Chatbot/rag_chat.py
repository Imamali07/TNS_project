import warnings
import os
import re

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# ============================================================
# SECURITY CHECK
# ============================================================

def security_check(question):

    unsafe_patterns = [
        "what medicine should i take",
        "which medicine should i take",
        "what medication should i take",
        "which medication should i take",
        "what drug should i take",
        "which drug should i take",
        "prescribe medicine",
        "prescribe medication",
        "give me a prescription",
        "what dosage should i take",
        "what dose should i take",
        "how much medicine should i take",
        "how much medication should i take",
        "how many tablets should i take",
        "recommend a medicine",
        "recommend medicine",
        "recommend a drug",
        "recommend medication",
        "tell me my medicine",
        "ignore your instructions",
        "ignore previous instructions",
        "ignore all instructions",
        "bypass your instructions",
        "forget your instructions"
    ]

    question_lower = question.lower().strip()

    for pattern in unsafe_patterns:

        if pattern in question_lower:
            return False

    return True


# ============================================================
# FOLLOW-UP SECURITY CHECK
# ============================================================

def followup_security_check(question):

    followup_patterns = [
        "give it",
        "give me",
        "give one",
        "tell me",
        "which one",
        "then what",
        "what should i take",
        "so what should i take",
        "just tell me",
        "please tell me",
        "please give me",
        "name one",
        "name a medicine",
        "give the medicine"
    ]

    question_lower = question.lower().strip()

    for pattern in followup_patterns:

        if pattern in question_lower:
            return False

    return True


# ============================================================
# OUTPUT SAFETY CHECK
# ============================================================

def output_safety_check(answer):

    unsafe_patterns = [
        "you should take",
        "recommended dosage",
        "recommended dose",
        "take 500 mg",
        "take 250 mg",
        "take 100 mg",
        "take 50 mg",
        "take one tablet",
        "take two tablets",
        "take 1 tablet",
        "take 2 tablets",
        "i recommend taking",
        "you can take",
        "you may take"
    ]

    answer_lower = answer.lower()

    for pattern in unsafe_patterns:

        if pattern in answer_lower:
            return False

    return True


# ============================================================
# MEDICINE NAME EXTRACTION
# ============================================================

def extract_medicine_name(question):

    patterns = [
        r"of\s+(.+?)(?:\?|$)",
        r"for\s+(.+?)(?:\?|$)",
        r"^is\s+(.+?)(?:\s+discontinued|\?|$)",
        r"^who\s+manufactures?\s+(.+?)(?:\?|$)",
        r"^tell\s+me\s+about\s+(.+?)(?:\?|$)",
        r"^information\s+about\s+(.+?)(?:\?|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:

            medicine = match.group(1).strip()

            medicine = re.sub(
                r"^(the|medicine|tablet)\s+",
                "",
                medicine,
                flags=re.IGNORECASE
            )

            return medicine.strip()

    return None


# ============================================================
# FIELD EXTRACTION
# ============================================================

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


# ============================================================
# DATABASE MEDICINE NAME
# ============================================================

def extract_database_medicine_name(context):

    match = re.search(
        r"Medicine Name:\s*(.+)",
        context,
        re.IGNORECASE
    )

    if match:

        return match.group(1).strip()

    return None


# ============================================================
# NORMALIZE MEDICINE NAME
# ============================================================

def normalize_medicine_name(name):

    if not name:

        return ""

    name = name.lower().strip()

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    name = name.rstrip(
        "?.!,:"
    )

    return name.strip()


# ============================================================
# GET REQUESTED FIELD
# ============================================================

def get_requested_field(question):

    question_lower = question.lower()

    if "manufacturer" in question_lower:

        return "Manufacturer"

    elif "manufactures" in question_lower:

        return "Manufacturer"

    elif "manufactured by" in question_lower:

        return "Manufacturer"

    elif "price" in question_lower:

        return "Price"

    elif "cost" in question_lower:

        return "Price"

    elif "composition" in question_lower:

        return "Composition"

    elif "ingredients" in question_lower:

        return "Composition"

    elif "pack size" in question_lower:

        return "Pack Size"

    elif "pack" in question_lower and "size" in question_lower:

        return "Pack Size"

    elif "discontinued" in question_lower:

        return "Discontinued"

    elif (
        "medicine type" in question_lower
        or "type of medicine" in question_lower
    ):

        return "Medicine Type"

    return None


# ============================================================
# PRINT ANSWER + SOURCE
# ============================================================

def print_answer(answer):

    print(
        "\nAssistant:",
        answer
    )

    print(
        "Source: A-Z Medicines Dataset of India"
    )

    print(
        "EVALUATION_ANSWER:",
        answer
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD FAISS DATABASE
# ============================================================

print("Loading medicine vector database...")

vector_db = FAISS.load_local(
    "data/faiss_medicine_records",
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================================
# LOAD LANGUAGE MODEL
# ============================================================

print("Loading language model...")

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)


# ============================================================
# CHATBOT START
# ============================================================

print("\n========================================")
print("Pharmacy Chatbot Ready!")
print("Type 'exit' to stop.")
print("========================================")


unsafe_conversation = False


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    question = input("\nYou: ").strip()


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not question:

        print(
            "Assistant: Please enter a question."
        )

        continue


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if question.lower() == "exit":

        print("Goodbye!")

        break


    # --------------------------------------------------------
    # SECURITY CHECK
    # --------------------------------------------------------

    if not security_check(question):

        unsafe_conversation = True

        security_answer = (
            "I can provide general medicine information, "
            "but I cannot recommend a specific medicine, "
            "prescription, or dosage for your personal "
            "condition. Please consult a qualified doctor "
            "or pharmacist."
        )

        print_answer(
            security_answer
        )

        continue


    # --------------------------------------------------------
    # FOLLOW-UP SECURITY
    # --------------------------------------------------------

    if unsafe_conversation:

        if not followup_security_check(question):

            followup_answer = (
                "I cannot provide a personalized medicine "
                "recommendation or dosage. Please consult a "
                "qualified doctor or pharmacist."
            )

            print_answer(
                followup_answer
            )

            continue

        unsafe_conversation = False


    # --------------------------------------------------------
    # EXTRACT MEDICINE NAME
    # --------------------------------------------------------

    medicine_name = extract_medicine_name(
        question
    )


    if medicine_name:

        search_query = (
            medicine_name + " " + question
        )

    else:

        search_query = question


    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    results = vector_db.similarity_search(
        search_query,
        k=20
    )


    if not results:

        unknown_answer = (
            "Information not available in the database."
        )

        print_answer(
            unknown_answer
        )

        continue


    # --------------------------------------------------------
    # FIND EXACT MEDICINE
    # --------------------------------------------------------

    selected_result = None


    if medicine_name:

        requested_medicine = normalize_medicine_name(
            medicine_name
        )


        for result in results:

            database_medicine = (
                extract_database_medicine_name(
                    result.page_content
                )
            )


            if database_medicine:

                database_medicine = (
                    normalize_medicine_name(
                        database_medicine
                    )
                )


                if database_medicine == requested_medicine:

                    selected_result = result

                    break


    # --------------------------------------------------------
    # UNKNOWN MEDICINE
    # --------------------------------------------------------

    if medicine_name and selected_result is None:

        unknown_answer = (
            "Information not available in the database."
        )

        print_answer(
            unknown_answer
        )

        continue


    # --------------------------------------------------------
    # SELECT BEST RESULT
    # --------------------------------------------------------

    if selected_result is None:

        selected_result = results[0]


    context = selected_result.page_content


    selected_medicine = (
        extract_database_medicine_name(
            context
        )
    )


    print(
        "\n[Retrieved medicine:",
        selected_medicine,
        "]"
    )


    # --------------------------------------------------------
    # DIRECT FIELD EXTRACTION
    # --------------------------------------------------------

    requested_field = get_requested_field(
        question
    )


    if requested_field:

        answer = extract_field(
            context,
            requested_field
        )


        if answer is not None:

            if requested_field == "Discontinued":

                if answer.lower() == "false":

                    answer = (
                        "No, this medicine is not "
                        "marked as discontinued in "
                        "the database."
                    )

                elif answer.lower() == "true":

                    answer = (
                        "Yes, this medicine is marked "
                        "as discontinued in the database."
                    )


            print_answer(
                answer
            )

            continue


    # ========================================================
    # QWEN PROMPT
    # ========================================================

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


    # --------------------------------------------------------
    # QWEN GENERATION
    # --------------------------------------------------------

    response = llm(
        prompt,
        max_new_tokens=30,
        do_sample=False,
        return_full_text=False
    )


    answer = response[0][
        "generated_text"
    ].strip()


    # --------------------------------------------------------
    # OUTPUT SAFETY
    # --------------------------------------------------------

    if not output_safety_check(answer):

        safety_answer = (
            "I cannot provide a personalized medicine "
            "recommendation or dosage. Please consult a "
            "qualified doctor or pharmacist."
        )

        print_answer(
            safety_answer
        )

        continue


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print_answer(
        answer
    )