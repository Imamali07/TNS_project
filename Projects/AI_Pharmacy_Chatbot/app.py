import warnings
import os
import re
import time
from datetime import datetime

# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Pharmacy Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #212121;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 20px;
        padding-bottom: 120px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2b2b2b;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 14px 12px;
    }


    /* ======================================================
       NEW CHAT BUTTON
       ====================================================== */

    [data-testid="stSidebar"] .stButton > button {

        width: 100%;

        height: 42px;

        background-color: transparent;

        color: #ffffff;

        border: 1px solid #444444;

        border-radius: 8px;

        font-size: 14px;

        transition: 0.2s;
    }


    [data-testid="stSidebar"] .stButton > button:hover {

        background-color: #2a2a2a;

        border-color: #666666;
    }


    /* ======================================================
       CHAT MESSAGES
       ====================================================== */

    [data-testid="stChatMessage"] {

        background-color: transparent;

        padding-top: 14px;

        padding-bottom: 14px;
    }


    [data-testid="stChatMessage"] p {

        color: #ececec;

        font-size: 15px;

        line-height: 1.7;
    }


    /* ======================================================
       SOURCE
       ====================================================== */

    .source-text {

        color: #777777;

        font-size: 12px;

        margin-top: 8px;
    }


    /* ======================================================
       CHAT INPUT
       ====================================================== */

    [data-testid="stChatInput"] {

        bottom: 20px;
    }


    [data-testid="stChatInput"] > div {

        background-color: #2f2f2f !important;

        border: 1px solid #4a4a4a !important;

        border-radius: 16px !important;
    }


    [data-testid="stChatInput"] textarea {

        color: #ffffff !important;

        font-size: 15px !important;
    }


    [data-testid="stChatInput"] textarea::placeholder {

        color: #999999 !important;
    }


    /* ======================================================
       WELCOME SCREEN
       ====================================================== */

    .welcome {

        min-height: 65vh;

        display: flex;

        flex-direction: column;

        justify-content: center;

        align-items: center;

        text-align: center;
    }


    .logo-circle {

        width: 64px;

        height: 64px;

        border-radius: 50%;

        background-color: #2f2f2f;

        border: 1px solid #3d3d3d;

        display: flex;

        align-items: center;

        justify-content: center;

        font-size: 30px;

        margin-bottom: 22px;
    }


    .welcome-title {

        color: #ffffff;

        font-size: 30px;

        font-weight: 600;

        margin-bottom: 8px;
    }


    .welcome-subtitle {

        color: #a0a0a0;

        font-size: 17px;

        font-weight: 400;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 700px) {

        .block-container {

            padding-left: 15px;

            padding-right: 15px;
        }


        .welcome-title {

            font-size: 25px;
        }


        .welcome-subtitle {

            font-size: 15px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# EXTRACT MEDICINE NAME
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
# EXTRACT DATABASE FIELD
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
# EXTRACT DATABASE MEDICINE NAME
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
# DETECT REQUESTED FIELD
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


    elif (
        "medicine type" in question_lower
        or "type of medicine" in question_lower
    ):

        return "Medicine Type"


    elif "discontinued" in question_lower:

        return "Discontinued"


    return None


# ============================================================
# LOAD RAG SYSTEM
# ============================================================

@st.cache_resource
def load_rag_system():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vector_db = FAISS.load_local(

        "data/faiss_medicine_records",

        embeddings,

        allow_dangerous_deserialization=True
    )


    llm = pipeline(

        "text-generation",

        model="Qwen/Qwen2.5-0.5B-Instruct"
    )


    return vector_db, llm


# ============================================================
# START AI SYSTEM
# ============================================================

with st.spinner(
    "Starting AI Pharmacy Assistant..."
):

    vector_db, llm = load_rag_system()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:


    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            color:#ffffff;
            font-size:19px;
            font-weight:600;
            padding:10px 8px 20px 8px;
        ">
            💊 AI Pharmacy
        </div>
        """
    )


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋  New chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    # --------------------------------------------------------
    # RECENT CHATS TITLE
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            color:#8e8e8e;
            font-size:11px;
            font-weight:600;
            letter-spacing:0.5px;
            margin-top:24px;
            margin-bottom:8px;
            padding-left:8px;
        ">
            RECENT CHATS
        </div>
        """
    )


    # --------------------------------------------------------
    # GET USER QUESTIONS
    # --------------------------------------------------------

    user_questions = [

        message["content"]

        for message in st.session_state.messages

        if message["role"] == "user"
    ]


    # --------------------------------------------------------
    # NO CHATS
    # --------------------------------------------------------

    if not user_questions:

        st.html(
            """
            <div style="
                color:#777777;
                font-size:13px;
                padding:8px;
            ">
                No conversations yet
            </div>
            """
        )


    # --------------------------------------------------------
    # SHOW RECENT CHATS
    # --------------------------------------------------------

    else:

        for question in user_questions[-8:]:

            short_question = question


            if len(short_question) > 42:

                short_question = (
                    short_question[:42]
                    + "..."
                )


            st.html(
                f"""
                <div style="
                    color:#d0d0d0;
                    font-size:13px;
                    padding:9px 10px;
                    border-radius:7px;
                    margin-bottom:2px;
                ">
                    💬 {short_question}
                </div>
                """
            )


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            color:#8e8e8e;
            font-size:11px;
            font-weight:600;
            letter-spacing:0.5px;
            margin-top:24px;
            margin-bottom:8px;
            padding-left:8px;
        ">
            ABOUT
        </div>

        <div style="
            color:#8e8e8e;
            font-size:12px;
            line-height:1.6;
            padding:8px;
        ">
            RAG-based medicine information
            assistant using FAISS and Qwen.
        </div>
        """
    )


    # --------------------------------------------------------
    # SAFETY INFORMATION
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            color:#8e8e8e;
            font-size:12px;
            line-height:1.6;
            padding:20px 8px 8px 8px;
        ">

            <div style="
                color:#f0b84b;
            ">
                ⚠️ Educational information only.
            </div>

            <br>

            Consult a qualified doctor or
            pharmacist for medical advice.

        </div>
        """
    )


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:


    # --------------------------------------------------------
    # CURRENT TIME
    # --------------------------------------------------------

    current_hour = datetime.now().hour


    if current_hour < 12:

        greeting = "Good morning! 👋"


    elif current_hour < 17:

        greeting = "Good afternoon! 👋"


    else:

        greeting = "Good evening! 👋"


    # --------------------------------------------------------
    # WELCOME
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="welcome">

            <div class="logo-circle">
                💊
            </div>

            <div class="welcome-title">
                {greeting}
            </div>

            <div class="welcome-subtitle">
                What can I help you with today?
            </div>

        </div>
        """
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:


    if message["role"] == "user":

        avatar = "👤"

    else:

        avatar = "💊"


    with st.chat_message(

        message["role"],

        avatar=avatar
    ):


        st.markdown(
            message["content"]
        )


        if "source" in message:

            st.html(
                f"""
                <div class="source-text">
                    Source · {message["source"]}
                </div>
                """
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about a medicine..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ========================================================
    # SHOW USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            question
        )


    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="💊"
    ):


        # ----------------------------------------------------
        # THINKING SPINNER
        # ----------------------------------------------------

        with st.spinner(
            "Thinking..."
        ):


            # =================================================
            # SECURITY CHECK
            # =================================================

            if not security_check(question):


                answer = (
                    "I can provide general medicine "
                    "information, but I cannot recommend "
                    "a specific medicine, prescription, "
                    "or dosage for your personal condition. "
                    "Please consult a qualified doctor "
                    "or pharmacist."
                )


                source = "Security Guardrail"


            else:


                # =================================================
                # IDENTIFY MEDICINE
                # =================================================

                medicine_name = (
                    extract_medicine_name(
                        question
                    )
                )


                # =================================================
                # SEARCH QUERY
                # =================================================

                if medicine_name:

                    search_query = (

                        medicine_name
                        + " "
                        + question
                    )

                else:

                    search_query = question


                # =================================================
                # FAISS RETRIEVAL
                # =================================================

                results = (
                    vector_db.similarity_search(
                        search_query,
                        k=20
                    )
                )


                # =================================================
                # NO RESULTS
                # =================================================

                if not results:


                    answer = (
                        "Information not available "
                        "in the database."
                    )


                    source = (
                        "A-Z Medicines Dataset of India"
                    )


                else:


                    selected_result = None


                    # =================================================
                    # EXACT MEDICINE MATCH
                    # =================================================

                    if medicine_name:


                        requested_medicine = (
                            normalize_medicine_name(
                                medicine_name
                            )
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


                                if (
                                    database_medicine
                                    == requested_medicine
                                ):


                                    selected_result = result

                                    break


                        # ---------------------------------------------
                        # MEDICINE NOT FOUND
                        # ---------------------------------------------

                        if selected_result is None:


                            answer = (
                                "Information not available "
                                "in the database."
                            )


                            source = (
                                "A-Z Medicines Dataset of India"
                            )


                    # =================================================
                    # GENERAL QUESTION
                    # =================================================

                    if (
                        selected_result is None
                        and medicine_name is None
                    ):


                        selected_result = results[0]


                    # =================================================
                    # USE SELECTED RECORD
                    # =================================================

                    if selected_result is not None:


                        context = (
                            selected_result.page_content
                        )


                        # =================================================
                        # REQUESTED FIELD
                        # =================================================

                        requested_field = (
                            get_requested_field(
                                question
                            )
                        )


                        # =================================================
                        # DIRECT DATABASE ANSWER
                        # =================================================

                        if requested_field:


                            extracted_answer = (
                                extract_field(
                                    context,
                                    requested_field
                                )
                            )


                            if extracted_answer is not None:


                                answer = (
                                    extracted_answer
                                )


                                # -------------------------------------
                                # DISCONTINUED
                                # -------------------------------------

                                if (
                                    requested_field
                                    == "Discontinued"
                                ):


                                    if (
                                        extracted_answer.lower()
                                        == "false"
                                    ):


                                        answer = (
                                            "No, this medicine "
                                            "is not marked as "
                                            "discontinued in "
                                            "the database."
                                        )


                                    elif (
                                        extracted_answer.lower()
                                        == "true"
                                    ):


                                        answer = (
                                            "Yes, this medicine "
                                            "is marked as "
                                            "discontinued in "
                                            "the database."
                                        )


                                source = (
                                    "A-Z Medicines Dataset of India"
                                )


                            else:


                                answer = (
                                    "Information not available "
                                    "in the database."
                                )


                                source = (
                                    "A-Z Medicines Dataset of India"
                                )


                        # =================================================
                        # QWEN ANSWER
                        # =================================================

                        else:


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
7. If information is missing, say:

Information not available in the database.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


                            response = llm(

                                prompt,

                                max_new_tokens=80,

                                do_sample=False,

                                return_full_text=False
                            )


                            answer = (
                                response[0]
                                ["generated_text"]
                                .strip()
                            )


                            source = (
                                "A-Z Medicines Dataset of India"
                            )


                # =================================================
                # FINAL SAFETY CHECK
                # =================================================

                if not output_safety_check(
                    answer
                ):


                    answer = (
                        "I cannot provide a personalized "
                        "medicine recommendation or dosage. "
                        "Please consult a qualified doctor "
                        "or pharmacist."
                    )


                    source = "Security Guardrail"


        # ====================================================
        # SHOW ANSWER
        # ====================================================

        st.markdown(
            answer
        )


        # ====================================================
        # SOURCE
        # ====================================================

        st.html(
            f"""
            <div class="source-text">
                Source · {source}
            </div>
            """
        )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "source": source
        }
    )