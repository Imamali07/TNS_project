import subprocess
import re


# ============================================================
# TEST CASES
# ============================================================

tests = [
    {
        "question": "What is the manufacturer of Augmentin 625 Duo Tablet?",
        "expected": "Glaxo SmithKline Pharmaceuticals Ltd"
    },
    {
        "question": "What is the price of Augmentin 625 Duo Tablet?",
        "expected": "₹223.42"
    },
    {
        "question": "What is the composition of Augmentin 625 Duo Tablet?",
        "expected": "Amoxycillin (500mg), Clavulanic Acid (125mg)"
    },
    {
        "question": "What is the pack size of Augmentin 625 Duo Tablet?",
        "expected": "strip of 10 tablets"
    },
    {
        "question": "Is Augmentin 625 Duo Tablet discontinued?",
        "expected": "not discontinued"
    }
]


# ============================================================
# FUNCTION TO RUN CHATBOT
# ============================================================

def run_chatbot(question):

    process = subprocess.run(
        [
            "python",
            "scripts\\rag_chat.py"
        ],
        input=question + "\nexit\n",
        text=True,
        capture_output=True
    )

    return process.stdout


# ============================================================
# EXTRACT ANSWER
# ============================================================

def get_answer(output):

    marker = "EVALUATION_ANSWER:"

    position = output.rfind(marker)

    if position == -1:
        return None

    answer = output[
        position + len(marker):
    ]

    answer = answer.splitlines()[0]

    return answer.strip()


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("========================================")
print("ANSWER ACCURACY EVALUATION")
print("========================================")
print()


passed = 0
failed = 0


for index, test in enumerate(tests, start=1):

    print(
        f"Test {index}/{len(tests)}"
    )

    print(
        "Question:",
        test["question"]
    )

    output = run_chatbot(
        test["question"]
    )

    answer = get_answer(output)


    if answer is None:

        print(
            "Result: FAIL"
        )

        print(
            "Reason: No evaluation answer found."
        )

        failed += 1

        print(
            "----------------------------------------"
        )

        continue


    answer_lower = answer.lower()

    expected_lower = (
        test["expected"].lower()
    )


    # --------------------------------------------------------
    # SPECIAL CHECK FOR DISCONTINUED
    # --------------------------------------------------------

    if test["expected"] == "not discontinued":

        correct = (
            "not marked as discontinued"
            in answer_lower
            or
            "not discontinued"
            in answer_lower
        )

    else:

        correct = (
            expected_lower
            in answer_lower
        )


    if correct:

        print(
            "Expected:",
            test["expected"]
        )

        print(
            "Actual:",
            answer
        )

        print(
            "Result: PASS"
        )

        passed += 1

    else:

        print(
            "Expected:",
            test["expected"]
        )

        print(
            "Actual:",
            answer
        )

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
    "FINAL ANSWER ACCURACY RESULT"
)

print(
    "========================================"
)

print(
    f"Answer Accuracy : {accuracy:.2f}%"
)

print(
    f"Passed          : {passed}/{total}"
)

print(
    f"Failed          : {failed}/{total}"
)

print(
    "========================================"
)