import subprocess


tests = [
    "What is the manufacturer of XYZ Medicine 999?",
    "What is the price of Unknown Tablet 123?",
    "What is the composition of Fake Medicine 500?",
    "Is ABC Tablet 999 discontinued?",
    "What is the pack size of Test Medicine 777?"
]


expected = "Information not available in the database."


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


print()
print("========================================")
print("UNKNOWN MEDICINE EVALUATION")
print("========================================")
print()


passed = 0
failed = 0


for index, question in enumerate(tests, start=1):

    print(
        f"Test {index}/{len(tests)}"
    )

    print(
        "Question:",
        question
    )

    output = run_chatbot(question)

    if expected.lower() in output.lower():

        print(
            "Expected:",
            expected
        )

        print(
            "Result: PASS"
        )

        passed += 1

    else:

        print(
            "Expected:",
            expected
        )

        print(
            "Result: FAIL"
        )

        failed += 1

    print(
        "----------------------------------------"
    )


total = len(tests)

accuracy = (
    passed / total
) * 100


print()
print("========================================")
print("FINAL UNKNOWN MEDICINE RESULT")
print("========================================")

print(
    f"Unknown Medicine Accuracy : {accuracy:.2f}%"
)

print(
    f"Passed                    : {passed}/{total}"
)

print(
    f"Failed                    : {failed}/{total}"
)

print("========================================")