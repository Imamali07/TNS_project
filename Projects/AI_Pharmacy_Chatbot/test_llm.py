from transformers import pipeline

print("Loading LLM...")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

print("LLM loaded successfully!")

messages = [
    {
        "role": "user",
        "content": "What is a medicine? Explain in simple words."
    }
]

result = generator(
    messages,
    max_new_tokens=100
)

print("\n========== LLM ANSWER ==========")
print(result[0]["generated_text"][-1]["content"])