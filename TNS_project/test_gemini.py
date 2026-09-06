import os
from dotenv import load_dotenv
from google import genai


# Load .env
load_dotenv()


# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found")
    exit()


# Create Gemini client
client = genai.Client(
    api_key=api_key
)


# Test Gemini
response = client.interactions.create(
    model="gemini-3.6-flash",
    input="Say hello in one short sentence."
)


print("✅ Gemini connection successful!")

print(response.output_text)