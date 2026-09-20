from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading medicine text data...")

loader = TextLoader(
    "data/cleaned_medicines.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Total documents loaded:", len(documents))

print("Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks created:", len(chunks))

print("\n========== FIRST CHUNK ==========")
print(chunks[0].page_content)

print("\n========== SECOND CHUNK ==========")
print(chunks[1].page_content)