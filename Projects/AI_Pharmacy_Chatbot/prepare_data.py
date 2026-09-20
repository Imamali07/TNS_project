import pandas as pd
import re
import os

INPUT_FILE = "data/A_Z_medicines_dataset_of_India.csv"
OUTPUT_FILE = "data/cleaned_medicines.txt"


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)


# Remove completely duplicate rows
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# Clean column names
df.columns = df.columns.str.strip()


# Clean important text columns
columns_to_clean = [
    "name",
    "manufacturer_name",
    "type",
    "pack_size_label",
    "short_composition1",
    "short_composition2"
]

for column in columns_to_clean:
    df[column] = df[column].apply(clean_text)


# Remove records without a medicine name
df = df[df["name"] != ""]

print("After removing missing medicine names:", df.shape)


# Remove duplicate medicines
df = df.drop_duplicates(
    subset=[
        "name",
        "manufacturer_name",
        "short_composition1",
        "short_composition2"
    ]
)

print("After removing duplicate medicines:", df.shape)


os.makedirs("data", exist_ok=True)

print("Creating RAG text data...")


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for _, row in df.iterrows():

        composition = row["short_composition1"]

        if row["short_composition2"]:
            composition += ", " + row["short_composition2"]

        document = f"""
MEDICINE INFORMATION

Medicine Name:
{row["name"]}

Manufacturer:
{row["manufacturer_name"]}

Medicine Type:
{row["type"]}

Composition:
{composition}

Pack Size:
{row["pack_size_label"]}

Price:
₹{row["price(₹)"]}

Discontinued:
{row["Is_discontinued"]}

Source:
A-Z Medicines Dataset of India

----------------------------------------
"""

        file.write(document)


print()
print("========================================")
print("DATA PREPARATION COMPLETED")
print("========================================")
print("Final records:", len(df))
print("Output file:", OUTPUT_FILE)
print("========================================")