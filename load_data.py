import os
import csv
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "enron_nlp")

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Safely set CSV field size limit
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

def load_emails(csv_file):
    """Load emails from CSV into MongoDB in batches."""
    try:
        batch_size = 1000
        batch = []

        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                batch.append(row)
                if len(batch) >= batch_size:
                    db.emails.insert_many(batch)
                    print(f"✅ Inserted {i} rows so far...")
                    batch = []

            # Insert remaining
            if batch:
                db.emails.insert_many(batch)

        print(f"🎉 Finished loading emails into {DB_NAME}.emails")

    except Exception as e:
        print("❌ Error loading emails:", e)

if __name__ == "__main__":
    load_emails("enron_dataset/enron_raw.csv")
