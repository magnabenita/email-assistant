from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

try:
    print("Databases:", client.list_database_names())
except Exception as e:
    print("❌ Error:", e)
