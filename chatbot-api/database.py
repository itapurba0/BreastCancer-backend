import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    raise RuntimeError("MONGO_DB_URL not set in chatbot-api/.env")

client = MongoClient(MONGO_DB_URL)
db = client.medicalChat

users_collection = db.users
sessions_collection = db.chat_sessions

users_collection.create_index("email", unique=True)
