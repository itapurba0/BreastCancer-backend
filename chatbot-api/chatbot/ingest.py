import os
import uuid
import re

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct



load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "medical_knowledge_base"

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "data"
)

# ---------------------------------------------------
# Connect Qdrant
# ---------------------------------------------------

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ---------------------------------------------------
# Embedding Model
# ---------------------------------------------------

embed_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ---------------------------------------------------
# Create Collection
# ---------------------------------------------------

if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("✅ Collection created")

# ---------------------------------------------------
# Clean Text
# ---------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ---------------------------------------------------
# Extract PDF
# ---------------------------------------------------

def extract_pdf_text(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + " "

    return clean_text(text)

# ---------------------------------------------------
# Chunking
# ---------------------------------------------------

def chunk_text(text, chunk_size=400, overlap=50):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size - overlap):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks

# ---------------------------------------------------
# Ingest PDFs
# ---------------------------------------------------

points = []

for filename in os.listdir(DATA_DIR):

    if filename.endswith(".pdf"):

        print(f"📄 Processing: {filename}")

        path = os.path.join(DATA_DIR, filename)

        text = extract_pdf_text(path)

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):

            embedding = embed_model.encode(
                chunk,
                normalize_embeddings=True
            ).tolist()

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": filename,
                        "chunk": idx
                    }
                )
            )

# ---------------------------------------------------
# Upload
# ---------------------------------------------------

qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"✅ Uploaded {len(points)} chunks")