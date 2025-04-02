# database_module.py

import chromadb
from google import genai
from google.genai import types
from data_module import documents

# Set up Google Gemini API key (ensure you store this securely)
GOOGLE_API_KEY = "AIzaSyA1Rnv5FsdF5Ex77cJEbg_-cCA7tMcFDt4"
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define embedding function using Gemini API
class GeminiEmbeddingFunction:
    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    def __call__(self, input_texts):
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input_texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [embedding.values for embedding in response.embeddings]

# Initialize ChromaDB and insert documents
DB_NAME = "googlecar_db"
embed_fn = GeminiEmbeddingFunction(document_mode=True)
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Add documents to ChromaDB
db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# Confirm data insertion
print(f"Total documents stored: {db.count()}")
