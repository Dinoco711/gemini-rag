import chromadb
from google import genai
from google.genai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings

# Set up Google Gemini API
GOOGLE_API_KEY = "AIzaSyA1Rnv5FsdF5Ex77cJEbg_-cCA7tMcFDt4"
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define retry condition for Gemini API calls
is_retriable = lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503}

# Define the embedding function for ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True  # True for storing documents, False for queries

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]

# Initialize ChromaDB
DB_NAME = "rag_chromadb"
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True  # Default mode for inserting documents

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Function to add documents
def add_documents(documents: list):
    ids = [str(i) for i in range(len(documents))]
    db.add(documents=documents, ids=ids)

# Function to retrieve relevant documents
def retrieve_documents(query: str, n_results: int = 3):
    embed_fn.document_mode = False  # Switch to query mode
    result = db.query(query_texts=[query], n_results=n_results)
    return result["documents"][0] if "documents" in result else []
