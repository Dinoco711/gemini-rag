import os
from flask import Flask, request, jsonify
import chromadb
from database_module import DB_NAME, embed_fn
import google_genai as genai

# Initialize Flask app
app = Flask(__name__)

# Initialize ChromaDB client and collection
db_client = chromadb.PersistentClient()
db = db_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Load Google Gemini API key
GOOGLE_API_KEY = os.getenv("AIzaSyA1Rnv5FsdF5Ex77cJEbg_-cCA7tMcFDt4")
genai.configure(api_key=GOOGLE_API_KEY)

def retrieve_relevant_docs(query_text, n_results=3):
    """Retrieve relevant documents from ChromaDB."""
    embed_fn.document_mode = False  # Switch to query mode
    result = db.query(query_texts=[query_text], n_results=n_results)
    return result["documents"][0] if result["documents"] else []

def generate_answer(query, retrieved_docs):
    """Generate an answer using Google Gemini."""
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided reference texts. 
    If relevant information is found, use it to form a comprehensive answer. Otherwise, state that you don't have enough information.
    
    QUESTION: {query}
    """
    for doc in retrieved_docs:
        prompt += f"\nPASSAGE: {doc}"
    
    response = genai.generate_content(model="gemini-1.5-flash", contents=prompt)
    return response.text

@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint to handle user queries."""
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query is required."}), 400
    
    retrieved_docs = retrieve_relevant_docs(query)
    answer = generate_answer(query, retrieved_docs)
    
    return jsonify({"query": query, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
