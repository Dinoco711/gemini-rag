import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from datetime import datetime
from database_module import DB_NAME  # Ensure this is the name used in database_module.py

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set Google Gemini API Key
GOOGLE_API_KEY = os.environ['AIzaSyA1Rnv5FsdF5Ex77cJEbg_-cCA7tMcFDt4']
client = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
db = chroma_client.get_collection(name=DB_NAME, embedding_function=DefaultEmbeddingFunction())

# Define chatbot system prompt
SYSTEM_PROMPT = """
You are NOVA, an advanced AI assistant for Nexobotics, designed to provide efficient and intelligent responses.
You utilize a **retrieval-augmented generation (RAG) system**, meaning you retrieve relevant context from a **vector database (ChromaDB)** before answering.
Your responses should be **concise, engaging, and helpful**.
Keep greetings short, and avoid unnecessary explanations unless requested.
"""

# Initialize chat history
chat_histories = {}

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    message = request.json.get('message')
    session_id = request.json.get('session_id', str(datetime.now()))

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    try:
        # Retrieve relevant documents from ChromaDB
        results = db.query(query_texts=[message], n_results=3)  # Fetch top 3 relevant docs
        retrieved_docs = "\n".join(results['documents'][0]) if results['documents'] else ""

        # Initialize chat history if not exists
        if session_id not in chat_histories:
            chat_histories[session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        # Add user query with retrieved context
        chat_histories[session_id].append({"role": "user", "content": f"Context: {retrieved_docs}\n\nUser: {message}"})

        # Generate response using Google Gemini
        response = client.chat.generate_content(
            model="gemini-pro",
            messages=chat_histories[session_id],
            temperature=0.7,
            max_output_tokens=1024
        )

        ai_response = response.candidates[0].content.parts[0].text
        chat_histories[session_id].append({"role": "assistant", "content": ai_response})

        return jsonify({'response': ai_response})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
