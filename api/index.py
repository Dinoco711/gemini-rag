"""
This is the main Flask application that implements a RAG-based chatbot using
Google's Gemini model and ChromaDB for vector storage.
"""

import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from vector_store import initialize_vector_store

app = Flask(__name__)

# Get API key from environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini model
MODEL_NAME = "gemini-1.5-pro"  # Use the latest available model
model = genai.GenerativeModel(MODEL_NAME)

# Initialize vector store with test documents
vector_store = initialize_vector_store(GOOGLE_API_KEY)

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>RAG Chatbot</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1 { color: #4285f4; }
                p { margin-bottom: 20px; }
                .endpoint { background-color: #f1f1f1; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>RAG Chatbot with Google Gemini</h1>
            <p>This is a Retrieval Augmented Generation chatbot using Google's Gemini model and ChromaDB.</p>
            <p>Use the <span class="endpoint">/chat</span> endpoint to interact with the chatbot.</p>
        </body>
    </html>
    """

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint for the RAG chatbot interaction.
    
    Expected JSON payload:
    {
        "query": "Your question here"
    }
    
    Returns:
    {
        "response": "The chatbot's response"
    }
    """
    # Get the query from the request
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    
    query = data['query']
    
    try:
        # Step 1: Retrieve relevant content from the vector store
        context = vector_store.query(query)
        
        # Step 2: Construct the prompt with retrieved context
        prompt = f"""
        You are an AI assistant focused on providing accurate information.
        
        Use the following retrieved context to answer the user's question. If the context doesn't
        contain relevant information, acknowledge that and provide a general answer based on your knowledge.
        
        Retrieved context:
        {context}
        
        User's question: {query}
        """
        
        # Step 3: Generate a response using Google's Gemini
        response = model.generate_content(prompt)
        
        # Step 4: Return the response
        return jsonify({"response": response.text})
    
    except Exception as e:
        app.logger.error(f"Error generating response: {str(e)}")
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500

if __name__ == '__main__':
    # Set debug to False in production
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
