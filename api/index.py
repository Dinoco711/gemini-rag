# index.py
import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
from rag import RAGSystem

# Load environment variables
load_dotenv()

# Set up Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Initialize RAG system
rag_system = RAGSystem()

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # Get relevant context from RAG system
    context = rag_system.get_context_for_query(user_message)
    
    # Create prompt with context (if available)
    if context:
        prompt = f"""
        Based on the following information and your knowledge, please answer the user's question:
        
        CONTEXT:
        {context}
        
        USER QUESTION:
        {user_message}
        
        Please provide a helpful, accurate response based on the context provided.
        """
    else:
        prompt = user_message
    
    # Generate AI Response with Gemini
    try:
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
