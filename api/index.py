from flask import Flask, request, jsonify
from database_module import add_documents, retrieve_documents
from google import genai

app = Flask(__name__)

# Set up Google Gemini API
GOOGLE_API_KEY = "AIzaSyA1Rnv5FsdF5Ex77cJEbg_-cCA7tMcFDt4"
client = genai.Client(api_key=GOOGLE_API_KEY)

# API Endpoint to add documents to ChromaDB
@app.route('/add_data', methods=['POST'])
def add_data():
    data = request.json.get("documents", [])
    if not data:
        return jsonify({"error": "No documents provided"}), 400
    add_documents(data)
    return jsonify({"message": "Documents added successfully!"})

# API Endpoint to handle user queries using RAG
@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    retrieved_docs = retrieve_documents(user_query)
    if not retrieved_docs:
        return jsonify({"response": "No relevant documents found."})

    # Build prompt with retrieved documents
    prompt = f"""
    You are an AI assistant answering user queries based on reference documents.
    Be concise but informative. If the documents do not provide enough details, state that clearly.

    QUESTION: {user_query}
    """
    for doc in retrieved_docs:
        prompt += f"\nPASSAGE: {doc}"

    # Generate response using Google Gemini API
    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return jsonify({"response": answer.text})

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
