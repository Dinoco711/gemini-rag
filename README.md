# RAG Chatbot with Google Gemini and Flask

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Google's Gemini models through the `google-generativeai` library, with ChromaDB as the vector store for document retrieval.

## Components

1. **test_data.py**: Contains simulated document data for testing purposes
2. **vector_store.py**: Handles ChromaDB integration for document embedding and retrieval
3. **index.py**: Main Flask application with the chatbot endpoint

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a Google AI API key from [Google AI Studio](https://ai.google.dev/)

3. Set your API key as an environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Running the Application

Start the Flask server:
```bash
python index.py
```

The server will run on `http://localhost:5000` by default.

## Using the Chatbot

Send a POST request to the `/chat` endpoint with a JSON payload containing your query:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?"}'
```

The response will be in JSON format:
```json
{
  "response": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans..."
}
```

## How It Works

1. When a query is received, the system searches ChromaDB for relevant document chunks
2. Retrieved context is combined with the user's query in a prompt to Google's Gemini model
3. Gemini generates a response based on both the retrieved context and its own knowledge
4. The response is returned to the user

## Customizing

- Add more documents to `test_data.py` to expand the knowledge base
- Adjust the number of results retrieved in `vector_store.py` by changing the `n_results` parameter
- Modify the prompt template in `index.py` to change how the chatbot responds 
