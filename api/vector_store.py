"""
This module handles the vector database operations using ChromaDB.
It provides functionality to embed documents and retrieve relevant context for queries.
"""

import os
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from test_data import documents

class VectorStore:
    def __init__(self, api_key, collection_name="rag_documents"):
        """
        Initialize the vector store with Google's Gemini for embeddings.
        
        Args:
            api_key (str): Google API key for Gemini
            collection_name (str): Name of the ChromaDB collection
        """
        # Configure the Google Gemini API
        genai.configure(api_key=api_key)
        
        # Set up ChromaDB client
        self.client = chromadb.Client()
        
        # Use a sentence_transformer as embedding function
        # You can also use Google's embedding model when available in the API
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{collection_name}' retrieved successfully.")
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{collection_name}' created successfully.")
    
    def add_documents(self, docs, metadatas=None):
        """
        Add documents to the vector store.
        
        Args:
            docs (list): List of document texts
            metadatas (list, optional): List of metadata dictionaries for each document
        """
        if not metadatas:
            metadatas = [{"source": f"document_{i}"} for i in range(len(docs))]
        
        # Add or update documents in the collection
        self.collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
        print(f"Added {len(docs)} documents to the collection.")
    
    def query(self, query_text, n_results=2):
        """
        Query the vector store for relevant document chunks.
        
        Args:
            query_text (str): The query text
            n_results (int): Number of results to return
            
        Returns:
            str: Concatenated relevant document chunks
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        if results and results['documents'] and results['documents'][0]:
            # Join the retrieved documents into a context string
            context = "\n\n".join(results['documents'][0])
            return context
        return ""

def initialize_vector_store(api_key):
    """
    Initialize the vector store and add test documents.
    
    Args:
        api_key (str): Google API key for Gemini
        
    Returns:
        VectorStore: Initialized vector store with test documents
    """
    vector_store = VectorStore(api_key)
    
    # Add the test documents to the vector store
    vector_store.add_documents(documents)
    
    return vector_store
