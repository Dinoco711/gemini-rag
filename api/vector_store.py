"""
This module handles the vector database operations using a simple in-memory vector store.
It provides functionality to embed documents and retrieve relevant context for queries.
"""

import os
import google.generativeai as genai
from test_data import documents
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorStore:
    def __init__(self, api_key):
        """
        Initialize a simple vector store using TF-IDF for document embeddings.
        
        Args:
            api_key (str): Google API key for Gemini (used for generation, not embeddings)
        """
        # Configure the Google Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.documents = []
        self.metadatas = []
        self.document_vectors = None
    
    def add_documents(self, docs, metadatas=None):
        """
        Add documents to the vector store.
        
        Args:
            docs (list): List of document texts
            metadatas (list, optional): List of metadata dictionaries for each document
        """
        if not metadatas:
            metadatas = [{"source": f"document_{i}"} for i in range(len(docs))]
        
        # Store documents and metadata
        self.documents.extend(docs)
        self.metadatas.extend(metadatas)
        
        # Fit and transform documents
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        print(f"Added {len(docs)} documents to the vector store.")
    
    def query(self, query_text, n_results=2):
        """
        Query the vector store for relevant document chunks.
        
        Args:
            query_text (str): The query text
            n_results (int): Number of results to return
            
        Returns:
            str: Concatenated relevant document chunks
        """
        if not self.documents:
            return ""
        
        # Transform query to vector
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get indices of top n_results
        top_indices = similarities.argsort()[-n_results:][::-1]
        
        # Get top documents
        top_documents = [self.documents[i] for i in top_indices]
        
        # Join the retrieved documents into a context string
        context = "\n\n".join(top_documents)
        return context

def initialize_vector_store(api_key):
    """
    Initialize the vector store and add test documents.
    
    Args:
        api_key (str): Google API key for Gemini
        
    Returns:
        SimpleVectorStore: Initialized vector store with test documents
    """
    vector_store = SimpleVectorStore(api_key)
    
    # Add the test documents to the vector store
    vector_store.add_documents(documents)
    
    return vector_store
