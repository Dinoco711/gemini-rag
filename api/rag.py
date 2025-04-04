# rag.py
import chromadb
import os
from chromadb.utils import embedding_functions
from data import get_knowledge_base

class RAGSystem:
    def __init__(self, collection_name="knowledge_collection"):
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        
        # Define embedding function (using default)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{collection_name}' loaded successfully.")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{collection_name}' created successfully.")
            
            # Populate database with knowledge base
            self._populate_database()
    
    def _populate_database(self):
        # Get knowledge base from data module
        knowledge_base = get_knowledge_base()
        
        # Add items to collection
        for i, text in enumerate(knowledge_base):
            self.collection.add(
                documents=[text],
                metadatas=[{"source": "knowledge_base"}],
                ids=[f"text_{i}"]
            )
        print(f"Added {len(knowledge_base)} items to the collection.")
    
    def query(self, question, n_results=3):
        # Query collection for relevant information
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        # Extract and return documents
        if results and results['documents']:
            return results['documents'][0]
        return []
    
    def get_context_for_query(self, question):
        # Get relevant context from database
        relevant_docs = self.query(question)
        
        # Combine into a single context string
        if relevant_docs:
            context = "\n\n".join(relevant_docs)
            return context
        return ""
