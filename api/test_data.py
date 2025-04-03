"""
This module contains simulated data for testing the RAG-based chatbot.
"""

document1 = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
that are programmed to think and learn like humans. The term may also be applied to any 
machine that exhibits traits associated with a human mind such as learning and problem-solving.

AI can be categorized as either weak or strong. Weak AI, also known as narrow AI, is designed 
to perform a narrow task (e.g. facial recognition). Strong AI, also known as artificial general 
intelligence (AGI), is an AI system with generalized human cognitive abilities.
"""

document2 = """
Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability 
to automatically learn and improve from experience without being explicitly programmed. 
Machine learning focuses on the development of computer programs that can access data and 
use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, 
or instruction, in order to look for patterns in data and make better decisions in the future 
based on the examples that we provide. The primary aim is to allow the computers to learn 
automatically without human intervention or assistance and adjust actions accordingly.
"""

document3 = """
Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers 
understand, interpret and manipulate human language. NLP draws from many disciplines, including 
computer science and computational linguistics, in its pursuit to fill the gap between human 
communication and computer understanding.

NLP tasks include text translation, sentiment analysis, speech recognition, and topic segmentation. 
Modern approaches to NLP are based on machine learning, especially statistical methods and deep learning.
"""

document4 = """
Reinforcement Learning (RL) is an area of machine learning concerned with how software agents 
ought to take actions in an environment in order to maximize the notion of cumulative reward. 
Reinforcement learning is one of three basic machine learning paradigms, alongside supervised 
learning and unsupervised learning.

Reinforcement learning differs from supervised learning in that correct input/output pairs 
need not be presented, and sub-optimal actions need not be explicitly corrected. Instead the 
focus is on finding a balance between exploration (of uncharted territory) and exploitation 
(of current knowledge).
"""

document5 = """
Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based methods and 
text generation models to create more accurate and contextually relevant outputs. In RAG systems,
when a query is received, relevant information is first retrieved from a knowledge base, and then
this information is used to augment the input to a generative model.

This approach helps ground the model's responses in factual information from the knowledge base,
reducing hallucinations and improving accuracy. RAG is particularly useful for question-answering
systems and chatbots that need access to specialized or up-to-date information.
"""

documents = [document1, document2, document3, document4, document5]
