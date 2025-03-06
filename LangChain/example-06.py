import os

import faiss
import numpy as np
import openai
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

# GPT-3 API key aus Umgebungsvariablen
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FAISS index
d = 768  # Dimension of the GPT-3 vectors
index = faiss.IndexFlatL2(d)

# Example documents
documents = [
    "Das Wetter ist sonnig und angenehm.",
    "Es regnet heute den ganzen Tag.",
    "Morgen wird es bewölkt sein."
]


# Function to get embeddings using OpenAI API
def get_embeddings(docs):
    doc_embeddings = []
    for doc in docs:
        response = openai.Embedding.create(model="text-embedding-ada-002-v2", input=[doc])
        embedding = response['data'][0]['embedding']
        doc_embeddings.append(embedding)
    return doc_embeddings


# Main execution
if __name__ == "__main__":
    # Create embeddings and insert them into the FAISS index
    embeddings = get_embeddings(documents)
    index.add(np.array(embeddings))
    
    # Check the number of vectors in the index
    print(f"Anzahl der Vektoren im Index: {index.ntotal}")
