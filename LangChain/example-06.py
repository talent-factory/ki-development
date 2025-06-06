import os

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

# OpenAI Client initialisieren
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FAISS index
dimensions = 1536  # Dimension of the text-embedding-ada-002 vectors
index = faiss.IndexFlatL2(dimensions)

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
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=doc
        )
        embedding = response.data[0].embedding
        doc_embeddings.append(embedding)
    return doc_embeddings


# Main execution
if __name__ == "__main__":
    # Create embeddings and insert them into the FAISS index
    embeddings = get_embeddings(documents)
    index.add(np.array(embeddings))
    
    # Check the number of vectors in the index
    print(f"Anzahl der Vektoren im Index: {index.ntotal}")

    # Zeige nur einen kleinen Teil des Vektors an
    print("Inhalt des Index (nur die ersten 10 Zahlen):")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc} -> {embeddings[i][:10]}")
       
