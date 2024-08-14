# Schritt 1: Installation von FastText
# Die FastText-Bibliothek kann mit pip installiert werden:
# pip install fasttext numpy

# Trainieren des FastText-Modells
import fasttext.util
import numpy as np
from numpy.linalg import norm

# Beispieltextdaten
sentences = [
    "das wetter ist heute sonnig",
    "es ist ein heller tag",
    "ich liebe kaltes winterwetter"
]

# FastText erwartet Textdaten aus einer Datei
with open('text_corpus.txt', 'w') as f:
    for sentence in sentences:
        f.write(f"{sentence}\n")

# Sicherstellen, dass die Datei nicht leer ist
with open('text_corpus.txt', 'r') as f:
    content = f.read()
    if not content.strip():
        raise ValueError("The text corpus is empty. Please provide valid text data.")

# Trainieren des FastText-Modells auf dem Korpus
try:
    model = fasttext.train_unsupervised('text_corpus.txt', model='skipgram', dim=100, ws=5, minCount=1)
except ValueError as e:
    print(f"Error in training FastText model: {e}")

# Berechnung von Embeddings für Beispielwörter
word1 = 'sonnig'
word2 = 'heller'
word3 = 'kaltes'

vector_sonnig = model.get_word_vector(word1)
vector_heller = model.get_word_vector(word2)
vector_kaltes = model.get_word_vector(word3)


# Schritt 4: Berechnung der semantischen Ähnlichkeit (z.B. Kosinus-Ähnlichkeit)
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


# Alternativer Algorithmus: Euklidische Distanz
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


distance_sonnig_heller = euclidean_distance(vector_sonnig, vector_heller)
distance_sonnig_kaltes = euclidean_distance(vector_sonnig, vector_kaltes)

print(f"Euklidische Distanz zwischen '{word1}' und '{word2}': {distance_sonnig_heller}")
print(f"Euklidische Distanz zwischen '{word1}' und '{word3}': {distance_sonnig_kaltes}")

similarity_sonnig_heller = cosine_similarity(vector_sonnig, vector_heller)
similarity_sonnig_kaltes = cosine_similarity(vector_sonnig, vector_kaltes)

print(f"Ähnlichkeit zwischen '{word1}' und '{word2}': {similarity_sonnig_heller}")
print(f"Ähnlichkeit zwischen '{word1}' und '{word3}': {similarity_sonnig_kaltes}")
