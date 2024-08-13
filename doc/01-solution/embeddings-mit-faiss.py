# Schritt 1: Installation von FAISS
# Die FAISS-Bibliothek kann mit pip installiert werden:
# pip install faiss-cpu

# Schritt 2: Vorbereitung der Embeddings.
# Wir verwenden die FastText-Bibliothek, um Embeddings zu erzeugen
import fasttext
import faiss
import numpy as np

# Beispieltextdaten
sentences = [
    "das wetter ist heute sonnig",
    "es ist ein heller tag",
    "ich liebe kaltes winterwetter"
]

# Speichern der Textdaten in einer Datei
with open('text_corpus.txt', 'w') as f:
    for sentence in sentences:
        f.write(f"{sentence}\n")

# Training des FastText-Modells auf dem Korpus
# Adding the `minCount` parameter
model = fasttext.train_unsupervised('text_corpus.txt', model='skipgram', dim=100, ws=5, minCount=1)

# Erzeugen und Sammeln der Vektoren (Embeddings) für die Wörter in Sätzen
words = ["sonnig", "heller", "kaltes"]
vectors = [model.get_word_vector(word) for word in words]

# Schritt 3: Erstellung des Vector Stores mit FAISS

# Konvertieren der Vektoren in ein Numpy-Array
vector_array = np.array(vectors)

# Erstellen eines FAISS-Index
dimension = vector_array.shape[1]  # Dimension der Vektoren
index = faiss.IndexFlatL2(dimension)  # L2-Distanzmetrikenstandard

# Hinzufügen der Vektoren zum Index
index.add(vector_array)

# Überprüfung der Anzahl der im Index gespeicherten Vektoren
print(f"Anzahl der Vektoren im Index: {index.ntotal}")

# Schritt 4: Durchführung von Abfragen
# Abfragevektor (Beispiel: Vektor für das Wort 'sonnig')
query_vector = model.get_word_vector('sonnig')

# Durchführung einer K-NN-Suche (Suche nach den 2 nächsten Nachbarn)
k = 2
D, I = index.search(np.array([query_vector]), k)

# Ausgabe der Ergebnisse
print(f"Indizes der nächsten Nachbarn: {I}")
print(f"Distanzen zu den nächsten Nachbarn: {D}")
print(f"Ähnlichste Wörter: {[words[i] for i in I[0]]}")
