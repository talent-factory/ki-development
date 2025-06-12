#  Released under MIT License
#
#  Copyright (c) 2025. Talent Factory GmbH
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights to
#  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
#  of the Software, and to permit persons to whom the Software is furnished to
#  do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.

"""
Beispielskript zur Demonstration von Word2Vec-Embeddings.

Lernziele
---------
1. Verständnis für das Training von Word2Vec-Modellen.
2. Interpretation von Wortvektoren und Ähnlichkeitsmessung.
3. Anwendung von PCA zur Visualisierung hochdimensionaler Vektoren.

Vorgehensweise
--------------
Das Skript ist in nummerierte Schritte gegliedert. Nutzen Sie die *Aufgabe*- und
*Didaktischer Hinweis*-Blöcke, um aktiv mit Ihren Student:innen zu arbeiten.
"""

# Schritt 0: Importe
# ------------------
# Drittanbieter-Bibliotheken werden hier gesammelt importiert.
# Dies fördert Übersichtlichkeit und erleichtert späteres Refactoring.
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Schritt 1: Vorbereitung der Trainingsdaten
# ------------------------------------------
# In produktiven Szenarien sollten Sie hier ein grosses, domänenspezifisches
# Korpus laden (z. B. über pandas). Für Demonstrationszwecke genügt eine
# kleine Liste von Token-Listen.
sentences = [
    ["das", "wetter", "ist", "heute", "sonnig"],
    ["es", "ist", "ein", "heller", "und", "sonniger", "tag"],
    ["ich", "liebe", "das", "kalte", "winterwetter"],
    ["dies", "ist", "ein", "weiterer", "sonniger", "tag"],
    ["kalt", "und", "windig", "aber", "schöner", "tag"],
]

# Aufgabe: Ergänzen Sie das Korpus mit mindestens zwei eigenen Sätzen und
# beobachten Sie später die Veränderungen in den Embeddings.


# Schritt 2: Training des Word2Vec-Modells
# ----------------------------------------
# Die Hyperparameter haben grossen Einfluss auf die Qualität der Embeddings.
# Experimentieren Sie gemeinsam mit der Lerngruppe mit `vector_size`,
# `window`, `min_count` und `epochs`.
model = Word2Vec(
    sentences,
    vector_size=100,  # Grösse des Embeddings
    window=5,         # Kontextfenster
    min_count=1,      # Mindesthäufigkeit eines Tokens
    workers=4,        # Anzahl CPU-Kerne
    epochs=10         # Anzahl Trainings-Epochen
)

# Didaktischer Hinweis: Erhöhen Sie `epochs`, um Overfitting zu diskutieren.


# Schritt 3: Exploration des Modells
# ----------------------------------
# 3a. Zugriff auf den Vektor eines Beispielwortes
print(f"\nWortvektor für 'sonnig':\n{model.wv['sonnig']}")

# 3b. Suche nach semantisch ähnlichen Wörtern
similar_words = model.wv.most_similar('sonnig', topn=3)
print(f"\nTop 3 ähnliche Wörter zu 'sonnig': {similar_words}")

# Aufgabe: Lassen Sie die Student:innen andere Zielwörter testen.


# Schritt 4: Visualisierung der Embeddings
# ----------------------------------------
# PCA reduziert die hochdimensionalen Embeddings auf zwei Dimensionen,
# wodurch sie visuell interpretierbar werden.
words = list(model.wv.index_to_key)
vectors = [model.wv[word] for word in words]

pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title('PCA von Word2Vec-Embeddings')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Didaktischer Hinweis: Besprechen Sie Clusterbildung und Ausreisser.


# Einstiegspunkt
# --------------
# Durch das `if __name__`-Konstrukt kann dieses Skript auch als Modul importiert
# werden, ohne dass der Trainings- und Visualisierungsteil sofort ausgeführt
# wird. Dies erleichtert z. B. automatisierte Tests.
if __name__ == '__main__':
    # In diesem einfachen Beispiel liegt die Programmlogik bereits auf
    # Modulebene. Bei komplexeren Projekten würde hier `main()` aufgerufen.
    pass
