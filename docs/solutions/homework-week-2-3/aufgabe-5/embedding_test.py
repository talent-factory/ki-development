#!/usr/bin/env python3
"""
Embedding Test: Verstehen wie Embeddings funktionieren

Dieses Skript demonstriert:
1. Erstellung von Embeddings für verschiedene Texte
2. Berechnung von Ähnlichkeiten
3. Visualisierung der Ergebnisse

Autor: Musterlösung für AI Development Kurs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd

# Optional: OpenAI für bessere Embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI nicht verfügbar - verwende Sentence Transformers")

# Sentence Transformers als Fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Sentence Transformers nicht verfügbar")

def get_openai_embeddings(texts, api_key=None):
    """Erstellt Embeddings mit OpenAI API"""
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API Key nicht gefunden")
    
    client = OpenAI(api_key=api_key)
    
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    return np.array(embeddings)

def get_sentence_transformer_embeddings(texts):
    """Erstellt Embeddings mit Sentence Transformers"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

def main():
    # Test-Texte definieren
    texts = [
        "Python ist eine Programmiersprache",           # Programmierung
        "JavaScript wird für Webentwicklung verwendet", # Programmierung  
        "Hunde sind treue Haustiere",                   # Tiere
        "Katzen sind unabhängige Tiere",                # Tiere
        "Machine Learning ist ein Teilbereich der KI"   # AI/ML
    ]
    
    labels = [
        "Python (Prog.)",
        "JavaScript (Prog.)", 
        "Hunde (Tiere)",
        "Katzen (Tiere)",
        "ML (AI)"
    ]
    
    print("🔍 Embedding Test gestartet...")
    print(f"📝 Analysiere {len(texts)} Texte")
    
    # Embeddings erstellen
    try:
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            print("🤖 Verwende OpenAI Embeddings...")
            embeddings = get_openai_embeddings(texts)
            model_name = "OpenAI Ada-002"
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🤗 Verwende Sentence Transformers...")
            embeddings = get_sentence_transformer_embeddings(texts)
            model_name = "Sentence-BERT"
        else:
            # Fallback: Zufällige Embeddings für Demo
            print("⚠️  Keine Embedding-Modelle verfügbar - verwende Zufallsdaten")
            embeddings = np.random.random((len(texts), 384))
            model_name = "Random (Demo)"
            
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der Embeddings: {e}")
        print("⚠️  Verwende Zufallsdaten für Demo...")
        embeddings = np.random.random((len(texts), 384))
        model_name = "Random (Demo)"
    
    print(f"✅ Embeddings erstellt mit {model_name}")
    print(f"📊 Shape: {embeddings.shape}")
    
    # Ähnlichkeiten berechnen
    similarities = cosine_similarity(embeddings)
    
    print("\n📈 Ähnlichkeits-Matrix (Cosine Similarity):")
    print("=" * 60)
    
    # Ähnlichkeits-Matrix als DataFrame für bessere Darstellung
    df_similarities = pd.DataFrame(
        similarities, 
        index=labels, 
        columns=labels
    )
    
    # Ähnlichkeiten ausgeben
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i < j:  # Nur obere Dreiecksmatrix
                sim = similarities[i][j]
                print(f"{label_i:20} ↔ {label_j:20} = {sim:.3f}")
    
    # Erwartete vs. tatsächliche Ähnlichkeiten analysieren
    print("\n🎯 Analyse der Ergebnisse:")
    print("=" * 40)
    
    # Programmiersprachen sollten ähnlich sein
    prog_similarity = similarities[0][1]  # Python ↔ JavaScript
    print(f"Programmiersprachen-Ähnlichkeit: {prog_similarity:.3f}")
    if prog_similarity > 0.5:
        print("✅ Erwartet: Programmiersprachen sind ähnlich")
    else:
        print("❌ Unerwartet: Programmiersprachen wenig ähnlich")
    
    # Tiere sollten ähnlich sein
    animal_similarity = similarities[2][3]  # Hunde ↔ Katzen
    print(f"Tier-Ähnlichkeit: {animal_similarity:.3f}")
    if animal_similarity > 0.5:
        print("✅ Erwartet: Tiere sind ähnlich")
    else:
        print("❌ Unerwartet: Tiere wenig ähnlich")
    
    # Cross-Domain sollte weniger ähnlich sein
    cross_similarity = similarities[0][2]  # Python ↔ Hunde
    print(f"Cross-Domain-Ähnlichkeit: {cross_similarity:.3f}")
    if cross_similarity < 0.3:
        print("✅ Erwartet: Cross-Domain wenig ähnlich")
    else:
        print("❌ Unerwartet: Cross-Domain zu ähnlich")
    
    # Visualisierung erstellen
    create_visualizations(embeddings, labels, similarities, model_name)
    
    print("\n🎉 Embedding Test abgeschlossen!")
    print("📊 Visualisierungen wurden erstellt")

def create_visualizations(embeddings, labels, similarities, model_name):
    """Erstellt Visualisierungen der Embeddings und Ähnlichkeiten"""
    
    # 1. Ähnlichkeits-Heatmap
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(
        similarities, 
        annot=True, 
        fmt='.3f',
        xticklabels=labels,
        yticklabels=labels,
        cmap='viridis',
        square=True
    )
    plt.title(f'Ähnlichkeits-Matrix\n({model_name})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 2. PCA-Visualisierung (2D)
    plt.subplot(1, 3, 2)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Farben für verschiedene Kategorien
    colors = ['red', 'red', 'blue', 'blue', 'green']
    categories = ['Programmierung', 'Programmierung', 'Tiere', 'Tiere', 'AI/ML']
    
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=colors[i], s=100, alpha=0.7)
        plt.annotate(labels[i], (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.title(f'PCA Visualisierung (2D)\nVarianz erklärt: {pca.explained_variance_ratio_.sum():.1%}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.grid(True, alpha=0.3)
    
    # 3. Ähnlichkeits-Verteilung
    plt.subplot(1, 3, 3)
    
    # Alle Ähnlichkeitswerte (ohne Diagonale)
    sim_values = []
    for i in range(len(similarities)):
        for j in range(i+1, len(similarities)):
            sim_values.append(similarities[i][j])
    
    plt.hist(sim_values, bins=10, alpha=0.7, edgecolor='black')
    plt.title('Verteilung der Ähnlichkeiten')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Häufigkeit')
    plt.axvline(np.mean(sim_values), color='red', linestyle='--', 
                label=f'Mittelwert: {np.mean(sim_values):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Zusätzliche Statistiken ausgeben
    print(f"\n📊 Embedding-Statistiken:")
    print(f"Dimensionen: {embeddings.shape[1]}")
    print(f"Durchschnittliche Ähnlichkeit: {np.mean(sim_values):.3f}")
    print(f"Standardabweichung: {np.std(sim_values):.3f}")
    print(f"Min/Max Ähnlichkeit: {np.min(sim_values):.3f} / {np.max(sim_values):.3f}")

if __name__ == "__main__":
    main()
