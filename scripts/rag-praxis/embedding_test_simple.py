"""
Einfacher Embedding-Test mit OpenAI (schneller für den Kurs)
Ausführung: uv run scripts/rag-praxis/embedding_test_simple.py

Verwendet OpenAI Embeddings - viel schneller als lokale Modelle
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Prüfe ob OpenAI verfügbar ist
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def get_openai_embeddings(texts):
    """Embeddings mit OpenAI API erstellen"""
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI nicht installiert. Installieren Sie mit: uv pip install openai")
        return None
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY nicht gesetzt")
        print("💡 Setzen Sie: export OPENAI_API_KEY='sk-...'")
        return None
    
    try:
        client = openai.OpenAI()
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        embeddings = []
        for item in response.data:
            embeddings.append(item.embedding)
        
        return np.array(embeddings)
    
    except Exception as e:
        print(f"❌ OpenAI Fehler: {e}")
        return None

def get_simple_embeddings(texts):
    """Einfache Bag-of-Words Embeddings (Fallback)"""
    print("🔄 Verwende einfache Bag-of-Words Embeddings...")
    
    # Alle Wörter sammeln
    all_words = set()
    for text in texts:
        words = text.lower().split()
        all_words.update(words)
    
    word_list = sorted(list(all_words))
    
    # Embeddings erstellen
    embeddings = []
    for text in texts:
        words = text.lower().split()
        vector = [1 if word in words else 0 for word in word_list]
        embeddings.append(vector)
    
    return np.array(embeddings)

def main():
    print("🚀 Schneller Embedding-Test startet...")
    print("=" * 50)
    
    # Test-Texte
    texte = [
        "Der König regiert das Land",
        "Die Königin herrscht über das Reich", 
        "Das Auto fährt schnell",
        "Künstliche Intelligenz verändert die Welt",
        "Machine Learning ist ein Teilbereich der KI",
        "Python ist eine Programmiersprache",
        "JavaScript wird für Webentwicklung verwendet"
    ]
    
    print(f"📊 Teste {len(texte)} Texte...")
    
    # Versuche OpenAI, dann Fallback
    embeddings = get_openai_embeddings(texte)
    
    if embeddings is None:
        print("⚠️  OpenAI nicht verfügbar, verwende einfache Methode...")
        embeddings = get_simple_embeddings(texte)
    else:
        print("✅ OpenAI Embeddings erstellt")
    
    print(f"📈 Embedding-Dimension: {embeddings.shape}")
    
    # Ähnlichkeitsmatrix
    similarity_matrix = cosine_similarity(embeddings)
    
    print("\n🎯 Interessante Ähnlichkeiten:")
    print("=" * 60)
    
    # Interessante Paare
    pairs = [
        (0, 1, "König vs Königin"),
        (3, 4, "KI vs Machine Learning"), 
        (5, 6, "Python vs JavaScript"),
        (0, 2, "König vs Auto (sollte niedrig sein)")
    ]
    
    for i, j, beschreibung in pairs:
        similarity = similarity_matrix[i][j]
        emoji = "🔥" if similarity > 0.7 else "👍" if similarity > 0.4 else "🤔" if similarity > 0.2 else "❌"
        
        print(f"\n{emoji} {beschreibung}")
        print(f"   Ähnlichkeit: {similarity:.3f}")
        print(f"   '{texte[i]}'")
        print(f"   '{texte[j]}'")
    
    # Top ähnlichste Paare
    print("\n🏆 TOP 3 ÄHNLICHSTE PAARE:")
    print("=" * 40)
    
    all_pairs = []
    for i in range(len(texte)):
        for j in range(i + 1, len(texte)):
            similarity = similarity_matrix[i][j]
            all_pairs.append((similarity, i, j))
    
    all_pairs.sort(reverse=True)
    
    for rank, (similarity, i, j) in enumerate(all_pairs[:3], 1):
        print(f"\n{rank}. Ähnlichkeit: {similarity:.3f}")
        print(f"   '{texte[i]}'")
        print(f"   '{texte[j]}'")
    
    print("\n" + "=" * 60)
    print("✅ Schneller Embedding-Test abgeschlossen!")
    print("\n💡 Erkenntnisse:")
    print("   • Embeddings messen semantische Ähnlichkeit")
    print("   • OpenAI Embeddings sind sehr präzise")
    print("   • Auch einfache Methoden zeigen Muster")
    print("   • Ähnlichkeit wird als Zahl zwischen 0-1 gemessen")

if __name__ == "__main__":
    main()
