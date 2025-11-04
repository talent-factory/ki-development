"""
Embedding-Test: Ähnlichkeit von Texten berechnen
Ausführung: uv run scripts/rag-praxis/embedding_test.py

Dieses Skript demonstriert:
- Wie Embeddings erstellt werden
- Wie Ähnlichkeit zwischen Texten berechnet wird
- Praktische Anwendung von Sentence Transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("🔍 Embedding-Test startet...")
    print("=" * 50)
    
    # Modell laden (läuft lokal, keine API nötig)
    print("📥 Lade Sentence Transformer Modell...")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("✅ Modell geladen")
    
    # Test-Texte für verschiedene Bereiche
    texte = [
        "Der König regiert das Land",
        "Die Königin herrscht über das Reich", 
        "Das Auto fährt schnell auf der Autobahn",
        "Künstliche Intelligenz verändert die Welt",
        "Machine Learning ist ein Teilbereich der KI",
        "Python ist eine Programmiersprache",
        "JavaScript wird für Webentwicklung verwendet",
        "Der Hund bellt laut im Garten",
        "Die Katze schläft auf dem Sofa"
    ]
    
    print(f"\n📊 Erstelle Embeddings für {len(texte)} Texte...")
    
    # Embeddings erstellen
    embeddings = model.encode(texte)
    print(f"✅ Embeddings erstellt: {embeddings.shape}")
    print(f"   Jeder Text wird als {embeddings.shape[1]}-dimensionaler Vektor dargestellt")
    
    # Ähnlichkeitsmatrix berechnen
    similarity_matrix = cosine_similarity(embeddings)
    
    # Ergebnisse anzeigen
    print("\n🎯 Ähnlichkeitsanalyse:")
    print("=" * 80)
    
    # Zeige nur interessante Vergleiche
    interesting_pairs = [
        (0, 1),  # König vs Königin
        (3, 4),  # KI vs ML
        (5, 6),  # Python vs JavaScript
        (7, 8),  # Hund vs Katze
        (0, 2),  # König vs Auto (sollte niedrig sein)
        (3, 7),  # KI vs Hund (sollte niedrig sein)
    ]
    
    for i, j in interesting_pairs:
        similarity = similarity_matrix[i][j]
        text1 = texte[i]
        text2 = texte[j]
        
        # Emoji basierend auf Ähnlichkeit
        if similarity > 0.7:
            emoji = "🔥"  # Sehr ähnlich
        elif similarity > 0.4:
            emoji = "👍"  # Ähnlich
        elif similarity > 0.2:
            emoji = "🤔"  # Etwas ähnlich
        else:
            emoji = "❌"  # Nicht ähnlich
        
        print(f"\n{emoji} Ähnlichkeit: {similarity:.3f}")
        print(f"   '{text1}'")
        print(f"   '{text2}'")
    
    # Finde die ähnlichsten und unähnlichsten Paare
    print("\n" + "=" * 80)
    print("🏆 TOP 3 ÄHNLICHSTE PAARE:")
    
    # Erstelle Liste aller Paare (ohne Selbstvergleiche)
    all_pairs = []
    for i in range(len(texte)):
        for j in range(i + 1, len(texte)):
            similarity = similarity_matrix[i][j]
            all_pairs.append((similarity, i, j))
    
    # Sortiere nach Ähnlichkeit
    all_pairs.sort(reverse=True)
    
    # Zeige Top 3
    for rank, (similarity, i, j) in enumerate(all_pairs[:3], 1):
        print(f"\n{rank}. Ähnlichkeit: {similarity:.3f}")
        print(f"   '{texte[i]}'")
        print(f"   '{texte[j]}'")
    
    print("\n" + "=" * 80)
    print("🔻 TOP 3 UNÄHNLICHSTE PAARE:")
    
    # Zeige Bottom 3
    for rank, (similarity, i, j) in enumerate(all_pairs[-3:], 1):
        print(f"\n{rank}. Ähnlichkeit: {similarity:.3f}")
        print(f"   '{texte[i]}'")
        print(f"   '{texte[j]}'")
    
    # Interaktiver Teil
    print("\n" + "=" * 80)
    print("🎮 INTERAKTIVER TEST")
    print("Geben Sie einen eigenen Text ein, um die Ähnlichkeit zu testen!")
    
    try:
        user_text = input("\nIhr Text: ").strip()
        if user_text:
            # Embedding für Benutzertext erstellen
            user_embedding = model.encode([user_text])
            
            # Ähnlichkeit zu allen anderen Texten berechnen
            similarities = cosine_similarity(user_embedding, embeddings)[0]
            
            print(f"\n🔍 Ähnlichkeit von '{user_text}' zu den Beispieltexten:")
            
            # Sortiere nach Ähnlichkeit
            sorted_indices = np.argsort(similarities)[::-1]
            
            for i, idx in enumerate(sorted_indices[:5]):  # Top 5
                similarity = similarities[idx]
                emoji = "🔥" if similarity > 0.5 else "👍" if similarity > 0.3 else "🤔"
                print(f"  {i+1}. {emoji} {similarity:.3f} - '{texte[idx]}'")
    
    except KeyboardInterrupt:
        print("\n\n👋 Test beendet!")
    
    print("\n" + "=" * 80)
    print("✅ Embedding-Test abgeschlossen!")
    print("\n💡 Was Sie gelernt haben:")
    print("   • Embeddings wandeln Text in Zahlen um")
    print("   • Ähnliche Texte haben ähnliche Embeddings")
    print("   • Cosinus-Ähnlichkeit misst die Ähnlichkeit (0-1)")
    print("   • Das funktioniert sprachübergreifend und semantisch")

if __name__ == "__main__":
    main()
