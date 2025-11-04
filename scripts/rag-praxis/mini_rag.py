"""
Mini-RAG System: Einfaches Retrieval Augmented Generation
Ausführung: uv run scripts/rag-praxis/mini_rag.py

Dieses Skript demonstriert:
- Aufbau einer Wissensbasis mit Embeddings
- Ähnlichkeitssuche mit FAISS
- Grundlagen des RAG-Prozesses
- Praktische Implementierung ohne externe APIs
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

class MiniRAG:
    def __init__(self):
        print("🚀 Mini-RAG System startet...")
        print("=" * 50)
        
        print("📥 Lade Sentence Transformer Modell...")
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        print("✅ Modell geladen")
    
    def add_documents(self, docs):
        """Dokumente zur Wissensbasis hinzufügen"""
        print(f"\n📚 Füge {len(docs)} Dokumente zur Wissensbasis hinzu...")
        
        start_time = time.time()
        self.documents.extend(docs)
        
        # Embeddings für alle Dokumente erstellen
        print("🔄 Erstelle Embeddings...")
        all_embeddings = self.model.encode(self.documents, show_progress_bar=True)
        self.embeddings = np.array(all_embeddings)
        
        # FAISS Index erstellen für schnelle Suche
        print("🔍 Erstelle FAISS-Index für schnelle Suche...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        elapsed_time = time.time() - start_time
        print(f"✅ {len(self.documents)} Dokumente indexiert in {elapsed_time:.2f}s")
        print(f"   Embedding-Dimension: {dimension}")
    
    def search(self, query, k=3):
        """Ähnlichste Dokumente finden"""
        if self.index is None:
            print("❌ Keine Dokumente in der Wissensbasis!")
            return []
        
        print(f"\n🔍 Suche nach: '{query}'")
        
        # Query in Embedding umwandeln
        query_embedding = self.model.encode([query])
        
        # Suche in FAISS Index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        # Ergebnisse zusammenstellen
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                # FAISS gibt L2-Distanz zurück, wir wollen Ähnlichkeit
                similarity = 1 / (1 + distance)
                results.append({
                    'document': self.documents[idx],
                    'similarity': similarity,
                    'rank': i + 1,
                    'distance': distance
                })
        
        return results
    
    def answer_question(self, question):
        """Frage mit RAG-Ansatz beantworten"""
        print("\n" + "=" * 80)
        print(f"❓ FRAGE: {question}")
        print("=" * 80)
        
        # 1. Relevante Dokumente finden (Retrieval-Phase)
        relevant_docs = self.search(question, k=3)
        
        if not relevant_docs:
            return "❌ Keine relevanten Dokumente gefunden."
        
        print("\n🎯 RETRIEVAL-ERGEBNISSE:")
        context = ""
        for doc in relevant_docs:
            print(f"\n  📄 Rang {doc['rank']} (Ähnlichkeit: {doc['similarity']:.3f})")
            
            # Zeige ersten Teil des Dokuments
            doc_preview = doc['document'][:150]
            if len(doc['document']) > 150:
                doc_preview += "..."
            
            print(f"     {doc_preview}")
            context += doc['document'] + "\n\n"
        
        # 2. Kontext für Generation vorbereiten
        print(f"\n💡 GENERATION-PHASE:")
        print("📝 Zusammengestellter Kontext:")
        context_preview = context[:300]
        if len(context) > 300:
            context_preview += "..."
        print(f"   {context_preview}")
        
        # 3. Einfache "Generation" (in echtem RAG würde hier LLM stehen)
        print(f"\n🤖 ANTWORT-GENERATION:")
        print("   In einem echten RAG-System würde jetzt ein LLM (wie GPT-4)")
        print("   basierend auf dem gefundenen Kontext eine Antwort generieren.")
        
        # Einfache regelbasierte "Antwort"
        answer = self._generate_simple_answer(question, relevant_docs)
        
        return answer
    
    def _generate_simple_answer(self, question, docs):
        """Einfache regelbasierte Antwortgenerierung (Ersatz für LLM)"""
        if not docs:
            return "Keine relevanten Informationen gefunden."
        
        best_doc = docs[0]  # Bestes Ergebnis
        
        # Einfache Antwort basierend auf bestem Dokument
        answer = f"""
🎯 ANTWORT (basierend auf bestem Match):

{best_doc['document']}

📊 Vertrauen: {best_doc['similarity']:.1%}
📚 Quelle: Dokument {best_doc['rank']} aus der Wissensbasis

💡 Hinweis: In einem echten RAG-System würde ein LLM diese Information 
   in natürlicher Sprache zusammenfassen und eine präzise Antwort formulieren.
"""
        return answer
    
    def show_knowledge_base(self):
        """Zeige alle Dokumente in der Wissensbasis"""
        print(f"\n📚 WISSENSBASIS ({len(self.documents)} Dokumente):")
        print("=" * 60)
        
        for i, doc in enumerate(self.documents, 1):
            preview = doc[:100]
            if len(doc) > 100:
                preview += "..."
            print(f"{i:2d}. {preview}")

def main():
    # RAG System initialisieren
    rag = MiniRAG()
    
    # Beispiel-Wissensbasis (erweitert)
    documents = [
        "Python ist eine vielseitige Programmiersprache, die 1991 von Guido van Rossum entwickelt wurde. Sie wird für Webentwicklung, Datenanalyse, künstliche Intelligenz und Automatisierung verwendet.",
        
        "Machine Learning ermöglicht es Computern, aus Daten zu lernen, ohne explizit programmiert zu werden. Es umfasst überwachtes, unüberwachtes und verstärkendes Lernen.",
        
        "RAG (Retrieval Augmented Generation) kombiniert Informationsabruf mit Textgenerierung. Es löst das Problem veralteter Trainingsdaten bei Large Language Models.",
        
        "Vector Stores wie FAISS, ChromaDB und Pinecone speichern Embeddings effizient und ermöglichen schnelle Ähnlichkeitssuche in großen Datensätzen.",
        
        "LangChain ist ein Framework für die Entwicklung von LLM-basierten Anwendungen. Es bietet Tools für Chains, Prompts, Memory und Agents.",
        
        "Embeddings wandeln Text in numerische Vektoren um, die semantische Ähnlichkeit erfassen. Sentence Transformers sind ein beliebtes Tool dafür.",
        
        "FAISS (Facebook AI Similarity Search) ist eine Bibliothek für effiziente Ähnlichkeitssuche und Clustering von dichten Vektoren.",
        
        "Streamlit ermöglicht es, schnell interaktive Web-Apps für Data Science und Machine Learning zu erstellen, ohne Frontend-Kenntnisse.",
        
        "OpenAI bietet APIs für GPT-Modelle und Embeddings. GPT-4 ist besonders gut für komplexe Reasoning-Aufgaben geeignet.",
        
        "Anthropic entwickelt Claude, einen AI-Assistenten, der auf Constitutional AI basiert und besonders sicher und hilfreich sein soll."
    ]
    
    # Dokumente zur Wissensbasis hinzufügen
    rag.add_documents(documents)
    
    # Wissensbasis anzeigen
    rag.show_knowledge_base()
    
    # Test-Fragen
    questions = [
        "Was ist Python?",
        "Wie funktioniert Machine Learning?",
        "Was sind Embeddings?",
        "Erkläre mir RAG",
        "Welche Vector Stores gibt es?",
        "Was ist der Unterschied zwischen OpenAI und Anthropic?"
    ]
    
    # Automatische Tests
    print("\n" + "🤖 AUTOMATISCHE TESTS".center(80, "="))
    
    for i, question in enumerate(questions, 1):
        print(f"\n🧪 TEST {i}/{len(questions)}")
        answer = rag.answer_question(question)
        print(answer)
        
        if i < len(questions):
            input("\n⏸️  Drücken Sie Enter für den nächsten Test...")
    
    # Interaktiver Teil
    print("\n" + "🎮 INTERAKTIVER MODUS".center(80, "="))
    print("Stellen Sie Ihre eigenen Fragen! (Leer lassen zum Beenden)")
    
    try:
        while True:
            user_question = input("\n❓ Ihre Frage: ").strip()
            
            if not user_question:
                break
                
            answer = rag.answer_question(user_question)
            print(answer)
    
    except KeyboardInterrupt:
        print("\n\n👋 RAG-System beendet!")
    
    print("\n" + "=" * 80)
    print("✅ Mini-RAG Demo abgeschlossen!")
    print("\n💡 Was Sie gelernt haben:")
    print("   • RAG = Retrieval (Suchen) + Generation (Antworten)")
    print("   • Embeddings ermöglichen semantische Suche")
    print("   • FAISS macht die Suche in großen Datenmengen schnell")
    print("   • Der Kontext verbessert die Qualität der Antworten")
    print("   • RAG löst das Problem veralteter LLM-Trainingsdaten")
    
    print("\n🚀 Nächste Schritte:")
    print("   • Integrieren Sie echte LLMs (OpenAI, Anthropic)")
    print("   • Laden Sie eigene Dokumente (PDFs, Websites)")
    print("   • Experimentieren Sie mit verschiedenen Embedding-Modellen")
    print("   • Bauen Sie eine Streamlit-App für Ihr RAG-System")

if __name__ == "__main__":
    main()
