"""
Schnelles Mini-RAG System mit OpenAI Embeddings
Ausführung: uv run scripts/rag-praxis/mini_rag_fast.py

Verwendet OpenAI für Embeddings und optional für Generation
Läuft sofort ohne lange Downloads
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# .env Datei laden
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("💡 Hinweis: python-dotenv nicht installiert, verwende Umgebungsvariablen")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class FastRAG:
    def __init__(self):
        print("🚀 Schnelles RAG-System startet...")
        print("=" * 50)
        
        # OpenAI Client setup
        self.client = None
        self._setup_openai()
        
        # RAG Komponenten
        self.documents = []
        self.embeddings = None
        print("✅ RAG-System bereit")
    
    def _setup_openai(self):
        """OpenAI Client einrichten"""
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI nicht installiert: uv pip install openai")
            return
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY nicht in .env gefunden")
            return
        
        try:
            self.client = openai.OpenAI()
            print("✅ OpenAI Client bereit")
        except Exception as e:
            print(f"❌ OpenAI Setup Fehler: {e}")
    
    def _get_embeddings(self, texts):
        """Embeddings mit OpenAI erstellen"""
        if not self.client:
            print("⚠️  Verwende einfache Bag-of-Words Embeddings...")
            return self._simple_embeddings(texts)
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return np.array(embeddings)
        
        except Exception as e:
            print(f"❌ OpenAI Embeddings Fehler: {e}")
            print("⚠️  Fallback zu einfachen Embeddings...")
            return self._simple_embeddings(texts)
    
    def _simple_embeddings(self, texts):
        """Einfache Bag-of-Words Embeddings als Fallback"""
        # Verwende das bereits erstellte Vokabular wenn verfügbar
        if hasattr(self, '_vocabulary'):
            word_list = self._vocabulary
        else:
            # Erstelle Vokabular aus allen Texten (Dokumente + Query)
            all_texts = self.documents + texts if hasattr(self, 'documents') else texts
            all_words = set()
            for text in all_texts:
                words = text.lower().split()
                all_words.update(words)
            word_list = sorted(list(all_words))
            self._vocabulary = word_list

        embeddings = []
        for text in texts:
            words = text.lower().split()
            vector = [1 if word in words else 0 for word in word_list]
            embeddings.append(vector)

        return np.array(embeddings)
    
    def add_documents(self, docs):
        """Dokumente zur Wissensbasis hinzufügen"""
        print(f"\n📚 Füge {len(docs)} Dokumente hinzu...")
        
        start_time = time.time()
        self.documents.extend(docs)
        
        # Embeddings erstellen
        print("🔄 Erstelle Embeddings...")
        self.embeddings = self._get_embeddings(self.documents)
        
        elapsed_time = time.time() - start_time
        print(f"✅ {len(self.documents)} Dokumente indexiert in {elapsed_time:.2f}s")
        print(f"   Embedding-Dimension: {self.embeddings.shape}")
    
    def search(self, query, k=3):
        """Ähnlichste Dokumente finden"""
        if self.embeddings is None:
            print("❌ Keine Dokumente in der Wissensbasis!")
            return []
        
        print(f"\n🔍 Suche nach: '{query}'")
        
        # Query Embedding erstellen
        query_embedding = self._get_embeddings([query])
        
        # Ähnlichkeit berechnen
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Top-k Ergebnisse
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            similarity = similarities[idx]
            results.append({
                'document': self.documents[idx],
                'similarity': similarity,
                'rank': i + 1
            })
        
        return results
    
    def answer_question(self, question, use_llm=True):
        """Frage mit RAG beantworten"""
        print("\n" + "=" * 80)
        print(f"❓ FRAGE: {question}")
        print("=" * 80)
        
        # 1. Retrieval Phase
        print("\n🔍 RETRIEVAL-PHASE:")
        relevant_docs = self.search(question, k=3)
        
        if not relevant_docs:
            return "❌ Keine relevanten Dokumente gefunden."
        
        # Kontext zusammenstellen
        context = ""
        for doc in relevant_docs:
            print(f"  📄 Rang {doc['rank']} (Ähnlichkeit: {doc['similarity']:.3f})")
            doc_preview = doc['document'][:100]
            if len(doc['document']) > 100:
                doc_preview += "..."
            print(f"     {doc_preview}")
            context += doc['document'] + "\n\n"
        
        # 2. Generation Phase
        print(f"\n🤖 GENERATION-PHASE:")
        
        if not use_llm or not self.client:
            print("   Modus: Nur Retrieval")
            return f"Relevante Informationen:\n\n{context}"
        
        # Mit OpenAI LLM generieren
        print("   Modus: RAG mit OpenAI")
        return self._generate_answer(question, context)
    
    def _generate_answer(self, question, context):
        """Antwort mit OpenAI generieren"""
        prompt = f"""Basierend auf dem folgenden Kontext, beantworte die Frage präzise und hilfreich.

Kontext:
{context}

Frage: {question}

Antwort:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"❌ OpenAI Generation Fehler: {e}\n\nFallback - Relevanter Kontext:\n{context}"
    
    def show_knowledge_base(self):
        """Wissensbasis anzeigen"""
        print(f"\n📚 WISSENSBASIS ({len(self.documents)} Dokumente):")
        print("=" * 60)
        
        for i, doc in enumerate(self.documents, 1):
            preview = doc[:80]
            if len(doc) > 80:
                preview += "..."
            print(f"{i:2d}. {preview}")

def main():
    # RAG System initialisieren
    rag = FastRAG()
    
    # Kompakte Wissensbasis für schnelle Demo
    documents = [
        "Python ist eine vielseitige Programmiersprache für Webentwicklung, Datenanalyse und KI.",
        "Machine Learning ermöglicht Computern das Lernen aus Daten ohne explizite Programmierung.",
        "RAG kombiniert Informationsabruf mit Textgenerierung für bessere AI-Antworten.",
        "Vector Stores wie FAISS und Pinecone speichern Embeddings für schnelle Suche.",
        "LangChain ist ein Framework für LLM-basierte Anwendungen mit Tools und Chains.",
        "OpenAI bietet APIs für GPT-Modelle und Embeddings wie text-embedding-ada-002.",
        "Streamlit ermöglicht schnelle Web-Apps für Data Science ohne Frontend-Kenntnisse.",
        "Embeddings wandeln Text in numerische Vektoren für semantische Ähnlichkeit um."
    ]
    
    # Dokumente hinzufügen
    rag.add_documents(documents)
    
    # Wissensbasis anzeigen
    rag.show_knowledge_base()
    
    # Test-Fragen
    questions = [
        "Was ist Python?",
        "Wie funktioniert RAG?",
        "Was sind Embeddings?",
        "Welche Tools gibt es für LLM-Apps?"
    ]
    
    print("\n" + "🧪 AUTOMATISCHE TESTS".center(80, "="))
    
    for i, question in enumerate(questions, 1):
        print(f"\n🎯 TEST {i}/{len(questions)}")
        
        # Teste sowohl Retrieval-only als auch mit LLM
        print("\n📄 NUR RETRIEVAL:")
        answer_retrieval = rag.answer_question(question, use_llm=False)
        print(f"Ergebnis: {answer_retrieval[:150]}...")
        
        if rag.client:
            print("\n🤖 MIT LLM:")
            answer_llm = rag.answer_question(question, use_llm=True)
            print(f"Ergebnis: {answer_llm}")
        
        if i < len(questions):
            input("\n⏸️  Enter für nächsten Test...")
    
    # Interaktiver Teil
    print("\n" + "🎮 INTERAKTIVER MODUS".center(80, "="))
    print("Stellen Sie eigene Fragen! (Leer lassen zum Beenden)")
    
    try:
        while True:
            user_question = input("\n❓ Ihre Frage: ").strip()
            
            if not user_question:
                break
            
            # Zeige beide Modi
            print("\n" + "-" * 40)
            answer = rag.answer_question(user_question, use_llm=True)
            print(f"\n🎯 Antwort: {answer}")
    
    except KeyboardInterrupt:
        print("\n\n👋 RAG-System beendet!")
    
    print("\n" + "=" * 80)
    print("✅ Schnelles RAG-Demo abgeschlossen!")
    print("\n💡 Vorteile dieser Implementierung:")
    print("   • Sofortiger Start (keine Downloads)")
    print("   • Professionelle OpenAI Embeddings")
    print("   • Echte LLM-Integration")
    print("   • Fallback-Mechanismen")
    print("   • Live-Demo tauglich")
    
    print("\n🚀 Nächste Schritte:")
    print("   • Eigene Dokumente hinzufügen")
    print("   • PDF-Support implementieren")
    print("   • Verschiedene LLM-Modelle testen")
    print("   • Streamlit-Interface bauen")

if __name__ == "__main__":
    main()
