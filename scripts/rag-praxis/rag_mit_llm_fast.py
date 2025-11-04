"""
Schnelles RAG mit echten LLMs - OpenAI/Anthropic Integration
Ausführung: uv run scripts/rag-praxis/rag_mit_llm_fast.py

Dieses Skript zeigt:
- Sofortige RAG-Integration mit echten LLMs
- OpenAI und Anthropic API Vergleich
- Qualitätsvergleich: Retrieval vs. echtes RAG
- Praktische Anwendung ohne Wartezeiten
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
    print("💡 Hinweis: python-dotenv nicht installiert")

# LLM Integrationen
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class FastRAGWithLLM:
    def __init__(self):
        print("🚀 Schnelles RAG mit LLM System startet...")
        print("=" * 60)
        
        # LLM Clients initialisieren
        self.openai_client = None
        self.anthropic_client = None
        self._setup_llm_clients()
        
        # RAG Komponenten
        self.documents = []
        self.embeddings = None
        
        print("✅ RAG-System bereit")
    
    def _setup_llm_clients(self):
        """LLM Clients basierend auf verfügbaren API-Keys einrichten"""
        
        # OpenAI Setup
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = openai.OpenAI()
                print("✅ OpenAI Client verfügbar")
            except Exception as e:
                print(f"⚠️  OpenAI Setup fehlgeschlagen: {e}")
        else:
            print("⚠️  OpenAI nicht verfügbar (API-Key oder Library fehlt)")
        
        # Anthropic Setup
        if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.anthropic_client = anthropic.Anthropic()
                print("✅ Anthropic Client verfügbar")
            except Exception as e:
                print(f"⚠️  Anthropic Setup fehlgeschlagen: {e}")
        else:
            print("⚠️  Anthropic nicht verfügbar (API-Key oder Library fehlt)")
        
        if not self.openai_client and not self.anthropic_client:
            print("💡 Hinweis: Ohne API-Keys läuft nur die Retrieval-Demo")
    
    def _get_embeddings(self, texts):
        """Embeddings mit OpenAI erstellen (oder Fallback)"""
        if not self.openai_client:
            return self._simple_embeddings(texts)
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return np.array(embeddings)
        
        except Exception as e:
            print(f"❌ OpenAI Embeddings Fehler: {e}")
            return self._simple_embeddings(texts)
    
    def _simple_embeddings(self, texts):
        """Einfache Bag-of-Words Embeddings als Fallback"""
        if hasattr(self, '_vocabulary'):
            word_list = self._vocabulary
        else:
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
        """Relevante Dokumente finden"""
        if self.embeddings is None:
            return []
        
        query_embedding = self._get_embeddings([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
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
    
    def generate_with_openai(self, question, context):
        """Antwort mit OpenAI generieren"""
        if not self.openai_client:
            return "❌ OpenAI nicht verfügbar"
        
        prompt = f"""Basierend auf dem folgenden Kontext, beantworte die Frage präzise und hilfreich.

Kontext:
{context}

Frage: {question}

Antwort:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ OpenAI Fehler: {e}"
    
    def generate_with_anthropic(self, question, context):
        """Antwort mit Anthropic Claude generieren"""
        if not self.anthropic_client:
            return "❌ Anthropic nicht verfügbar"
        
        prompt = f"""Basierend auf dem folgenden Kontext, beantworte die Frage präzise und hilfreich.

Kontext:
{context}

Frage: {question}

Antwort:"""
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"❌ Anthropic Fehler: {e}"
    
    def answer_question(self, question, use_llm=True, llm_provider="openai"):
        """Frage mit vollständigem RAG beantworten"""
        print("\n" + "=" * 80)
        print(f"❓ FRAGE: {question}")
        print("=" * 80)
        
        # 1. Retrieval Phase
        print("\n🔍 RETRIEVAL PHASE:")
        relevant_docs = self.search(question, k=3)
        
        if not relevant_docs:
            return "❌ Keine relevanten Dokumente gefunden."
        
        # Kontext zusammenstellen
        context = ""
        for doc in relevant_docs:
            print(f"  📄 Rang {doc['rank']} (Ähnlichkeit: {doc['similarity']:.3f})")
            doc_preview = doc['document'][:80]
            if len(doc['document']) > 80:
                doc_preview += "..."
            print(f"     {doc_preview}")
            context += doc['document'] + "\n\n"
        
        # 2. Generation Phase
        print("\n🤖 GENERATION PHASE:")
        
        if not use_llm:
            print("   Modus: Nur Retrieval (kein LLM)")
            return f"Relevante Informationen gefunden:\n\n{context}"
        
        # Mit LLM generieren
        print(f"   Modus: RAG mit {llm_provider.upper()}")
        
        if llm_provider == "openai":
            answer = self.generate_with_openai(question, context)
        elif llm_provider == "anthropic":
            answer = self.generate_with_anthropic(question, context)
        else:
            answer = "❌ Unbekannter LLM Provider"
        
        return answer
    
    def compare_llms(self, question):
        """Vergleiche OpenAI vs. Anthropic für dieselbe Frage"""
        print("\n" + "🔬 LLM-VERGLEICH".center(80, "="))
        print(f"Frage: {question}")
        
        providers = []
        if self.openai_client:
            providers.append(("OpenAI GPT-3.5", "openai"))
        if self.anthropic_client:
            providers.append(("Anthropic Claude", "anthropic"))
        
        if not providers:
            print("❌ Keine LLMs verfügbar für Vergleich")
            return
        
        for name, provider in providers:
            print(f"\n📊 {name}:")
            print("-" * 50)
            
            start_time = time.time()
            answer = self.answer_question(question, True, provider)
            elapsed = time.time() - start_time
            
            print(f"\n⏱️  Zeit: {elapsed:.2f}s")
            print(f"📝 Antwort: {answer}")
            
            if len(providers) > 1:
                input("\n⏸️  Enter für nächsten LLM...")

def main():
    # System initialisieren
    rag = FastRAGWithLLM()
    
    # Kompakte Wissensbasis für schnelle Demo
    documents = [
        "Python ist eine vielseitige Programmiersprache für Webentwicklung, Datenanalyse und KI. Entwickelt von Guido van Rossum 1991.",
        "Machine Learning ermöglicht Computern das Lernen aus Daten ohne explizite Programmierung. Teilbereich der künstlichen Intelligenz.",
        "RAG kombiniert Informationsabruf mit Textgenerierung für bessere AI-Antworten. Löst Problem veralteter Trainingsdaten.",
        "Vector Stores wie FAISS und Pinecone speichern Embeddings für schnelle Ähnlichkeitssuche in ML-Anwendungen.",
        "OpenAI entwickelt GPT-4, DALL-E und Codex. Gegründet 2015 in San Francisco, führend in AI-Forschung.",
        "Anthropic wurde 2021 von Ex-OpenAI-Mitarbeitern gegründet. Entwickelt Claude AI-Assistenten mit Constitutional AI.",
        "LangChain ist ein Framework für LLM-Anwendungen mit Tools für Chains, Agents und Datenintegration.",
        "Embeddings sind numerische Repräsentationen von Text in hochdimensionalen Vektorräumen für semantische Ähnlichkeit."
    ]
    
    # Dokumente hinzufügen
    rag.add_documents(documents)
    
    # Test-Fragen
    questions = [
        "Was ist der Unterschied zwischen OpenAI und Anthropic?",
        "Wie funktioniert RAG?",
        "Welche Vorteile hat Python für Machine Learning?"
    ]
    
    print("\n" + "🧪 SCHNELLE TESTS".center(80, "="))
    
    # Teste erste Frage mit verfügbaren LLMs
    test_question = questions[0]
    
    if rag.openai_client or rag.anthropic_client:
        # LLM-Vergleich
        rag.compare_llms(test_question)
    else:
        # Nur Retrieval
        print(f"\n🎯 Test (nur Retrieval): {test_question}")
        answer = rag.answer_question(test_question, False)
        print(f"\n📄 Retrieval-Only Antwort:\n{answer}")
    
    print("\n" + "=" * 80)
    print("✅ Schnelles RAG mit LLM Demo abgeschlossen!")
    print("\n💡 Erkenntnisse:")
    print("   • RAG + LLM liefert natürlichere Antworten als nur Retrieval")
    print("   • OpenAI und Anthropic haben unterschiedliche Antwort-Stile")
    print("   • Sofortige Performance dank API-basierter Embeddings")
    print("   • RAG löst das Problem veralteter Trainingsdaten")

if __name__ == "__main__":
    main()
