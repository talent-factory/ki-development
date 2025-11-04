"""
RAG mit echtem LLM: Vollständiges Retrieval Augmented Generation
Ausführung: uv run scripts/rag-praxis/rag_mit_llm.py

Dieses Skript zeigt:
- Integration von Mini-RAG mit echten LLMs
- OpenAI und Anthropic API Integration
- Qualitätsvergleich: Nur Retrieval vs. echtes RAG
- Praktische Anwendung mit verschiedenen Modellen
"""

import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# Optional: LLM Integrationen (nur wenn API-Keys verfügbar)
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

class RAGMitLLM:
    def __init__(self):
        print("🚀 RAG mit LLM System startet...")
        print("=" * 50)
        
        # Sentence Transformer für Embeddings
        print("📥 Lade Embedding-Modell...")
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # LLM Clients initialisieren
        self.openai_client = None
        self.anthropic_client = None
        
        self._setup_llm_clients()
        
        # RAG Komponenten
        self.documents = []
        self.embeddings = None
        self.index = None
        
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
    
    def add_documents(self, docs):
        """Dokumente zur Wissensbasis hinzufügen"""
        print(f"\n📚 Füge {len(docs)} Dokumente hinzu...")
        
        self.documents.extend(docs)
        
        # Embeddings erstellen
        all_embeddings = self.model.encode(self.documents, show_progress_bar=True)
        self.embeddings = np.array(all_embeddings)
        
        # FAISS Index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ {len(self.documents)} Dokumente indexiert")
    
    def search(self, query, k=3):
        """Relevante Dokumente finden"""
        if self.index is None:
            return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                similarity = 1 / (1 + distance)
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
            print(f"     {doc['document'][:100]}...")
            context += doc['document'] + "\n\n"
        
        # 2. Generation Phase
        print(f"\n🤖 GENERATION PHASE:")
        
        if not use_llm:
            # Nur Retrieval (wie Mini-RAG)
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
    
    def compare_approaches(self, question):
        """Vergleiche verschiedene Ansätze für dieselbe Frage"""
        print("\n" + "🔬 VERGLEICHSANALYSE".center(80, "="))
        print(f"Frage: {question}")
        
        approaches = [
            ("Nur Retrieval", False, None),
            ("RAG + OpenAI", True, "openai"),
            ("RAG + Anthropic", True, "anthropic")
        ]
        
        for name, use_llm, provider in approaches:
            print(f"\n📊 {name}:")
            print("-" * 40)
            
            start_time = time.time()
            answer = self.answer_question(question, use_llm, provider)
            elapsed = time.time() - start_time
            
            print(f"⏱️  Zeit: {elapsed:.2f}s")
            print(f"📝 Antwort: {answer[:200]}...")
            
            if len(approaches) > 1:
                input("\n⏸️  Enter für nächsten Ansatz...")

def main():
    # System initialisieren
    rag = RAGMitLLM()
    
    # Erweiterte Wissensbasis
    documents = [
        "Python ist eine interpretierte, objektorientierte Programmiersprache mit dynamischer Semantik. Sie wurde 1991 von Guido van Rossum entwickelt und ist bekannt für ihre einfache, lesbare Syntax.",
        
        "Machine Learning ist ein Teilbereich der künstlichen Intelligenz, der Algorithmen verwendet, um Muster in Daten zu erkennen und Vorhersagen zu treffen, ohne explizit programmiert zu werden.",
        
        "RAG (Retrieval Augmented Generation) ist eine Technik, die Large Language Models mit externen Wissensquellen verbindet, um aktuellere und präzisere Antworten zu generieren.",
        
        "Vector Databases wie FAISS, Pinecone und Weaviate sind speziell für die Speicherung und Suche von hochdimensionalen Vektoren optimiert, die in ML-Anwendungen verwendet werden.",
        
        "OpenAI entwickelt fortschrittliche AI-Systeme wie GPT-4, DALL-E und Codex. Das Unternehmen wurde 2015 gegründet und hat seinen Sitz in San Francisco.",
        
        "Anthropic wurde 2021 von ehemaligen OpenAI-Mitarbeitern gegründet und entwickelt Claude, einen AI-Assistenten, der auf Constitutional AI basiert.",
        
        "LangChain ist ein Framework für die Entwicklung von Anwendungen mit Large Language Models. Es bietet Tools für Chains, Agents, Memory und Integration mit verschiedenen Datenquellen.",
        
        "Embeddings sind numerische Repräsentationen von Text, Bildern oder anderen Daten in einem hochdimensionalen Vektorraum, die semantische Ähnlichkeiten erfassen.",
        
        "FAISS (Facebook AI Similarity Search) ist eine Open-Source-Bibliothek für effiziente Ähnlichkeitssuche und Clustering von dichten Vektoren, entwickelt von Meta AI.",
        
        "Streamlit ist ein Python-Framework für die schnelle Erstellung von Web-Apps für Data Science und Machine Learning, ohne Frontend-Kenntnisse zu benötigen."
    ]
    
    # Dokumente hinzufügen
    rag.add_documents(documents)
    
    # Test-Fragen
    questions = [
        "Was ist der Unterschied zwischen OpenAI und Anthropic?",
        "Wie funktioniert RAG?",
        "Welche Vorteile hat Python für Machine Learning?",
        "Was sind die besten Vector Databases?"
    ]
    
    print("\n" + "🧪 EINZELTESTS".center(80, "="))
    
    # Teste jede Frage mit verfügbaren LLMs
    for question in questions:
        print(f"\n🎯 Test: {question}")
        
        # Versuche verschiedene Ansätze
        if rag.openai_client:
            answer = rag.answer_question(question, True, "openai")
            print(f"\n✅ OpenAI Antwort:\n{answer}")
        
        if rag.anthropic_client:
            answer = rag.answer_question(question, True, "anthropic")
            print(f"\n✅ Anthropic Antwort:\n{answer}")
        
        if not rag.openai_client and not rag.anthropic_client:
            answer = rag.answer_question(question, False)
            print(f"\n📄 Retrieval-Only:\n{answer}")
        
        input("\n⏸️  Enter für nächste Frage...")
    
    # Vergleichsanalyse
    if rag.openai_client or rag.anthropic_client:
        print("\n" + "🔬 VERGLEICHSANALYSE".center(80, "="))
        rag.compare_approaches("Erkläre mir RAG in einfachen Worten")
    
    print("\n" + "=" * 80)
    print("✅ RAG mit LLM Demo abgeschlossen!")
    print("\n💡 Erkenntnisse:")
    print("   • RAG + LLM liefert natürlichere Antworten als nur Retrieval")
    print("   • Verschiedene LLMs haben unterschiedliche Stärken")
    print("   • Der Kontext aus der Retrieval-Phase ist entscheidend")
    print("   • RAG löst das Problem veralteter Trainingsdaten")

if __name__ == "__main__":
    main()
