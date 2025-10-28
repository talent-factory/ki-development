#!/usr/bin/env python3
"""
Mini-RAG Prototyp: Einfache Retrieval Augmented Generation

Dieses Skript implementiert einen grundlegenden RAG-Workflow:
1. Dokumente einlesen und in Chunks aufteilen
2. Embeddings für alle Chunks erstellen
3. Benutzer-Frage als Embedding
4. Ähnlichste Chunks finden (Retrieval)
5. Kontext + Frage an LLM senden (Generation)

Installation: uv pip install -r requirements.txt
Autor: Musterlösung für AI Development Kurs
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# Embedding-Modelle
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class Document:
    """Repräsentiert ein Dokument mit Metadaten"""
    content: str
    filename: str
    chunk_id: int
    embedding: np.ndarray = None

class MiniRAG:
    """Einfache RAG-Implementierung"""
    
    def __init__(self, embedding_model="sentence-transformers"):
        """
        Initialisiert das RAG-System
        
        Args:
            embedding_model: "openai" oder "sentence-transformers"
        """
        self.documents: List[Document] = []
        self.embedding_model_type = embedding_model
        
        # Embedding-Modell initialisieren
        if embedding_model == "openai" and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_model = None
            print("🤖 OpenAI Embeddings initialisiert")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.openai_client = None
            print("🤗 Sentence Transformers initialisiert")
        else:
            raise ValueError("Kein Embedding-Modell verfügbar!")
    
    def load_documents(self, doc_directory: str, chunk_size: int = 500):
        """
        Lädt Dokumente aus einem Verzeichnis
        
        Args:
            doc_directory: Pfad zum Dokumenten-Verzeichnis
            chunk_size: Maximale Anzahl Zeichen pro Chunk
        """
        doc_path = Path(doc_directory)
        if not doc_path.exists():
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {doc_directory}")
        
        print(f"📁 Lade Dokumente aus: {doc_directory}")
        
        for file_path in doc_path.glob("*.txt"):
            print(f"📄 Verarbeite: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Text in Chunks aufteilen
            chunks = self._split_text(content, chunk_size)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    filename=file_path.name,
                    chunk_id=i
                )
                self.documents.append(doc)
        
        print(f"✅ {len(self.documents)} Chunks aus {len(list(doc_path.glob('*.txt')))} Dateien geladen")
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Teilt Text in Chunks auf (einfache Implementierung)
        
        Args:
            text: Zu teilender Text
            chunk_size: Maximale Chunk-Größe
            
        Returns:
            Liste von Text-Chunks
        """
        # Einfache Aufteilung nach Absätzen und Sätzen
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_embeddings(self):
        """Erstellt Embeddings für alle Dokumente"""
        print("🔄 Erstelle Embeddings...")
        
        texts = [doc.content for doc in self.documents]
        
        if self.embedding_model_type == "openai" and self.openai_client:
            embeddings = self._get_openai_embeddings(texts)
        else:
            embeddings = self._get_sentence_transformer_embeddings(texts)
        
        # Embeddings zu Dokumenten hinzufügen
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding
        
        print(f"✅ Embeddings für {len(self.documents)} Chunks erstellt")
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Erstellt OpenAI Embeddings"""
        embeddings = []
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  📊 Progress: {i}/{len(texts)}")
            
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        
        return np.array(embeddings)
    
    def _get_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Erstellt Sentence Transformer Embeddings"""
        return self.embedding_model.encode(texts, show_progress_bar=True)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Findet die ähnlichsten Dokumente zu einer Anfrage
        
        Args:
            query: Benutzer-Anfrage
            top_k: Anzahl der zurückzugebenden Dokumente
            
        Returns:
            Liste von (Document, Similarity-Score) Tupeln
        """
        if not self.documents or not self.documents[0].embedding is not None:
            raise ValueError("Keine Embeddings verfügbar! Rufe create_embeddings() auf.")
        
        # Query-Embedding erstellen
        if self.embedding_model_type == "openai" and self.openai_client:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
        else:
            query_embedding = self.embedding_model.encode([query])
        
        # Ähnlichkeiten berechnen
        doc_embeddings = np.array([doc.embedding for doc in self.documents])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Top-K ähnlichste Dokumente finden
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], similarities[idx]))
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        Generiert eine Antwort basierend auf Kontext und Frage
        
        Args:
            query: Benutzer-Frage
            context_docs: Relevante Dokumente als Kontext
            
        Returns:
            Generierte Antwort
        """
        # Kontext zusammenstellen
        context = "\n\n".join([
            f"Dokument {i+1} ({doc.filename}):\n{doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Prompt erstellen
        prompt = f"""Basierend auf den folgenden Dokumenten, beantworte die Frage präzise und hilfreich.

KONTEXT:
{context}

FRAGE: {query}

ANTWORT:"""
        
        # LLM-Aufruf (vereinfacht für Demo)
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Du bist ein hilfreicher Assistent, der Fragen basierend auf gegebenen Dokumenten beantwortet."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Fehler bei LLM-Aufruf: {e}\n\nBasierend auf den Dokumenten kann ich folgende Informationen finden:\n{context[:500]}..."
        else:
            # Fallback: Einfache Antwort ohne LLM
            return f"[DEMO-MODUS] Basierend auf den gefundenen Dokumenten:\n\n{context[:500]}...\n\nFrage: {query}"
    
    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> Dict:
        """
        Vollständiger RAG-Workflow: Retrieve + Generate
        
        Args:
            question: Benutzer-Frage
            top_k: Anzahl der zu verwendenden Dokumente
            verbose: Detaillierte Ausgabe
            
        Returns:
            Dictionary mit Antwort und Metadaten
        """
        if verbose:
            print(f"\n🔍 Frage: {question}")
            print("=" * 50)
        
        # 1. Retrieval
        retrieved_docs = self.retrieve(question, top_k)
        
        if verbose:
            print(f"📚 Gefundene Dokumente ({top_k}):")
            for i, (doc, score) in enumerate(retrieved_docs):
                print(f"  {i+1}. {doc.filename} (Chunk {doc.chunk_id}) - Ähnlichkeit: {score:.3f}")
                print(f"     Preview: {doc.content[:100]}...")
        
        # 2. Generation
        context_docs = [doc for doc, _ in retrieved_docs]
        answer = self.generate_answer(question, context_docs)
        
        if verbose:
            print(f"\n💡 Antwort:")
            print(answer)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": [
                {
                    "filename": doc.filename,
                    "chunk_id": doc.chunk_id,
                    "similarity": score,
                    "content": doc.content
                }
                for doc, score in retrieved_docs
            ]
        }
    
    def save_index(self, filepath: str):
        """Speichert den Index für spätere Verwendung"""
        index_data = {
            "documents": [
                {
                    "content": doc.content,
                    "filename": doc.filename,
                    "chunk_id": doc.chunk_id,
                    "embedding": doc.embedding.tolist() if doc.embedding is not None else None
                }
                for doc in self.documents
            ],
            "embedding_model_type": self.embedding_model_type
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Index gespeichert: {filepath}")

def main():
    """Demo des Mini-RAG Systems"""
    print("🚀 Mini-RAG Prototyp gestartet")
    print("=" * 40)
    
    # RAG-System initialisieren
    try:
        if os.getenv("OPENAI_API_KEY"):
            rag = MiniRAG(embedding_model="openai")
        else:
            rag = MiniRAG(embedding_model="sentence-transformers")
    except Exception as e:
        print(f"❌ Fehler bei Initialisierung: {e}")
        return
    
    # Dokumente laden
    doc_directory = "test-dokumente"
    
    # Test-Dokumente erstellen falls nicht vorhanden
    if not Path(doc_directory).exists():
        create_test_documents(doc_directory)
    
    try:
        rag.load_documents(doc_directory)
        rag.create_embeddings()
    except Exception as e:
        print(f"❌ Fehler beim Laden der Dokumente: {e}")
        return
    
    # Test-Fragen
    test_questions = [
        "Was ist Machine Learning?",
        "Wie funktioniert Python?",
        "Welche Vorteile hat Cloud Computing?",
        "Was sind die Grundlagen der Webentwicklung?",
        "Erkläre mir Datenbanken"
    ]
    
    print("\n🧪 Teste RAG-System mit verschiedenen Fragen:")
    print("=" * 50)
    
    for question in test_questions:
        try:
            result = rag.query(question, top_k=2, verbose=True)
            print("\n" + "="*50)
        except Exception as e:
            print(f"❌ Fehler bei Frage '{question}': {e}")
    
    # Index speichern
    rag.save_index("mini_rag_index.json")
    
    print("\n🎉 Mini-RAG Demo abgeschlossen!")

def create_test_documents(doc_directory: str):
    """Erstellt Test-Dokumente für die Demo"""
    Path(doc_directory).mkdir(exist_ok=True)
    
    documents = {
        "machine_learning.txt": """
Machine Learning Grundlagen

Machine Learning ist ein Teilbereich der Künstlichen Intelligenz (KI), der es Computern ermöglicht, aus Daten zu lernen und Vorhersagen zu treffen, ohne explizit programmiert zu werden.

Haupttypen des Machine Learning:

1. Supervised Learning (Überwachtes Lernen)
Bei diesem Ansatz wird das Modell mit gelabelten Daten trainiert. Das bedeutet, dass sowohl die Eingabedaten als auch die gewünschten Ausgaben bekannt sind.

2. Unsupervised Learning (Unüberwachtes Lernen)
Hier arbeitet das Modell mit ungelabelten Daten und versucht, Muster und Strukturen in den Daten zu entdecken.

3. Reinforcement Learning (Verstärkendes Lernen)
Das Modell lernt durch Interaktion mit einer Umgebung und erhält Belohnungen oder Strafen für seine Aktionen.

Anwendungen von Machine Learning:
- Bilderkennung und Computer Vision
- Natürliche Sprachverarbeitung
- Empfehlungssysteme
- Medizinische Diagnose
- Autonome Fahrzeuge
""",
        
        "python_programming.txt": """
Python Programmierung

Python ist eine hochrangige, interpretierte Programmiersprache, die für ihre Einfachheit und Lesbarkeit bekannt ist. Sie wurde von Guido van Rossum entwickelt und 1991 erstmals veröffentlicht.

Hauptmerkmale von Python:

1. Einfache Syntax
Python verwendet eine klare und intuitive Syntax, die es Anfängern leicht macht, die Sprache zu erlernen.

2. Vielseitigkeit
Python kann für verschiedene Anwendungen verwendet werden:
- Webentwicklung (Django, Flask)
- Datenanalyse (Pandas, NumPy)
- Machine Learning (Scikit-learn, TensorFlow)
- Automatisierung und Scripting

3. Große Community
Python hat eine aktive Community, die kontinuierlich Bibliotheken und Tools entwickelt.

4. Plattformunabhängigkeit
Python-Code kann auf verschiedenen Betriebssystemen ausgeführt werden.

Grundlegende Python-Konzepte:
- Variablen und Datentypen
- Kontrollstrukturen (if, for, while)
- Funktionen und Module
- Objektorientierte Programmierung
- Exception Handling
""",
        
        "cloud_computing.txt": """
Cloud Computing

Cloud Computing bezeichnet die Bereitstellung von IT-Ressourcen über das Internet. Anstatt eigene Hardware und Software zu besitzen und zu warten, können Unternehmen diese Ressourcen von Cloud-Anbietern mieten.

Service-Modelle:

1. Infrastructure as a Service (IaaS)
Bereitstellung von virtuellen Maschinen, Speicher und Netzwerk-Ressourcen.

2. Platform as a Service (PaaS)
Bereitstellung einer Entwicklungsplattform mit Tools und Services für die Anwendungsentwicklung.

3. Software as a Service (SaaS)
Bereitstellung von fertigen Anwendungen über das Internet.

Vorteile von Cloud Computing:

1. Kosteneffizienz
Reduzierte Investitionskosten für Hardware und IT-Infrastruktur.

2. Skalierbarkeit
Ressourcen können je nach Bedarf schnell hoch- oder herunterskaliert werden.

3. Flexibilität
Zugriff auf Ressourcen von überall und zu jeder Zeit.

4. Automatische Updates
Software und Sicherheitsupdates werden automatisch vom Anbieter durchgeführt.

5. Disaster Recovery
Integrierte Backup- und Wiederherstellungslösungen.

Hauptanbieter:
- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform (GCP)
""",
        
        "web_development.txt": """
Webentwicklung Grundlagen

Webentwicklung umfasst die Erstellung und Wartung von Websites und Webanwendungen. Sie gliedert sich in verschiedene Bereiche:

Frontend-Entwicklung:
Das Frontend ist der Teil einer Website, den Benutzer sehen und mit dem sie interagieren.

Technologien:
1. HTML (HyperText Markup Language)
Strukturiert den Inhalt einer Webseite.

2. CSS (Cascading Style Sheets)
Definiert das Aussehen und Layout der Webseite.

3. JavaScript
Fügt Interaktivität und dynamische Funktionen hinzu.

Backend-Entwicklung:
Das Backend verarbeitet Daten und Geschäftslogik im Hintergrund.

Technologien:
- Programmiersprachen: Python, Java, PHP, Node.js
- Datenbanken: MySQL, PostgreSQL, MongoDB
- Server: Apache, Nginx

Full-Stack-Entwicklung:
Full-Stack-Entwickler arbeiten sowohl im Frontend als auch im Backend.

Moderne Entwicklungsansätze:
1. Responsive Design
Websites passen sich verschiedenen Bildschirmgrößen an.

2. Progressive Web Apps (PWAs)
Webanwendungen mit app-ähnlichen Funktionen.

3. Single Page Applications (SPAs)
Dynamische Webanwendungen, die ohne Seitenneuladen funktionieren.

Frameworks und Tools:
- Frontend: React, Vue.js, Angular
- Backend: Django, Express.js, Spring
- Versionskontrolle: Git
- Build-Tools: Webpack, Vite
"""
    }
    
    for filename, content in documents.items():
        with open(Path(doc_directory) / filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"📝 {len(documents)} Test-Dokumente erstellt in: {doc_directory}")

if __name__ == "__main__":
    main()
