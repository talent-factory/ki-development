"""
Wissensbasis für den AI Development Kurs-Assistenten
"""

COURSE_KNOWLEDGE = """
# AI Development Kurs - Wissensbasis

## Kursübersicht
Der AI Development Kurs umfasst 6 Abende mit je 4 Lektionen (50 Min + 10 Min Pause).

### Abend 1: Grundlagen und No-Code Tools
- Einführung in AI/ML Konzepte
- No-Code/Low-Code AI-Plattformen
- Flowise und Langflow
- Praktische Übungen mit AI-Tools

### Abend 2: Python Setup und Deployment
- Python-Entwicklungsumgebung einrichten
- Git/GitHub Grundlagen
- Streamlit AI-Anwendungen entwickeln
- Erstes Deployment auf Streamlit Cloud
- Einführung in Vektordatenbanken

### Abend 3: LangChain und RAG
- LangChain Framework
- Retrieval Augmented Generation (RAG)
- Vector Stores und Embeddings
- Praktische RAG-Implementierung

### Abend 4: Fortgeschrittene Konzepte
- Erweiterte LangChain Patterns
- Multi-Agent Systeme
- Performance Optimierung

### Abend 5: Eigene Projekte
- Projektplanung und -umsetzung
- Individuelle Betreuung
- Code Reviews

### Abend 6: Präsentationen
- Projektpräsentationen
- Feedback und Diskussion
- Ausblick und Weiterbildung

## Wichtige Konzepte

### Large Language Models (LLMs)
- GPT-4, Claude, Gemini
- Prompt Engineering
- API-Integration

### Vector Stores
- Speicherung von Embeddings
- Similarity Search
- FAISS, ChromaDB, Pinecone

### Embeddings
- Text-zu-Vektor Transformation
- Semantische Ähnlichkeit
- OpenAI Embeddings, Sentence Transformers

### RAG (Retrieval Augmented Generation)
- Kombination von Retrieval und Generation
- Kontextuelle Antworten
- Wissensbasis-Integration

### No-Code/Low-Code Plattformen
- Flowise: LangChain-basierte visuelle Entwicklung
- Langflow: Python-basierte, model-agnostische Plattform
- n8n: Open-Source Automation mit AI-Integration
- Make AI: Kommerzielle Workflow-Automation

### Python für AI
- Streamlit für Web-Apps
- LangChain für LLM-Anwendungen
- OpenAI und Anthropic APIs
- Vector Databases (FAISS, ChromaDB)

## Tools und Technologien
- Python, Streamlit
- LangChain, OpenAI API, Anthropic API
- Git/GitHub
- Flowise, Langflow
- Vector Databases
- Deployment auf Streamlit Cloud

## Häufige Fragen

### Wie installiere ich Python?
1. Python von python.org herunterladen
2. Während Installation "Add to PATH" aktivieren
3. Installation mit `python --version` testen

### Wie erstelle ich ein GitHub Repository?
1. Auf GitHub.com anmelden
2. "New repository" klicken
3. Repository-Name eingeben
4. "Create repository" klicken

### Wie deploye ich auf Streamlit Cloud?
1. Code auf GitHub pushen
2. share.streamlit.io besuchen
3. Mit GitHub Account anmelden
4. Repository auswählen und deployen

### Was ist der Unterschied zwischen Flowise und Langflow?
- Flowise: Mehr Integrationen, einfacher für Anfänger
- Langflow: Flexibler, Python-basiert, größere Community

### Wie funktioniert RAG?
1. Dokumente in Embeddings umwandeln
2. In Vector Store speichern
3. Bei Anfrage ähnliche Dokumente finden
4. Kontext an LLM senden für Antwort
"""

def get_system_prompt():
    """Erstellt den System-Prompt für den AI Assistenten"""
    return f"""Du bist ein hilfreicher AI-Assistent für den AI Development Kurs. 
    
Deine Aufgabe ist es, Fragen der Kursteilnehmer zu beantworten basierend auf folgender Wissensbasis:

{COURSE_KNOWLEDGE}

Antworte immer:
- Freundlich und hilfsbereit
- Auf Deutsch
- Präzise und verständlich
- Mit praktischen Beispielen wenn möglich
- Ehrlich, wenn du etwas nicht weißt

Wenn eine Frage nicht direkt mit dem Kurs zusammenhängt, verweise höflich auf den Kurskontext zurück.

Verwende Emojis und Markdown-Formatierung für bessere Lesbarkeit.
"""
