# LangChain Konzept-Map

## Zentral: LangChain

**Definition:** Ein Framework für die Entwicklung von Anwendungen mit Large Language Models (LLMs), das modulare Komponenten und Abstraktionen bereitstellt.

## Hauptäste und Konzepte

### 1. Chains
**Definition:** Sequenzen von Aufrufen, die verschiedene Komponenten miteinander verbinden.

**Anwendungsbeispiel:** 
- SimpleChain: Prompt → LLM → Output
- SequentialChain: Mehrere Schritte nacheinander
- RouterChain: Bedingte Verzweigung basierend auf Input

**Verbindungen:**
- Nutzt Prompts als Input
- Kann Memory für Kontext verwenden
- Kann Tools für externe Funktionen einbinden

### 2. Prompts
**Definition:** Template-System für strukturierte und wiederverwendbare Prompt-Erstellung.

**Anwendungsbeispiel:**
```python
PromptTemplate(
    input_variables=["topic"],
    template="Erkläre mir {topic} in einfachen Worten."
)
```

**Verbindungen:**
- Zentrale Eingabe für Chains
- Kann dynamische Variablen enthalten
- Arbeitet mit Memory für Kontext

### 3. Memory
**Definition:** Komponenten zur Speicherung und Verwaltung von Konversationshistorie und Zustand.

**Anwendungsbeispiel:**
- ConversationBufferMemory: Speichert gesamte Konversation
- ConversationSummaryMemory: Fasst lange Gespräche zusammen
- VectorStoreRetrieverMemory: Nutzt Vektordatenbank für Kontext

**Verbindungen:**
- Erweitert Chains um Zustandsverwaltung
- Arbeitet mit Vector Stores für persistente Speicherung
- Beeinflusst Prompt-Generierung

### 4. Agents
**Definition:** Autonome Entitäten, die Tools verwenden und Entscheidungen treffen können.

**Anwendungsbeispiel:**
- ReAct Agent: Reasoning + Acting Pattern
- Conversational Agent: Für Chatbots
- Plan-and-Execute Agent: Für komplexe Aufgaben

**Verbindungen:**
- Nutzt Tools für externe Funktionen
- Verwendet Chains für Entscheidungsfindung
- Kann Memory für Kontext verwenden

### 5. Tools
**Definition:** Externe Funktionen und APIs, die von Agents verwendet werden können.

**Anwendungsbeispiel:**
- Web Search Tool
- Calculator Tool
- Database Query Tool
- Custom Python Functions

**Verbindungen:**
- Werden von Agents orchestriert
- Erweitern Capabilities von LLMs
- Können mit Vector Stores interagieren

### 6. Vector Stores
**Definition:** Speichersysteme für Embeddings und semantische Suche.

**Anwendungsbeispiel:**
- FAISS für lokale Entwicklung
- Pinecone für Cloud-Deployment
- ChromaDB für einfache Persistierung

**Verbindungen:**
- Arbeitet eng mit Embeddings zusammen
- Zentral für RAG-Implementierungen
- Kann von Memory-Komponenten genutzt werden

### 7. Embeddings
**Definition:** Vektorrepräsentationen von Text für semantische Ähnlichkeitsberechnungen.

**Anwendungsbeispiel:**
- OpenAI Embeddings
- HuggingFace Embeddings
- Custom Embedding Models

**Verbindungen:**
- Essentiell für Vector Stores
- Ermöglicht semantische Suche
- Basis für RAG-Systeme

## Beziehungen zwischen Komponenten

```
LangChain Framework
├── Chains (Orchestrierung)
│   ├── verwendet → Prompts
│   ├── nutzt → Memory
│   └── kann einbinden → Tools
├── Agents (Autonome Akteure)
│   ├── verwendet → Tools
│   ├── nutzt → Chains
│   └── kann nutzen → Memory
└── RAG-Pipeline
    ├── Vector Stores ← speichert → Embeddings
    ├── Memory ← kann nutzen → Vector Stores
    └── Chains ← orchestriert → RAG-Prozess
```

## Kernvorteile von LangChain

1. **Modularität:** Wiederverwendbare Komponenten
2. **Abstraktion:** Vereinfacht komplexe LLM-Workflows
3. **Flexibilität:** Unterstützt verschiedene LLM-Provider
4. **Skalierbarkeit:** Von Prototyp bis Produktion
5. **Community:** Grosse Entwicklergemeinschaft und Ecosystem

## Anwendungsszenarien

- **Chatbots** mit Memory und Tools
- **RAG-Systeme** für Wissensdatenbanken
- **Automatisierte Workflows** mit Agents
- **Datenanalyse** mit Tool-Integration
- **Content-Generierung** mit Template-System
