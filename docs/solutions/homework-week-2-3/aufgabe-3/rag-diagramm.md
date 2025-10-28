# RAG-Prozess Flussdiagramm

## Überblick

Retrieval Augmented Generation (RAG) kombiniert die Stärken von Informationsretrieval und Textgenerierung, um präzise und kontextbezogene Antworten zu liefern.

## Flussdiagramm des RAG-Prozesses

```
┌─────────────────────┐
│   1. Dokumente/     │
│   Wissensbasis      │
│                     │
│ • PDF-Dateien       │
│ • Webseiten         │
│ • Datenbanken       │
│ • APIs              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   2. Embedding-     │
│   Modell            │
│                     │
│ • OpenAI Ada-002    │
│ • Sentence-BERT     │
│ • Custom Models     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   3. Vektordaten-   │
│   bank              │
│                     │
│ • FAISS             │
│ • Pinecone          │
│ • ChromaDB          │
│ • Weaviate          │
└─────────────────────┘
           ▲
           │ (Indexierung)
           │
┌─────────────────────┐
│   4. Benutzer-      │
│   Anfrage           │
│                     │
│ "Wie funktioniert  │
│  maschinelles       │
│  Lernen?"           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   5. Query          │
│   Embedding         │
│                     │
│ Anfrage wird in     │
│ Vektor umgewandelt  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   6. Similarity     │
│   Search            │
│                     │
│ Cosine Similarity   │
│ zwischen Query und  │
│ gespeicherten Docs  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   7. Kontext-       │
│   Retrieval         │
│                     │
│ Top-K ähnlichste    │
│ Dokumente werden    │
│ ausgewählt          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   8. LLM mit        │
│   Prompt + Kontext  │
│                     │
│ "Basierend auf:     │
│ [Kontext]           │
│ Beantworte: [Frage]"│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   9. Generierte     │
│   Antwort           │
│                     │
│ Kontextbasierte,    │
│ präzise Antwort     │
│ mit Quellenangaben  │
└─────────────────────┘
```

## Detaillierte Schritt-Erklärungen

### 1. Dokumente/Wissensbasis
**Was passiert:** Sammlung und Vorbereitung der Datenquellen
**Erklärung:** Alle relevanten Dokumente werden gesammelt, bereinigt und in verarbeitbare Chunks aufgeteilt. Typische Chunk-Grösse: 200-1000 Tokens mit Überlappung.

### 2. Embedding-Modell
**Was passiert:** Transformation von Text in numerische Vektoren
**Erklärung:** Jeder Dokumenten-Chunk wird durch ein Embedding-Modell in einen hochdimensionalen Vektor umgewandelt, der die semantische Bedeutung repräsentiert.

### 3. Vektordatenbank
**Was passiert:** Speicherung und Indexierung der Embeddings
**Erklärung:** Die Vektoren werden in einer spezialisierten Datenbank gespeichert, die für schnelle Ähnlichkeitssuchen optimiert ist.

### 4. Benutzer-Anfrage
**Was passiert:** Eingabe der Frage durch den Nutzer
**Erklärung:** Der Benutzer stellt eine natürlichsprachliche Frage, die das System beantworten soll.

### 5. Query Embedding
**Was passiert:** Umwandlung der Frage in einen Vektor
**Erklärung:** Die Benutzeranfrage wird mit demselben Embedding-Modell in einen Vektor transformiert, um Konsistenz zu gewährleisten.

### 6. Similarity Search
**Was passiert:** Suche nach ähnlichen Dokumenten
**Erklärung:** Das System berechnet die Ähnlichkeit zwischen dem Query-Vektor und allen gespeicherten Dokumenten-Vektoren, meist mit Cosine Similarity.

### 7. Kontext-Retrieval
**Was passiert:** Auswahl der relevantesten Dokumente
**Erklärung:** Die Top-K ähnlichsten Dokumente (meist 3-10) werden als Kontext für die Antwortgenerierung ausgewählt.

### 8. LLM mit Prompt + Kontext
**Was passiert:** Generierung der Antwort basierend auf Kontext
**Erklärung:** Ein Large Language Model erhält einen strukturierten Prompt mit der ursprünglichen Frage und dem abgerufenen Kontext.

### 9. Generierte Antwort
**Was passiert:** Ausgabe der kontextbasierten Antwort
**Erklärung:** Das LLM generiert eine Antwort, die auf den abgerufenen Dokumenten basiert und idealerweise Quellenangaben enthält.

## Technische Details

### Embedding-Dimensionen
- **OpenAI Ada-002:** 1536 Dimensionen
- **Sentence-BERT:** 384-768 Dimensionen
- **Custom Models:** Variabel

### Similarity-Metriken
- **Cosine Similarity:** Meist verwendet
- **Euclidean Distance:** Alternative
- **Dot Product:** Für normalisierte Vektoren

### Chunk-Strategien
- **Fixed Size:** Feste Anzahl Tokens
- **Semantic Chunking:** Basierend auf Absätzen/Sätzen
- **Sliding Window:** Mit Überlappung

## Vorteile von RAG

1. **Aktuelle Informationen:** Kann mit neuen Daten erweitert werden
2. **Nachvollziehbarkeit:** Quellenangaben möglich
3. **Kosteneffizienz:** Kein Fine-Tuning nötig
4. **Flexibilität:** Verschiedene Datenquellen kombinierbar
5. **Präzision:** Reduziert Halluzinationen durch Kontext

## Herausforderungen

1. **Chunk-Qualität:** Optimale Segmentierung schwierig
2. **Embedding-Qualität:** Modell muss zur Domäne passen
3. **Retrieval-Präzision:** Relevante Dokumente finden
4. **Kontext-Länge:** LLM-Token-Limits beachten
5. **Latenz:** Zusätzliche Schritte erhöhen Antwortzeit
