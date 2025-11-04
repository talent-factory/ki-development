# RAG in der Praxis - Kursunterlagen

## 📚 Übersicht

Diese Unterlagen begleiten den Kursabend "RAG in der Praxis" und führen Sie schrittweise von der Theorie zur praktischen Implementierung von Retrieval Augmented Generation (RAG) Systemen.

## 🎯 Lernziele

Nach diesem Kursabend können Sie:
- ✅ Das RAG-Konzept erklären und die einzelnen Schritte benennen
- ✅ Embeddings erstellen und deren Ähnlichkeit berechnen  
- ✅ Ein einfaches RAG-System implementieren und testen
- ✅ Anwendungsfälle für RAG in Ihrem Arbeitsbereich identifizieren
- ✅ Die Verbindung zwischen Theorie und Praxis herstellen

## 📋 Kursstruktur (4 x 50 Minuten)

### 🕐 Lektion 1: RAG-Grundlagen & Embeddings verstehen
- **Theorie**: Was ist RAG? Warum brauchen wir es?
- **Praxis**: `embedding_test_simple.py` - Sofortige Ähnlichkeit mit OpenAI

### 🕑 Lektion 2: Schnelles RAG System
- **Theorie**: RAG-Architektur verstehen
- **Praxis**: `mini_rag_fast.py` - Vollständiges RAG in < 2 Sekunden

### 🕒 Lektion 3: LLM-Vergleich & Integration
- **Theorie**: OpenAI vs. Anthropic
- **Praxis**: Verschiedene Modelle testen und vergleichen

### 🕓 Lektion 4: Anwendungsfälle & Ausblick
- **Praxis**: Eigene Anwendungsfälle entwickeln
- **Diskussion**: Nächste Schritte und Herausforderungen

## 🛠️ Setup & Installation

### Voraussetzungen
- Python 3.10+
- `uv` Package Manager (bereits installiert)
- Optional: OpenAI/Anthropic API-Keys für Lektion 3

### Installation
```bash
# Aus dem Projektverzeichnis ai-development:
cd /path/to/ai-development

# Abhängigkeiten installieren
uv pip install -r scripts/rag-praxis/requirements.txt

# Test: Embedding-Demo ausführen
uv run scripts/rag-praxis/embedding_test.py
```

## 📁 Dateien im Überblick

| Datei | Beschreibung | Lektion |
|-------|-------------|---------|
| `06-rag-praxis.adoc` | Theoretische Grundlagen (PDF-Export) | 1-4 |
| `embedding_test.py` | Embeddings & Ähnlichkeit verstehen | 1 |
| `mini_rag.py` | Einfaches RAG-System ohne LLM | 2 |
| `rag_mit_llm.py` | Vollständiges RAG mit OpenAI/Anthropic | 3 |
| `requirements.txt` | Python-Abhängigkeiten | Setup |

## 🚀 Schnellstart

### 1. Embeddings testen
```bash
uv run scripts/rag-praxis/embedding_test.py
```
**Was passiert**: Berechnet Ähnlichkeit zwischen verschiedenen Texten

### 2. Mini-RAG ausprobieren  
```bash
uv run scripts/rag-praxis/mini_rag.py
```
**Was passiert**: Baut eine Wissensbasis auf und beantwortet Fragen

### 3. Echtes RAG (mit API-Keys)
```bash
# API-Keys setzen (optional)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

uv run scripts/rag-praxis/rag_mit_llm.py
```
**Was passiert**: Integriert echte LLMs für natürliche Antworten

## 💡 Wichtige Konzepte

### RAG-Prozess
```
Benutzeranfrage → Embedding → Ähnlichkeitssuche → Relevante Dokumente → LLM + Kontext → Antwort
```

### Embeddings
- **Was**: Numerische Repräsentation von Text (z.B. 384-dimensionaler Vektor)
- **Warum**: Ermöglicht semantische Ähnlichkeitssuche
- **Wie**: Sentence Transformers (läuft lokal, keine API nötig)

### Vector Stores
- **FAISS**: Schnelle lokale Suche (Facebook AI)
- **ChromaDB**: Einfache Integration, gut für Prototyping
- **Pinecone**: Cloud-basiert, skaliert automatisch

## 🎮 Interaktive Übungen

### Übung 1: Eigene Texte testen
Modifizieren Sie `embedding_test.py`:
- Fügen Sie Texte aus Ihrem Arbeitsbereich hinzu
- Testen Sie verschiedene Sprachen
- Experimentieren Sie mit Fachbegriffen

### Übung 2: Wissensbasis erweitern
Erweitern Sie `mini_rag.py`:
- Laden Sie eigene Dokumente
- Implementieren Sie PDF-Support
- Verbessern Sie das Text-Chunking

### Übung 3: Anwendungsfall entwickeln
Denken Sie an Ihren Arbeitsbereich:
- Welche Dokumente könnten von RAG profitieren?
- Welche Fragen stellen Kollegen/Kunden häufig?
- Wie aktuell müssen die Informationen sein?

## 🔧 Troubleshooting

### Häufige Probleme

**Problem**: `ModuleNotFoundError: No module named 'sentence_transformers'`
```bash
# Lösung: Abhängigkeiten installieren
uv pip install -r scripts/rag-praxis/requirements.txt
```

**Problem**: `uv run` hängt oder findet Datei nicht
```bash
# Lösung: Aus Projektverzeichnis ausführen
cd /path/to/ai-development
uv run scripts/rag-praxis/embedding_test.py
```

**Problem**: Langsame erste Ausführung
```bash
# Normal: Sentence Transformer lädt Modell beim ersten Mal herunter
# Dauert ~1-2 Minuten, danach ist es gecacht
```

### API-Key Setup (optional für Lektion 3)

**OpenAI**:
```bash
export OPENAI_API_KEY="sk-..."
```

**Anthropic**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 📊 Erwartete Ausgaben

### embedding_test.py
```
🔍 Embedding-Test startet...
✅ Modell geladen
📊 Embeddings erstellt: (9, 384)

🎯 Ähnlichkeitsanalyse:
🔥 Ähnlichkeit: 0.756
   'Der König regiert das Land'
   'Die Königin herrscht über das Reich'
```

### mini_rag.py
```
🚀 Mini-RAG System startet...
📚 Füge 10 Dokumente zur Wissensbasis hinzu...
✅ 10 Dokumente indexiert in 2.34s

❓ FRAGE: Was ist Python?
🎯 RETRIEVAL-ERGEBNISSE:
  📄 Rang 1 (Ähnlichkeit: 0.892)
     Python ist eine vielseitige Programmiersprache...
```

## 🎯 Nächste Schritte

Nach dem Kurs können Sie:

1. **Experimentieren**: Testen Sie RAG mit Ihren eigenen Dokumenten
2. **Erweitern**: Fügen Sie PDF-Support und besseres Chunking hinzu
3. **Integrieren**: Verbinden Sie mit OpenAI/Anthropic für echte Antworten  
4. **Anwenden**: Identifizieren Sie einen konkreten Anwendungsfall
5. **Skalieren**: Nutzen Sie Cloud Vector Stores für größere Projekte

## 📚 Weiterführende Ressourcen

- **LangChain Dokumentation**: https://python.langchain.com/
- **Sentence Transformers**: https://www.sbert.net/
- **FAISS Tutorial**: https://github.com/facebookresearch/faiss/wiki
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **Vector Database Vergleich**: Siehe `homework-week-2-3.adoc`

---

**Viel Erfolg beim Experimentieren mit RAG! 🚀**
