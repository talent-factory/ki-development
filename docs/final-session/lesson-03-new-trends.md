# 🧠 Lektion 3: Neue AI-Tools & LLM-Evaluation

**Zeit:** 20:00 - 20:50 (50 Minuten)  
**Ziel:** Aktuelle AI-Entwicklungen verstehen und neue Tools praktisch erkunden

## 🎯 **Lernziele**
Nach dieser Lektion können die Teilnehmenden:
- Aktuelle AI-Trends und deren Relevanz einschätzen
- Neue Tools (codemap, brepl) installieren und nutzen
- LLM-Evaluation-Methoden verstehen und anwenden
- AI-Coding-Assistenten systematisch bewerten

## ⏰ **Zeitplan (50 Min)**

### 📰 **Phase 1: Aktuelle AI-Entwicklungen (15 Min)**
```
20:00-20:15  AI News vom 25.11.2025 - Was ist relevant?
```

#### **Claude Opus 4.5 - Was ist neu? (8 Min)**
```
🔥 BREAKING: Claude Opus 4.5 Release
- Verbesserte Code-Generierung (+40% Genauigkeit)
- Längerer Kontext (200k Tokens)
- Bessere Multimodalität
- Schwierigkeiten bei der Evaluation neuer LLMs
```

**Diskussion:** "Warum wird LLM-Evaluation immer schwieriger?"

**Antworten sammeln:**
- Modelle werden zu ähnlich in Standard-Benchmarks
- Subjektive Qualitätsunterschiede schwer messbar
- Spezialisierung auf verschiedene Aufgaben
- Gaming der Benchmarks durch Training

#### **AI-Coding-Assistenten Framework (7 Min)**
```
📊 Framework für Produktivitätsmessung:
1. Code-Qualität (Bugs, Performance, Maintainability)
2. Entwicklungsgeschwindigkeit (Lines/Hour, Features/Sprint)
3. Lernkurve (Time to Productivity)
4. Entwicklerzufriedenheit (Survey-basiert)
```

**Live-Umfrage:** "Wie würdet ihr eure Produktivität mit AI-Tools bewerten?"

### 🛠️ **Phase 2: Neue Tools Exploration (20 Min)**
```
20:15-20:35  codemap & brepl - Hands-On Installation
```

#### **Tool 1: codemap - "Project Brain for AI" (10 Min)**
```bash
# Installation
npm install -g @jordancoin/codemap

# Projekt-Analyse
cd /path/to/your/project
codemap analyze

# AI-Kontext generieren
codemap context --output context.md
```

**Was macht codemap?**
- Analysiert Projektstruktur automatisch
- Generiert AI-optimierte Kontext-Beschreibungen
- Reduziert Token-Verbrauch bei LLM-Anfragen
- Verbessert Code-Verständnis für AI

**Live-Demo:** Dozent zeigt codemap mit einem echten Projekt.

**Hands-On:** Teilnehmende analysieren ihr eigenes Projekt oder Demo-Projekt.

#### **Tool 2: brepl - "Universal REPL Bridge" (10 Min)**
```bash
# Installation
npm install -g @maximerivest/brepl

# Starten
brepl

# LLM-Integration testen
> connect claude
> ask "explain this Python function"
> tab-completion test
```

**Was macht brepl?**
- Universelle REPL für verschiedene LLMs
- Tab-Completion für AI-Befehle
- Interaktive Prompts mit Kontext
- TUI (Terminal User Interface) Support

**Live-Demo:** Dozent zeigt brepl mit verschiedenen LLMs.

**Hands-On:** Teilnehmende testen brepl mit ihren API-Keys.

### 📊 **Phase 3: LLM-Evaluation Framework (15 Min)**
```
20:35-20:50  Praktische LLM-Bewertung
```

#### **Evaluation-Methoden verstehen (8 Min)**

**1. Quantitative Metriken:**
```
- BLEU Score (Text-Ähnlichkeit)
- Code-Kompilierbarkeit (%)
- Performance-Benchmarks
- Token-Effizienz
```

**2. Qualitative Bewertung:**
```
- Code-Lesbarkeit (1-10)
- Lösungskreativität
- Fehlerbehandlung
- Best-Practice-Einhaltung
```

**3. Praktische Tests:**
```bash
# Gleiche Aufgabe an verschiedene LLMs
claude "Create a Python REST API with FastAPI"
gh copilot suggest "create REST API with Python"
gemini generate "Python FastAPI REST API example"
```

#### **Live-Evaluation Experiment (7 Min)**

**Aufgabe fuer alle:** "Erstellt eine Python-Funktion zur Passwort-Validierung"

**Test mit 3 Tools:**
1. Claude Code: `claude "create password validation function"`
2. GitHub Copilot: `gh copilot suggest "Python password validation"`
3. Gemini: `gemini generate "password validation Python function"`

**Bewertungskriterien:**
- Sicherheit (Regex-Pattern, Länge, Komplexität)
- Code-Qualität (Lesbarkeit, Kommentare)
- Vollständigkeit (Edge Cases, Error Handling)
- Performance (Effizienz)

**Ergebnisse vergleichen und diskutieren:**
"Welche Lösung ist am besten? Warum?"

## 🎓 **Lernzielkontrolle**

### ✅ **Praktische Bewertung**
```
Aufgabe: LLM-Vergleich durchführen
1. Wählt eine Programmieraufgabe
2. Testet mit mindestens 2 verschiedenen LLMs
3. Bewertet nach dem gelernten Framework
4. Dokumentiert Ergebnisse
```

### 📊 **Bewertungsmatrix**
| Kriterium | Claude | Copilot | Gemini | Gewichtung |
|-----------|--------|---------|--------|------------|
| Code-Qualität | ?/10 | ?/10 | ?/10 | 30% |
| Vollständigkeit | ?/10 | ?/10 | ?/10 | 25% |
| Sicherheit | ?/10 | ?/10 | ?/10 | 25% |
| Lesbarkeit | ?/10 | ?/10 | ?/10 | 20% |

### 🔍 **Erfolgskriterien**
- [ ] codemap erfolgreich installiert und getestet
- [ ] brepl installiert und mit LLM verbunden
- [ ] LLM-Evaluation durchgeführt und dokumentiert
- [ ] Aktuelle AI-Trends verstanden und diskutiert

## 💡 **Reflexionsfragen**
1. "Welche neuen Tools werden euren Workflow am meisten verändern?"
2. "Wie wichtig ist objektive LLM-Evaluation für euch?"
3. "Welche AI-Trends seht ihr als Game-Changer?"

## 🔗 **Übergang zu Lektion 4**
"Ihr habt jetzt einen Überblick über die neuesten Entwicklungen. Zeit, selbst kreativ zu werden und eigene Slash Commands zu entwickeln!"

---

**Pause: 20:50 - 21:00 (10 Minuten)**
