# [KI Professional – Fachrichtung Development](https://ibaw.ch/bildungsangebote/informatik/digital-collaboration-transformation/ki-professional-development/#rowno5)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/talent-factory/ki-development)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Daniel%20Senften-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dsenften/)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-blue?logo=github&logoColor=white)](https://github.com/talent-factory/ki-development)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
&nbsp;
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

KI-Anwendungen entwickeln, ohne nur eine Zeile Code zu schreiben? Ja, das ist möglich. Erfahren Sie, wie Sie KI-gesteuerte Apps mit No-Code-Tools erstellen und entwickeln Sie dabei ein umfangreiches Verständnis der dahinter liegenden maschinellen Lernprinzipien – damit Sie diese erfolgreich für Ihre Projekte umsetzen können.

Entdecken Sie Konzepte sowie Anwendungsfelder von Machine Learning (ML), Large Language Models (LLM), Vector Databases und Retrieval Augmented Generation (RAG). Verstehen Sie Low-Code-Entwicklung, Anwendungsfelder und die Integration in eigene Apps. Mit dem erworbenen Wissen wenden Sie die verschiedenen Tools an, um Ihre eigenen Ideen erfolgsbringend in eine Anwendung zu überführen – darunter Q&A-Systeme für eigene Daten, Chatbots und Agents sowie die Interaktion mit APIs. Die Zukunft der KI beginnt hier!

## 🚀 Projekt-Setup

Dieses Projekt verwendet `uv` als Paketmanager und Build-System. `uv` ist ein schneller, zuverlässiger und kompatibler Ersatz für pip und pip-tools, der in Rust geschrieben wurde.

### Voraussetzungen

- Python 3.10 oder höher
- `uv` installieren:

  ```bash
  pip install uv
  ```

### Installation

1. Virtuelle Umgebung erstellen und aktivieren:

   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   # ODER
   .\.venv\Scripts\activate  # Windows
   ```

2. Abhängigkeiten installieren:

   ```bash
   uv pip install -r requirements.txt
   ```

### Entwicklung

- **Neue Abhängigkeit hinzufügen**:

  ```bash
  uv pip install <paketname>
  ```

- **Aktualisierte Abhängigkeiten speichern**:

  ```bash
  uv pip freeze > requirements.txt
  ```

- **Streamlit-App starten**:

  ```bash
  uv run streamlit run streamlit/<app_name>.py
  ```

## LERNZIELE

---

Am Ende dieses Lerngangs …

- kennen Sie die Grundlagen der künstlichen Intelligenz (engl. Artificial Intelligence, AI).
- beherrschen Sie die Terminologie im Bereich der KI.
- verstehen Sie die ethischen, rechtlichen und Datenschutz-Aspekte im Kontext von KI.
- sind Sie in der Lage, Nutzen und Gefahren der KI-Technologie zu erkennen sowie die Möglichkeiten und Grenzen zu verstehen.
- verstehen Sie die KI-gesteuerte Programmierung und sind Sie in der Lage, KI-Tools zur Codeerstellung effizient zu nutzen.
- können Sie Codes von KI-Systemen verstehen, dokumentieren und Fehler im Code mithilfe von KI beheben.
- sind Sie in der Lage, die Vor- und Nachteile lokaler Sprachmodelle zu erörtern und können diese installieren sowie konfigurieren.
- können Sie ein einfaches eigenes Machine-Learning-Modell erstellen, trainieren, überprüfen und exportieren.
- erkennen Sie die vielfältigen Anwendungsfelder der Low-Code-Entwicklung mit LLM-basierten Anwendungen.
- verstehen Sie die Systemarchitektur von [LangChain](https://www.langchain.com/) basierten Apps und sind vertraut mit Low- Code-Frameworks wie [FlowiseAI](https://flowiseai.com/) und [Langflow](https://www.langflow.org/) ([GitHub](https://github.com/logspace-ai/langflow)).
- haben Sie eigene Anwendungen erstellt– darunter Q&A-Systeme für eigene Daten, Chatbots und Agents sowie die Interaktion mit APIs.

## INHALT

---

### Einführung in die KI-Basics

- Grundlagen der künstlichen Intelligenz
- Ethische, rechtliche und Datenschutzaspekte im Kontext von KI
- Nutzen und Gefahren von KI, um fundierte Entscheidungen zu treffen
- Möglichkeiten und Grenzen der KI-Technologie
- Technologie und Funktionsweise von KI-Systemen
- Wichtige Begriffe und Terminologien

### Training von Machine-Learning-Modellen ohne Code

- Möglichkeiten und Grenzen
- Erstellung des eigenen Machine-Learning-Modells
- Training, Überprüfung und Export des Machine-Learning-Modells
- Integration Ihres Machine-Learning-Modells in eine Webseite

### Grundlagen LLM-basierter Applikationen

- Möglichkeiten und Grenzen
- Kennenlernen und verwenden verschiedener Sprachmodelle lokal und in der Cloud
- Anwendungsgesteuerte Kommunikation mit Sprachmodellen
- Abrechnung und Kosten
- Fine-Tuning und Embedding

### Entwicklung von LLM-basierten Applikationen

- Low-Code-Entwicklung von LLM/RAG-basierten Anwendungen
- Identifizierung von potenziellen Anwendungsfeldern und Use Cases
- Verständnis der Systemarchitektur von LangChain basierten Apps
- Nutzung von Low-Code-Frameworks wie FlowiseAI und Langflow
- Praxistransfer und Erstellung eigener Anwendungen, darunter Q&A für eigene Daten, Chatbots, Agents und API-Interaktionen

## 🤝 Beiträge

Wir freuen uns über Beiträge! Bitte lesen Sie unsere [Contributing Guidelines](CONTRIBUTING.md) und [Code of Conduct](CODE_OF_CONDUCT.md) bevor Sie einen Pull Request erstellen.

### Wie Sie beitragen können

- 🐛 Fehler melden über [Issues](https://github.com/talent-factory/ki-development/issues)
- 💡 Neue Features vorschlagen
- 📖 Dokumentation verbessern
- 🧪 Beispiele und Tutorials hinzufügen
- 🔧 Code-Verbesserungen einreichen

Siehe [CONTRIBUTING.md](CONTRIBUTING.md) für detaillierte Informationen.

## 📄 Lizenz

Dieses Projekt ist unter der [MIT-Lizenz](LICENSE) lizenziert. Sie dürfen den Code frei verwenden, modifizieren und verteilen.

## 🔒 Sicherheit

Sicherheit ist uns wichtig. Wenn Sie eine Sicherheitslücke entdecken, lesen Sie bitte unsere [Sicherheitsrichtlinie](SECURITY.md) für Informationen zur verantwortungsvollen Meldung.

## 📞 Kontakt

- **LinkedIn**: [Daniel Senften](https://www.linkedin.com/in/dsenften/)
- **GitHub Issues**: [Problem melden](https://github.com/talent-factory/ki-development/issues)
- **Kurs-Website**: [IBAW KI Professional](https://ibaw.ch/bildungsangebote/informatik/digital-collaboration-transformation/ki-professional-development/)

## 🙏 Danksagungen

Vielen Dank an alle [Contributors](https://github.com/talent-factory/ki-development/graphs/contributors), die zu diesem Projekt beigetragen haben!

---

**Hinweis**: Dieses Repository wird aktiv für Bildungszwecke verwendet. Fragen und Diskussionen sind willkommen!
