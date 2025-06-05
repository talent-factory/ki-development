# Pydantic AI Beispiele

Diese Beispiele zeigen die schrittweise Entwicklung eines einfachen Chatbots mit Pydantic AI, wobei der Fokus auf einer guten Projektstruktur und Wiederverwendbarkeit liegt.

## Übersicht der Beispieldateien

### 01_simple_tool_agent.py

#### Grundlegende Verwendung

- Demonstriert die einfache Verwendung eines Pydantic AI Agenten
- Liest API-Keys aus der `.env`-Datei im Projektverzeichnis
- Startet eine einfache Konversation mit dem Modell

### 02_simple_tool_agent.py

#### Verbesserte Pfadauflösung

- Erweitert das erste Beispiel um eine robuste Pfadauflösung
- Ermöglicht das Starten des Skripts aus beliebigen Unterverzeichnissen
- Fügt eine Chat-Schleife für interaktive Konversationen hinzu
- Behält den Kontext zwischen den Nachrichten bei

### 03_simple_tool_agent.py

#### Modulare Struktur

- Lagert die Pfad- und Konfigurationslogik in eine wiederverwendbare Bibliothek aus
- Verwendet das `utils`-Modul für projektweite Hilfsfunktionen
- Zeigt Best Practices für die Projektstruktur auf
- Vereinfacht die Wartung und Wiederverwendung von Code

## Voraussetzungen

- Python 3.8 oder höher
- Installierte Abhängigkeiten aus der `requirements.txt`
- Gültige API-Keys in der `.env`-Datei

## Verwendung

1. Installieren Sie die erforderlichen Pakete:

   ```bash
   pip install -r requirements.txt
   ```

2. Erstellen Sie eine `.env`-Datei im Projektstammverzeichnis mit Ihren API-Keys:

   ```env
   OPENAI_API_KEY=ihr_api_schluessel
   ```

3. Führen Sie eines der Beispiele aus:

   ```bash
   python src/pydantic/tool/01_simple_tool_agent.py
   ```

## Nächste Schritte

- Erweitern Sie die Funktionalität durch Hinzufügen weiterer Tools
- Implementieren Sie eine Benutzeroberfläche mit Streamlit oder einer Webanwendung
- Fügen Sie Logging und Fehlerbehandlung hinzu
