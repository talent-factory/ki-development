# ChatBot mit OpenAI, LangChain und Streamlit

Ein einfacher ChatBot, der mit OpenAI, LangChain und Streamlit erstellt wurde.

## Voraussetzungen

- Python 3.8+
- OpenAI API-Schlüssel

## Installation

1. Repository klonen
2. In das Projektverzeichnis wechseln:

   ```bash
   cd streamlit
   ```

3. Virtuelle Umgebung erstellen und aktivieren:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # oder
   .\venv\Scripts\activate  # Windows
   ```

4. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```

## Verwendung

1. Starte die Anwendung:

   ```bash
   streamlit run chatbot.py
   ```

2. Öffne deinen Browser und navigiere zu `http://localhost:8501`
3. Gib deinen OpenAI API-Schlüssel in der Seitenleiste ein
4. Beginne mit dem Chatten!

## Funktionen

- Unterstützung für verschiedene OpenAI-Modelle (GPT-3.5, GPT-4)
- Anpassbare Kreativität (Temperatur)
- Chat-Historie
- Responsives Design

## Umgebungsvariablen

Du kannst deinen OpenAI API-Schlüssel auch in einer `.env`-Datei speichern:

```bash
OPENAI_API_KEY=dein-api-schluessel
```

## Lizenz

MIT
