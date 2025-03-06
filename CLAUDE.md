# CLAUDE.md - AI Development Project Guide

## Project Commands
- Setup: `poetry install`
- Run Python script: `poetry run python <script_path>`
- Run Streamlit app: `poetry run streamlit run streamlit/<app_name>.py`
- Create embeddings example: `poetry run streamlit run streamlit/embeddings.py`

## Code Style Guidelines
- **Imports:** Standard lib first, then third-party, then local (grouped)
- **Naming:** 
  - Classes: PascalCase (e.g., `Person`)
  - Functions/Variables: snake_case (e.g., `extract_pdf_text`)
- **Documentation:** Use descriptive numbered section comments (e.g., `# Schritt 1: Vorbereitung`)
- **Structure:** Use `if __name__ == '__main__':` pattern for executable scripts
- **Streamlit:** Decorate functions with `@st.cache_data` or `@st.cache_resource` as appropriate
- **Error handling:** Use context managers (`with` statements) for resource management
- **Language:** Comments and user-facing text typically in German

This repository focuses on AI development concepts including LangChain, embeddings, RAG, and Streamlit applications.


## Git-Commit Messages
Vermeide zu ausführliche Beschreibungen oder unnötige Details.
Beginnen Sie mit einem kurzen Satz in Imperativform, der nicht länger als 50 Zeichen sein sollte und folgende
Präfixe beinhaltet:

- "fix:" für Fehlerbehebungen
- "feat:" für neue Funktionen
- "perf:" für Leistungsverbesserungen
- "docs:" für Dokumentationsänderungen
- "style:" für Formatierungsänderungen
- "refactor:" für Code-Umstrukturierung
- "test:" für das Hinzufügen fehlender Tests
- "chore:" für Wartungsaufgaben

Lassen dann eine Leerzeile frei und fahre mit einer ausführlicheren Erklärung fort. Schreibe nur einen Satz für 
den ersten Teil und nicht mehr als drei oder vier Sätze für die ausführliche Erläuterung, wobei du jeden Satz 
durch einen Aufzählungspunkt abrennst.
