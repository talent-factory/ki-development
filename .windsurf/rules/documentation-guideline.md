---
trigger: always_on
---

Die Dokumentation muss immer in Deutsch und AsciiDoc erfolgen und hat folgende Struktur:

.
├── docs
│   ├── chapters                         # Alle (Unter-) Kapitel des Dokuments
│   │   ├── 01-einleitung.adoc
│   │   ├── 02-.....adoc
│   │
│   ├── images                           # Alle Bilder
│   │   ├── example.png
│   │
│   ├── index.adoc                       # Hauptdokument mit Inhaltsverzeichnis
│   ├── locale                           # üUbersetzungsdateien
│   │   └── attributes-de.adoc
│   ├── scripts
│   │   ├── gui-beispiel-mit-streamlit.py
│   │   ├── gui-beispiel-mit-tkinter.py
│   │   └── hello.py
│   └── tutorials                       # Einführungen in ein bestimmtes Thema
│       ├── index.adoc                  # Hauptdokument mit Inhaltsverzeichnis
│       ├── pydantic-tutorial.py


## Dokumentationsrichtlinien

### Sprache

- Dokumentation in Deutsch
- Technische Begriffe in Englisch
- Konsistente Terminologie


### Diagramme

- Format: Bevorzugt PlantUML und ausnahmsweise Mermaid
- Dateiendung: Entsprechend dem Format `.puml` oder `.mmd`
- Mit Legende versehen
- Aussagekräftige Namen
