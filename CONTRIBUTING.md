# Beiträge zum KI Professional Development Projekt

Vielen Dank für Ihr Interesse, zu diesem Projekt beizutragen! Wir freuen uns über jede Art von Beitrag.

## 🎯 Wie kann ich beitragen?

### Fehler melden

Wenn Sie einen Fehler gefunden haben:

1. Prüfen Sie, ob der Fehler bereits in den [Issues](https://github.com/talent-factory/ki-development/issues) gemeldet wurde
2. Falls nicht, erstellen Sie ein neues Issue mit:
   - Beschreibender Titel
   - Detaillierte Beschreibung des Problems
   - Schritte zur Reproduktion
   - Erwartetes vs. tatsächliches Verhalten
   - Screenshots (falls relevant)
   - Ihre Umgebung (Python-Version, Betriebssystem, etc.)

### Verbesserungen vorschlagen

Haben Sie eine Idee für eine Verbesserung?

1. Erstellen Sie ein Issue mit dem Label "enhancement"
2. Beschreiben Sie Ihre Idee detailliert
3. Erklären Sie, warum diese Änderung wertvoll wäre
4. Warten Sie auf Feedback vom Team

### Code beitragen

#### Vorbereitung

1. Forken Sie das Repository
2. Klonen Sie Ihren Fork:
   ```bash
   git clone https://github.com/IHR-USERNAME/ki-development.git
   cd ki-development
   ```

3. Richten Sie die Entwicklungsumgebung ein:
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   # ODER
   .\.venv\Scripts\activate  # Windows

   uv pip install -r requirements.txt
   ```

#### Änderungen vornehmen

1. Erstellen Sie einen Feature-Branch:
   ```bash
   git checkout -b feature/ihre-feature-beschreibung
   ```

2. Nehmen Sie Ihre Änderungen vor und committen Sie sie:
   ```bash
   git add .
   git commit -m "feat: Beschreibung Ihrer Änderung"
   ```

3. Pushen Sie zu Ihrem Fork:
   ```bash
   git push origin feature/ihre-feature-beschreibung
   ```

4. Erstellen Sie einen Pull Request über die GitHub-Weboberfläche

#### Pull Request Guidelines

- **Ein Pull Request = Eine Funktion/Ein Fix**: Halten Sie PRs fokussiert
- **Beschreibende Titel**: Verwenden Sie klare, beschreibende Titel
- **Detaillierte Beschreibung**: Erklären Sie, was und warum
- **Tests**: Stellen Sie sicher, dass Ihre Änderungen funktionieren
- **Dokumentation**: Aktualisieren Sie Dokumentation wenn nötig

#### Commit-Message-Konventionen

Wir folgen den [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - Neue Funktionalität
- `fix:` - Fehlerbehebung
- `docs:` - Dokumentationsänderungen
- `style:` - Formatierung, keine Code-Änderungen
- `refactor:` - Code-Umstrukturierung ohne Funktionsänderung
- `test:` - Hinzufügen oder Korrigieren von Tests
- `chore:` - Wartungsaufgaben, Build-Prozess, etc.

Beispiele:
```
feat: Add new Q&A system example
fix: Correct API endpoint in chatbot
docs: Update installation instructions
```

## 📋 Branch-Strategie

- `main` - Stabile Production-Version
- `develop` - Entwicklungsbranch (Standard)
- `feature/*` - Feature-Branches
- `fix/*` - Bugfix-Branches

**Wichtig:** Die Branches `main` und `develop` sind geschützt und erfordern:
- Mindestens 1 Approval vor dem Merge
- Pull Request Review
- Lineare Historie (keine Merge Commits)

## 🔒 Branch-Protection

Die Hauptbranches sind geschützt:
- Direktes Pushen ist nicht erlaubt
- Pull Requests müssen reviewt werden
- Force Pushes sind deaktiviert
- Branch-Löschung ist deaktiviert

## 🧪 Code-Qualität

- Testen Sie Ihre Änderungen gründlich
- Folgen Sie Python Best Practices (PEP 8)
- Kommentieren Sie komplexen Code
- Halten Sie Funktionen klein und fokussiert

## 📚 Dokumentation

- Aktualisieren Sie die README.md bei relevanten Änderungen
- Fügen Sie Docstrings zu neuen Funktionen hinzu
- Kommentieren Sie nicht-offensichtlichen Code
- Aktualisieren Sie Beispiele wenn nötig

## 💬 Kommunikation

- Seien Sie respektvoll und konstruktiv
- Folgen Sie unserem [Code of Conduct](CODE_OF_CONDUCT.md)
- Stellen Sie Fragen in Issues oder Discussions
- Seien Sie offen für Feedback

## 🎓 Erste Schritte für Anfänger

Neu im Open-Source-Bereich? Kein Problem!

1. Schauen Sie sich Issues mit dem Label "good first issue" an
2. Lesen Sie die Dokumentation
3. Stellen Sie Fragen - wir helfen gerne!
4. Starten Sie mit kleinen Änderungen

## 📝 Lizenz

Durch das Beitragen zum Projekt stimmen Sie zu, dass Ihre Beiträge unter der [MIT-Lizenz](LICENSE) lizenziert werden.

## 🙏 Anerkennung

Alle Contributors werden in der README.md aufgeführt. Vielen Dank für Ihre Unterstützung!

---

Bei Fragen kontaktieren Sie uns über:
- GitHub Issues
- GitHub Discussions
- [LinkedIn](https://www.linkedin.com/in/dsenften/)
