# Sicherheitsrichtlinie

## Unterstützte Versionen

Wir unterstützen derzeit die folgenden Versionen mit Sicherheitsupdates:

| Version | Unterstützt          |
| ------- | -------------------- |
| develop | :white_check_mark:   |
| main    | :white_check_mark:   |

## Meldung von Sicherheitslücken

Die Sicherheit unseres Projekts ist uns sehr wichtig. Wenn Sie eine Sicherheitslücke entdeckt haben, helfen Sie uns bitte, diese verantwortungsvoll zu beheben.

### Bitte NICHT:

- Öffentliche GitHub Issues für Sicherheitsprobleme erstellen
- Sicherheitslücken öffentlich diskutieren
- Die Schwachstelle ausnutzen

### Stattdessen:

1. **Private Meldung**: Nutzen Sie GitHubs Private Vulnerability Reporting:
   - Gehen Sie zu: https://github.com/talent-factory/ki-development/security/advisories/new
   - Oder melden Sie sich per E-Mail (siehe unten)

2. **Informationen bereitstellen**:
   - Art der Schwachstelle
   - Schritte zur Reproduktion
   - Betroffene Versionen/Komponenten
   - Mögliche Auswirkungen
   - Vorschläge für Fixes (optional)

3. **Kontakt**:
   - GitHub Security Advisory (bevorzugt)
   - E-Mail: [daniel.senften@talent-factory.ch](mailto:daniel.senften@talent-factory.ch)

### Was Sie erwarten können:

- **Bestätigung**: Innerhalb von 48 Stunden
- **Erste Bewertung**: Innerhalb von 5 Werktagen
- **Status-Updates**: Regelmässige Kommunikation während der Behebung
- **Anerkennung**: Mit Ihrer Erlaubnis werden Sie in den Release Notes erwähnt

## Sicherheits-Best-Practices für Benutzer

### API-Schlüssel-Verwaltung

**NIEMALS** API-Schlüssel in Code oder öffentlichen Repositories committen:

```bash
# .env Datei für lokale Entwicklung (NICHT committen!)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Stellen Sie sicher, dass `.env` in `.gitignore` steht:
```
.env
.env.local
*.key
```

### Umgebungsvariablen

Verwenden Sie Umgebungsvariablen für sensible Daten:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
```

### Virtuelle Umgebungen

Verwenden Sie immer virtuelle Umgebungen:

```bash
uv venv
source .venv/bin/activate
```

### Abhängigkeiten aktualisieren

Halten Sie Dependencies aktuell:

```bash
uv pip list --outdated
uv pip install --upgrade paket-name
```

### Code-Überprüfung

- Überprüfen Sie Code aus unbekannten Quellen
- Seien Sie vorsichtig bei externen Dependencies
- Nutzen Sie Tools wie `pip-audit` zur Sicherheitsüberprüfung

## Sicherheitsüberprüfungen

Wir führen regelmässig durch:

- Dependency Scanning
- Code Reviews
- Security Audits
- Penetration Tests (bei Bedarf)

## Bekannte Einschränkungen

### API-Kosten

- Keine automatische Kostenbegrenzung
- Benutzer sind selbst für API-Kosten verantwortlich
- Überwachen Sie Ihre API-Nutzung regelmässig

### Datenschutz

- Sensible Daten nie an externe APIs senden
- Lokale Modelle für vertrauliche Informationen verwenden
- DSGVO-Konformität bei Verwendung von Cloud-Services beachten

## Compliance

Dieses Projekt behandelt:

- **Keine persönlichen Daten** standardmässig
- **Benutzerverantwortung** für API-Nutzung
- **Lokale Verarbeitung** wo möglich

Bei der Verwendung externer APIs (OpenAI, Anthropic, etc.):

- Lesen Sie deren Privacy Policies
- Verstehen Sie Datenverarbeitungsrichtlinien
- Halten Sie gesetzliche Anforderungen ein

## Sicherheitsempfehlungen für Contributors

1. **Code-Reviews**: Alle PRs werden auf Sicherheitsprobleme überprüft
2. **Secrets Scanning**: Automatisch in CI/CD aktiviert
3. **Dependencies**: Regelmässige Updates und Audits
4. **Best Practices**: Folgen Sie Python-Security-Guidelines

## Kontakt

Für sicherheitsrelevante Fragen:

- GitHub Security Advisory: https://github.com/talent-factory/ki-development/security/advisories
- E-Mail: [daniel.senften@talent-factory.ch](mailto:daniel.senften@talent-factory.ch)
- Issues (für nicht-sicherheitskritische Themen): https://github.com/talent-factory/ki-development/issues

## Richtlinie für Responsible Disclosure

Wir folgen den Prinzipien der verantwortungsvollen Offenlegung:

1. Sicherheitsforscher melden Schwachstellen privat
2. Wir bestätigen und bewerten die Meldung
3. Wir arbeiten an einem Fix
4. Koordinierte öffentliche Bekanntgabe nach dem Fix
5. Anerkennung des Forschers (mit Zustimmung)

Vielen Dank, dass Sie helfen, dieses Projekt sicher zu halten!
