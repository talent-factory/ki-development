# Slash Commands & AI-Workflows
## Professionelle Entwicklungsworkflows mit AI-Agenten

**Kursabend 2: Von RAG zu praktischen Workflows**  
*4 Lektionen à 50 Minuten | 3 Teilnehmende*

---

## 🎯 **Lernziele**

Nach diesem Kursabend können Sie:
- ✅ **Slash Commands** verstehen und effektiv nutzen
- ✅ **AI-Workflows** für tägliche Entwicklungsarbeit einsetzen
- ✅ **Eigene Commands** erstellen und anpassen
- ✅ **Professionelle Git-Workflows** mit AI automatisieren
- ✅ **Projekt-Management** von PRD bis Tasks durchführen

---

## 📋 **Kursaufbau**

### **🔄 Anknüpfung an letzten Abend:**
```
Letzter Abend: RAG-Grundlagen (technisch) → PRD-Demo (zu grosser Sprung)
Heute: Slash Commands & AI-Workflows (praktisch anwendbar)
```

### **📚 Lektion 1: Einführung & Setup (50 Min)**
- **10 Min:** Rückblick letzter Abend + Brücke zu heute
- **15 Min:** Was sind Slash Commands? Warum wichtig?
- **15 Min:** dotfiles Repo erkunden (GitHub Tour)
- **10 Min:** Installation durchführen (alle Teilnehmenden)

### **⚡ Lektion 2: Git & PR Workflows (50 Min)**
- **20 Min:** `/commit` - Professionelle Git Commits mit AI
- **20 Min:** `/create-pr` - Pull Request Workflow automatisieren
- **10 Min:** Hands-On: Eigene Commits & PRs erstellen

### **🚀 Lektion 3: Projekt-Management (50 Min)**
- **20 Min:** `/project:create-prd` - Vertiefen vom letzten Mal
- **20 Min:** `/project:create-plan` - Von PRD zu konkreten Tasks
- **10 Min:** Hands-On: Mini-Projekt komplett durchplanen

### **🛠️ Lektion 4: Eigene Commands erstellen (50 Min)**
- **15 Min:** Command-Struktur verstehen (.md Format)
- **25 Min:** Eigenen Command für ihren Workflow erstellen
- **10 Min:** Sharing, Best Practices & Ausblick

---

## 📖 **Lektion 1: Einführung & Setup**

### **🔄 Rückblick & Brücke (10 Min)**

**Letzter Abend - Was wir gelernt haben:**
- ✅ **RAG-Grundlagen**: Embeddings, Retrieval, LLM-Integration
- ✅ **Praktische Umsetzung**: 3 funktionierende Python-Skripte
- ✅ **PRD-Erstellung**: Mit `/project:create-prd` Command

**Der Sprung war gross:**
```
RAG-Details (technisch) → PRD-Erstellung (abstrakt)
```

**Heute bauen wir die Brücke:**
```
RAG-Verständnis → Praktische AI-Workflows → Tägliche Produktivität
```

### **💡 Was sind Slash Commands? (15 Min)**

**Definition:**
Slash Commands sind **vordefinierte AI-Prompts** die komplexe Workflows automatisieren.

**Warum sind sie wichtig?**
1. **Konsistenz** - Gleiche Qualität bei jedem Aufruf
2. **Effizienz** - Keine Prompt-Entwicklung jedes Mal
3. **Best Practices** - Bewährte Workflows eingebaut
4. **Skalierbarkeit** - Einmal erstellt, überall nutzbar

**Beispiel-Vergleich:**

**❌ Ohne Slash Command:**
```
"Kannst du mir helfen einen Git Commit zu erstellen? 
Ich habe folgende Änderungen gemacht... 
Bitte achte auf Conventional Commits Format... 
Und prüfe auch ob alles korrekt ist..."
```

**✅ Mit Slash Command:**
```
/commit
```
→ Automatisch: Änderungen analysieren, Conventional Commits, Pre-Commit Checks, professionelle Nachricht

### **🌍 dotfiles Repository erkunden (15 Min)**

**Repository-Übersicht:**
- **URL:** https://github.com/talent-factory/dotfiles
- **Zweck:** Professionelle AI-Agent Konfigurationen
- **4 AI Agents:** Augment Code, Claude Code, GitHub Copilot, Windsurf
- **Cross-Platform:** macOS, Linux, Windows

**Wichtige Verzeichnisse:**
```
dotfiles/
├── agents/
│   ├── claude/commands/          # Slash Commands für Claude Code
│   ├── augment/commands/         # Commands für Augment Code  
│   ├── copilot/prompts/          # Prompts für GitHub Copilot
│   └── windsurf/workflows/       # Workflows für Windsurf
├── install/                      # Installation System
└── README.md                     # Dokumentation
```

**Verfügbare Claude Code Commands:**
- `/commit` - Professionelle Git Commits
- `/create-pr` - Pull Request Erstellung
- `/project:create-prd` - PRD Generation (kennen Sie bereits!)
- `/project:create-plan` - Projekt-Planung
- `/develop:check-agents` - Konfiguration validieren
- `/skills:build-skill` - Eigene Skills erstellen

### **⚙️ Installation durchführen (10 Min)**

**Schritt 1: Repository klonen**
```bash
# macOS/Linux
git clone https://github.com/talent-factory/dotfiles.git ~/.dotfiles
cd ~/.dotfiles

# Windows
git clone https://github.com/talent-factory/dotfiles.git $env:USERPROFILE\.dotfiles
cd $env:USERPROFILE\.dotfiles
```

**Schritt 2: Interaktive Installation**
```bash
# macOS/Linux
./install.sh --interactive

# Windows  
.\install.ps1 -Interactive
```

**Installation-Optionen wählen:**
1. **Installation Target:** Home Directory (empfohlen)
2. **Agent Selection:** Claude Code auswählen
3. **Installation Method:** Symlink (empfohlen)
4. **Bestätigen:** Installation durchführen

**Schritt 3: Testen**
- Claude Code neu starten
- `/` eingeben → Commands sollten verfügbar sein
- `/commit` testen (falls Git-Repository vorhanden)

---

## ⚡ **Lektion 2: Git & PR Workflows**

### **📝 `/commit` - Professionelle Git Commits (20 Min)**

**Was macht der Command?**
1. **Änderungen analysieren** - Welche Dateien wurden geändert?
2. **Conventional Commits** - Automatisches Format (feat:, fix:, docs:)
3. **Pre-Commit Checks** - Syntax, Tests, Linting
4. **Professionelle Nachricht** - Klar, präzise, aussagekräftig

**Live-Demo:**
```bash
# Beispiel-Änderung machen
echo "# Test" > test.md
git add test.md

# In Claude Code
/commit
```

**Erwartete Ausgabe:**
```
📝 Analysiere Git-Änderungen...

Gefundene Änderungen:
- test.md (neu): Markdown-Datei hinzugefügt

🎯 Vorgeschlagener Commit:
docs: add test markdown file

Add basic test.md file for documentation purposes.

✅ Pre-Commit Checks:
- Syntax: OK
- Dateigrösse: OK  
- Keine Secrets: OK

Soll ich diesen Commit erstellen? (y/n)
```

**Conventional Commits Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Häufige Types:**
- `feat:` - Neue Funktionalität
- `fix:` - Bugfix
- `docs:` - Dokumentation
- `style:` - Formatierung
- `refactor:` - Code-Refactoring
- `test:` - Tests
- `chore:` - Build/Tools

### **🔀 `/create-pr` - Pull Request Workflow (20 Min)**

**Was macht der Command?**
1. **Branch-Management** - Automatische Branch-Erstellung
2. **PR-Beschreibung** - Template-basiert, vollständig
3. **Reviewer-Vorschläge** - Basierend auf Code-Änderungen
4. **Labels & Milestones** - Automatische Kategorisierung

**Workflow-Schritte:**
```
1. Feature-Branch erstellen
2. Änderungen committen  
3. /create-pr ausführen
4. PR automatisch erstellt
5. Team-Benachrichtigung
```

**Live-Demo:**
```bash
# Feature-Branch erstellen
git checkout -b feature/slash-commands-demo

# Änderungen machen
echo "# Slash Commands Demo" > demo.md
git add demo.md
git commit -m "docs: add slash commands demo"

# In Claude Code
/create-pr
```

**Erwartete Ausgabe:**
```
🚀 Pull Request Workflow startet...

📋 PR-Details:
- Branch: feature/slash-commands-demo
- Target: main
- Commits: 1
- Dateien: 1

📝 Generiere PR-Beschreibung...

## 📋 Änderungen
- Neue Dokumentation für Slash Commands Demo

## 🎯 Zweck  
Demonstration der Slash Commands Funktionalität

## ✅ Checklist
- [x] Code getestet
- [x] Dokumentation aktualisiert
- [ ] Review angefordert

🔗 PR erstellt: https://github.com/user/repo/pull/123
```

### **🛠️ Hands-On: Eigene Commits & PRs (10 Min)**

**Aufgabe für Teilnehmende:**
1. **Eigenes Repository** verwenden (oder Test-Repository erstellen)
2. **Kleine Änderung** machen (README.md erweitern)
3. **`/commit` verwenden** für professionellen Commit
4. **`/create-pr` testen** (falls GitHub-Repository)

**Hilfestellung:**
- Bei Problemen: Command-Ausgabe analysieren
- Fehler-Behandlung: Was tun wenn Commands nicht funktionieren?
- Best Practices: Wann welchen Command verwenden?

---

## 🚀 **Lektion 3: Projekt-Management**

### **📋 `/project:create-prd` - Vertiefen (20 Min)**

**Rückblick letzter Abend:**
Sie haben bereits gesehen wie aus einem einfachen Prompt ein umfangreiches PRD entsteht.

**Heute vertiefen wir:**
1. **PRD-Struktur verstehen** - Was macht ein gutes PRD aus?
2. **Prompt-Optimierung** - Bessere Eingaben = bessere Ergebnisse  
3. **Iterative Verbesserung** - PRD schrittweise verfeinern

**PRD-Anatomie:**
```markdown
# Product Requirements Document

## 1. Executive Summary
- Vision, Ziele, Erfolgsmetriken

## 2. Problem Statement  
- Welches Problem lösen wir?
- Für wen lösen wir es?

## 3. Solution Overview
- Lösungsansatz
- Kernfunktionalitäten

## 4. User Stories & Acceptance Criteria
- Detaillierte Anforderungen
- Testbare Kriterien

## 5. Technical Requirements
- Architektur-Überlegungen
- Technologie-Stack

## 6. Success Metrics
- Messbare Erfolgskriterien
- KPIs und Monitoring
```

**Prompt-Optimierung Beispiel:**

**❌ Schwacher Prompt:**
```
"Erstelle ein PRD für eine App"
```

**✅ Starker Prompt:**
```
"Erstelle ein PRD für eine mobile Lern-App für Programmierer.

Zielgruppe: Junior-Entwickler (1-3 Jahre Erfahrung)
Problem: Schwierig, kontinuierlich neue Technologien zu lernen
Lösung: Gamifizierte Micro-Learning Sessions (5-10 Min)
Platform: iOS/Android, später Web
Monetarisierung: Freemium mit Premium-Inhalten

Fokus auf: User Experience, Engagement, Lernfortschritt"
```

### **📊 `/project:create-plan` - Von PRD zu Tasks (20 Min)**

**Was macht der Command?**
1. **PRD analysieren** - Anforderungen extrahieren
2. **Epics definieren** - Grosse Funktionsbereiche
3. **User Stories** - Detaillierte Anforderungen
4. **Tasks aufteilen** - Umsetzbare Arbeitsschritte
5. **Abhängigkeiten** - Reihenfolge und Prioritäten

**Workflow:**
```
PRD → Epics → User Stories → Tasks → Sprint Planning
```

**Live-Demo:**
```
# In Claude Code mit vorhandenem PRD
/project:create-plan

# Oder mit PRD-Datei
/project:create-plan docs/prd.md
```

**Erwartete Ausgabe:**
```
📋 Analysiere PRD: Mobile Lern-App...

🎯 Identifizierte Epics:
1. User Authentication & Onboarding
2. Learning Content Management  
3. Gamification System
4. Progress Tracking
5. Social Features

📝 Epic 1: User Authentication & Onboarding
├── US-001: Als neuer User möchte ich mich registrieren
│   ├── Task: UI/UX Design für Registrierung
│   ├── Task: Backend API für User-Erstellung
│   └── Task: Email-Verifizierung implementieren
├── US-002: Als User möchte ich mich anmelden
│   ├── Task: Login-Formular erstellen
│   └── Task: JWT-Authentication implementieren

🔗 Abhängigkeiten:
- US-001 → US-002 (Registrierung vor Login)
- Epic 1 → Epic 2 (Auth vor Content)

📊 Geschätzte Entwicklungszeit: 8-12 Wochen
```

**Integration mit Tools:**
- **Linear:** Automatische Ticket-Erstellung
- **Jira:** Export als Jira-Format
- **GitHub Issues:** Direkte Issue-Erstellung
- **Notion:** Strukturierte Dokumentation

### **🛠️ Hands-On: Mini-Projekt planen (10 Min)**

**Aufgabe für Teilnehmende:**
1. **Eigene Projekt-Idee** entwickeln (5 Min Brainstorming)
2. **PRD erstellen** mit `/project:create-prd`
3. **Projekt-Plan** generieren mit `/project:create-plan`
4. **Ergebnisse vergleichen** und diskutieren

**Projekt-Ideen als Inspiration:**
- **Personal Dashboard** - Übersicht über eigene Projekte/Tasks
- **Code Snippet Manager** - Sammlung wiederverwendbarer Code-Teile
- **Learning Tracker** - Fortschritt bei Online-Kursen verfolgen
- **Team Chat Bot** - Slack/Discord Bot für Team-Produktivität

**Diskussion:**
- Welche Epics wurden identifiziert?
- Sind die Tasks realistisch geschätzt?
- Welche Abhängigkeiten wurden erkannt?
- Was würden Sie anders machen?

---

## 🛠️ **Lektion 4: Eigene Commands erstellen**

### **📝 Command-Struktur verstehen (15 Min)**

**Slash Commands sind Markdown-Dateien:**
```markdown
# Command-Name

Kurze Beschreibung was der Command macht.

## Schritte:

1. Schritt 1 beschreibung
2. Schritt 2 beschreibung  
3. Schritt 3 beschreibung

## Beispiel:

Konkretes Beispiel der Verwendung.

## Hinweise:

- Wichtige Punkte
- Häufige Fehler vermeiden
```

**Beispiel: Einfacher `/test` Command:**
```markdown
# Test Command

Führt Tests für das aktuelle Projekt aus und gibt eine Zusammenfassung.

## Schritte:

1. Erkenne das Projekt-Type (Python, JavaScript, etc.)
2. Führe entsprechende Test-Commands aus
3. Analysiere Test-Ergebnisse
4. Erstelle übersichtliche Zusammenfassung

## Beispiel:

```bash
# Python Projekt
pytest --verbose --coverage

# JavaScript Projekt  
npm test

# Ergebnis-Zusammenfassung
✅ 15 Tests bestanden
❌ 2 Tests fehlgeschlagen
📊 Coverage: 87%
```

## Hinweise:

- Funktioniert mit pytest, jest, mocha
- Zeigt Coverage-Report wenn verfügbar
- Schlägt Fixes für fehlgeschlagene Tests vor
```

**Command-Kategorien:**
- **Development:** `/test`, `/deploy`, `/review`
- **Project:** `/create-prd`, `/create-plan`, `/estimate`
- **Git:** `/commit`, `/create-pr`, `/release`
- **Documentation:** `/readme`, `/changelog`, `/api-docs`

### **🔧 Eigenen Command erstellen (25 Min)**

**Schritt 1: Command-Idee entwickeln (5 Min)**

**Brainstorming-Fragen:**
- Welche Aufgabe machen Sie regelmässig?
- Was dauert immer zu lange?
- Wo machen Sie oft Fehler?
- Was vergessen Sie häufig?

**Beliebte Command-Ideen:**
- `/readme` - README.md automatisch generieren
- `/deploy` - Deployment-Prozess automatisieren  
- `/review` - Code-Review Checkliste abarbeiten
- `/estimate` - Aufwand für Tasks schätzen
- `/standup` - Daily Standup Notizen erstellen

**Schritt 2: Command schreiben (15 Min)**

**Template verwenden:**
```markdown
# /[command-name]

[Kurze Beschreibung - was macht der Command?]

## Schritte:

1. [Erster Schritt]
2. [Zweiter Schritt]
3. [Dritter Schritt]
4. [Letzter Schritt]

## Beispiel:

[Konkretes Beispiel mit Input/Output]

## Hinweise:

- [Wichtiger Punkt 1]
- [Wichtiger Punkt 2]
- [Häufiger Fehler vermeiden]
```

**Live-Beispiel: `/readme` Command erstellen**
```markdown
# /readme

Generiert eine professionelle README.md für das aktuelle Projekt basierend auf Code-Analyse.

## Schritte:

1. Analysiere Projekt-Struktur und Dateien
2. Erkenne Technologie-Stack und Dependencies
3. Identifiziere Haupt-Features aus Code
4. Generiere Installation-Anweisungen
5. Erstelle Usage-Beispiele
6. Füge Badges und Metadaten hinzu

## Beispiel:

Für ein Python-Projekt mit FastAPI:

```markdown
# Project Name

Brief description of what this project does.

## 🚀 Features

- Feature 1 (detected from routes)
- Feature 2 (detected from models)
- Feature 3 (detected from tests)

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Usage

```python
from main import app
# Usage example
```

## 🧪 Testing

```bash
pytest
```
```

## Hinweise:

- Analysiert package.json, requirements.txt, etc.
- Erkennt Framework automatisch (React, FastAPI, etc.)
- Generiert passende Badges (Build Status, Coverage)
- Berücksichtigt bestehende README-Teile
```

**Schritt 3: Command testen (5 Min)**

1. **Datei speichern** als `~/.claude/commands/readme.md`
2. **Claude Code neu starten**
3. **Command testen** mit `/readme`
4. **Ergebnis bewerten** und anpassen

### **📤 Sharing & Best Practices (10 Min)**

**Command teilen:**
1. **GitHub Gist** - Einzelne Commands
2. **Fork dotfiles** - Eigene Command-Sammlung
3. **Pull Request** - Contribution zum Haupt-Repository
4. **Team-Repository** - Firmen-spezifische Commands

**Best Practices:**
- ✅ **Klare Beschreibung** - Was macht der Command?
- ✅ **Konkrete Schritte** - Nachvollziehbare Anweisungen
- ✅ **Beispiele** - Input/Output zeigen
- ✅ **Fehler-Behandlung** - Was tun wenn etwas schief geht?
- ✅ **Konsistente Struktur** - Gleiche Markdown-Struktur

**Häufige Fehler vermeiden:**
- ❌ Zu vage Beschreibungen
- ❌ Fehlende Beispiele
- ❌ Zu komplexe Commands (lieber aufteilen)
- ❌ Keine Fehler-Behandlung

**Ausblick:**
- **Advanced Commands** - Mit Parametern und Optionen
- **Command-Chains** - Commands die andere Commands aufrufen
- **Integration** - Mit externen APIs und Tools
- **Team-Workflows** - Commands für Zusammenarbeit

---

## 🎯 **Zusammenfassung & Ausblick**

### **Was haben wir heute gelernt?**

1. **✅ Slash Commands verstehen**
   - Vordefinierte AI-Prompts für Workflows
   - Konsistenz, Effizienz, Best Practices

2. **✅ Git & PR Workflows automatisieren**
   - `/commit` für professionelle Commits
   - `/create-pr` für strukturierte Pull Requests

3. **✅ Projekt-Management mit AI**
   - `/project:create-prd` für Requirements
   - `/project:create-plan` für Task-Breakdown

4. **✅ Eigene Commands erstellen**
   - Markdown-basierte Struktur
   - Von Idee bis funktionierendem Command

### **Praktischer Nutzen:**

**Sofort anwendbar:**
- Professionellere Git-Commits
- Strukturierte Projekt-Planung  
- Automatisierte Workflows
- Eigene Produktivitäts-Tools

**Langfristige Vorteile:**
- Konsistente Arbeitsweise
- Weniger Fehler
- Mehr Zeit für kreative Arbeit
- Bessere Team-Zusammenarbeit

### **Nächste Schritte:**

1. **Commands täglich nutzen** - Integration in Workflow
2. **Eigene Commands entwickeln** - Für spezifische Bedürfnisse
3. **Team-Commands erstellen** - Für Zusammenarbeit
4. **Advanced Features** - Parameter, Chains, Integrationen

### **Ressourcen:**

- **dotfiles Repository:** https://github.com/talent-factory/dotfiles
- **Dokumentation:** CLAUDE.md, INSTALLATION.md
- **Community:** GitHub Discussions für Fragen
- **Inspiration:** Bestehende Commands als Vorlage

---

**🎉 Herzlichen Glückwunsch!**  
Sie haben erfolgreich den Sprung von technischen RAG-Details zu praktischen AI-Workflows geschafft und können jetzt professionelle Entwicklungstools effektiv nutzen!
