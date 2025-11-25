# ⚡ Lektion 2: CLI-Tools in Aktion + Git Workflows

**Zeit:** 19:00 - 19:50 (50 Minuten)  
**Ziel:** Praktische Anwendung der CLI-Tools in realen Entwicklungsworkflows

## 🎯 **Lernziele**
Nach dieser Lektion können die Teilnehmenden:
- Jedes CLI-Tool für spezifische Aufgaben gezielt einsetzen
- AI-unterstützte Git-Workflows durchführen
- Die Stärken und Schwächen verschiedener Tools bewerten
- Tools in bestehende Entwicklungsprozesse integrieren

## ⏰ **Zeitplan (50 Min)**

### 🚀 **Phase 1: CLI-Tools Hands-On (20 Min)**
```
19:00-19:20  Live-Coding mit allen 4 Tools
```

#### **GitHub Copilot CLI - Code-Suggestions (5 Min)**
```bash
# Praktische Szenarien
gh copilot suggest "deploy a Python app to AWS Lambda"
gh copilot suggest "create a Docker container for a React app"
gh copilot explain "git rebase -i HEAD~3"

# Live-Übung für Teilnehmende
gh copilot suggest "setup CI/CD pipeline for Python project"
```

**Pädagogischer Fokus:** Zeigen, wie Copilot komplexe DevOps-Aufgaben vereinfacht.

#### **Claude Code - Konversationelle Entwicklung (5 Min)**
```bash
# Interaktive Code-Analyse
claude "Analyze this Python function for performance issues"
claude "Refactor this code to use modern Python patterns"
claude "Generate unit tests for this function"

# Live-Übung
claude "Help me optimize this database query"
```

**Pädagogischer Fokus:** Konversationeller Ansatz vs. Command-basiert.

#### **Gemini CLI - Multimodale Analyse (5 Min)**
```bash
# Text-Generierung
gemini generate "Write a README for a Python ML project"
gemini chat "Explain the difference between REST and GraphQL"

# Live-Übung
gemini generate "Create a Python script for data visualization"
```

**Pädagogischer Fokus:** Google's Stärken in Wissensverarbeitung.

#### **Augment Code - Workflow-Automation (5 Min)**
```bash
# Workflow-Management
auggie workflow list
auggie workflow create "python-setup"
auggie status

# Live-Übung
auggie workflow run "test-and-deploy"
```

**Pädagogischer Fokus:** Team-orientierte Entwicklung und Automation.

### 🔄 **Phase 2: AI-unterstützte Git Workflows (20 Min)**
```
19:20-19:40  Git + AI = Produktivitätsboost
```

#### **Intelligente Commit Messages (8 Min)**
```bash
# Traditionell vs. AI-unterstützt
git add .
git commit -m "fix stuff"  # ❌ Schlecht

# Mit AI-Unterstützung
/commit "Implement user authentication with JWT tokens"  # ✅ Besser
gh copilot suggest "write a good commit message for authentication feature"
```

**Live-Demo:** Dozent zeigt schlechte vs. gute Commit Messages.

#### **Pull Request Automation (8 Min)**
```bash
# PR-Erstellung mit Kontext
/create-pr "Feature: Add user authentication system

This PR implements:
- JWT token-based authentication
- User registration and login endpoints
- Password hashing with bcrypt
- Session management

Closes #123"
```

**Hands-On:** Teilnehmende erstellen eigene PR mit AI-Hilfe.

#### **Code Review mit AI (4 Min)**
```bash
# Code-Analyse vor Commit
claude "Review this code for security issues"
gh copilot explain "What does this function do?"
```

### 🔧 **Phase 3: Integration in Entwicklungsworkflows (10 Min)**
```
19:40-19:50  Workflow-Integration & Best Practices
```

#### **Workflow-Vergleichstabelle (5 Min)**

| Aufgabe | Traditionell | Mit AI-CLI | Zeitersparnis |
|---------|-------------|------------|---------------|
| **Commit Message** | 2-3 Min denken | `/commit` 30 Sek | 80% |
| **PR Description** | 5-10 Min schreiben | `/create-pr` 2 Min | 70% |
| **Code Erklärung** | Dokumentation suchen | `gh copilot explain` | 90% |
| **Deployment** | Docs lesen | `gh copilot suggest` | 60% |

#### **Integration Best Practices (5 Min)**
```bash
# Täglicher Workflow mit AI
1. claude "Plan today's development tasks"
2. gh copilot suggest "setup development environment"
3. # Entwicklung mit AI-Unterstützung
4. /commit "Descriptive commit message"
5. /create-pr "Detailed PR description"
```

**Diskussion:** "Wie würdet ihr diese Tools in euren aktuellen Workflow integrieren?"

## 🎓 **Lernzielkontrolle**

### ✅ **Praktische Übung (Alle Teilnehmenden)**
```
Aufgabe: Erstellt einen kleinen Python-Script mit AI-Hilfe
1. claude "Create a Python script that reads CSV and creates a chart"
2. Implementiert den Code
3. gh copilot suggest "add error handling to Python script"
4. /commit "Add data visualization script with error handling"
```

### 🔍 **Erfolgskriterien**
- [ ] Mindestens 2 CLI-Tools praktisch angewendet
- [ ] Einen AI-generierten Commit erstellt
- [ ] Workflow-Integration verstanden
- [ ] Tool-Vergleich kann artikuliert werden

## 💡 **Reflexionsfragen**
1. "Welches Tool passt am besten zu eurem Arbeitsstil?"
2. "Wo seht ihr die grössten Produktivitätsgewinne?"
3. "Welche Bedenken habt ihr bei der AI-Integration?"

## 🔗 **Übergang zu Lektion 3**
"Ihr habt jetzt die Grundlagen gemeistert. Als nächstes schauen wir uns die neuesten Entwicklungen in der AI-Welt an und wie sie unsere Arbeit beeinflussen werden!"

---

**Pause: 19:50 - 20:00 (10 Minuten)**
