# 📋 CLI-Installation Checkliste

**Ziel:** Systematische Installation aller AI-CLI Tools mit Troubleshooting

## 🔧 **Voraussetzungen prüfen**

### ✅ **System-Requirements**
```bash
# Node.js Version prüfen (mindestens v16)
node --version

# npm Version prüfen
npm --version

# Git Installation prüfen
git --version

# Internet-Verbindung testen
ping google.com
```

**Erwartete Ausgaben:**
- Node.js: v18.0.0 oder höher
- npm: v8.0.0 oder höher
- Git: v2.30.0 oder höher

## 🚀 **Installation Schritt-für-Schritt**

### **1. GitHub CLI (Copilot-Alternative)**

#### Installation:
```bash
# GitHub CLI installieren (falls noch nicht vorhanden)
# macOS: brew install gh
# Windows: winget install GitHub.cli
# Linux: siehe https://github.com/cli/cli#installation
```

#### Setup:
```bash
# GitHub CLI authentifizieren
gh auth login

# Copilot-Berechtigung aktivieren (optional)
gh auth refresh -s copilot
```

#### Test:
```bash
gh --version
gh repo list --limit 3
```

**Hinweis:** GitHub Copilot CLI hat derzeit Authentifizierungsprobleme. Wir nutzen GitHub CLI als stabile Alternative.

#### ✅ **Erfolgskriterium:**
- [ ] Version wird angezeigt
- [ ] Suggestion wird generiert

#### ❌ **Häufige Probleme:**
```
Problem: "gh: command not found"
Lösung: GitHub CLI installieren: https://cli.github.com/

Problem: "Authentication failed"
Lösung: gh auth login erneut ausführen

Problem: "Copilot subscription required"
Lösung: GitHub Copilot Subscription aktivieren
```

---

### **2. Claude Code CLI**

#### Installation:
```bash
npm install -g @anthropic-ai/claude-code
```

#### Setup:
```bash
claude auth login
```

#### Test:
```bash
claude --version
claude "Hello, can you help me with Python?"
```

#### ✅ **Erfolgskriterium:**
- [ ] Version wird angezeigt
- [ ] Chat-Response wird generiert

#### ❌ **Häufige Probleme:**
```
Problem: "API key required"
Lösung: Anthropic API Key in .env oder claude config setzen

Problem: "Rate limit exceeded"
Lösung: Warten oder API Key Limits prüfen

Problem: "Network error"
Lösung: Proxy-Einstellungen prüfen
```

---

### **3. Gemini CLI**

#### Installation:
```bash
npm install -g @google/gemini-cli
```

#### Setup:
```bash
# API Key setzen (aus Google AI Studio)
gemini config set api-key YOUR_GEMINI_API_KEY
```

#### Test:
```bash
gemini --version
gemini chat "What is machine learning?"
```

#### ✅ **Erfolgskriterium:**
- [ ] Version wird angezeigt
- [ ] Chat funktioniert

#### ❌ **Häufige Probleme:**
```
Problem: "Invalid API key"
Lösung: Neuen API Key aus Google AI Studio generieren

Problem: "Service unavailable"
Lösung: Region/VPN prüfen, Gemini ist nicht überall verfügbar

Problem: "Quota exceeded"
Lösung: API Limits in Google Cloud Console prüfen
```

---

### **4. Augment Code CLI**

#### Installation:
```bash
npm install -g @augmentcode/auggie
```

#### Setup:
```bash
auggie login
```

#### Test:
```bash
auggie --version
auggie status
```

#### ✅ **Erfolgskriterium:**
- [ ] Version wird angezeigt
- [ ] Status zeigt "authenticated"

#### ❌ **Häufige Probleme:**
```
Problem: "Login failed"
Lösung: Augment Code Account erstellen/verifizieren

Problem: "Command not found"
Lösung: npm global path prüfen: npm config get prefix

Problem: "Permission denied"
Lösung: sudo npm install -g (nur als letzter Ausweg)
```

## 🔍 **Finale Verifikation**

### **Alle Tools testen:**
```bash
# Versions-Check
echo "=== CLI Tools Versions ==="
gh --version 2>/dev/null && echo "✅ GitHub CLI" || echo "❌ GitHub CLI"
claude --version 2>/dev/null && echo "✅ Claude Code" || echo "❌ Claude Code"
gemini --version 2>/dev/null && echo "✅ Gemini CLI" || echo "❌ Gemini CLI"
auggie --version 2>/dev/null && echo "✅ Augment Code" || echo "❌ Augment Code"
```

### **Funktionalitäts-Test:**
```bash
# Schneller Funktionstest
gh repo list --limit 3  # GitHub CLI Test
claude "Say hello" | head -3
gemini chat "Hi there" | head -3
auggie status | head -3
```

## 🆘 **Troubleshooting Guide**

### **Allgemeine Probleme:**

#### **npm Permission Errors:**
```bash
# npm global Verzeichnis prüfen
npm config get prefix

# Falls nötig, npm prefix ändern
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### **Network/Proxy Issues:**
```bash
# npm Proxy konfigurieren (falls nötig)
npm config set proxy http://proxy.company.com:8080
npm config set https-proxy http://proxy.company.com:8080

# Proxy entfernen
npm config delete proxy
npm config delete https-proxy
```

#### **API Key Management:**
```bash
# .env Datei erstellen
touch ~/.env
echo "ANTHROPIC_API_KEY=your_key_here" >> ~/.env
echo "GOOGLE_API_KEY=your_key_here" >> ~/.env
```

### **Backup-Plan:**
Falls Installation fehlschlägt:
1. **Web-Interfaces nutzen:** claude.ai, chat.openai.com
2. **VS Code Extensions:** GitHub Copilot, Claude Dev
3. **Alternative CLIs:** ollama, openai-cli

## ✅ **Erfolgreiche Installation Checkliste**

- [ ] Alle 4 CLI-Tools installiert
- [ ] Authentifizierung für alle Tools erfolgreich
- [ ] Mindestens ein Test-Command pro Tool ausgeführt
- [ ] Keine Error-Messages bei Version-Check
- [ ] API-Keys sicher gespeichert

**Bei Problemen:** Dozent um Hilfe bitten! 🙋‍♂️
