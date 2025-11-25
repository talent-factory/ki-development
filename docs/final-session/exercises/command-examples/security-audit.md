# /security-audit: Automatische Sicherheitsanalyse

Führe eine umfassende Sicherheitsanalyse des bereitgestellten Codes durch und identifiziere potenzielle Schwachstellen.

## Aufgaben:
1. **Code-Sicherheit analysieren**
   - Input-Validierung prüfen
   - SQL-Injection Risiken identifizieren
   - XSS-Vulnerabilities finden
   - Authentication/Authorization Schwächen

2. **Dependency-Sicherheit prüfen**
   - Veraltete Packages identifizieren
   - Bekannte CVEs in Dependencies
   - Unsichere Konfigurationen

3. **Best-Practice Compliance**
   - OWASP Top 10 Compliance
   - Secure Coding Standards
   - Data Protection Compliance

## Sicherheits-Checkliste:

### 🔒 **Input Validation**
- [ ] Alle User-Inputs validiert
- [ ] Parameterized Queries verwendet
- [ ] File Upload Restrictions
- [ ] Input Sanitization implementiert

### 🛡️ **Authentication & Authorization**
- [ ] Sichere Passwort-Policies
- [ ] Session Management sicher
- [ ] JWT Token Validation
- [ ] Role-based Access Control

### 🔐 **Data Protection**
- [ ] Sensitive Daten verschlüsselt
- [ ] Sichere Datenübertragung (HTTPS)
- [ ] Keine Secrets im Code
- [ ] Secure Headers gesetzt

### ⚠️ **Common Vulnerabilities**
- [ ] SQL Injection Prevention
- [ ] XSS Protection
- [ ] CSRF Protection
- [ ] Path Traversal Prevention

## Output Format:

### Sicherheits-Report:
```markdown
# 🔒 Security Audit Report

## 📊 Zusammenfassung
- **Kritische Schwachstellen:** X
- **Hohe Risiken:** X
- **Mittlere Risiken:** X
- **Niedrige Risiken:** X
- **Sicherheits-Score:** X/100

## 🚨 Kritische Schwachstellen

### 1. SQL Injection Risk (CRITICAL)
**Datei:** `user_controller.py:45`
**Problem:** Direkte String-Interpolation in SQL Query
```python
# ❌ UNSICHER
query = f"SELECT * FROM users WHERE id = {user_id}"
```
**Lösung:**
```python
# ✅ SICHER
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

### 2. Hardcoded API Key (CRITICAL)
**Datei:** `config.py:12`
**Problem:** API Key im Quellcode
```python
# ❌ UNSICHER
API_KEY = "sk-1234567890abcdef"
```
**Lösung:**
```python
# ✅ SICHER
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY environment variable required")
```

## ⚠️ Hohe Risiken

### 1. Missing Input Validation
**Datei:** `api/routes.py:23`
**Problem:** Keine Validierung von User-Input
**Empfehlung:** Input-Schema mit Pydantic/Joi implementieren

### 2. Weak Password Policy
**Datei:** `auth/password.py:15`
**Problem:** Minimale Passwort-Anforderungen
**Empfehlung:** Stärkere Passwort-Policy implementieren

## 🔧 Empfohlene Fixes

### Sofort (Kritisch):
1. SQL Queries parametrisieren
2. Secrets aus Code entfernen
3. Input Validation implementieren

### Diese Woche (Hoch):
1. Security Headers hinzufügen
2. Rate Limiting implementieren
3. Logging für Security Events

### Nächster Sprint (Mittel):
1. Dependency Updates
2. Security Tests erweitern
3. Code Review Guidelines
```

## Beispiel-Anwendung:

### Input Code:
```python
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route('/user/<user_id>')
def get_user(user_id):
    # Potenzielle SQL Injection
    conn = sqlite3.connect('users.db')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = conn.execute(query).fetchone()
    return {"user": result}

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # Keine Input-Validierung
    # Keine Rate Limiting
    # Passwort nicht gehashed
    
    if username == "admin" and password == "password123":
        return {"token": "hardcoded-jwt-token"}
    return {"error": "Invalid credentials"}
```

### Security Audit Output:
```markdown
# 🔒 Security Audit Report

## 📊 Zusammenfassung
- **Kritische Schwachstellen:** 3
- **Hohe Risiken:** 2
- **Mittlere Risiken:** 1
- **Sicherheits-Score:** 25/100 ⚠️ KRITISCH

## 🚨 Kritische Schwachstellen

### 1. SQL Injection (Line 9)
**Risiko:** CRITICAL
**CWE:** CWE-89
**Beschreibung:** Direkte String-Interpolation ermöglicht SQL Injection
**Fix:** Parameterized Queries verwenden

### 2. Hardcoded Credentials (Line 21)
**Risiko:** CRITICAL  
**CWE:** CWE-798
**Beschreibung:** Hardcoded Admin-Passwort im Quellcode
**Fix:** Sichere Credential-Storage implementieren

### 3. Hardcoded JWT Token (Line 22)
**Risiko:** CRITICAL
**CWE:** CWE-798
**Beschreibung:** Statischer JWT Token ohne Expiration
**Fix:** Dynamische JWT-Generierung mit Expiration

## ⚠️ Hohe Risiken

### 1. Missing Input Validation
**Risiko:** HIGH
**Beschreibung:** Keine Validierung von User-Inputs
**Fix:** Input-Schema mit Flask-WTF implementieren

### 2. No Rate Limiting
**Risiko:** HIGH
**Beschreibung:** Login-Endpoint ohne Rate Limiting
**Fix:** Flask-Limiter implementieren

## 🔧 Sichere Version:
```python
from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import sqlite3
import hashlib
import jwt
import os

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/user/<int:user_id>')
def get_user(user_id):
    # ✅ Sichere parametrisierte Query
    conn = sqlite3.connect('users.db')
    query = "SELECT id, username, email FROM users WHERE id = ?"
    result = conn.execute(query, (user_id,)).fetchone()
    return {"user": result}

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    # ✅ Input Validation
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    if not username or not password:
        return {"error": "Username and password required"}, 400
    
    # ✅ Sichere Passwort-Verifikation
    user = authenticate_user(username, password)
    if user:
        token = jwt.encode({
            'user_id': user['id'],
            'exp': datetime.utcnow() + timedelta(hours=1)
        }, os.getenv('JWT_SECRET'), algorithm='HS256')
        return {"token": token}
    
    return {"error": "Invalid credentials"}, 401
```
```

## Tools Integration:
- **bandit** (Python): `bandit -r .`
- **eslint-security** (JavaScript): `npm audit`
- **semgrep**: `semgrep --config=auto .`
- **CodeQL**: GitHub Security Tab
- **Snyk**: `snyk test`
