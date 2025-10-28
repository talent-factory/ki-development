# RAG-Anwendungsfall: Firmen-Wissensdatenbank für IT-Support

## 1. Problemstellung

**Herausforderung:** IT-Support-Teams in mittelständischen Unternehmen verbringen täglich Stunden damit, in verschiedenen Dokumenten, Wikis und Handbüchern nach Lösungen für technische Probleme zu suchen. Informationen sind über verschiedene Systeme verteilt (Confluence, SharePoint, PDF-Handbücher, Ticket-Systeme), was zu ineffizienter Problemlösung und inkonsistenten Antworten führt.

**Ziel:** Ein intelligenter Assistent, der sofort präzise Antworten auf technische Fragen liefert und dabei auf das gesamte Firmenwissen zugreift.

## 2. Zielgruppe

### Primäre Nutzer:
- **IT-Support-Mitarbeiter (Level 1 & 2)**
- **System-Administratoren**
- **Help-Desk-Teams**

### Sekundäre Nutzer:
- **Neue Mitarbeiter** (Onboarding)
- **Externe Dienstleister** (mit eingeschränktem Zugang)
- **IT-Manager** (für Reporting und Wissenslücken-Analyse)

## 3. Datenquellen

### Interne Dokumentation:
- **Confluence-Wiki** (Prozesse, Anleitungen)
- **SharePoint-Dokumente** (Policies, Standards)
- **PDF-Handbücher** (Hersteller-Dokumentation)
- **Ticket-System-Historie** (JIRA/ServiceNow)
- **Runbooks** (Incident-Response-Prozeduren)

### Externe Quellen:
- **Hersteller-Dokumentation** (Microsoft, Cisco, VMware)
- **Knowledge-Base-Artikel** (von Software-Anbietern)
- **Best-Practice-Guides** (ITIL, Security-Standards)

### Strukturierte Daten:
- **Asset-Management-System** (Hardware/Software-Inventar)
- **Monitoring-Logs** (System-Status, Alerts)
- **Konfigurationsdatenbank** (CMDB)

## 4. Beispiel-Fragen

### Technische Problemlösung:
1. **"Outlook kann keine E-Mails senden, Fehlercode 0x800CCC0E"**
2. **"Wie setze ich das Passwort für den lokalen Administrator auf Windows Server 2019 zurück?"**
3. **"VPN-Verbindung bricht nach 10 Minuten ab, was sind mögliche Ursachen?"**
4. **"Drucker HP LaserJet 4050 druckt nur leere Seiten"**
5. **"Wie konfiguriere ich VLAN 100 auf Cisco Switch 2960?"**

### Prozess-Fragen:
1. **"Welche Schritte sind bei einem Sicherheitsvorfall zu befolgen?"**
2. **"Wie beantrage ich neue Software für einen Mitarbeiter?"**
3. **"Was ist das Eskalationsverfahren bei kritischen System-Ausfällen?"**

## 5. Erwartete Antworten

### Beispiel-Antwort für Outlook-Problem:

```
**Problem:** Outlook Fehlercode 0x800CCC0E (E-Mail kann nicht gesendet werden)

**Lösungsschritte:**
1. **SMTP-Einstellungen prüfen:**
   - Server: mail.firma.com
   - Port: 587 (TLS) oder 465 (SSL)
   - Authentifizierung: Aktiviert

2. **Firewall/Antivirus prüfen:**
   - Outlook in Firewall-Ausnahmen hinzufügen
   - E-Mail-Schutz temporär deaktivieren

3. **Konto neu konfigurieren:**
   - Systemsteuerung → Mail → Profile → Neues Profil

**Eskalation:** Bei weiterhin bestehenden Problemen → Level 2 Support

**Quellen:** 
- Confluence: "E-Mail-Troubleshooting Guide v2.3"
- Ticket #INC-2023-4567 (ähnlicher Fall)
```

### Beispiel-Antwort für VLAN-Konfiguration:

```
**VLAN 100 Konfiguration auf Cisco 2960:**

**Befehle:**
```
enable
configure terminal
vlan 100
name Marketing_VLAN
exit
interface fastethernet 0/10
switchport mode access
switchport access vlan 100
exit
```

**Verifikation:**
```
show vlan brief
show interface fa0/10 switchport
```

**Sicherheitshinweis:** Änderungen nur während Wartungsfenster durchführen

**Quelle:** "Cisco Switch Configuration Standard v1.2"
```

## 6. Herausforderungen

### Technische Herausforderungen:
- **Dokumenten-Qualität:** Veraltete oder widersprüchliche Informationen
- **Multi-Format-Support:** PDFs, Word-Docs, Wiki-Markup, Code-Snippets
- **Sicherheit:** Sensible Informationen müssen geschützt bleiben
- **Performance:** Schnelle Antworten bei grossen Datenmengen

### Organisatorische Herausforderungen:
- **Daten-Governance:** Wer pflegt welche Informationen?
- **Zugriffskontrolle:** Nicht alle Infos für alle Nutzer
- **Change Management:** Akzeptanz bei IT-Teams
- **Qualitätssicherung:** Wie wird Genauigkeit sichergestellt?

### Fachliche Herausforderungen:
- **Kontext-Verständnis:** Technische Begriffe und Abkürzungen
- **Versionierung:** Verschiedene Software-Versionen berücksichtigen
- **Umgebungs-Spezifika:** Test- vs. Produktions-Umgebung
- **Compliance:** GDPR, ISO 27001, interne Richtlinien

## 7. Implementierungsstrategie

### Phase 1: MVP (3 Monate)
- **Scope:** Top 50 häufigste Support-Fragen
- **Datenquellen:** Confluence + PDF-Handbücher
- **Nutzer:** 5 Support-Mitarbeiter (Pilot)

### Phase 2: Erweiterung (6 Monate)
- **Scope:** Vollständige Wissensbasis
- **Datenquellen:** Alle internen Quellen
- **Nutzer:** Gesamtes IT-Team (20 Personen)

### Phase 3: Integration (12 Monate)
- **Scope:** Ticket-System-Integration
- **Features:** Automatische Lösungsvorschläge
- **Nutzer:** Alle IT-Stakeholder

## 8. Erfolgs-Metriken

### Effizienz-Metriken:
- **Durchschnittliche Lösungszeit:** Reduktion um 40%
- **First-Call-Resolution-Rate:** Steigerung von 60% auf 80%
- **Dokumenten-Suchzeit:** Reduktion von 15 auf 3 Minuten

### Qualitäts-Metriken:
- **Antwort-Genauigkeit:** >90% korrekte Lösungsvorschläge
- **Nutzer-Zufriedenheit:** >4.5/5 Sterne
- **Wissenslücken-Identifikation:** Automatische Erkennung fehlender Dokumentation

### Business-Metriken:
- **Kosteneinsparung:** 30% weniger Eskalationen an Level 2
- **Mitarbeiter-Produktivität:** 25% mehr gelöste Tickets pro Tag
- **Onboarding-Zeit:** 50% schnellere Einarbeitung neuer Mitarbeiter

## 9. ROI-Berechnung

### Kosten:
- **Entwicklung:** €50.000 (6 Monate)
- **Infrastruktur:** €1.000/Monat (Cloud-Hosting)
- **Wartung:** €10.000/Jahr

### Einsparungen:
- **Zeitersparnis:** 2h/Tag × 20 Mitarbeiter × €50/h = €2.000/Tag
- **Weniger Eskalationen:** €500/Monat
- **Schnelleres Onboarding:** €5.000/neuer Mitarbeiter

**ROI:** 400% im ersten Jahr

---

*Dieser Anwendungsfall zeigt, wie RAG konkrete Geschäftsprobleme lösen und messbaren Wert schaffen kann.*
