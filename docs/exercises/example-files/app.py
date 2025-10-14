import streamlit as st
import os
from course_knowledge import get_system_prompt

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="AI Kurs-Assistent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_ai_client():
    """Initialisiert den AI-Client basierend auf verfügbaren API-Schlüsseln"""
    
    # Prüfe OpenAI API-Schlüssel
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            return client, "openai"
        except ImportError:
            st.error("OpenAI-Bibliothek nicht installiert!")
            return None, None
    
    elif anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            return client, "anthropic"
        except ImportError:
            st.error("Anthropic-Bibliothek nicht installiert!")
            return None, None
    
    else:
        st.error("Kein API-Schlüssel gefunden! Bitte konfigurieren Sie OPENAI_API_KEY oder ANTHROPIC_API_KEY.")
        return None, None

def get_ai_response(client, client_type, messages):
    """Sendet Anfrage an AI-Service und gibt Antwort zurück"""
    
    try:
        if client_type == "openai":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif client_type == "anthropic":
            # Anthropic erwartet separaten System-Prompt
            system_message = messages[0]["content"] if messages[0]["role"] == "system" else ""
            user_messages = [msg for msg in messages if msg["role"] != "system"]
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                system=system_message,
                messages=user_messages
            )
            return response.content[0].text
    
    except Exception as e:
        return f"Fehler bei der AI-Anfrage: {str(e)}"

def main():
    """Hauptfunktion der Streamlit-App"""
    
    # Header
    st.title("🤖 AI Kurs-Assistent")
    st.markdown("*Ihr persönlicher Assistent für den AI Development Kurs*")
    
    # Sidebar mit Informationen
    with st.sidebar:
        st.header("ℹ️ Über den Assistenten")
        st.markdown("""
        Dieser AI-Assistent kann Ihnen helfen bei:
        
        - **Kursinhalten** und Konzepten
        - **Übungen** und Aufgaben  
        - **Tools** und Technologien
        - **Troubleshooting** bei Problemen
        
        Stellen Sie einfach Ihre Frage!
        """)
        
        st.header("📚 Kursübersicht")
        st.markdown("""
        **Abend 1:** Grundlagen & No-Code  
        **Abend 2:** Python & Deployment  
        **Abend 3:** LangChain & RAG  
        **Abend 4:** Fortgeschrittene Konzepte  
        **Abend 5:** Eigene Projekte  
        **Abend 6:** Präsentationen  
        """)
    
    # AI-Client initialisieren
    client, client_type = init_ai_client()
    
    if not client:
        st.stop()
    
    # Chat-Interface
    st.header("💬 Stellen Sie Ihre Frage")
    
    # Session State für Chat-Verlauf
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat-Verlauf anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Neue Nachricht
    if prompt := st.chat_input("Ihre Frage zum AI Development Kurs..."):
        # Benutzernachricht hinzufügen
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI-Antwort generieren
        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                # Nachrichten für AI vorbereiten
                ai_messages = [
                    {"role": "system", "content": get_system_prompt()}
                ] + st.session_state.messages
                
                response = get_ai_response(client, client_type, ai_messages)
                st.markdown(response)
        
        # AI-Antwort zu Session State hinzufügen
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Reset-Button
    if st.button("🗑️ Chat zurücksetzen"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
