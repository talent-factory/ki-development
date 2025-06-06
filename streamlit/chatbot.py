import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

# Streamlit Konfiguration
st.set_page_config(page_title="ChatBot mit OpenAI", page_icon="🤖")
st.title("ChatBot mit OpenAI")

# Initialisiere Chat-Verlauf im Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hallo! Ich bin dein KI-Assistent. Wie kann ich dir heute helfen?"}
    ]

# Sidebar für API-Schlüssel und Einstellungen
with st.sidebar:
    st.header("Einstellungen")
    openai_api_key = st.text_input("OpenAI API-Schlüssel", type="password")
    
    # Modell-Auswahl
    model_name = st.selectbox(
        "Modell auswählen",
        ["gpt-3.5-turbo", "gpt-4"]
    )
    
    # Temperatur für Kreativität
    temperature = st.slider("Kreativität", 0.0, 1.0, 0.7, 0.1)
    
    # Clear Button
    if st.button("Chat leeren"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hallo! Ich bin dein KI-Assistent. Wie kann ich dir heute helfen?"}
        ]
        st.rerun()

# Zeige den Chat-Verlauf an
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Benutzereingabe
if prompt := st.chat_input("Schreibe deine Nachricht..."):
    if not openai_api_key:
        st.warning("Bitte gib deinen OpenAI API-Schlüssel ein.")
        st.stop()
    
    # Füge Benutzernachricht zum Verlauf hinzu
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Zeige die Benutzernachricht an
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Zeige Ladeindikator während der Antwort generiert wird
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Initialisiere das Chat-Modell
            chat = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=model_name,
                temperature=temperature,
                streaming=True
            )
            
            # Konvertiere den Nachrichtenverlauf in das Format für LangChain
            messages_for_llm = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages_for_llm.append(HumanMessage(content=msg["content"]))
                else:
                    messages_for_llm.append(AIMessage(content=msg["content"]))
            
            # Generiere die Antwort
            for chunk in chat.stream(messages_for_llm):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"Ein Fehler ist aufgetreten: {str(e)}"
            st.error(error_msg)
            full_response = error_msg
    
    # Füge die Antwort zum Verlauf hinzu
    st.session_state.messages.append({"role": "assistant", "content": full_response})
