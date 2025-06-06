import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langdetect import detect

# Streamlit Konfiguration
st.set_page_config(page_title="Deutsch-Englisch Übersetzer", page_icon="🌍")
st.title("Deutsch-Englisch Übersetzer")

# Initialisiere Chat-Verlauf
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar für API-Schlüssel und Einstellungen
with st.sidebar:
    st.header("Einstellungen")
    openai_api_key = st.text_input("OpenAI API-Schlüssel", type="password")
    
    # Übersetzungsrichtung
    translation_direction = st.radio(
        "Übersetzungsrichtung",
        ["Automatisch erkennen", "Deutsch → Englisch", "Englisch → Deutsch"],
        index=0
    )

# Funktion zur Spracherkennung
def detect_language(text):
    try:
        lang = detect(text)
        return "de" if lang == "de" else "en"
    except:
        return "en"

# Funktion zur Übersetzung
def translate_text(text, source_lang, target_lang, api_key):
    try:
        chat = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        system_message = f"""
        Du bist ein professioneller Übersetzer. Übersetze den folgenden Text von {source_lang} nach {target_lang}.
        Gib NUR die Übersetzung zurück, ohne zusätzlichen Text oder Anführungszeichen.
        """
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=text)
        ]
        
        response = chat.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"Fehler bei der Übersetzung: {str(e)}"

# Hauptbereich
st.write("Geben Sie einen Text ein, den Sie übersetzen möchten:")

# Texteingabe
user_input = st.text_area("Eingabe", height=150, label_visibility="collapsed")

if st.button("Übersetzen"):
    if not openai_api_key:
        st.warning("Bitte geben Sie Ihren OpenAI API-Schlüssel in der Seitenleiste ein.")
        st.stop()
    
    if not user_input.strip():
        st.warning("Bitte geben Sie einen Text zum Übersetzen ein.")
        st.stop()
    
    # Bestimme die Übersetzungsrichtung
    if translation_direction == "Automatisch erkennen":
        detected_lang = detect_language(user_input)
        source_lang = "Deutsch" if detected_lang == "de" else "Englisch"
        target_lang = "Englisch" if detected_lang == "de" else "Deutsch"
    else:
        if "Deutsch → Englisch" in translation_direction:
            source_lang, target_lang = "Deutsch", "Englisch"
        else:
            source_lang, target_lang = "Englisch", "Deutsch"
    
    # Übersetzung durchführen
    with st.spinner(f"Übersetze von {source_lang} nach {target_lang}..."):
        translated_text = translate_text(
            user_input, 
            source_lang, 
            target_lang, 
            openai_api_key
        )
    
    # Ergebnis anzeigen
    st.subheader("Übersetzung:")
    st.write(translated_text)
    
    # Verlauf aktualisieren
    st.session_state.messages.append({
        "original": user_input,
        "translated": translated_text,
        "direction": f"{source_lang} → {target_lang}"
    })

# Verlauf anzeigen (max. 5 Einträge)
if st.session_state.messages:
    st.divider()
    st.subheader("Letzte Übersetzungen")
    
    for i, msg in enumerate(reversed(st.session_state.messages[-5:]), 1):
        with st.expander(f"Übersetzung {i} ({msg['direction']})"):
            st.write(f"**Original:** {msg['original']}")
            st.write(f"**Übersetzung:** {msg['translated']}")
