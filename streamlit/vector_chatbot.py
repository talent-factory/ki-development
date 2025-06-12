import os
import tempfile
from typing import List
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

# Streamlit Konfiguration
st.set_page_config(
    page_title="CharBot mit Dokumentenanalyse",
    page_icon="🤖",
    layout="wide"
)

# Initialisiere Session State Variablen
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hallo! Ich bin dein KI-Assistent. Lade Dokumente hoch, um damit zu interagieren."}
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

init_session_state()

# Hilfsfunktion zum Laden von Dokumenten
def load_document(file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif file.name.lower().endswith('.txt'):
            loader = TextLoader(tmp_file_path)
        else:
            # Versuche unstrukturierte Dateien zu laden
            loader = UnstructuredFileLoader(tmp_file_path)
        
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {str(e)}")
        return []
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Dokumente verarbeiten und in Vektorspeicher laden
def process_documents(documents: List[Document], openai_api_key: str):
    if not documents:
        return None
    
    # Dokumente in Chunks aufteilen
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Vektorspeicher erstellen
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

# Konversationskette initialisieren
def init_conversation_chain(vectorstore, model_name: str, temperature: float, openai_api_key: str):
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain

# UI Komponenten
st.title("🤖 CharBot mit Dokumentenanalyse")

# Sidebar für Einstellungen und Dokumentenupload
with st.sidebar:
    st.header("Einstellungen")
    
    # API-Schlüssel
    openai_api_key = st.text_input(
        "OpenAI API-Schlüssel",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    
    # Modell-Auswahl
    model_name = st.selectbox(
        "Modell auswählen",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    
    # Temperatur
    temperature = st.slider(
        "Kreativität",
        0.0, 1.0, 0.7, 0.1,
        help="Höhere Werte machen die Antworten kreativer, können aber auch ungenauer werden."
    )
    
    # Dokumenten-Upload
    st.markdown("---")
    st.subheader("Dokumente hochladen")
    uploaded_files = st.file_uploader(
        "Lade Dokumente hoch (PDF, TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    process_button = st.button("Dokumente verarbeiten")
    
    if process_button and uploaded_files:
        if not openai_api_key:
            st.error("Bitte gib deinen OpenAI API-Schlüssel ein.")
        else:
            with st.spinner("Verarbeite Dokumente..."):
                all_documents = []
                for uploaded_file in uploaded_files:
                    documents = load_document(uploaded_file)
                    all_documents.extend(documents)
                
                if all_documents:
                    vectorstore = process_documents(all_documents, openai_api_key)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.conversation_chain = init_conversation_chain(
                            vectorstore, model_name, temperature, openai_api_key
                        )
                        st.success(f"{len(all_documents)} Dokument(e) erfolgreich geladen und verarbeitet!")
                else:
                    st.error("Keine Dokumente konnten verarbeitet werden.")
    
    # Chat leeren Button
    if st.button("Chat leeren"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hallo! Ich bin dein KI-Assistent. Lade Dokumente hoch, um damit zu interagieren."}
        ]
        st.rerun()

# Chat-Container
chat_container = st.container()

# Chat-Verlauf anzeigen
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Benutzereingabe
if prompt := st.chat_input("Stelle eine Frage zu deinen Dokumenten..."):
    if not openai_api_key:
        st.error("Bitte gib deinen OpenAI API-Schlüssel in der Seitenleiste ein.")
        st.stop()
    
    if not st.session_state.vectorstore:
        st.error("Bitte lade zuerst Dokumente hoch und klicke auf 'Dokumente verarbeiten'.")
        st.stop()
    
    # Benutzernachricht zum Verlauf hinzufügen
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Benutzernachricht anzeigen
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Antwort generieren
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Konversationskette aufrufen
                # Use invoke() instead of __call__ and handle the output properly
                result = st.session_state.conversation_chain.invoke({"question": prompt})
                full_response = result["answer"]
                
                # Quellen anzeigen
                if "source_documents" in result and result["source_documents"]:
                    full_response += "\n\n**Quellen:**\n"
                    sources = set()
                    for doc in result["source_documents"]:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            source = doc.metadata['source']
                            if source not in sources:
                                sources.add(source)
                                full_response += f"- {os.path.basename(source)}\n"
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_msg = f"Ein Fehler ist aufgetreten: {str(e)}"
                st.error(error_msg)
                full_response = error_msg
    
    # Antwort zum Verlauf hinzufügen
    st.session_state.messages.append({"role": "assistant", "content": full_response})
