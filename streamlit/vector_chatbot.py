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
load_dotenv(override=True)

# Hole API-Key aus .env Datei
env_api_key = os.getenv("OPENAI_API_KEY", "")

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
            {"role": "assistant",
             "content": "Hallo! Ich bin dein KI-Assistent. Lade Dokumente hoch, um damit zu interagieren."}
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None


init_session_state()


# Hilfsfunktion zum Laden von Dokumenten
def load_document(file) -> List[Document]:
    filename = file.name
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Füge Metadaten für PDF-Dokumente hinzu
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': filename,
                    'page': i + 1,  # PDF-Seitennummerierung beginnt bei 1
                    'document_type': 'pdf',
                    'total_pages': len(documents)
                })
                
        elif filename.lower().endswith('.txt'):
            loader = TextLoader(tmp_file_path)
            documents = loader.load()
            
            # Füge Metadaten für Textdokumente hinzu
            for doc in documents:
                doc.metadata.update({
                    'source': filename,
                    'document_type': 'text',
                    'page': 1  # Textdateien haben keine Seitenzahlen
                })
        else:
            # Versuche unstrukturierte Dateien zu laden
            loader = UnstructuredFileLoader(tmp_file_path)
            documents = loader.load()
            
            # Füge grundlegende Metadaten für andere Dokumenttypen hinzu
            for doc in documents:
                doc.metadata.update({
                    'source': filename,
                    'document_type': 'other'
                })

        return documents
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei {filename}: {str(e)}")
        return []
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


# Dokumente verarbeiten und in Vektorspeicher laden
def process_documents(documents: List[Document], openai_api_key: str):
    if not documents:
        st.warning("Keine Dokumente zum Verarbeiten gefunden.")
        return None

    try:
        # Dokumente in Chunks aufteilen
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.error("Konnte keine verwertbaren Textteile aus den Dokumenten extrahieren.")
            return None
            
        # Überprüfen, ob der Text in den Chunks nicht leer ist
        valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not valid_chunks:
            st.error("Die Dokumente enthalten keinen lesbaren Text.")
            return None

        # Vektorspeicher erstellen
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Teste die Embedding-Erstellung mit einem kleinen Text
        try:
            test_embedding = embeddings.embed_query("Test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Ungültige Embedding-Antwort vom API-Server")
        except Exception as e:
            st.error(f"Fehler beim Testen der Embedding-API: {str(e)}")
            return None
            
        vectorstore = FAISS.from_documents(valid_chunks, embeddings)
        return vectorstore
        
    except Exception as e:
        st.error(f"Fehler bei der Dokumentenverarbeitung: {str(e)}")
        return None


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
        value=env_api_key or "",
        help="Legen Sie den API-Schlüssel in der .env Datei fest oder geben Sie ihn hier ein."
    )
    
    # Zeige Warnung, falls kein API-Key vorhanden ist
    if not (env_api_key or openai_api_key):
        st.warning("Bitte geben Sie einen OpenAI API-Schlüssel ein oder konfigurieren Sie ihn in der .env Datei.")

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
        # Verwende den eingegebenen Schlüssel oder den aus der .env Datei
        effective_api_key = openai_api_key or env_api_key
        if not effective_api_key:
            st.error("Bitte geben Sie einen gültigen OpenAI API-Schlüssel ein.")
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
                            vectorstore, model_name, temperature, effective_api_key
                        )
                        st.success(f"{len(all_documents)} Dokument(e) erfolgreich geladen und verarbeitet!")
                else:
                    st.error("Keine Dokumente konnten verarbeitet werden.")

    # Chat leeren Button
    if st.button("Chat leeren"):
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hallo! Ich bin dein KI-Assistent. Lade Dokumente hoch, um damit zu interagieren."}
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
    # Verwende den eingegebenen Schlüssel oder den aus der .env Datei
    effective_api_key = openai_api_key or env_api_key
    if not effective_api_key:
        st.error("Bitte geben Sie einen gültigen OpenAI API-Schlüssel in der Seitenleiste ein oder konfigurieren Sie ihn in der .env Datei.")
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

                # Quellen mit Metadaten anzeigen
                if "source_documents" in result and result["source_documents"]:
                    full_response += "\n\n**Quellen:**\n"
                    sources = set()
                    
                    for doc in result["source_documents"]:
                        if not hasattr(doc, 'metadata'):
                            continue
                            
                        metadata = doc.metadata
                        source_info = []
                        
                        # Dokumentname
                        doc_name = os.path.basename(metadata.get('source', 'Unbekannte Quelle'))
                        source_info.append(f"**{doc_name}**")
                        
                        # Seitenzahl (falls vorhanden)
                        if 'page' in metadata and metadata['page']:
                            source_info.append(f"Seite {metadata['page']}")
                            
                            # Gesamtseitenzahl (falls vorhanden)
                            if 'total_pages' in metadata and metadata['total_pages'] > 1:
                                source_info[-1] += f" von {metadata['total_pages']}"
                        
                        # Dokumenttyp (falls vorhanden)
                        if 'document_type' in metadata:
                            doc_type = metadata['document_type'].upper()
                            source_info.append(f"({doc_type})")
                        
                        source_str = ", ".join(source_info)
                        
                        if source_str not in sources:
                            sources.add(source_str)
                            full_response += f"- {source_str}\n"

                message_placeholder.markdown(full_response)

            except Exception as e:
                error_msg = f"Ein Fehler ist aufgetreten: {str(e)}"
                st.error(error_msg)
                full_response = error_msg

    # Antwort zum Verlauf hinzufügen
    st.session_state.messages.append({"role": "assistant", "content": full_response})
