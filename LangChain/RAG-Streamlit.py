import os

import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI


# Funktion zum Extrahieren des Textes aus dem PDF
@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with st.spinner(f"Extrahiere Text aus {pdf.name}..."):
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text


# Funktion zum Aufteilen des Textes in Chunks
@st.cache_data
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Funktion zum Erstellen des Vektorspeichers
@st.cache_resource
def get_vectorstore(text_chunks):
    with st.spinner("Erstelle Vektorspeicher..."):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore


# Funktion zum Erstellen der Konversationskette
@st.cache_resource
def get_conversation_chain(vectorstore):
    with st.spinner("Erstelle Konversationsmodell..."):
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain


# Hauptfunktion für die Streamlit-App
def main():
    st.set_page_config(page_title="FAQ - Häufig gestellte Fragen", page_icon=":books:")
    st.header("FAQ - Häufig gestellte Fragen 💬")

    # PDF-Upload
    pdf_docs = st.file_uploader("Lade alle relevanten PDF-Dateien hoch", accept_multiple_files=True)

    if pdf_docs:
        # Text extrahieren
        raw_text = get_pdf_text(pdf_docs)

        # Text in Chunks aufteilen
        text_chunks = get_text_chunks(raw_text)

        # Vektorspeicher erstellen
        vectorstore = get_vectorstore(text_chunks)

        # Konversationskette erstellen
        conversation = get_conversation_chain(vectorstore)

        # Chat-Interface
        user_question = st.text_input("Stelle eine Frage zu deinem Dokument:")
        if user_question:
            response = conversation({"question": user_question})
            st.write(response["answer"])


if __name__ == '__main__':
    main()
